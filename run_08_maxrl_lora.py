#!/usr/bin/env python3
"""
run_08_maxrl_lora.py

MaxRL-style RLVR with LoRA for the ATP curriculum pipeline.

This script consumes a frontier theorem pool, starts from the best LoRA SFT
adapter produced by run_07, samples whole-proof rollouts, verifies them with
Lean, computes MaxRL advantages, and updates only the LoRA adapter.

Pipeline context:
  run_04: mine frontier tasks and search jobs
  run_05: solve holes/prefixes with encoder-guided search
  run_06: assemble verified trajectories
  run_07: LoRA SFT on verified trajectories
  run_08: LoRA MaxRL/RLVR on frontier tasks

Typical use:
  python run_08_maxrl_lora.py \
    --frontier-train-path outputs/run_04_mine_failed_problems/frontier_theorems_train.jsonl \
    --policy-adapter-path outputs/run_07_sft_lora/best_adapter \
    --base-model deepseek-ai/DeepSeek-Prover-V1.5-SFT \
    --lean-project-root . \
    --output-dir outputs/run_08_maxrl_lora

Dry run:
  python run_08_maxrl_lora.py --frontier-train-path data/frontier.jsonl --dry-run

Notes:
  * MaxRL differs from GRPO mainly in the group advantage normalization:
      GRPO:  (r_i - mean_reward) / std_reward
      MaxRL: (r_i - mean_reward) / mean_reward
    Groups with zero successes are assigned zero advantage and skipped.
  * This script implements a clipped PPO/GRPO-style token-ratio objective with
    MaxRL advantages and optional KL-to-reference/old-policy regularization.
  * The default reference mode is old_policy to avoid loading a second 7B model.
    If memory allows, use --reference-mode separate_model and provide a frozen
    reference adapter/model.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------
# Config
# -----------------------------


@dataclass
class ExperimentConfig:
    name: str = "run_08_maxrl_lora_v1"
    seed: int = 42
    dry_run: bool = False
    debug: bool = False
    resume: bool = True


@dataclass
class ModelConfig:
    base_model_name_or_path: str = "deepseek-ai/DeepSeek-Prover-V1.5-SFT"
    tokenizer_name_or_path: Optional[str] = None
    policy_adapter_path: Optional[str] = "outputs/run_07_sft_lora/best_adapter"
    reference_adapter_path: Optional[str] = "outputs/run_07_sft_lora/best_adapter"
    reference_model_name_or_path: Optional[str] = None
    reference_mode: str = "old_policy"  # old_policy | separate_model | none
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    use_cache: bool = False
    device_map: Optional[str] = None


@dataclass
class LoraConfigSpec:
    enabled: bool = True
    mode: str = "continue_sft_adapter"  # continue_sft_adapter | fresh_adapter
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    modules_to_save: Optional[List[str]] = None
    use_qlora: bool = False
    qlora_compute_dtype: str = "bfloat16"
    qlora_quant_type: str = "nf4"
    qlora_double_quant: bool = True


@dataclass
class DataConfig:
    frontier_train_path: str = "outputs/run_04_mine_failed_problems/frontier_theorems_train.jsonl"
    heldout_path: Optional[str] = None
    max_train_theorems: Optional[int] = 10000
    max_eval_theorems: int = 300
    exclude_ids_path: Optional[str] = None
    prompt_field: Optional[str] = None
    formal_statement_field: str = "formal_statement"
    header_field: str = "imports"
    theorem_id_field: str = "parent_theorem_id"
    allow_final_trajectories_as_tasks: bool = True
    shuffle: bool = True


@dataclass
class FrontierFilterConfig:
    enabled: bool = False
    cached_estimates_path: Optional[str] = None
    estimate_pass_rate_before_training: bool = False
    estimate_rollouts_per_theorem: int = 16
    keep_min_successes: int = 1
    keep_max_successes: int = 8
    hard_zero_success_fraction: float = 0.05
    easy_all_success_fraction: float = 0.0
    refresh_every_steps: int = 200


@dataclass
class PromptConfig:
    format: str = "deepseek_prover_completion"
    include_imports: bool = True
    include_natural_language_statement: bool = False
    end_with: str = ":= by"
    system_prefix: str = "Complete the following Lean 4 theorem. Return only Lean proof code."
    strip_comments: bool = False


@dataclass
class RolloutConfig:
    rollouts_per_prompt: int = 16
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    do_sample: bool = True
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\ntheorem", "\n\nlemma", "#check"])
    forbid_strings: List[str] = field(default_factory=lambda: ["sorry", "admit"])
    generation_batch_size: int = 8
    prompt_max_length: int = 3072
    max_total_length: int = 4096
    pad_to_multiple_of: int = 8


@dataclass
class RewardConfig:
    type: str = "lean_binary"
    require_compiles: bool = True
    require_no_sorry: bool = True
    require_no_admit: bool = True
    syntax_error_reward: float = 0.0
    timeout_reward: float = 0.0
    success_reward: float = 1.0
    lean_infra_error_policy: str = "retry_then_skip"  # retry_then_skip | fail | reward_zero
    retries_on_infra_error: int = 1


@dataclass
class LeanConfig:
    project_root: str = "."
    command: str = "lake env lean"
    max_workers: int = 16
    timeout_sec: int = 20
    cache_results: bool = True
    temp_dir: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class MaxRLConfig:
    enabled: bool = True
    advantage: str = "mean_reward_normalized"
    skip_zero_success_groups: bool = True
    epsilon_mean_reward: float = 1.0e-6
    advantage_clip: float = 8.0
    use_control_variate: bool = True
    normalize_group_advantages_after_clip: bool = False


@dataclass
class LossConfig:
    objective: str = "maxrl_clipped_policy_gradient"
    token_level_ratio: bool = True
    sequence_level_advantage: bool = True
    clip_range: float = 0.2
    kl_to_reference: bool = True
    kl_beta: float = 0.02
    adaptive_kl: bool = True
    kl_target: float = 0.05
    kl_horizon: int = 100
    completion_only_loss: bool = True
    max_ratio: float = 10.0


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/run_08_maxrl_lora"
    overwrite_output_dir: bool = False
    max_rl_steps: int = 1000
    prompts_per_step: int = 8
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    ppo_epochs_per_batch: int = 1
    learning_rate: float = 1.0e-5
    min_learning_rate: float = 1.0e-6
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 20
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optimizer: str = "paged_adamw_8bit"
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 5
    max_seq_length: int = 4096
    tf32: bool = True
    torch_compile: bool = False
    dataloader_seed_offset: int = 0


@dataclass
class EvalConfig:
    run_eval: bool = True
    eval_steps: int = 100
    rollout_passk: List[int] = field(default_factory=lambda: [1, 8, 32])
    eval_temperature: float = 0.7
    eval_top_p: float = 0.95
    eval_max_new_tokens: int = 1024
    max_eval_theorems: int = 300
    generation_batch_size: int = 8
    stop_after_first_success_for_passk: bool = False


@dataclass
class CheckpointSelectionConfig:
    primary_metric: str = "heldout_pass_at_8"
    secondary_metric: str = "heldout_pass_at_32"
    reject_if_kl_above: float = 0.15
    reject_if_pass32_drops_more_than: float = 0.03
    save_best_adapter: bool = True


@dataclass
class OutputConfig:
    rollout_logs: str = "rollout_logs.jsonl"
    reward_logs: str = "reward_logs.jsonl"
    train_metrics: str = "train_metrics.jsonl"
    eval_reports: str = "eval_passk_reports.jsonl"
    selected_adapter_dir: str = "best_maxrl_adapter"
    config_file: str = "resolved_config.yaml"
    tasks_file: str = "rl_train_tasks.jsonl"


@dataclass
class RunConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    data: DataConfig = field(default_factory=DataConfig)
    frontier_filter: FrontierFilterConfig = field(default_factory=FrontierFilterConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    lean: LeanConfig = field(default_factory=LeanConfig)
    maxrl: MaxRLConfig = field(default_factory=MaxRLConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint_selection: CheckpointSelectionConfig = field(default_factory=CheckpointSelectionConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)


# -----------------------------
# Utilities
# -----------------------------


def dataclass_to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def dict_to_dataclass(cls, data: Dict[str, Any]):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        field_type = f.type
        default_obj = getattr(cls(), f.name) if dataclasses.is_dataclass(cls) else None
        if dataclasses.is_dataclass(default_obj) and isinstance(val, dict):
            kwargs[f.name] = dict_to_dataclass(type(default_obj), val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def normalize_ws(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").replace("\r\n", "\n").split("\n")).strip()


def contains_forbidden(text: str, forbid: Sequence[str]) -> bool:
    low = text.lower()
    return any(s.lower() in low for s in forbid)


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml_config(path: Path, cfg: RunConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclass_to_dict(cfg)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        write_json(path.with_suffix(".json"), data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# -----------------------------
# Task representation
# -----------------------------


@dataclass
class RLTask:
    task_id: str
    prompt: str
    formal_statement: str
    header: str = ""
    natural_language_statement: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def first_present(row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return default


def format_import_header(value: Any) -> str:
    """Normalize dataset import metadata into Lean header lines."""
    if value is None:
        return ""
    raw_lines: List[str] = []
    if isinstance(value, str):
        raw_lines = [line.strip() for line in normalize_ws(value).splitlines()]
    elif isinstance(value, (list, tuple)):
        raw_lines = [str(x).strip() for x in value if str(x).strip()]
    else:
        return ""

    lines: List[str] = []
    for line in raw_lines:
        if not line:
            continue
        if re.match(r"^(import|open|set_option|namespace|universe)\b", line):
            lines.append(line)
        elif re.match(r"^[A-Za-z0-9_.]+$", line):
            lines.append(f"import {line}")
        else:
            lines.append(line)
    return "\n".join(lines)


def header_from_row(row: Dict[str, Any], cfg: RunConfig) -> str:
    for key in [cfg.data.header_field, "header", "imports", "import_header"]:
        header = format_import_header(row.get(key))
        if header:
            return header
    return ""


def split_theorem_prompt_from_code(code: str) -> Tuple[str, str]:
    code = normalize_ws(code)
    marker = re.search(r":=\s*by\b", code)
    if marker:
        prompt = code[: marker.end()].rstrip() + "\n"
        completion = code[marker.end():].strip("\n")
        return prompt, completion
    return code + "\n", ""


def make_prompt_from_row(row: Dict[str, Any], cfg: RunConfig) -> Optional[RLTask]:
    dc = cfg.data
    pc = cfg.prompt

    task_id = str(
        row.get(dc.theorem_id_field)
        or row.get("theorem_id")
        or row.get("id")
        or row.get("item_id")
        or row.get("job_id")
        or stable_hash(json.dumps(row, sort_keys=True, ensure_ascii=False))
    )

    if dc.prompt_field and isinstance(row.get(dc.prompt_field), str):
        prompt = row[dc.prompt_field].strip()
        if not prompt.endswith("\n"):
            prompt += "\n"
        formal_statement = row.get(dc.formal_statement_field, "") or prompt
        header = header_from_row(row, cfg)
        return RLTask(task_id=task_id, prompt=prompt, formal_statement=formal_statement, header=header, metadata=row)

    for key in ["prompt", "theorem_prompt", "input", "query"]:
        if isinstance(row.get(key), str) and row[key].strip():
            prompt = row[key].strip()
            if not prompt.endswith("\n"):
                prompt += "\n"
            formal_statement = row.get(dc.formal_statement_field, "") or prompt
            header = header_from_row(row, cfg)
            return RLTask(task_id=task_id, prompt=prompt, formal_statement=formal_statement, header=header, metadata=row)

    full_code = first_present(row, ["full_theorem_code", "theorem_code", "code", "lean_code"])
    if full_code:
        prompt, _completion = split_theorem_prompt_from_code(full_code)
        return RLTask(task_id=task_id, prompt=prompt, formal_statement=prompt, header="", metadata=row)

    formal_statement = first_present(row, [dc.formal_statement_field, "formal_statement", "statement", "theorem"])
    if not formal_statement:
        return None

    header = header_from_row(row, cfg)
    nl = first_present(row, ["natural_language_statement", "nl_statement", "problem"])

    stmt = normalize_ws(formal_statement)
    prompt_stmt, _existing_completion = split_theorem_prompt_from_code(stmt)
    if _existing_completion:
        stmt = prompt_stmt.rstrip()
    elif not re.search(r":=\s*by\b", stmt):
        # If the statement ends before the proof body, append := by.
        stmt = stmt.rstrip()
        if stmt.endswith(":="):
            stmt += " by"
        elif not stmt.endswith("by"):
            stmt += " := by"

    parts = []
    if pc.include_imports and header:
        parts.append(normalize_ws(header))
    if pc.include_natural_language_statement and nl:
        parts.append("/- " + nl.replace("-/", "") + " -/")
    parts.append(stmt)
    prompt = "\n\n".join(p for p in parts if p).strip() + "\n"
    return RLTask(task_id=task_id, prompt=prompt, formal_statement=stmt, header=header, natural_language_statement=nl, metadata=row)


def load_tasks(cfg: RunConfig) -> List[RLTask]:
    path = Path(cfg.data.frontier_train_path)
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"No tasks found at {path}")

    exclude = set()
    if cfg.data.exclude_ids_path:
        for row in read_jsonl(Path(cfg.data.exclude_ids_path)):
            for key in ["id", "theorem_id", "parent_theorem_id", "task_id"]:
                if row.get(key) is not None:
                    exclude.add(str(row[key]))

    tasks: List[RLTask] = []
    seen = set()
    for row in rows:
        task = make_prompt_from_row(row, cfg)
        if task is None:
            continue
        if task.task_id in exclude:
            continue
        h = stable_hash(task.prompt)
        if h in seen:
            continue
        seen.add(h)
        tasks.append(task)

    if cfg.data.shuffle:
        random.shuffle(tasks)
    if cfg.data.max_train_theorems:
        tasks = tasks[: cfg.data.max_train_theorems]
    return tasks


def load_eval_tasks(cfg: RunConfig) -> List[RLTask]:
    if cfg.data.heldout_path:
        rows = read_jsonl(Path(cfg.data.heldout_path))
        tasks = []
        seen = set()
        for row in rows:
            task = make_prompt_from_row(row, cfg)
            if task is None:
                continue
            h = stable_hash(task.prompt)
            if h in seen:
                continue
            seen.add(h)
            tasks.append(task)
        return tasks[: cfg.eval.max_eval_theorems]
    train = load_tasks(cfg)
    return train[-min(len(train), cfg.eval.max_eval_theorems):]


def task_metadata_counts(tasks: Sequence[RLTask], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task in tasks:
        value = task.metadata.get(key, "unknown")
        if value is None or value == "":
            value = "unknown"
        value = str(value)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


# -----------------------------
# Lean verifier
# -----------------------------


@dataclass
class LeanResult:
    ok: bool
    reward: Optional[float]
    status: str
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    elapsed_sec: float = 0.0


class LeanVerifier:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.cache: Dict[str, LeanResult] = {}
        self.project_root = Path(cfg.lean.project_root).resolve()
        self.temp_dir = Path(cfg.lean.temp_dir).resolve() if cfg.lean.temp_dir else self.project_root / ".run08_tmp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def assemble_code(self, prompt: str, completion: str) -> str:
        text = prompt.rstrip() + "\n" + completion.strip() + "\n"
        return text

    def verify_one(self, prompt: str, completion: str) -> LeanResult:
        code = self.assemble_code(prompt, completion)
        if self.cfg.reward.require_no_sorry and re.search(r"\bsorry\b", code):
            return LeanResult(False, 0.0, "contains_sorry")
        if self.cfg.reward.require_no_admit and re.search(r"\badmit\b", code):
            return LeanResult(False, 0.0, "contains_admit")

        key = stable_hash(code)
        if self.cfg.lean.cache_results and key in self.cache:
            return self.cache[key]

        result = self._run_lean(code, key)
        if self.cfg.lean.cache_results:
            self.cache[key] = result
        return result

    def _run_lean(self, code: str, key: str) -> LeanResult:
        path = self.temp_dir / f"run08_{key}.lean"
        path.write_text(code, encoding="utf-8")
        cmd = self.cfg.lean.command.split() + [str(path)]
        env = os.environ.copy()
        env.update(self.cfg.lean.extra_env)
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.cfg.lean.timeout_sec,
            )
            elapsed = time.time() - start
            ok = proc.returncode == 0
            status = "ok" if ok else "lean_error"
            reward = self.cfg.reward.success_reward if ok else 0.0
            return LeanResult(ok, reward, status, proc.stdout, proc.stderr, proc.returncode, elapsed)
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            return LeanResult(False, self.cfg.reward.timeout_reward, "timeout", e.stdout or "", e.stderr or "", None, elapsed)
        except Exception as e:  # pragma: no cover
            elapsed = time.time() - start
            if self.cfg.reward.lean_infra_error_policy == "fail":
                raise
            reward = None if self.cfg.reward.lean_infra_error_policy == "retry_then_skip" else 0.0
            return LeanResult(False, reward, f"infra_error:{type(e).__name__}:{e}", "", "", None, elapsed)

    def verify_batch(self, prompts: Sequence[str], completions: Sequence[str]) -> List[LeanResult]:
        assert len(prompts) == len(completions)
        if not prompts:
            return []
        results: List[Optional[LeanResult]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.lean.max_workers)) as ex:
            fut_to_i = {ex.submit(self.verify_one, p, c): i for i, (p, c) in enumerate(zip(prompts, completions))}
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # pragma: no cover
                    if self.cfg.reward.lean_infra_error_policy == "fail":
                        raise
                    results[i] = LeanResult(False, None, f"infra_error:{type(e).__name__}:{e}")
        return [r if r is not None else LeanResult(False, None, "missing") for r in results]


# -----------------------------
# Model utilities
# -----------------------------


def require_torch_stack():
    try:
        import torch
        import transformers
    except Exception as e:  # pragma: no cover
        raise RuntimeError("run_08 requires torch and transformers installed.") from e
    return torch, transformers


def dtype_from_string(torch, name: str):
    name = (name or "").lower()
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["fp32", "float32"]:
        return torch.float32
    return torch.bfloat16


def build_model_and_tokenizer(cfg: RunConfig):
    torch, transformers = require_torch_stack()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if cfg.lora.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError("QLoRA requires bitsandbytes-compatible transformers.") from e
        compute_dtype = dtype_from_string(torch, cfg.lora.qlora_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=cfg.lora.qlora_quant_type,
            bnb_4bit_use_double_quant=cfg.lora.qlora_double_quant,
        )

    model_kwargs = dict(
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=dtype_from_string(torch, cfg.model.torch_dtype),
        quantization_config=quantization_config,
    )
    if cfg.model.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation
    if cfg.model.device_map:
        model_kwargs["device_map"] = cfg.model.device_map

    model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_name_or_path, **model_kwargs)
    model.config.use_cache = cfg.model.use_cache

    if cfg.lora.enabled:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        if cfg.lora.use_qlora:
            model = prepare_model_for_kbit_training(model)
        if cfg.lora.mode == "continue_sft_adapter" and cfg.model.policy_adapter_path:
            model = PeftModel.from_pretrained(model, cfg.model.policy_adapter_path, is_trainable=True)
        else:
            peft_cfg = LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                bias=cfg.lora.bias,
                task_type="CAUSAL_LM",
                target_modules=cfg.lora.target_modules,
                modules_to_save=cfg.lora.modules_to_save,
            )
            model = get_peft_model(model, peft_cfg)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if not cfg.model.device_map:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    ref_model = None
    if cfg.model.reference_mode == "separate_model":
        ref_kwargs = dict(model_kwargs)
        ref_model_name = cfg.model.reference_model_name_or_path or cfg.model.base_model_name_or_path
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, **ref_kwargs)
        if cfg.lora.enabled and cfg.model.reference_adapter_path:
            try:
                from peft import PeftModel
                ref_model = PeftModel.from_pretrained(ref_model, cfg.model.reference_adapter_path, is_trainable=False)
            except Exception as e:
                print(f"[warn] failed to load reference adapter: {e}", file=sys.stderr)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        if not cfg.model.device_map:
            device = next(model.parameters()).device
            ref_model.to(device)

    return model, tokenizer, ref_model


def build_optimizer(model, cfg: RunConfig):
    torch, _transformers = require_torch_stack()
    params = [p for p in model.parameters() if p.requires_grad]
    optim_name = cfg.training.optimizer.lower()
    if "8bit" in optim_name or "paged" in optim_name:
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        except Exception as e:
            print(f"[warn] bitsandbytes optimizer unavailable ({e}); falling back to AdamW", file=sys.stderr)
    return torch.optim.AdamW(params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)


def build_scheduler(optimizer, cfg: RunConfig):
    torch, _ = require_torch_stack()
    rollouts_per_rl_step = max(1, cfg.training.prompts_per_step * cfg.rollout.rollouts_per_prompt)
    minibatches_per_epoch = math.ceil(rollouts_per_rl_step / max(1, cfg.training.mini_batch_size))
    microbatches_per_rl_step = minibatches_per_epoch * max(1, cfg.training.ppo_epochs_per_batch)
    updates_per_rl_step = max(1, math.ceil(microbatches_per_rl_step / max(1, cfg.training.gradient_accumulation_steps)))
    total_optimizer_steps = max(1, cfg.training.max_rl_steps * updates_per_rl_step)
    warmup_optimizer_steps = min(
        total_optimizer_steps,
        max(0, cfg.training.warmup_steps * updates_per_rl_step),
    )

    if cfg.training.lr_scheduler_type.lower() == "cosine" and cfg.training.min_learning_rate > 0:
        min_factor = min(1.0, max(0.0, cfg.training.min_learning_rate / max(cfg.training.learning_rate, 1e-12)))

        def lr_lambda(current_step: int) -> float:
            if warmup_optimizer_steps > 0 and current_step < warmup_optimizer_steps:
                return max(min_factor, float(current_step) / float(max(1, warmup_optimizer_steps)))
            progress = float(current_step - warmup_optimizer_steps) / float(max(1, total_optimizer_steps - warmup_optimizer_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    try:
        from transformers import get_scheduler
    except Exception:
        return None
    return get_scheduler(
        name=cfg.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_optimizer_steps,
        num_training_steps=total_optimizer_steps,
    )


# -----------------------------
# Generation and log-probs
# -----------------------------


def truncate_at_stop(text: str, stops: Sequence[str]) -> str:
    best = None
    for s in stops:
        idx = text.find(s)
        if idx >= 0:
            best = idx if best is None else min(best, idx)
    if best is not None:
        text = text[:best]
    return text.strip("\n")


def generate_completions(model, tokenizer, prompts: Sequence[str], cfg: RunConfig) -> List[str]:
    torch, _ = require_torch_stack()
    model.eval()
    out: List[str] = []
    bsz = max(1, cfg.rollout.generation_batch_size)
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    for start in range(0, len(prompts), bsz):
        batch_prompts = list(prompts[start:start + bsz])
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.rollout.prompt_max_length,
            pad_to_multiple_of=cfg.rollout.pad_to_multiple_of,
        ).to(device)
        generation_kwargs = dict(
            do_sample=cfg.rollout.do_sample,
            temperature=cfg.rollout.temperature,
            top_p=cfg.rollout.top_p,
            max_new_tokens=cfg.rollout.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if cfg.rollout.top_k and cfg.rollout.top_k > 0:
            generation_kwargs["top_k"] = cfg.rollout.top_k
        with torch.no_grad():
            gen = model.generate(
                **enc,
                **generation_kwargs,
            )
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for i, seq in enumerate(gen):
            # Because generation used left padding, generated tokens start after full padded input length.
            gen_tokens = seq[enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            text = truncate_at_stop(text, cfg.rollout.stop_sequences)
            out.append(text)
    return out


def make_scoring_batch(tokenizer, prompts: Sequence[str], completions: Sequence[str], cfg: RunConfig):
    torch, _ = require_torch_stack()
    tokenizer.padding_side = "right"
    input_ids_list = []
    attn_list = []
    label_mask_list = []
    prompt_lens = []

    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_text = prompt + completion
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None and len(full_ids) < cfg.training.max_seq_length:
            full_ids = full_ids + [tokenizer.eos_token_id]
        if len(full_ids) > cfg.training.max_seq_length:
            full_ids = full_ids[: cfg.training.max_seq_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        mask = [0] * len(full_ids)
        for j in range(prompt_len, len(full_ids)):
            mask[j] = 1
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        attn_list.append(torch.ones(len(full_ids), dtype=torch.long))
        label_mask_list.append(torch.tensor(mask, dtype=torch.bool))
        prompt_lens.append(prompt_len)

    max_len = max(len(x) for x in input_ids_list) if input_ids_list else 0
    if cfg.rollout.pad_to_multiple_of and max_len % cfg.rollout.pad_to_multiple_of != 0:
        max_len = ((max_len // cfg.rollout.pad_to_multiple_of) + 1) * cfg.rollout.pad_to_multiple_of
    pad_id = tokenizer.pad_token_id
    ids = []
    attn = []
    masks = []
    for x, a, m in zip(input_ids_list, attn_list, label_mask_list):
        pad = max_len - len(x)
        ids.append(torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)]))
        attn.append(torch.cat([a, torch.zeros(pad, dtype=torch.long)]))
        masks.append(torch.cat([m, torch.zeros(pad, dtype=torch.bool)]))
    return {
        "input_ids": torch.stack(ids) if ids else torch.empty((0, 0), dtype=torch.long),
        "attention_mask": torch.stack(attn) if attn else torch.empty((0, 0), dtype=torch.long),
        "label_mask": torch.stack(masks) if masks else torch.empty((0, 0), dtype=torch.bool),
        "prompt_lens": prompt_lens,
    }


def token_logprobs_from_logits(logits, input_ids):
    import torch
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    tok_logp = torch.gather(logp, dim=-1, index=target).squeeze(-1)
    pad_first = torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=tok_logp.dtype)
    return torch.cat([pad_first, tok_logp], dim=1)


def compute_logprobs(model, tokenizer, prompts: Sequence[str], completions: Sequence[str], cfg: RunConfig, batch_size: int = 1):
    torch, _ = require_torch_stack()
    device = next(model.parameters()).device
    model.eval()
    all_logps = []
    all_masks = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = make_scoring_batch(tokenizer, prompts[start:start + batch_size], completions[start:start + batch_size], cfg)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_mask = batch["label_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            tok_logp = token_logprobs_from_logits(out.logits, input_ids)
            all_logps.append(tok_logp.detach().cpu())
            all_masks.append(label_mask.detach().cpu())
    return all_logps, all_masks


# -----------------------------
# MaxRL advantages and loss
# -----------------------------


def compute_maxrl_advantages(rewards: List[float], group_size: int, cfg: RunConfig):
    import torch
    r = torch.tensor(rewards, dtype=torch.float32)
    if len(r) % group_size != 0:
        raise ValueError("Number of rewards must be divisible by rollouts_per_prompt")
    grouped = r.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    adv = torch.zeros_like(grouped)
    nonzero = mean.squeeze(1) > 0
    if nonzero.any():
        adv[nonzero] = (grouped[nonzero] - mean[nonzero]) / (mean[nonzero] + cfg.maxrl.epsilon_mean_reward)
    if cfg.maxrl.advantage_clip and cfg.maxrl.advantage_clip > 0:
        adv = adv.clamp(-cfg.maxrl.advantage_clip, cfg.maxrl.advantage_clip)
    if cfg.maxrl.normalize_group_advantages_after_clip:
        for i in range(adv.shape[0]):
            if nonzero[i]:
                std = adv[i].std(unbiased=False)
                if std > 1e-6:
                    adv[i] = (adv[i] - adv[i].mean()) / (std + 1e-6)
    return adv.reshape(-1), mean.reshape(-1), nonzero


def approx_kl(new_logp, ref_logp):
    # Non-negative approximation used in several RLHF implementations:
    # exp(ref-new) - (ref-new) - 1.
    import torch
    diff = ref_logp - new_logp
    return torch.exp(diff).clamp(max=100.0) - diff - 1.0


def compute_policy_loss(model, tokenizer, prompts, completions, old_logps_chunks, ref_logps_chunks, advantages, cfg: RunConfig):
    torch, _ = require_torch_stack()
    device = next(model.parameters()).device
    model.train()

    batch = make_scoring_batch(tokenizer, prompts, completions, cfg)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    label_mask = batch["label_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    new_logp = token_logprobs_from_logits(out.logits, input_ids)

    # old_logps_chunks/ref_logps_chunks here are tensors for the same batch, not all chunks.
    old_logp = old_logps_chunks.to(device)
    if ref_logps_chunks is None:
        ref_logp = old_logp
    else:
        ref_logp = ref_logps_chunks.to(device)

    adv = advantages.to(device).view(-1, 1)
    ratio = torch.exp((new_logp - old_logp).clamp(min=-20, max=20))
    ratio = ratio.clamp(max=cfg.loss.max_ratio)
    unclipped = ratio * adv
    clipped_ratio = ratio.clamp(1.0 - cfg.loss.clip_range, 1.0 + cfg.loss.clip_range)
    clipped = clipped_ratio * adv
    pg_obj = torch.minimum(unclipped, clipped)

    mask = label_mask.float()
    denom = mask.sum().clamp_min(1.0)
    pg_loss = -(pg_obj * mask).sum() / denom

    kl = approx_kl(new_logp, ref_logp)
    kl_loss = (kl * mask).sum() / denom
    total = pg_loss + (cfg.loss.kl_beta * kl_loss if cfg.loss.kl_to_reference else 0.0)

    with torch.no_grad():
        entropy_like = -(torch.exp(new_logp).clamp(max=1.0) * new_logp * mask).sum() / denom
        mean_ratio = (ratio * mask).sum() / denom
    return total, {
        "pg_loss": float(pg_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "loss": float(total.detach().cpu()),
        "mean_ratio": float(mean_ratio.detach().cpu()),
        "entropy_like": float(entropy_like.detach().cpu()),
    }


# -----------------------------
# Training and evaluation
# -----------------------------


def sample_tasks(tasks: Sequence[RLTask], n: int, step: int, cfg: RunConfig) -> List[RLTask]:
    rng = random.Random(cfg.experiment.seed + step + cfg.training.dataloader_seed_offset)
    if len(tasks) <= n:
        return list(tasks)
    return rng.sample(list(tasks), n)


def build_rollout_batch(tasks: Sequence[RLTask], cfg: RunConfig) -> Tuple[List[RLTask], List[str]]:
    repeated_tasks = []
    prompts = []
    for task in tasks:
        for _ in range(cfg.rollout.rollouts_per_prompt):
            repeated_tasks.append(task)
            prompts.append(task.prompt)
    return repeated_tasks, prompts


def rewards_from_lean_results(results: Sequence[LeanResult]) -> Tuple[List[float], List[str]]:
    rewards = []
    statuses = []
    for r in results:
        rewards.append(float(r.reward) if r.reward is not None else 0.0)
        statuses.append(r.status)
    return rewards, statuses


def save_adapter(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(path), safe_serialization=True)
    except TypeError:
        model.save_pretrained(str(path))


def enforce_save_limit(output_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    ckpts = sorted([p for p in output_dir.glob("checkpoint-step-*") if p.is_dir()], key=lambda p: int(p.name.split("-")[-1]))
    while len(ckpts) > limit:
        shutil.rmtree(ckpts.pop(0), ignore_errors=True)


def evaluate_passk(model, tokenizer, verifier: LeanVerifier, eval_tasks: Sequence[RLTask], cfg: RunConfig, step: int) -> Dict[str, Any]:
    if not eval_tasks:
        return {}
    old_temp, old_top_p, old_n, old_bsz, old_new = (
        cfg.rollout.temperature,
        cfg.rollout.top_p,
        cfg.rollout.rollouts_per_prompt,
        cfg.rollout.generation_batch_size,
        cfg.rollout.max_new_tokens,
    )
    max_k = max(cfg.eval.rollout_passk)
    cfg.rollout.temperature = cfg.eval.eval_temperature
    cfg.rollout.top_p = cfg.eval.eval_top_p
    cfg.rollout.rollouts_per_prompt = max_k
    cfg.rollout.generation_batch_size = cfg.eval.generation_batch_size
    cfg.rollout.max_new_tokens = cfg.eval.eval_max_new_tokens

    selected = list(eval_tasks)[: cfg.eval.max_eval_theorems]
    per_task = []
    try:
        for task in selected:
            prompts = [task.prompt] * max_k
            completions = generate_completions(model, tokenizer, prompts, cfg)
            results = verifier.verify_batch(prompts, completions)
            rewards, statuses = rewards_from_lean_results(results)
            row = {"task_id": task.task_id, "rewards": rewards, "statuses": statuses}
            per_task.append(row)
    finally:
        cfg.rollout.temperature = old_temp
        cfg.rollout.top_p = old_top_p
        cfg.rollout.rollouts_per_prompt = old_n
        cfg.rollout.generation_batch_size = old_bsz
        cfg.rollout.max_new_tokens = old_new

    metrics: Dict[str, Any] = {"step": step, "num_eval_tasks": len(selected)}
    for k in cfg.eval.rollout_passk:
        succ = 0
        for row in per_task:
            if any(r > 0 for r in row["rewards"][:k]):
                succ += 1
        metrics[f"pass_at_{k}"] = succ / max(1, len(selected))
    metrics["mean_reward_at_max_k"] = sum(sum(row["rewards"]) for row in per_task) / max(1, len(selected) * max_k)
    return {"metrics": metrics, "per_task": per_task}


def adjust_kl_beta(cfg: RunConfig, observed_kl: float) -> None:
    if not cfg.loss.adaptive_kl or cfg.loss.kl_target <= 0:
        return
    target = cfg.loss.kl_target
    if observed_kl > 1.5 * target:
        cfg.loss.kl_beta *= 1.2
    elif observed_kl < target / 1.5:
        cfg.loss.kl_beta /= 1.2
    cfg.loss.kl_beta = float(max(1e-5, min(1.0, cfg.loss.kl_beta)))


def train(cfg: RunConfig) -> None:
    output_dir = Path(cfg.training.output_dir)
    if output_dir.exists() and cfg.training.overwrite_output_dir:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(output_dir / cfg.outputs.config_file, cfg)

    if cfg.experiment.dry_run:
        random.seed(cfg.experiment.seed)
    else:
        set_seed(cfg.experiment.seed)
    tasks = load_tasks(cfg)
    if cfg.eval.run_eval and not cfg.data.heldout_path:
        print(
            "[warn] no --heldout-path provided; eval will use a shuffled training subset "
            "and should not be reported as held-out pass@k.",
            file=sys.stderr,
        )
    eval_tasks = load_eval_tasks(cfg) if cfg.eval.run_eval else []
    write_jsonl(output_dir / cfg.outputs.tasks_file, [dataclasses.asdict(t) for t in tasks])

    print(f"Loaded {len(tasks)} train tasks and {len(eval_tasks)} eval tasks.")
    print("Train task sources:", json.dumps(task_metadata_counts(tasks, "source"), ensure_ascii=False))
    if eval_tasks:
        print("Eval task sources:", json.dumps(task_metadata_counts(eval_tasks, "source"), ensure_ascii=False))
    if cfg.experiment.dry_run:
        print("Dry run only. No model is loaded.")
        return

    torch, _transformers = require_torch_stack()
    if cfg.training.tf32 and hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    model, tokenizer, ref_model = build_model_and_tokenizer(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    verifier = LeanVerifier(cfg)

    train_metrics_path = output_dir / cfg.outputs.train_metrics
    rollout_logs_path = output_dir / cfg.outputs.rollout_logs
    reward_logs_path = output_dir / cfg.outputs.reward_logs
    eval_reports_path = output_dir / cfg.outputs.eval_reports

    best_metric = -float("inf")
    best_step = 0

    global_micro_step = 0
    optimizer_updates = 0
    for step in range(1, cfg.training.max_rl_steps + 1):
        t0 = time.time()
        selected_tasks = sample_tasks(tasks, cfg.training.prompts_per_step, step, cfg)
        repeated_tasks, prompts = build_rollout_batch(selected_tasks, cfg)

        completions = generate_completions(model, tokenizer, prompts, cfg)
        # LeanVerifier assigns zero reward to forbidden strings before invoking Lean.

        lean_results = verifier.verify_batch(prompts, completions)
        rewards, statuses = rewards_from_lean_results(lean_results)
        advantages, group_means, nonzero_groups = compute_maxrl_advantages(rewards, cfg.rollout.rollouts_per_prompt, cfg)

        # Compute old-policy logprobs before updating.
        old_chunks, mask_chunks = compute_logprobs(model, tokenizer, prompts, completions, cfg, batch_size=max(1, cfg.training.mini_batch_size))
        if cfg.model.reference_mode == "separate_model" and ref_model is not None:
            ref_chunks, _ = compute_logprobs(ref_model, tokenizer, prompts, completions, cfg, batch_size=max(1, cfg.training.mini_batch_size))
        elif cfg.model.reference_mode == "none":
            ref_chunks = [None] * len(old_chunks)
        else:
            ref_chunks = old_chunks

        # Flatten chunk bookkeeping.
        # Each chunk may contain multiple rows; mini-batch again with identical chunking to align logprobs.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_stats: List[Dict[str, float]] = []
        bsz = max(1, cfg.training.mini_batch_size)
        chunk_idx = 0
        for epoch in range(max(1, cfg.training.ppo_epochs_per_batch)):
            chunk_idx = 0
            for start in range(0, len(prompts), bsz):
                end = start + bsz
                mb_prompts = prompts[start:end]
                mb_completions = completions[start:end]
                mb_adv = advantages[start:end]
                if torch.all(mb_adv == 0):
                    chunk_idx += 1
                    continue
                old_lp = old_chunks[chunk_idx]
                ref_lp = None if ref_chunks[chunk_idx] is None else ref_chunks[chunk_idx]
                loss, stats = compute_policy_loss(model, tokenizer, mb_prompts, mb_completions, old_lp, ref_lp, mb_adv, cfg)
                (loss / max(1, cfg.training.gradient_accumulation_steps)).backward()
                loss_stats.append(stats)
                global_micro_step += 1
                if global_micro_step % cfg.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.training.max_grad_norm)
                    optimizer.step()
                    optimizer_updates += 1
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                chunk_idx += 1

        # Flush pending grads after a step.
        if global_micro_step % cfg.training.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.training.max_grad_norm)
            optimizer.step()
            optimizer_updates += 1
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_micro_step = 0

        elapsed = time.time() - t0
        group_rewards = [sum(rewards[i:i + cfg.rollout.rollouts_per_prompt]) for i in range(0, len(rewards), cfg.rollout.rollouts_per_prompt)]
        zero_groups = sum(1 for x in group_rewards if x == 0)
        all_success_groups = sum(1 for x in group_rewards if x == cfg.rollout.rollouts_per_prompt)
        mean_reward = sum(rewards) / max(1, len(rewards))
        status_counts: Dict[str, int] = {}
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        group_success_hist: Dict[str, int] = {}
        for success_count in group_rewards:
            key = str(int(success_count))
            group_success_hist[key] = group_success_hist.get(key, 0) + 1
        adv_cpu = advantages.detach().cpu()
        active_adv_mask = adv_cpu != 0
        active_rollout_rate = float(active_adv_mask.float().mean().item()) if adv_cpu.numel() else 0.0
        advantage_mean = float(adv_cpu.mean().item()) if adv_cpu.numel() else 0.0
        advantage_abs_mean = float(adv_cpu.abs().mean().item()) if adv_cpu.numel() else 0.0
        advantage_min = float(adv_cpu.min().item()) if adv_cpu.numel() else 0.0
        advantage_max = float(adv_cpu.max().item()) if adv_cpu.numel() else 0.0
        observed_kl = sum(s.get("kl", 0.0) for s in loss_stats) / max(1, len(loss_stats))
        adjust_kl_beta(cfg, observed_kl)
        mean_loss = sum(s.get("loss", 0.0) for s in loss_stats) / max(1, len(loss_stats))
        mean_pg_loss = sum(s.get("pg_loss", 0.0) for s in loss_stats) / max(1, len(loss_stats))

        metrics = {
            "step": step,
            "elapsed_sec": elapsed,
            "num_prompts": len(selected_tasks),
            "num_rollouts": len(rewards),
            "optimizer_updates": optimizer_updates,
            "mean_reward": mean_reward,
            "num_successes": int(sum(rewards)),
            "zero_success_group_rate": zero_groups / max(1, len(group_rewards)),
            "all_success_group_rate": all_success_groups / max(1, len(group_rewards)),
            "nonzero_group_rate": 1.0 - zero_groups / max(1, len(group_rewards)),
            "active_rollout_rate": active_rollout_rate,
            "mean_group_successes": sum(group_rewards) / max(1, len(group_rewards)),
            "group_success_hist": group_success_hist,
            "status_counts": status_counts,
            "advantage_mean": advantage_mean,
            "advantage_abs_mean": advantage_abs_mean,
            "advantage_min": advantage_min,
            "advantage_max": advantage_max,
            "loss": mean_loss,
            "pg_loss": mean_pg_loss,
            "kl": observed_kl,
            "kl_beta": cfg.loss.kl_beta,
            "lr": optimizer.param_groups[0].get("lr", None),
        }
        append_jsonl(train_metrics_path, [metrics])

        reward_rows = []
        rollout_rows = []
        for i, (task, completion, reward, status) in enumerate(zip(repeated_tasks, completions, rewards, statuses)):
            reward_rows.append({
                "step": step,
                "task_id": task.task_id,
                "rollout_index": i % cfg.rollout.rollouts_per_prompt,
                "reward": reward,
                "status": status,
            })
            if cfg.experiment.debug:
                rollout_rows.append({
                    "step": step,
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "completion": completion,
                    "reward": reward,
                    "status": status,
                })
        append_jsonl(reward_logs_path, reward_rows)
        if rollout_rows:
            append_jsonl(rollout_logs_path, rollout_rows)

        if step % cfg.training.logging_steps == 0:
            print(json.dumps(metrics, ensure_ascii=False))

        if cfg.eval.run_eval and step % cfg.eval.eval_steps == 0:
            ev = evaluate_passk(model, tokenizer, verifier, eval_tasks, cfg, step)
            if ev:
                append_jsonl(eval_reports_path, [ev])
                ev_metrics = ev["metrics"]
                primary = ev_metrics.get(cfg.checkpoint_selection.primary_metric.replace("heldout_", ""))
                if primary is None:
                    # fallback for metric names like heldout_pass_at_8
                    primary = ev_metrics.get("pass_at_8") or ev_metrics.get("pass_at_1") or 0.0
                if primary > best_metric:
                    best_metric = float(primary)
                    best_step = step
                    if cfg.checkpoint_selection.save_best_adapter:
                        save_adapter(model, output_dir / cfg.outputs.selected_adapter_dir)
                print("EVAL", json.dumps(ev_metrics, ensure_ascii=False))

        if step % cfg.training.save_steps == 0:
            ckpt = output_dir / f"checkpoint-step-{step}"
            save_adapter(model, ckpt)
            enforce_save_limit(output_dir, cfg.training.save_total_limit)

    final_dir = output_dir / "final_adapter"
    save_adapter(model, final_dir)
    report = {
        "best_step": best_step,
        "best_metric": best_metric,
        "final_adapter": str(final_dir),
        "best_adapter": str(output_dir / cfg.outputs.selected_adapter_dir),
        "num_train_tasks": len(tasks),
        "num_eval_tasks": len(eval_tasks),
    }
    write_json(output_dir / "run_report.json", report)
    print("Training complete.", json.dumps(report, ensure_ascii=False, indent=2))


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LoRA MaxRL/RLVR on Lean frontier tasks.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--frontier-train-path", type=str, default=None)
    p.add_argument("--heldout-path", type=str, default=None)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--policy-adapter-path", type=str, default=None)
    p.add_argument("--reference-adapter-path", type=str, default=None)
    p.add_argument("--reference-mode", type=str, choices=["old_policy", "separate_model", "none"], default=None)
    p.add_argument("--lean-project-root", type=str, default=None)
    p.add_argument("--lean-command", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")

    p.add_argument("--rollouts-per-prompt", type=int, default=None)
    p.add_argument("--prompts-per-step", type=int, default=None)
    p.add_argument("--max-rl-steps", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--kl-beta", type=float, default=None)
    p.add_argument("--advantage-clip", type=float, default=None)
    p.add_argument("--clip-range", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--run-eval", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--timeout-sec", type=int, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg_dict = dataclass_to_dict(RunConfig())
    file_cfg = load_yaml_config(args.config)
    deep_update(cfg_dict, file_cfg)

    def set_if(path: Sequence[str], value: Any):
        if value is None:
            return
        d = cfg_dict
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value

    set_if(["data", "frontier_train_path"], args.frontier_train_path)
    set_if(["data", "heldout_path"], args.heldout_path)
    set_if(["model", "base_model_name_or_path"], args.base_model)
    set_if(["model", "policy_adapter_path"], args.policy_adapter_path)
    set_if(["model", "reference_adapter_path"], args.reference_adapter_path)
    set_if(["model", "reference_mode"], args.reference_mode)
    set_if(["lean", "project_root"], args.lean_project_root)
    set_if(["lean", "command"], args.lean_command)
    set_if(["training", "output_dir"], args.output_dir)
    set_if(["rollout", "rollouts_per_prompt"], args.rollouts_per_prompt)
    set_if(["training", "prompts_per_step"], args.prompts_per_step)
    set_if(["training", "max_rl_steps"], args.max_rl_steps)
    set_if(["rollout", "max_new_tokens"], args.max_new_tokens)
    set_if(["rollout", "temperature"], args.temperature)
    set_if(["rollout", "top_p"], args.top_p)
    set_if(["training", "learning_rate"], args.learning_rate)
    set_if(["loss", "kl_beta"], args.kl_beta)
    set_if(["maxrl", "advantage_clip"], args.advantage_clip)
    set_if(["loss", "clip_range"], args.clip_range)
    set_if(["lora", "r"], args.lora_r)
    set_if(["lora", "alpha"], args.lora_alpha)
    set_if(["eval", "eval_steps"], args.eval_steps)
    set_if(["training", "eval_steps"], args.eval_steps)
    set_if(["training", "save_steps"], args.save_steps)
    set_if(["lean", "max_workers"], args.max_workers)
    set_if(["lean", "timeout_sec"], args.timeout_sec)

    if args.overwrite:
        cfg_dict["training"]["overwrite_output_dir"] = True
    if args.dry_run:
        cfg_dict["experiment"]["dry_run"] = True
    if args.debug:
        cfg_dict["experiment"]["debug"] = True
    if args.qlora:
        cfg_dict["lora"]["use_qlora"] = True
    if args.no_lora:
        cfg_dict["lora"]["enabled"] = False
    if args.run_eval:
        cfg_dict["eval"]["run_eval"] = True
    if args.no_eval:
        cfg_dict["eval"]["run_eval"] = False

    return dict_to_dataclass(RunConfig, cfg_dict)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    train(cfg)


if __name__ == "__main__":
    main()

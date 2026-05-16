#!/usr/bin/env python3
"""
run_07_sft.py

LoRA/QLoRA SFT stage for the ATP curriculum pipeline.

This script consumes verified trajectories assembled by run_06 and trains a
DeepSeek-Prover-style student model with completion-only supervised fine-tuning.
It does not mine, search, or call Lean proof search. It only builds a clean SFT
corpus from verified proofs, optionally mixes in general Lean proof data, and
trains PEFT adapters.

Typical use:
  python run_07_sft.py \
    --run06-dir outputs/run_06_assemble_curriculum \
    --general-lean-data data/general_lean_sft.jsonl \
    --base-model deepseek-ai/DeepSeek-Prover-V1.5-SFT \
    --output-dir outputs/run_07_sft_lora

Dry run:
  python run_07_sft.py --run06-dir outputs/run_06_assemble_curriculum --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
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
class ModelConfig:
    base_model_name_or_path: str = "deepseek-ai/DeepSeek-Prover-V1.5-SFT"
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    use_cache: bool = False


@dataclass
class DataConfig:
    run06_dir: str = "outputs/run_06_assemble_curriculum"
    final_trajectories_file: str = "final_trajectories.jsonl"
    final_curriculum_items_file: str = "final_curriculum_items.jsonl"
    use_final_trajectories: bool = True
    use_final_curriculum_items: bool = True
    general_lean_data: Optional[str] = None
    output_processed_train: str = "sft_train_dataset.jsonl"
    output_processed_val: str = "sft_val_dataset.jsonl"
    output_processed_test: str = "sft_test_dataset.jsonl"
    reject_sorry: bool = True
    reject_admit: bool = True
    require_verified: bool = True
    max_seq_length: int = 4096
    min_completion_tokens_approx: int = 8
    max_completion_chars: int = 60000
    split_by: str = "parent_theorem_id"
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    preserve_comments: bool = True
    include_natural_language_statement: bool = False
    add_eos_token: bool = True
    skip_overlength_examples: bool = True
    strip_trailing_whitespace: bool = True
    max_variants_per_theorem: int = 4
    keep_files_if_no_new_data: bool = False


@dataclass
class MixtureConfig:
    enabled: bool = True
    mined_curriculum_weight: float = 0.35
    general_lean_weight: float = 0.55
    original_successful_weight: float = 0.10
    mcsp_assembled_weight: float = 2.0
    prefix_suffix_weight: float = 1.3
    full_solved_direct_weight: float = 1.0
    general_lean_example_weight: float = 1.0
    max_effective_repeat_per_example: int = 4
    max_train_examples: Optional[int] = None
    max_eval_examples: Optional[int] = None


@dataclass
class LoraConfigSpec:
    enabled: bool = True
    use_qlora: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: Optional[List[str]] = None
    qlora_compute_dtype: str = "bfloat16"
    qlora_quant_type: str = "nf4"
    qlora_double_quant: bool = True


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/run_07_sft_lora"
    overwrite_output_dir: bool = False
    num_train_epochs: float = 2.0
    learning_rate: float = 1.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    per_device_eval_batch_size: int = 1
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 4
    report_to: str = "none"
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    group_by_length: bool = False
    packing: bool = False  # This script performs explicit tokenization; packing is not implemented.
    completion_only_loss: bool = True
    resume_from_checkpoint: Optional[str] = None
    save_safetensors: bool = True
    merge_lora_after_training: bool = False
    torch_compile: bool = False
    tf32: bool = True


@dataclass
class LeanEvalConfig:
    enabled: bool = False
    command: Optional[str] = None
    run_after_training: bool = False
    max_eval_examples: int = 200


@dataclass
class ExperimentConfig:
    name: str = "run_07_lora_mined_curriculum_v1"
    seed: int = 42
    dry_run: bool = False
    debug: bool = False


@dataclass
class RunConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lean_eval: LeanEvalConfig = field(default_factory=LeanEvalConfig)


# -----------------------------
# Utility
# -----------------------------


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def dataclass_to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def dict_to_dataclass(cls, data: Dict[str, Any]):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if dataclasses.is_dataclass(f.type):
            kwargs[f.name] = dict_to_dataclass(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: Optional[str]) -> RunConfig:
    cfg_dict = dataclass_to_dict(RunConfig())
    if path:
        if yaml is None:
            raise RuntimeError("PyYAML is required for --config. Install pyyaml or omit --config.")
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        deep_update(cfg_dict, user_cfg)
    return build_run_config(cfg_dict)


def build_run_config(d: Dict[str, Any]) -> RunConfig:
    return RunConfig(
        experiment=ExperimentConfig(**d.get("experiment", {})),
        model=ModelConfig(**d.get("model", {})),
        data=DataConfig(**d.get("data", {})),
        mixture=MixtureConfig(**d.get("mixture", {})),
        lora=LoraConfigSpec(**d.get("lora", {})),
        training=TrainingConfig(**d.get("training", {})),
        lean_eval=LeanEvalConfig(**d.get("lean_eval", {})),
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSONL at {p}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:24]


def normalize_ws(text: str, strip_trailing: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_trailing:
        text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip("\n") + "\n"


def contains_forbidden(text: str, reject_sorry: bool, reject_admit: bool) -> Optional[str]:
    lowered = text.lower()
    if reject_sorry and re.search(r"\bsorry\b", lowered):
        return "contains_sorry"
    if reject_admit and re.search(r"\badmit\b", lowered):
        return "contains_admit"
    return None


def approx_token_count(text: str) -> int:
    # Conservative fallback before tokenizer is loaded.
    return max(1, len(text) // 4)


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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# -----------------------------
# SFT example construction
# -----------------------------


@dataclass
class SFTExample:
    id: str
    parent_theorem_id: str
    source_type: str
    prompt: str
    completion: str
    text: str
    theorem_hash: str
    proof_hash: str
    weight: float
    metadata: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


PROOF_SPLIT_RE = re.compile(r"(?s)(.*?\b(?:theorem|lemma|example)\b.*?:=\s*by\s*)(.*)")


def first_present(record: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        v = record.get(k)
        if v is not None and v != "":
            return v
    return default


def nested_first_present(record: Dict[str, Any], paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        cur: Any = record
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and cur not in (None, ""):
            return cur
    return default


def extract_code_fields(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    full_code = first_present(
        record,
        [
            "full_theorem_code",
            "assembled_theorem_code",
            "theorem_code",
            "lean_code",
            "code",
        ],
    )
    if full_code is None:
        full_code = nested_first_present(record, [["output", "full_theorem_code"], ["trajectory", "full_theorem_code"]])

    formal_statement = first_present(
        record,
        ["formal_statement", "statement", "theorem_statement", "lean_statement"],
    )
    if formal_statement is None:
        formal_statement = nested_first_present(record, [["input_context", "formal_statement"], ["source_job", "formal_statement"]])

    proof_body = first_present(
        record,
        [
            "full_proof_body",
            "proof_body",
            "proof",
            "target_proof",
            "completion",
            "suffix_text",
            "replacement_proof",
        ],
    )
    if proof_body is None:
        proof_body = nested_first_present(
            record,
            [
                ["output", "full_proof_body"],
                ["trajectory", "full_proof_body"],
                ["input_context", "target_proof"],
            ],
        )

    header = first_present(record, ["imports", "header", "lean_header", "prelude"], "")
    if header is None:
        header = ""
    return full_code, formal_statement, proof_body, header


def build_prompt_completion(
    record: Dict[str, Any], data_cfg: DataConfig
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    full_code, formal_statement, proof_body, header = extract_code_fields(record)

    if full_code:
        full_code = normalize_ws(str(full_code), data_cfg.strip_trailing_whitespace)
        m = PROOF_SPLIT_RE.match(full_code)
        if m:
            prompt = normalize_ws(m.group(1), data_cfg.strip_trailing_whitespace)
            completion = normalize_ws(m.group(2), data_cfg.strip_trailing_whitespace)
            return prompt, completion, full_code
        # If full_code is not splitable but a proof body exists, fall through.

    if formal_statement and proof_body:
        statement = normalize_ws(str(formal_statement), data_cfg.strip_trailing_whitespace)
        proof = normalize_ws(str(proof_body), data_cfg.strip_trailing_whitespace)
        if ":= by" in statement:
            prompt = statement
        else:
            statement = statement.rstrip()
            if statement.endswith(":="):
                prompt = statement + " by\n"
            elif statement.endswith("by"):
                prompt = statement + "\n"
            else:
                prompt = statement + " := by\n"
        header = normalize_ws(str(header or ""), data_cfg.strip_trailing_whitespace) if header else ""
        full = normalize_ws(header + prompt + proof, data_cfg.strip_trailing_whitespace)
        return normalize_ws(header + prompt, data_cfg.strip_trailing_whitespace), proof, full

    return None, None, None


def is_verified_record(record: Dict[str, Any]) -> bool:
    if record.get("verified") is True:
        return True
    if record.get("verified_full") is True:
        return True
    if record.get("lean_verified") is True:
        return True
    status = str(record.get("status", "")).upper()
    if status in {"VERIFIED", "SOLVED_FULL_PARENT", "SOLVED_CONTEXTUAL", "FULLY_VERIFIED"}:
        return True
    quality = str(record.get("quality_class", "")).lower()
    if quality in {"verified", "high_verified"}:
        return True
    return False


def infer_source_type(record: Dict[str, Any]) -> str:
    for key in ["source_type", "item_type", "trajectory_type", "job_type", "type"]:
        v = record.get(key)
        if v:
            s = str(v).lower()
            if "mcsp" in s or "hole" in s or "sorry" in s:
                return "mcsp_assembled"
            if "prefix" in s or "suffix" in s:
                return "prefix_suffix"
            if "full" in s or "direct" in s or "solved" in s:
                return "full_solved_direct"
            return s
    if record.get("holes_filled") or record.get("semi_proof_text"):
        return "mcsp_assembled"
    if record.get("prefix_text") or record.get("suffix_text"):
        return "prefix_suffix"
    return "full_solved_direct"


def parent_id(record: Dict[str, Any], fallback_text: str) -> str:
    pid = first_present(
        record,
        ["parent_theorem_id", "theorem_id", "problem_id", "id", "name", "source_id"],
    )
    if pid is None:
        pid = nested_first_present(record, [["source_job", "parent_theorem_id"], ["metadata", "parent_theorem_id"]])
    return str(pid) if pid is not None else stable_hash(fallback_text)


def source_weight(source_type: str, mix_cfg: MixtureConfig) -> float:
    if source_type == "mcsp_assembled":
        return mix_cfg.mcsp_assembled_weight
    if source_type == "prefix_suffix":
        return mix_cfg.prefix_suffix_weight
    if source_type == "general_lean":
        return mix_cfg.general_lean_example_weight
    return mix_cfg.full_solved_direct_weight


def record_to_sft_example(
    record: Dict[str, Any], data_cfg: DataConfig, mix_cfg: MixtureConfig, source_override: Optional[str] = None
) -> Tuple[Optional[SFTExample], Optional[str]]:
    if data_cfg.require_verified and source_override != "general_lean" and not is_verified_record(record):
        return None, "not_verified"

    prompt, completion, full_code = build_prompt_completion(record, data_cfg)
    if not prompt or not completion:
        return None, "cannot_extract_prompt_completion"

    if data_cfg.include_natural_language_statement:
        nl = first_present(record, ["natural_language_statement", "nl_statement", "problem_text"])
        if nl:
            prompt = f"/-\n{str(nl).strip()}\n-/\n" + prompt

    if len(completion) > data_cfg.max_completion_chars:
        return None, "completion_too_long_chars"

    forbidden = contains_forbidden(prompt + completion, data_cfg.reject_sorry, data_cfg.reject_admit)
    if forbidden:
        return None, forbidden

    if approx_token_count(completion) < data_cfg.min_completion_tokens_approx:
        return None, "completion_too_short"

    prompt = normalize_ws(prompt, data_cfg.strip_trailing_whitespace)
    completion = normalize_ws(completion, data_cfg.strip_trailing_whitespace)
    text = prompt + completion

    stype = source_override or infer_source_type(record)
    pid = parent_id(record, prompt)
    thash = stable_hash(prompt)
    phash = stable_hash(completion)
    eid = first_present(record, ["item_id", "trajectory_id", "job_id", "id"], f"sft_{thash}_{phash}")

    return SFTExample(
        id=str(eid),
        parent_theorem_id=pid,
        source_type=stype,
        prompt=prompt,
        completion=completion,
        text=text,
        theorem_hash=thash,
        proof_hash=phash,
        weight=source_weight(stype, mix_cfg),
        metadata={
            "original_id": record.get("id"),
            "source_type": stype,
            "quality_class": record.get("quality_class"),
            "verified": is_verified_record(record) or source_override == "general_lean",
        },
    ), None


def load_mined_examples(cfg: RunConfig) -> Tuple[List[SFTExample], Dict[str, int]]:
    run06 = Path(cfg.data.run06_dir)
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)

    if cfg.data.use_final_trajectories:
        p = run06 / cfg.data.final_trajectories_file
        rs = read_jsonl(p)
        rows.extend(rs)
        counts[f"loaded_{p.name}"] += len(rs)
    if cfg.data.use_final_curriculum_items:
        p = run06 / cfg.data.final_curriculum_items_file
        rs = read_jsonl(p)
        rows.extend(rs)
        counts[f"loaded_{p.name}"] += len(rs)

    examples: List[SFTExample] = []
    reject_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        ex, reason = record_to_sft_example(r, cfg.data, cfg.mixture)
        if ex:
            examples.append(ex)
        else:
            reject_counts[reason or "unknown"] += 1
    for k, v in reject_counts.items():
        counts[f"rejected_{k}"] += v
    counts["accepted_mined"] = len(examples)
    return examples, dict(counts)


def load_general_examples(cfg: RunConfig) -> Tuple[List[SFTExample], Dict[str, int]]:
    counts: Dict[str, int] = {}
    if not cfg.data.general_lean_data:
        return [], counts
    rows = read_jsonl(cfg.data.general_lean_data)
    counts["loaded_general"] = len(rows)
    examples: List[SFTExample] = []
    reject_counts: Dict[str, int] = defaultdict(int)
    # General data may not have verified=True but should be trusted as an existing proof corpus.
    data_cfg = dataclasses.replace(cfg.data, require_verified=False)
    for r in rows:
        ex, reason = record_to_sft_example(r, data_cfg, cfg.mixture, source_override="general_lean")
        if ex:
            examples.append(ex)
        else:
            reject_counts[reason or "unknown"] += 1
    for k, v in reject_counts.items():
        counts[f"general_rejected_{k}"] = v
    counts["accepted_general"] = len(examples)
    return examples, counts


def deduplicate_and_cap(examples: List[SFTExample], max_variants: int) -> List[SFTExample]:
    seen = set()
    by_theorem: Dict[str, List[SFTExample]] = defaultdict(list)
    for ex in examples:
        key = (ex.theorem_hash, ex.proof_hash)
        if key in seen:
            continue
        seen.add(key)
        by_theorem[ex.theorem_hash].append(ex)

    def score(ex: SFTExample) -> Tuple[float, int, int]:
        struct_bonus = sum(ex.completion.count(tok) for tok in ["have ", "suffices ", "calc", "cases ", "induction "])
        length = len(ex.completion)
        # Prefer higher weight, then structural richness, then shorter proof.
        return (ex.weight, struct_bonus, -length)

    capped: List[SFTExample] = []
    for _, group in by_theorem.items():
        group = sorted(group, key=score, reverse=True)
        capped.extend(group[:max_variants])
    return capped


def group_split(
    examples: List[SFTExample], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[SFTExample], List[SFTExample], List[SFTExample]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    rng = random.Random(seed)
    groups: Dict[str, List[SFTExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.parent_theorem_id].append(ex)
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    n = len(group_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train : n_train + n_val])
    test_ids = set(group_ids[n_train + n_val :])
    train = [ex for gid in train_ids for ex in groups[gid]]
    val = [ex for gid in val_ids for ex in groups[gid]]
    test = [ex for gid in test_ids for ex in groups[gid]]
    return train, val, test


def materialize_weighted(examples: List[SFTExample], cfg: RunConfig, split_name: str) -> List[SFTExample]:
    if not cfg.mixture.enabled:
        return examples
    rng = random.Random(cfg.experiment.seed + (1 if split_name == "train" else 2))
    materialized: List[SFTExample] = []
    for ex in examples:
        repeat = max(1, min(cfg.mixture.max_effective_repeat_per_example, int(math.floor(ex.weight))))
        frac = ex.weight - math.floor(ex.weight)
        materialized.extend([ex] * repeat)
        if frac > 0 and rng.random() < frac and repeat < cfg.mixture.max_effective_repeat_per_example:
            materialized.append(ex)
    rng.shuffle(materialized)
    if split_name == "train" and cfg.mixture.max_train_examples:
        materialized = materialized[: cfg.mixture.max_train_examples]
    if split_name != "train" and cfg.mixture.max_eval_examples:
        materialized = materialized[: cfg.mixture.max_eval_examples]
    return materialized


# -----------------------------
# Tokenization dataset
# -----------------------------


class PromptCompletionDataset:
    def __init__(self, examples: List[SFTExample], tokenizer: Any, cfg: RunConfig, split_name: str):
        self.examples = examples
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.split_name = split_name
        self.items: List[Dict[str, Any]] = []
        self.skipped: Dict[str, int] = defaultdict(int)
        self._build()

    def _encode_no_special(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build(self) -> None:
        eos = self.tokenizer.eos_token_id
        for ex in self.examples:
            prompt_ids = self._encode_no_special(ex.prompt)
            completion_ids = self._encode_no_special(ex.completion)
            if self.cfg.data.add_eos_token and eos is not None:
                if not completion_ids or completion_ids[-1] != eos:
                    completion_ids = completion_ids + [eos]
            input_ids = prompt_ids + completion_ids
            if len(input_ids) > self.cfg.data.max_seq_length:
                if self.cfg.data.skip_overlength_examples:
                    self.skipped["overlength"] += 1
                    continue
                input_ids = input_ids[: self.cfg.data.max_seq_length]
            labels = list(input_ids)
            if self.cfg.training.completion_only_loss:
                prompt_len = min(len(prompt_ids), len(labels))
                labels[:prompt_len] = [-100] * prompt_len
            if all(x == -100 for x in labels):
                self.skipped["all_labels_masked"] += 1
                continue
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "metadata": {
                        "id": ex.id,
                        "parent_theorem_id": ex.parent_theorem_id,
                        "source_type": ex.source_type,
                    },
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        return {k: v for k, v in item.items() if k != "metadata"}


def make_collator(tokenizer: Any):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


# -----------------------------
# Training
# -----------------------------


def dtype_from_string(name: str):
    import torch

    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return None


def filter_init_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def load_model_and_tokenizer(cfg: RunConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "torch_dtype": dtype_from_string(cfg.model.torch_dtype),
        "use_cache": cfg.model.use_cache,
    }
    if cfg.model.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation

    if cfg.lora.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("QLoRA requires bitsandbytes-compatible transformers BitsAndBytesConfig.") from e
        compute_dtype = dtype_from_string(cfg.lora.qlora_compute_dtype) or torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.lora.qlora_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.lora.qlora_double_quant,
        )
        model_kwargs["device_map"] = "auto"

    # Some model implementations may not support flash_attention_2. Retry without it if needed.
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_name_or_path, **model_kwargs)
    except Exception as e:
        if "attn_implementation" in model_kwargs:
            print(f"[warn] model load failed with attn_implementation={model_kwargs['attn_implementation']}: {e}")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_name_or_path, **model_kwargs)
        else:
            raise

    if cfg.training.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model, "config"):
            model.config.use_cache = False

    if cfg.lora.enabled:
        from peft import LoraConfig, get_peft_model

        if cfg.lora.use_qlora:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.training.gradient_checkpointing)

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

    return model, tokenizer


def build_training_args(cfg: RunConfig):
    from transformers import TrainingArguments

    kwargs: Dict[str, Any] = {
        "output_dir": cfg.training.output_dir,
        "overwrite_output_dir": cfg.training.overwrite_output_dir,
        "num_train_epochs": cfg.training.num_train_epochs,
        "learning_rate": cfg.training.learning_rate,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "warmup_ratio": cfg.training.warmup_ratio,
        "weight_decay": cfg.training.weight_decay,
        "max_grad_norm": cfg.training.max_grad_norm,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
        "bf16": cfg.training.bf16,
        "fp16": cfg.training.fp16,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "optim": cfg.training.optim,
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "save_total_limit": cfg.training.save_total_limit,
        "report_to": [] if cfg.training.report_to == "none" else cfg.training.report_to.split(","),
        "dataloader_num_workers": cfg.training.dataloader_num_workers,
        "remove_unused_columns": cfg.training.remove_unused_columns,
        "group_by_length": cfg.training.group_by_length,
        "save_safetensors": cfg.training.save_safetensors,
        "torch_compile": cfg.training.torch_compile,
        "tf32": cfg.training.tf32,
        "load_best_model_at_end": False,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = cfg.training.eval_steps
    return TrainingArguments(**filter_init_kwargs(TrainingArguments, kwargs))


def train(cfg: RunConfig, train_ds: PromptCompletionDataset, val_ds: PromptCompletionDataset) -> None:
    from transformers import Trainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    args = build_training_args(cfg)
    collator = make_collator(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    if cfg.training.merge_lora_after_training and cfg.lora.enabled:
        try:
            merged_dir = Path(cfg.training.output_dir) / "merged_model"
            ensure_dir(merged_dir)
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_dir, safe_serialization=cfg.training.save_safetensors)
            tokenizer.save_pretrained(merged_dir)
            print(f"[ok] merged LoRA model saved to {merged_dir}")
        except Exception as e:
            print(f"[warn] failed to merge LoRA adapter: {e}")


# -----------------------------
# Optional external eval
# -----------------------------


def run_optional_lean_eval(cfg: RunConfig) -> Optional[int]:
    if not cfg.lean_eval.enabled or not cfg.lean_eval.run_after_training or not cfg.lean_eval.command:
        return None
    cmd = cfg.lean_eval.command.format(output_dir=cfg.training.output_dir)
    print(f"[eval] running: {cmd}")
    return subprocess.call(cmd, shell=True)


# -----------------------------
# Main
# -----------------------------


def apply_cli_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    if args.run06_dir:
        cfg.data.run06_dir = args.run06_dir
    if args.general_lean_data:
        cfg.data.general_lean_data = args.general_lean_data
    if args.base_model:
        cfg.model.base_model_name_or_path = args.base_model
        cfg.model.tokenizer_name_or_path = args.base_model
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.max_seq_length:
        cfg.data.max_seq_length = args.max_seq_length
    if args.lora_r:
        cfg.lora.r = args.lora_r
    if args.lora_alpha:
        cfg.lora.alpha = args.lora_alpha
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.epochs:
        cfg.training.num_train_epochs = args.epochs
    if args.gradient_accumulation_steps:
        cfg.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.per_device_train_batch_size:
        cfg.training.per_device_train_batch_size = args.per_device_train_batch_size
    if args.dry_run:
        cfg.experiment.dry_run = True
    if args.overwrite:
        cfg.training.overwrite_output_dir = True
    if args.qlora:
        cfg.lora.use_qlora = True
    if args.no_lora:
        cfg.lora.enabled = False
    if args.no_flash_attention:
        cfg.model.attn_implementation = None
    if args.debug:
        cfg.experiment.debug = True
    return cfg


def build_and_save_datasets(cfg: RunConfig) -> Tuple[PromptCompletionDataset, PromptCompletionDataset, PromptCompletionDataset, Dict[str, Any]]:
    out = ensure_dir(cfg.training.output_dir)
    mined, mined_counts = load_mined_examples(cfg)
    general, general_counts = load_general_examples(cfg)
    all_examples = mined + general
    if not all_examples and cfg.data.keep_files_if_no_new_data:
        print("[warn] no examples loaded, but keep_files_if_no_new_data=true")
    if not all_examples:
        raise RuntimeError(
            "No SFT examples were loaded. Check run06_dir and expected files: "
            f"{cfg.data.final_trajectories_file}, {cfg.data.final_curriculum_items_file}."
        )

    deduped = deduplicate_and_cap(all_examples, cfg.data.max_variants_per_theorem)
    train_ex, val_ex, test_ex = group_split(
        deduped,
        cfg.data.train_ratio,
        cfg.data.val_ratio,
        cfg.data.test_ratio,
        cfg.data.seed,
    )
    train_ex = materialize_weighted(train_ex, cfg, "train")
    val_ex = materialize_weighted(val_ex, cfg, "val")
    test_ex = materialize_weighted(test_ex, cfg, "test")

    write_jsonl(out / cfg.data.output_processed_train, [ex.to_json() for ex in train_ex])
    write_jsonl(out / cfg.data.output_processed_val, [ex.to_json() for ex in val_ex])
    write_jsonl(out / cfg.data.output_processed_test, [ex.to_json() for ex in test_ex])

    # Tokenizer may be slow to load, but doing it here catches sequence-length issues before training.
    from transformers import AutoTokenizer

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = PromptCompletionDataset(train_ex, tokenizer, cfg, "train")
    val_ds = PromptCompletionDataset(val_ex, tokenizer, cfg, "val")
    test_ds = PromptCompletionDataset(test_ex, tokenizer, cfg, "test")

    report = {
        "experiment": cfg.experiment.name,
        "loaded": {**mined_counts, **general_counts},
        "num_examples_raw": len(all_examples),
        "num_examples_deduped": len(deduped),
        "num_examples_train_materialized": len(train_ex),
        "num_examples_val_materialized": len(val_ex),
        "num_examples_test_materialized": len(test_ex),
        "num_tokenized_train": len(train_ds),
        "num_tokenized_val": len(val_ds),
        "num_tokenized_test": len(test_ds),
        "tokenization_skipped_train": dict(train_ds.skipped),
        "tokenization_skipped_val": dict(val_ds.skipped),
        "tokenization_skipped_test": dict(test_ds.skipped),
        "source_counts_deduped": dict(source_count(deduped)),
        "config": dataclass_to_dict(cfg),
    }
    write_json(out / "dataset_report.json", report)
    return train_ds, val_ds, test_ds, report


def source_count(examples: List[SFTExample]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.source_type] += 1
    return counts


def save_config(cfg: RunConfig) -> None:
    out = ensure_dir(cfg.training.output_dir)
    cfg_dict = dataclass_to_dict(cfg)
    write_json(out / "training_config.json", cfg_dict)
    if yaml is not None:
        with open(out / "training_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, allow_unicode=True, sort_keys=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA/QLoRA SFT on verified ATP trajectories from run_06.")
    p.add_argument("--config", type=str, default=None, help="YAML config path.")
    p.add_argument("--run06-dir", type=str, default=None)
    p.add_argument("--general-lean-data", type=str, default=None)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--per-device-train-batch-size", type=int, default=None)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--no-flash-attention", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    set_seed(cfg.experiment.seed)

    out = Path(cfg.training.output_dir)
    if out.exists() and cfg.training.overwrite_output_dir and not cfg.experiment.dry_run:
        # Be conservative: only delete known SFT output directory if it contains our config/report or is empty.
        if any((out / name).exists() for name in ["training_config.json", "dataset_report.json"]) or not any(out.iterdir()):
            shutil.rmtree(out)
    ensure_dir(out)
    save_config(cfg)

    print(f"[run_07] output_dir={cfg.training.output_dir}")
    print(f"[run_07] base_model={cfg.model.base_model_name_or_path}")
    print(f"[run_07] run06_dir={cfg.data.run06_dir}")

    train_ds, val_ds, test_ds, report = build_and_save_datasets(cfg)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2, ensure_ascii=False))

    if cfg.experiment.dry_run:
        print("[dry-run] datasets built; training skipped.")
        return

    if len(train_ds) == 0:
        raise RuntimeError("Tokenized train dataset is empty after filtering/truncation.")

    train(cfg, train_ds, val_ds)
    rc = run_optional_lean_eval(cfg)
    if rc is not None and rc != 0:
        print(f"[warn] optional Lean eval command exited with code {rc}")
    print("[ok] run_07 SFT finished.")


if __name__ == "__main__":
    main()

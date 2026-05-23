#!/usr/bin/env python3
"""

Сбор учебных данных из доказательств Lean через 2 канала:

1.MCSP, Поиск в полудоказательствах (для нелинейных структур)
Вырезает блоки доказательств, заменяя их на заглушки `sorry`,
проверяет компиляцию основного кода с этой заглушкой

2. Fallback по префиксам (для линейных доказательств)
Извлекает рабочие префиксы из незавершенных цепочек,
генерирует продолжения через адаптивный сэмплинг (16 -> 32 -> 64).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
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
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_WHOLE_SEQ_PATH = "outputs/domain_selected_whole_seq_and_sorry_datasets/domain_20x_whole_seq_trajectories.jsonl"



@dataclass
class ExperimentConfig:
    experiment_name: str = "mine_failed_problems_v1"
    seed: int = 42
    save_every: int = 250
    resume: bool = True
    dry_run: bool = False


@dataclass
class DatasetConfig:
    source_name: str = "lean_workbook_plus"
    domain: str = "inequalities_real_algebra"
    max_theorems: Optional[int] = 5000
    theorem_id_key: str = "base_task_id"
    formal_statement_key: str = "formal_statement"
    natural_language_key: str = "natural_language_statement"
    imports_key: str = "imports"
    proof_key_candidates: List[str] = field(default_factory=lambda: ["proof", "tactic", "proof_text", "generated_proof"])
    accept_nested_trajectory_schema: bool = True
    attempt_status_filter: List[str] = field(default_factory=lambda: ["failed"])
    include_symbols: List[str] = field(default_factory=lambda: ["\u211d", "\u2102", "\u2115", "\u2124", "Real", "\u2264", "<", "\u2265", ">", "^2", "abs"])
    include_tactics_hint: List[str] = field(default_factory=lambda: ["nlinarith", "linarith", "ring", "positivity", "norm_num", "field_simp"])
    exclude_symbols: List[str] = field(default_factory=lambda: ["Measure", "Topology", "CategoryTheory"])
    apply_domain_filter: bool = False


@dataclass
class IOConfig:
    theorems_jsonl: str = DEFAULT_WHOLE_SEQ_PATH
    attempts_jsonl: Optional[str] = DEFAULT_WHOLE_SEQ_PATH
    out_dir: str = "outputs/run_04_mine_failed_problems"
    write_attempts: bool = True
    write_search_jobs: bool = True
    write_frontier_theorems: bool = True
    overwrite: bool = False


@dataclass
class LeanConfig:
    project_root: str = "."
    use_lake_env: bool = True
    lean_cmd: str = "lean"
    timeout_full_proof_sec: int = 20
    timeout_suffix_sec: int = 15
    max_workers: int = 1
    default_imports: List[str] = field(default_factory=lambda: ["Mathlib"])
    allow_sorry_in_mcsp_check: bool = True
    disallow_sorry_in_final_proofs: bool = True
    keep_temp_files: bool = False
    verbose_errors: bool = False


@dataclass
class FullGenerationConfig:
    enabled: bool = False
    model_name: str = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
    engine: str = "none"  # vllm | none
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    samples_per_theorem: int = 16
    structured_cot_samples: int = 8
    flat_samples: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1024
    batch_size: int = 8
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\ntheorem ", "\n\nlemma ", "#check"])


@dataclass
class CompletionConfig:
    enabled: bool = False
    n_probe: int = 16
    n_main: int = 32
    n_confirm: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 768
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\ntheorem ", "\n\nlemma ", "#check"])


@dataclass
class ProofStyleConfig:
    min_structural_blocks_for_mcsp: int = 1
    min_nested_by_blocks_for_mcsp: int = 1
    max_automation_line_ratio_for_mcsp: float = 0.85
    max_lines_for_flat: int = 5
    structural_markers: List[str] = field(default_factory=lambda: [
        "have ", "suffices ", "show ", "calc", "constructor", "cases ", "induction ", "refine ", "·", " by"
    ])
    automation_markers: List[str] = field(default_factory=lambda: [
        "simp", "simp_all", "aesop", "omega", "linarith", "nlinarith", "norm_num", "ring"
    ])


@dataclass
class MCSPConfig:
    enabled: bool = True
    take_all_sorry_compilable_sketches: bool = True
    max_holes_per_proof: int = 8
    reject_single_giant_hole: bool = True
    reject_if_hole_ratio_above: float = 0.80
    min_hole_body_lines: int = 1
    max_hole_body_lines: int = 40
    allow_line_holes: bool = True
    allow_block_holes: bool = True
    allow_structural_block_holes: bool = True
    search_max_nodes_strong: int = 512
    search_max_nodes_weak: int = 256
    search_max_nodes_giant: int = 64
    search_max_depth_strong: int = 16
    search_max_depth_weak: int = 12
    search_max_depth_giant: int = 8
    expansions_per_node: int = 4
    use_encoder_ranker: bool = True
    accept_solved_at_main: int = 1
    accept_solved_at_confirm: int = 1
    short_solution_len_threshold: int = 8


@dataclass
class PrefixFallbackConfig:
    enabled: bool = True
    max_prefixes_per_attempt: int = 5
    include_first_error_prefix: bool = True
    include_after_structural_tactic: bool = True
    include_after_branching_tactic: bool = True
    include_after_valid_progress_window: bool = True
    include_random_valid_prefix: int = 1
    min_prefix_lines: int = 1
    max_prefix_lines: int = 40
    short_suffix_len_threshold: int = 8
    probe_accept_solved_count: int = 2
    escalate_if_solved_count: int = 1
    escalate_if_valid_partial_rate: float = 0.35
    escalate_if_max_valid_suffix_steps: int = 4
    confirm_if_solved_count_main: int = 1
    confirm_if_valid_partial_rate_main: float = 0.50


@dataclass
class QualityConfig:
    min_quality_for_curriculum: str = "medium"  # high | medium
    save_uncertain_for_later: bool = True
    deduplicate: bool = True
    wilson_z: float = 1.96
    score_solved_weight: float = 3.0
    score_wilson_weight: float = 1.5
    score_valid_partial_weight: float = 1.0
    score_short_suffix_weight: float = 0.7
    score_unique_suffix_weight: float = 0.5
    score_duplicate_penalty: float = 0.7
    score_prefix_len_penalty: float = 0.3


@dataclass
class EncoderRankerConfig:
    enabled: bool = False
    ranker_path: Optional[str] = None  # e.g. "src.search:score_candidates"
    higher_is_better: bool = True


@dataclass
class RunConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    io: IOConfig = field(default_factory=IOConfig)
    lean: LeanConfig = field(default_factory=LeanConfig)
    full_generation: FullGenerationConfig = field(default_factory=FullGenerationConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    proof_style: ProofStyleConfig = field(default_factory=ProofStyleConfig)
    mcsp: MCSPConfig = field(default_factory=MCSPConfig)
    prefix_fallback: PrefixFallbackConfig = field(default_factory=PrefixFallbackConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    encoder_ranker: EncoderRankerConfig = field(default_factory=EncoderRankerConfig)


# -----------------------------
# Utility functions
# -----------------------------


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: str, prefix: str = "id") -> str:
    h = sha1_text("\n---\n".join(parts))[:16]
    return f"{prefix}_{h}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    lines = text.splitlines()
    return "\n".join(pad + line if line.strip() else line for line in lines)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:lean|lean4)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def remove_comments_for_hash(text: str) -> str:
    text = re.sub(r"--.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def line_count(text: str) -> int:
    return len([x for x in text.splitlines() if x.strip()])


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def nested_get(obj: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def first_present_nested(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        val = nested_get(obj, key, None) if "." in key else obj.get(key)
        if val is not None and val != "":
            return val
    return default


def coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def record_theorem_id(rec: Dict[str, Any], cfg: RunConfig) -> str:
    val = first_present_nested(rec, [
        cfg.dataset.theorem_id_key,
        "theorem_id",
        "base_task_id",
        "task_id",
        "id",
        "problem.problem_id",
        "problem_id",
        "trajectory_id",
    ])
    return str(val) if val else stable_id(json.dumps(rec, sort_keys=True, ensure_ascii=False), prefix="thm")


def record_formal_statement(rec: Dict[str, Any], cfg: RunConfig) -> str:
    val = first_present_nested(rec, [
        cfg.dataset.formal_statement_key,
        "formal",
        "statement",
        "theorem",
        "problem.formal_statement",
    ])
    return coerce_text(val)


def record_natural_language(rec: Dict[str, Any], cfg: RunConfig) -> str:
    val = first_present_nested(rec, [
        cfg.dataset.natural_language_key,
        "nl_statement",
        "natural_language",
        "problem.natural_language_statement",
    ])
    if not val and isinstance(rec.get("problem"), str):
        val = rec.get("problem")
    return coerce_text(val)


def record_proof_text(rec: Dict[str, Any], cfg: RunConfig) -> str:
    keys = [
        "proof_text",
        "generated_proof",
        "proof",
        "tactic",
        "completion",
        "solution.tactic_script",
        "failure.original_failed_whole_seq_script",
        "original_failed_whole_seq_script",
        "sorry_solution.tactic_script",
    ] + cfg.dataset.proof_key_candidates
    val = first_present_nested(rec, keys)
    if isinstance(val, list):
        return "\n".join(str(x) for x in val)
    return coerce_text(val)


def load_config(path: Optional[str]) -> RunConfig:
    cfg = RunConfig()
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml or use JSON.") from e
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
    return update_dataclass(cfg, raw)


def update_dataclass(obj: Any, patch: Dict[str, Any]) -> Any:
    for k, v in patch.items():
        if not hasattr(obj, k):
            raise KeyError(f"Unknown config key: {k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            setattr(obj, k, update_dataclass(cur, v))
        else:
            setattr(obj, k, v)
    return obj

# Lean proof assembly and verification


@dataclass
class LeanResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    elapsed_sec: float
    file_path: Optional[str] = None
    has_sorry_warning: bool = False
    error_summary: str = ""


class LeanVerifier:
    def __init__(self, cfg: LeanConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root).resolve()
        if cfg.use_lake_env:
            self.cmd_prefix = ["lake", "env", cfg.lean_cmd]
        else:
            self.cmd_prefix = [cfg.lean_cmd]

    def verify_code(self, code: str, timeout_sec: int, allow_sorry: bool) -> LeanResult:
        with tempfile.NamedTemporaryFile("w", suffix=".lean", encoding="utf-8", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        start = time.time()
        try:
            proc = subprocess.run(
                self.cmd_prefix + [tmp_path],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
            elapsed = time.time() - start
            combined = proc.stdout + "\n" + proc.stderr
            has_sorry_warning = "declaration uses 'sorry'" in combined or "uses 'sorry'" in combined
            ok = proc.returncode == 0
            if ok and not allow_sorry and has_sorry_warning:
                ok = False
            summary = summarize_lean_error(combined)
            return LeanResult(ok, proc.stdout, proc.stderr, proc.returncode, elapsed, tmp_path if self.cfg.keep_temp_files else None, has_sorry_warning, summary)
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            return LeanResult(False, e.stdout or "", e.stderr or "", 124, elapsed, tmp_path if self.cfg.keep_temp_files else None, False, "timeout")
        except Exception as e:
            elapsed = time.time() - start
            return LeanResult(False, "", repr(e), 1, elapsed, tmp_path if self.cfg.keep_temp_files else None, False, repr(e))
        finally:
            if not self.cfg.keep_temp_files:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def summarize_lean_error(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    important = []
    for l in lines:
        low = l.lower()
        if "error:" in low or "warning:" in low or "unknown" in low or "failed" in low or "unsolved" in low or "timeout" in low:
            important.append(l)
    return " | ".join(important[:5])


def theorem_header_from_formal(formal_statement: str) -> str:
    s = strip_code_fences(formal_statement).strip()
    s = re.sub(r"^import\s+.*?$", "", s, flags=re.MULTILINE).strip()
    m = re.search(r"(?s)(.*?)(:=\s*by\b)", s)
    if m:
        return m.group(1).rstrip() + " := by"
    if re.search(r"\bby\s*$", s):
        return s
    if re.search(r"\b(theorem|lemma|example)\b", s):
        return s.rstrip() + " := by"
    raise ValueError("Could not parse Lean statement")


def extract_existing_proof_from_formal(formal_statement: str) -> Optional[str]:
    s = strip_code_fences(formal_statement).strip()
    m = re.search(r"(?s):=\s*by\s*(.*)$", s)
    if not m:
        return None
    body = m.group(1).strip()
    return body or None


def extract_imports(record: Dict[str, Any], cfg: LeanConfig, imports_key: str = "imports") -> List[str]:
    raw = first_present_nested(record, [imports_key, "problem.imports", "imports"])
    if isinstance(raw, list):
        imports = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str) and raw.strip():
        imports = [x.strip() for x in raw.splitlines() if x.strip()]
    else:
        imports = cfg.default_imports
    cleaned = []
    for imp in imports:
        if imp.startswith("import "):
            cleaned.append(imp)
        else:
            cleaned.append(f"import {imp}")
    return cleaned


def build_lean_code(imports: Sequence[str], formal_statement: str, proof_body_or_full: str) -> str:
    proof = strip_code_fences(proof_body_or_full).strip()
    proof = strip_surrounding_theorem_if_present(proof)
    header = theorem_header_from_formal(formal_statement)
    code = "\n".join(imports).strip() + "\n\n" + header + "\n" + indent(proof, 2) + "\n"
    return code


def strip_surrounding_theorem_if_present(text: str) -> str:
    s = text.strip()
    m = re.search(r"(?s)\b(theorem|lemma|example)\b.*?:=\s*by\s*(.*)$", s)
    if m:
        return m.group(2).strip()
    if s.startswith("by\n") or s == "by" or s.startswith("by "):
        return re.sub(r"^by\s*", "", s, count=1).strip()
    return s


def proof_with_sorry_after_prefix(prefix_body: str) -> str:
    prefix_body = prefix_body.strip()
    if not prefix_body:
        return "sorry"
    return prefix_body.rstrip() + "\n" + "sorry"


# Generator adapters

@dataclass
class Generation:
    text: str
    logprob: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGenerator:
    def generate(self, prompts: List[str], n: int, temperature: float, top_p: float, max_new_tokens: int, stop: Sequence[str]) -> List[List[Generation]]:
        raise NotImplementedError


class NoOpGenerator(BaseGenerator):
    def generate(self, prompts: List[str], n: int, temperature: float, top_p: float, max_new_tokens: int, stop: Sequence[str]) -> List[List[Generation]]:
        raise RuntimeError(
            "No generator configured."
        )


class VLLMGenerator(BaseGenerator):
    def __init__(self, cfg: FullGenerationConfig):
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError("vLLM is not installed, but full_generation.engine='vllm'.") from e
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=cfg.model_name,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel_size,
            max_model_len=cfg.max_model_len,
            trust_remote_code=True,
        )

    def generate(self, prompts: List[str], n: int, temperature: float, top_p: float, max_new_tokens: int, stop: Sequence[str]) -> List[List[Generation]]:
        params = self.SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=list(stop) if stop else None)
        outputs = self.llm.generate(prompts, params)
        result: List[List[Generation]] = []
        for out in outputs:
            gens = []
            for o in out.outputs:
                gens.append(Generation(text=o.text, logprob=None, metadata={"finish_reason": getattr(o, "finish_reason", None)}))
            result.append(gens)
        return result


def make_generator(cfg: RunConfig) -> BaseGenerator:
    if cfg.full_generation.engine == "vllm":
        return VLLMGenerator(cfg.full_generation)
    return NoOpGenerator()



STRUCTURED_COT_INSTRUCTION = """Complete the following Lean 4 theorem.
Use meaningful intermediate Lean blocks where useful: `have`, `suffices`, `show`, `calc`, `constructor`, `cases`, `induction`, or `refine`.
Avoid replacing the whole proof by one large automation call unless it is genuinely direct.
Return only Lean proof code after `by`; do not include markdown fences.
"""

FLAT_INSTRUCTION = """Complete the following Lean 4 theorem.
Return only Lean proof code after `by`; do not include markdown fences.
"""

HOLE_FILL_INSTRUCTION = """Fill the `sorry` hole in this Lean 4 proof.
Return only the Lean code that should replace this exact `sorry`; do not include markdown fences or repeat the theorem.
"""

PREFIX_CONTINUE_INSTRUCTION = """Continue this Lean 4 proof from the valid prefix.
Return only the remaining Lean proof code after the prefix; do not include markdown fences or repeat the theorem.
"""


def prompt_full_proof(record: Dict[str, Any], mode: str) -> str:
    formal = record["formal_statement"]
    nl = record.get("natural_language_statement", "")
    header = theorem_header_from_formal(formal)
    instruction = STRUCTURED_COT_INSTRUCTION if mode == "structured_cot" else FLAT_INSTRUCTION
    parts = [instruction]
    if nl:
        parts.append(f"Natural-language statement:\n{nl}")
    parts.append(f"Lean theorem:\n{header}")
    return "\n\n".join(parts)


def prompt_fill_hole(formal_statement: str, semi_proof_body: str, hole_marker: str = "sorry") -> str:
    header = theorem_header_from_formal(formal_statement)
    return f"{HOLE_FILL_INSTRUCTION}\n\nLean theorem with hole:\n{header}\n{indent(semi_proof_body, 2)}\n"


def prompt_continue_prefix(formal_statement: str, prefix_body: str, state_comment: Optional[str] = None) -> str:
    header = theorem_header_from_formal(formal_statement)
    parts = [PREFIX_CONTINUE_INSTRUCTION]
    if state_comment:
        parts.append(f"Current Lean state/comment:\n{state_comment}")
    parts.append(f"Lean theorem prefix:\n{header}\n{indent(prefix_body.strip(), 2)}")
    return "\n\n".join(parts)


#  opt encoder ranker



class EncoderRanker:
    def __init__(self, cfg: EncoderRankerConfig):
        self.cfg = cfg
        self.fn: Optional[Callable[..., List[float]]] = None
        if cfg.enabled and cfg.ranker_path:
            mod_name, fn_name = cfg.ranker_path.split(":", 1)
            mod = importlib.import_module(mod_name)
            self.fn = getattr(mod, fn_name)

    def rank(self, context: Dict[str, Any], candidates: List[str]) -> List[str]:
        if not candidates:
            return candidates
        if not self.fn:
            return candidates
        try:
            scores = self.fn(context, candidates)
            pairs = list(zip(candidates, scores))
            pairs.sort(key=lambda x: x[1], reverse=self.cfg.higher_is_better)
            return [p[0] for p in pairs]
        except Exception:
            return candidates


# Dataset and attempts


@dataclass
class TheoremRecord:
    theorem_id: str
    formal_statement: str
    natural_language_statement: str = ""
    imports: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProofAttempt:
    attempt_id: str
    theorem_id: str
    proof_text: str
    mode: str = "unknown"
    source: str = "input"
    raw: Dict[str, Any] = field(default_factory=dict)


def load_theorems(path: str, cfg: RunConfig) -> List[TheoremRecord]:
    rows: List[TheoremRecord] = []
    seen: set[str] = set()
    for rec in read_jsonl(path):
        tid = record_theorem_id(rec, cfg)
        if tid in seen:
            continue
        formal = record_formal_statement(rec, cfg)
        if not formal:
            continue
        nl = record_natural_language(rec, cfg)
        imports = extract_imports(rec, cfg.lean, cfg.dataset.imports_key)
        tr = TheoremRecord(tid, formal, nl, imports, rec)
        if cfg.dataset.apply_domain_filter and not domain_filter(tr, cfg.dataset):
            continue
        rows.append(tr)
        seen.add(tid)
        if cfg.dataset.max_theorems and len(rows) >= cfg.dataset.max_theorems:
            break
    return rows


def domain_filter(tr: TheoremRecord, cfg: DatasetConfig) -> bool:
    text = tr.formal_statement + "\n" + tr.natural_language_statement + "\n" + json.dumps(tr.raw, ensure_ascii=False)
    if cfg.exclude_symbols and any(x in text for x in cfg.exclude_symbols):
        return False
    if cfg.include_symbols and any(x in text for x in cfg.include_symbols):
        return True
    if cfg.include_tactics_hint and any(x in text for x in cfg.include_tactics_hint):
        return True
    return False


def load_attempts(path: str, cfg: RunConfig) -> Dict[str, List[ProofAttempt]]:
    attempts: Dict[str, List[ProofAttempt]] = {}
    for rec in read_jsonl(path):
        status = rec.get("status")
        if cfg.dataset.attempt_status_filter and status is not None and str(status) not in set(cfg.dataset.attempt_status_filter):
            continue
        theorem_id = record_theorem_id(rec, cfg)
        if not theorem_id:
            continue
        proof_text = record_proof_text(rec, cfg)
        if not proof_text:
            continue
        aid = str(first_present_nested(rec, ["attempt_id", "trajectory_id", "id"], None) or stable_id(theorem_id, proof_text, prefix="attempt"))
        attempts.setdefault(theorem_id, []).append(
            ProofAttempt(
                aid,
                theorem_id,
                proof_text,
                str(first_present_nested(rec, ["mode", "trajectory_kind"], "input")),
                str(first_present_nested(rec, ["source", "problem.source"], "input")),
                rec,
            )
        )
    return attempts


def generate_full_attempts(theorems: List[TheoremRecord], generator: BaseGenerator, cfg: RunConfig) -> Dict[str, List[ProofAttempt]]:
    attempts: Dict[str, List[ProofAttempt]] = {}
    gen_cfg = cfg.full_generation
    batch: List[Tuple[TheoremRecord, str, str]] = []
    for tr in theorems:
        for i in range(gen_cfg.structured_cot_samples):
            batch.append((tr, "structured_cot", prompt_full_proof(dataclasses.asdict(tr), "structured_cot")))
        for i in range(gen_cfg.flat_samples):
            batch.append((tr, "flat_non_cot", prompt_full_proof(dataclasses.asdict(tr), "flat_non_cot")))
    for start in range(0, len(batch), gen_cfg.batch_size):
        sub = batch[start:start + gen_cfg.batch_size]
        prompts = [x[2] for x in sub]
        outputs = generator.generate(prompts, n=1, temperature=gen_cfg.temperature, top_p=gen_cfg.top_p, max_new_tokens=gen_cfg.max_new_tokens, stop=gen_cfg.stop_sequences)
        for (tr, mode, _), outs in zip(sub, outputs):
            for g in outs:
                proof = strip_code_fences(g.text)
                aid = stable_id(tr.theorem_id, mode, proof, str(time.time()), prefix="attempt")
                attempts.setdefault(tr.theorem_id, []).append(ProofAttempt(aid, tr.theorem_id, proof, mode, "generated", {"logprob": g.logprob, **g.metadata}))
    return attempts

# Proof style classification

@dataclass
class ProofStyle:
    label: str
    num_lines: int
    structural_blocks: int
    nested_by_blocks: int
    automation_lines: int
    automation_ratio: float
    has_branching: bool
    marker_counts: Dict[str, int]


def classify_proof_style(proof_body: str, cfg: ProofStyleConfig) -> ProofStyle:
    body = strip_surrounding_theorem_if_present(strip_code_fences(proof_body))
    lines = [l for l in body.splitlines() if l.strip()]
    marker_counts: Dict[str, int] = {}
    structural_blocks = 0
    for m in cfg.structural_markers:
        c = body.count(m)
        marker_counts[m.strip()] = c
        structural_blocks += c
    nested_by_blocks = len(re.findall(r":=\s*by\b|\bby\s*$", body, flags=re.MULTILINE))
    automation_lines = 0
    for l in lines:
        if any(re.search(rf"\b{re.escape(m)}\b", l) for m in cfg.automation_markers):
            automation_lines += 1
    automation_ratio = automation_lines / max(1, len(lines))
    has_branching = any(re.search(rf"\b{x}\b", body) for x in ["cases", "induction", "constructor", "rcases", "by_cases"])
    if len(lines) <= cfg.max_lines_for_flat and structural_blocks < cfg.min_structural_blocks_for_mcsp:
        label = "FLAT_TACTIC"
    elif structural_blocks >= cfg.min_structural_blocks_for_mcsp and nested_by_blocks >= cfg.min_nested_by_blocks_for_mcsp and automation_ratio <= cfg.max_automation_line_ratio_for_mcsp:
        label = "MCSP_CANDIDATE"
        if structural_blocks >= 4 and has_branching:
            label = "STRONG_MCSP_CANDIDATE"
    elif structural_blocks > 0:
        label = "SHALLOW_STRUCTURED"
    else:
        label = "FLAT_TACTIC"
    return ProofStyle(label, len(lines), structural_blocks, nested_by_blocks, automation_lines, automation_ratio, has_branching, marker_counts)


# MCSP stubs extraction

@dataclass
class HoleCandidate:
    hole_id: str
    theorem_id: str
    attempt_id: str
    semi_proof_body: str
    original_block: str
    hole_kind: str
    start_line: int
    end_line: int
    proof_style_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_hole_candidates(tr: TheoremRecord, attempt: ProofAttempt, style: ProofStyle, cfg: RunConfig) -> List[HoleCandidate]:
    body = strip_surrounding_theorem_if_present(strip_code_fences(attempt.proof_text)).strip()
    if not body:
        return []
    holes: List[HoleCandidate] = []
    if cfg.mcsp.allow_structural_block_holes:
        holes.extend(extract_structural_block_holes(tr, attempt, body, style, cfg))
    if cfg.mcsp.allow_block_holes:
        holes.extend(extract_indented_block_holes(tr, attempt, body, style, cfg))
    if cfg.mcsp.allow_line_holes:
        holes.extend(extract_line_holes(tr, attempt, body, style, cfg))
    dedup: Dict[str, HoleCandidate] = {}
    for h in holes:
        key = sha1_text(normalize_ws(h.semi_proof_body))
        if key not in dedup:
            dedup[key] = h
    result = list(dedup.values())
    result.sort(key=lambda h: (h.start_line, h.end_line - h.start_line))
    return result[:cfg.mcsp.max_holes_per_proof]


def extract_line_holes(tr: TheoremRecord, attempt: ProofAttempt, body: str, style: ProofStyle, cfg: RunConfig) -> List[HoleCandidate]:
    lines = body.splitlines()
    holes: List[HoleCandidate] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped in {"·", "case"}:
            continue
        if "sorry" in stripped:
            continue
        if not is_meaningful_hole_line(stripped, cfg):
            continue
        new_lines = list(lines)
        prefix_spaces = len(line) - len(line.lstrip(" "))
        new_lines[i] = " " * prefix_spaces + "sorry"
        original = line
        hole_id = stable_id(tr.theorem_id, attempt.attempt_id, str(i), original, prefix="hole")
        holes.append(HoleCandidate(hole_id, tr.theorem_id, attempt.attempt_id, "\n".join(new_lines), original, "line", i + 1, i + 1, style.label))
    return holes


def is_meaningful_hole_line(line: str, cfg: RunConfig) -> bool:
    if line.startswith("--"):
        return False
    if line.startswith("import "):
        return False
    if line in {"by", "exact?"}:
        return False
    if any(line.startswith(x) for x in ["have ", "suffices ", "show ", "calc", "constructor", "cases ", "induction ", "refine "]):
        return True
    if any(re.search(rf"\b{re.escape(x)}\b", line) for x in cfg.proof_style.automation_markers):
        return True
    return len(line.split()) >= 2


def extract_indented_block_holes(tr: TheoremRecord, attempt: ProofAttempt, body: str, style: ProofStyle, cfg: RunConfig) -> List[HoleCandidate]:
    lines = body.splitlines()
    holes: List[HoleCandidate] = []
    for start in range(len(lines)):
        line = lines[start]
        stripped = line.strip()
        if not stripped or "sorry" in stripped:
            continue
        if not any(stripped.startswith(x) for x in ["have ", "suffices ", "show ", "case ", "·", "calc"]):
            continue
        base_indent = len(line) - len(line.lstrip(" "))
        end = start
        for j in range(start + 1, len(lines)):
            sj = lines[j]
            if not sj.strip():
                end = j
                continue
            ij = len(sj) - len(sj.lstrip(" "))
            if ij <= base_indent and not sj.lstrip().startswith(("·", "case ")):
                break
            end = j
        if end < start:
            continue
        block_len = end - start + 1
        if block_len < cfg.mcsp.min_hole_body_lines or block_len > cfg.mcsp.max_hole_body_lines:
            continue
        new_lines = lines[:start] + [" " * base_indent + "sorry"] + lines[end + 1:]
        original = "\n".join(lines[start:end + 1])
        hole_id = stable_id(tr.theorem_id, attempt.attempt_id, str(start), str(end), original, prefix="hole")
        holes.append(HoleCandidate(hole_id, tr.theorem_id, attempt.attempt_id, "\n".join(new_lines), original, "block", start + 1, end + 1, style.label))
    return holes


def extract_structural_block_holes(tr: TheoremRecord, attempt: ProofAttempt, body: str, style: ProofStyle, cfg: RunConfig) -> List[HoleCandidate]:
    lines = body.splitlines()
    holes: List[HoleCandidate] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not any(stripped.startswith(x) for x in ["have ", "suffices ", "show "]):
            continue
        if ":= by" not in stripped and stripped != "by":
            continue
        base_indent = len(line) - len(line.lstrip(" "))
        end = i
        for j in range(i + 1, len(lines)):
            if not lines[j].strip():
                end = j
                continue
            ij = len(lines[j]) - len(lines[j].lstrip(" "))
            if ij <= base_indent:
                break
            end = j
        block_len = end - i + 1
        if block_len > cfg.mcsp.max_hole_body_lines:
            continue
        new_lines = lines[:i] + [" " * base_indent + "sorry"] + lines[end + 1:]
        original = "\n".join(lines[i:end + 1])
        hole_id = stable_id(tr.theorem_id, attempt.attempt_id, "struct", str(i), str(end), original, prefix="hole")
        holes.append(HoleCandidate(hole_id, tr.theorem_id, attempt.attempt_id, "\n".join(new_lines), original, "structural_block", i + 1, end + 1, style.label))
    return holes


def hole_ratio(hole: HoleCandidate, attempt_body: str) -> float:
    return line_count(hole.original_block) / max(1, line_count(attempt_body))


# Prefix extraction

@dataclass
class PrefixCandidate:
    prefix_id: str
    theorem_id: str
    attempt_id: str
    prefix_body: str
    reason: str
    end_line: int
    verification: Optional[LeanResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_prefix_candidates(tr: TheoremRecord, attempt: ProofAttempt, verifier: LeanVerifier, cfg: RunConfig) -> List[PrefixCandidate]:
    body = strip_surrounding_theorem_if_present(strip_code_fences(attempt.proof_text)).strip()
    lines = [l for l in body.splitlines() if l.strip()]
    if not lines:
        return []
    candidate_end_lines: List[Tuple[int, str]] = []
    if cfg.prefix_fallback.include_after_structural_tactic:
        for i, l in enumerate(lines):
            s = l.strip()
            if any(s.startswith(x) for x in ["have ", "suffices ", "show ", "calc", "refine "]):
                candidate_end_lines.append((i + 1, "after_structural_tactic"))
    if cfg.prefix_fallback.include_after_branching_tactic:
        for i, l in enumerate(lines):
            s = l.strip()
            if any(re.search(rf"\b{x}\b", s) for x in ["cases", "induction", "constructor", "rcases", "by_cases"]):
                candidate_end_lines.append((i + 1, "after_branching_tactic"))
    if cfg.prefix_fallback.include_after_valid_progress_window:
        for k in [3, 5, 8, 12]:
            if k <= len(lines):
                candidate_end_lines.append((k, f"progress_window_{k}"))
    if cfg.prefix_fallback.include_random_valid_prefix > 0:
        choices = list(range(cfg.prefix_fallback.min_prefix_lines, min(len(lines), cfg.prefix_fallback.max_prefix_lines) + 1))
        random.shuffle(choices)
        for k in choices[:cfg.prefix_fallback.include_random_valid_prefix]:
            candidate_end_lines.append((k, "random_valid_prefix"))
    # Last valid prefix before first error is approximated by scanning prefix + sorry.
    if cfg.prefix_fallback.include_first_error_prefix:
        last_valid = find_last_valid_prefix_line(tr, lines, verifier, cfg)
        if last_valid:
            candidate_end_lines.append((last_valid, "last_valid_before_error"))
    # Deduplicate and verify.
    seen: set[str] = set()
    results: List[PrefixCandidate] = []
    for end_line, reason in candidate_end_lines:
        if end_line < cfg.prefix_fallback.min_prefix_lines or end_line > min(len(lines), cfg.prefix_fallback.max_prefix_lines):
            continue
        prefix = "\n".join(lines[:end_line]).strip()
        key = sha1_text(normalize_ws(prefix))
        if key in seen:
            continue
        seen.add(key)
        proof_with_sorry = proof_with_sorry_after_prefix(prefix)
        code = build_lean_code(tr.imports, tr.formal_statement, proof_with_sorry)
        ver = verifier.verify_code(code, cfg.lean.timeout_suffix_sec, allow_sorry=True)
        if not ver.ok:
            continue
        pid = stable_id(tr.theorem_id, attempt.attempt_id, str(end_line), reason, prefix, prefix="prefix")
        results.append(PrefixCandidate(pid, tr.theorem_id, attempt.attempt_id, prefix, reason, end_line, ver))
        if len(results) >= cfg.prefix_fallback.max_prefixes_per_attempt:
            break
    return results


def find_last_valid_prefix_line(tr: TheoremRecord, lines: List[str], verifier: LeanVerifier, cfg: RunConfig) -> Optional[int]:
    last = None
    upper = min(len(lines), cfg.prefix_fallback.max_prefix_lines)
    for k in range(1, upper + 1):
        prefix = "\n".join(lines[:k])
        code = build_lean_code(tr.imports, tr.formal_statement, proof_with_sorry_after_prefix(prefix))
        ver = verifier.verify_code(code, cfg.lean.timeout_suffix_sec, allow_sorry=True)
        if ver.ok:
            last = k
        else:
            break
    return last


@dataclass
class CandidateSolution:
    text: str
    ok: bool
    verification: LeanResult
    valid_prefix_steps: int = 0
    source: str = "generated"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchStats:
    n_sampled: int = 0
    solved_count: int = 0
    solved_ratio: float = 0.0
    wilson_lower_bound: float = 0.0
    valid_partial_count: int = 0
    valid_partial_rate: float = 0.0
    max_valid_suffix_steps: int = 0
    mean_valid_suffix_steps: float = 0.0
    shortest_solved_suffix_len: Optional[int] = None
    duplicate_rate: float = 0.0
    unique_solution_count: int = 0
    quality_class: str = "reject"
    quality_score: float = 0.0


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def compute_search_stats(solutions: List[CandidateSolution], cfg: RunConfig, prefix_len: int = 0) -> SearchStats:
    n = len(solutions)
    solved = [s for s in solutions if s.ok]
    valid_steps = [s.valid_prefix_steps for s in solutions]
    unique_hashes = {sha1_text(remove_comments_for_hash(s.text)) for s in solved}
    all_hashes = [sha1_text(remove_comments_for_hash(s.text)) for s in solutions]
    duplicate_rate = 1.0 - (len(set(all_hashes)) / max(1, len(all_hashes)))
    shortest = min([line_count(s.text) for s in solved], default=None)
    valid_partial_count = sum(1 for s in solutions if s.valid_prefix_steps > 0 or s.ok)
    valid_partial_rate = valid_partial_count / max(1, n)
    solved_ratio = len(solved) / max(1, n)
    wlb = wilson_lower_bound(len(solved), n, cfg.quality.wilson_z)
    shortest_bonus = 0.0 if shortest is None else 1.0 / (1.0 + shortest)
    unique_suffix_bonus = len(unique_hashes) / max(1, len(solved)) if solved else 0.0
    score = (
        cfg.quality.score_solved_weight * solved_ratio
        + cfg.quality.score_wilson_weight * wlb
        + cfg.quality.score_valid_partial_weight * valid_partial_rate
        + cfg.quality.score_short_suffix_weight * shortest_bonus
        + cfg.quality.score_unique_suffix_weight * unique_suffix_bonus
        - cfg.quality.score_duplicate_penalty * duplicate_rate
        - cfg.quality.score_prefix_len_penalty * min(1.0, prefix_len / 40.0)
    )
    if len(solved) >= 2 or (len(solved) >= 1 and shortest is not None and shortest <= cfg.prefix_fallback.short_suffix_len_threshold):
        q = "high"
    elif len(solved) >= 1:
        q = "medium"
    elif valid_partial_rate >= 0.50 and max(valid_steps, default=0) >= 5:
        q = "uncertain"
    else:
        q = "reject"
    return SearchStats(
        n_sampled=n,
        solved_count=len(solved),
        solved_ratio=solved_ratio,
        wilson_lower_bound=wlb,
        valid_partial_count=valid_partial_count,
        valid_partial_rate=valid_partial_rate,
        max_valid_suffix_steps=max(valid_steps, default=0),
        mean_valid_suffix_steps=sum(valid_steps) / max(1, len(valid_steps)),
        shortest_solved_suffix_len=shortest,
        duplicate_rate=duplicate_rate,
        unique_solution_count=len(unique_hashes),
        quality_class=q,
        quality_score=score,
    )


def should_escalate_prefix(stats: SearchStats, stage: str, cfg: RunConfig) -> bool:
    pf = cfg.prefix_fallback
    if stage == "probe_to_main":
        return (
            stats.solved_count >= pf.escalate_if_solved_count
            or stats.valid_partial_rate >= pf.escalate_if_valid_partial_rate
            or stats.max_valid_suffix_steps >= pf.escalate_if_max_valid_suffix_steps
        )
    if stage == "main_to_confirm":
        return stats.solved_count >= pf.confirm_if_solved_count_main or stats.valid_partial_rate >= pf.confirm_if_valid_partial_rate_main
    return False


def should_escalate_hole(stats: SearchStats, stage: str, cfg: RunConfig) -> bool:
    if stage == "probe_to_main":
        return stats.solved_count >= 1 or stats.valid_partial_rate >= 0.25 or stats.max_valid_suffix_steps >= 3
    if stage == "main_to_confirm":
        return stats.solved_count >= 1 or stats.valid_partial_rate >= 0.50
    return False


def verify_replacement_for_hole(verifier: LeanVerifier, tr: TheoremRecord, hole: HoleCandidate, replacement: str, cfg: RunConfig, allow_sorry: bool = False) -> LeanResult:
    repl = strip_surrounding_theorem_if_present(strip_code_fences(replacement)).strip()
    if not repl:
        repl = "skip"
    semi = replace_first_sorry(hole.semi_proof_body, repl)
    code = build_lean_code(tr.imports, tr.formal_statement, semi)
    return verifier.verify_code(code, cfg.lean.timeout_suffix_sec, allow_sorry=allow_sorry)


def replace_first_sorry(text: str, replacement: str) -> str:
    m = re.search(r"\bsorry\b", text)
    if not m:
        return text
    before = text[:m.start()]
    line_start = before.rfind("\n") + 1
    indent_len = len(before[line_start:])
    repl = indent(replacement.strip(), indent_len).lstrip() if indent_len > 0 else replacement.strip()
    return text[:m.start()] + repl + text[m.end():]


def verify_prefix_continuation(verifier: LeanVerifier, tr: TheoremRecord, prefix: PrefixCandidate, suffix: str, cfg: RunConfig) -> LeanResult:
    suffix = strip_surrounding_theorem_if_present(strip_code_fences(suffix)).strip()
    full = prefix.prefix_body.rstrip() + "\n" + suffix
    code = build_lean_code(tr.imports, tr.formal_statement, full)
    return verifier.verify_code(code, cfg.lean.timeout_suffix_sec, allow_sorry=False)


def count_valid_suffix_prefix_steps(verifier: LeanVerifier, tr: TheoremRecord, prefix_body: str, suffix: str, cfg: RunConfig) -> int:
    lines = [l for l in strip_surrounding_theorem_if_present(strip_code_fences(suffix)).splitlines() if l.strip()]
    valid = 0
    for k in range(1, min(len(lines), 12) + 1):
        body = prefix_body.rstrip() + "\n" + "\n".join(lines[:k]) + "\n" + "sorry"
        code = build_lean_code(tr.imports, tr.formal_statement, body)
        ver = verifier.verify_code(code, cfg.lean.timeout_suffix_sec, allow_sorry=True)
        if ver.ok:
            valid = k
        else:
            break
    return valid


def sample_prefix_continuations(
    tr: TheoremRecord,
    prefix: PrefixCandidate,
    generator: BaseGenerator,
    verifier: LeanVerifier,
    cfg: RunConfig,
    total_n: int,
) -> List[CandidateSolution]:
    prompt = prompt_continue_prefix(tr.formal_statement, prefix.prefix_body)
    gens = generator.generate([prompt], n=total_n, temperature=cfg.completion.temperature, top_p=cfg.completion.top_p, max_new_tokens=cfg.completion.max_new_tokens, stop=cfg.completion.stop_sequences)[0]
    sols: List[CandidateSolution] = []
    for g in gens:
        text = strip_code_fences(g.text)
        ver = verify_prefix_continuation(verifier, tr, prefix, text, cfg)
        valid_steps = 0 if ver.ok else count_valid_suffix_prefix_steps(verifier, tr, prefix.prefix_body, text, cfg)
        sols.append(CandidateSolution(text, ver.ok, ver, valid_steps, metadata=g.metadata))
    return sols


def sample_hole_replacements(
    tr: TheoremRecord,
    hole: HoleCandidate,
    generator: BaseGenerator,
    verifier: LeanVerifier,
    ranker: EncoderRanker,
    cfg: RunConfig,
    total_n: int,
) -> List[CandidateSolution]:
    prompt = prompt_fill_hole(tr.formal_statement, hole.semi_proof_body)
    gens = generator.generate([prompt], n=total_n, temperature=cfg.completion.temperature, top_p=cfg.completion.top_p, max_new_tokens=min(cfg.completion.max_new_tokens, 512), stop=cfg.completion.stop_sequences)[0]
    cand_texts = [strip_code_fences(g.text) for g in gens]
    cand_texts = ranker.rank({"theorem_id": tr.theorem_id, "hole_id": hole.hole_id, "semi_proof_body": hole.semi_proof_body}, cand_texts)
    sols: List[CandidateSolution] = []
    for text in cand_texts:
        ver = verify_replacement_for_hole(verifier, tr, hole, text, cfg, allow_sorry=False)
        valid_steps = 0
        if not ver.ok:
            valid_steps = count_valid_hole_replacement_prefix_steps(verifier, tr, hole, text, cfg)
        sols.append(CandidateSolution(text, ver.ok, ver, valid_steps))
    return sols


def count_valid_hole_replacement_prefix_steps(verifier: LeanVerifier, tr: TheoremRecord, hole: HoleCandidate, replacement: str, cfg: RunConfig) -> int:
    lines = [l for l in strip_surrounding_theorem_if_present(strip_code_fences(replacement)).splitlines() if l.strip()]
    valid = 0
    for k in range(1, min(len(lines), 12) + 1):
        partial = "\n".join(lines[:k]) + "\n" + "sorry"
        ver = verify_replacement_for_hole(verifier, tr, hole, partial, cfg, allow_sorry=True)
        if ver.ok:
            valid = k
        else:
            break
    return valid


def adaptive_search_prefix(tr: TheoremRecord, prefix: PrefixCandidate, generator: BaseGenerator, verifier: LeanVerifier, cfg: RunConfig) -> Tuple[List[CandidateSolution], SearchStats]:
    sols: List[CandidateSolution] = []
    sols.extend(sample_prefix_continuations(tr, prefix, generator, verifier, cfg, cfg.completion.n_probe))
    stats = compute_search_stats(sols, cfg, prefix_len=line_count(prefix.prefix_body))
    if should_escalate_prefix(stats, "probe_to_main", cfg) and len(sols) < cfg.completion.n_main:
        sols.extend(sample_prefix_continuations(tr, prefix, generator, verifier, cfg, cfg.completion.n_main - len(sols)))
        stats = compute_search_stats(sols, cfg, prefix_len=line_count(prefix.prefix_body))
    if should_escalate_prefix(stats, "main_to_confirm", cfg) and len(sols) < cfg.completion.n_confirm:
        sols.extend(sample_prefix_continuations(tr, prefix, generator, verifier, cfg, cfg.completion.n_confirm - len(sols)))
        stats = compute_search_stats(sols, cfg, prefix_len=line_count(prefix.prefix_body))
    return sols, stats


def adaptive_search_hole(tr: TheoremRecord, hole: HoleCandidate, generator: BaseGenerator, verifier: LeanVerifier, ranker: EncoderRanker, cfg: RunConfig) -> Tuple[List[CandidateSolution], SearchStats]:
    sols: List[CandidateSolution] = []
    sols.extend(sample_hole_replacements(tr, hole, generator, verifier, ranker, cfg, cfg.completion.n_probe))
    stats = compute_search_stats(sols, cfg, prefix_len=0)
    if should_escalate_hole(stats, "probe_to_main", cfg) and len(sols) < cfg.completion.n_main:
        sols.extend(sample_hole_replacements(tr, hole, generator, verifier, ranker, cfg, cfg.completion.n_main - len(sols)))
        stats = compute_search_stats(sols, cfg, prefix_len=0)
    if should_escalate_hole(stats, "main_to_confirm", cfg) and len(sols) < cfg.completion.n_confirm:
        sols.extend(sample_hole_replacements(tr, hole, generator, verifier, ranker, cfg, cfg.completion.n_confirm - len(sols)))
        stats = compute_search_stats(sols, cfg, prefix_len=0)
    return sols, stats


# -----------------------------
# Output serialization
# -----------------------------


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return dict(obj)


def lean_result_public(ver: Optional[LeanResult], verbose: bool = False) -> Optional[Dict[str, Any]]:
    if ver is None:
        return None
    d = {
        "ok": ver.ok,
        "returncode": ver.returncode,
        "elapsed_sec": ver.elapsed_sec,
        "has_sorry_warning": ver.has_sorry_warning,
        "error_summary": ver.error_summary,
    }
    if verbose:
        d["stdout"] = ver.stdout
        d["stderr"] = ver.stderr
        d["file_path"] = ver.file_path
    return d


def stats_to_dict(stats: SearchStats) -> Dict[str, Any]:
    return dataclasses.asdict(stats)


def best_solution(solutions: List[CandidateSolution]) -> Optional[CandidateSolution]:
    solved = [s for s in solutions if s.ok]
    if not solved:
        return None
    solved.sort(key=lambda s: (line_count(s.text), len(s.text)))
    return solved[0]


def curriculum_item_from_hole(tr: TheoremRecord, hole: HoleCandidate, sol: CandidateSolution, stats: SearchStats) -> Dict[str, Any]:
    return {
        "item_id": stable_id(hole.hole_id, sol.text, prefix="item"),
        "item_type": "mcsp_hole_subtask",
        "source_dataset": tr.raw.get("source_name") or "unknown",
        "domain": tr.raw.get("domain") or "unknown",
        "parent_theorem_id": tr.theorem_id,
        "attempt_id": hole.attempt_id,
        "hole_id": hole.hole_id,
        "formal_statement": tr.formal_statement,
        "natural_language_statement": tr.natural_language_statement,
        "input_context": {
            "semi_proof_body": hole.semi_proof_body,
            "original_block": hole.original_block,
            "hole_kind": hole.hole_kind,
            "start_line": hole.start_line,
            "end_line": hole.end_line,
        },
        "target_proof": sol.text,
        "verified": True,
        "uses_sorry": False,
        "quality_class": stats.quality_class,
        "quality_score": stats.quality_score,
        "search_stats": stats_to_dict(stats),
    }


def curriculum_item_from_prefix(tr: TheoremRecord, prefix: PrefixCandidate, sol: CandidateSolution, stats: SearchStats) -> Dict[str, Any]:
    return {
        "item_id": stable_id(prefix.prefix_id, sol.text, prefix="item"),
        "item_type": "prefix_with_verified_suffix",
        "source_dataset": tr.raw.get("source_name") or "unknown",
        "domain": tr.raw.get("domain") or "unknown",
        "parent_theorem_id": tr.theorem_id,
        "attempt_id": prefix.attempt_id,
        "prefix_id": prefix.prefix_id,
        "formal_statement": tr.formal_statement,
        "natural_language_statement": tr.natural_language_statement,
        "input_context": {
            "prefix_body": prefix.prefix_body,
            "reason": prefix.reason,
            "end_line": prefix.end_line,
        },
        "target_proof": prefix.prefix_body.rstrip() + "\n" + sol.text.strip(),
        "target_suffix": sol.text,
        "verified": True,
        "uses_sorry": False,
        "quality_class": stats.quality_class,
        "quality_score": stats.quality_score,
        "search_stats": stats_to_dict(stats),
    }


def infer_goal_from_original_block(block: str) -> str:
    text = strip_code_fences(block).strip()
    m = re.search(r"(?s)^\s*have\s+(?:\S+\s+)?(?::\s*(.*?)\s*)?:=\s*by\b", text)
    if m and m.group(1):
        return normalize_ws(m.group(1))
    m = re.search(r"(?s)^\s*show\s+(.*?)\s*(?::=\s*by)?$", text)
    if m:
        return normalize_ws(m.group(1))
    return ""


def proof_prefix_before_line(proof_body: str, start_line: int) -> str:
    lines = strip_surrounding_theorem_if_present(proof_body).splitlines()
    return "\n".join(lines[: max(0, start_line - 1)]).strip()



class Miner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        random.seed(cfg.experiment.seed)
        self.out_dir = ensure_dir(cfg.io.out_dir)
        if cfg.io.overwrite and self.out_dir.exists():
            shutil.rmtree(self.out_dir)
            ensure_dir(self.out_dir)
        self.paths = {
            "curriculum": self.out_dir / "mined_curriculum_items.jsonl",
            "mcsp_holes": self.out_dir / "mcsp_holes.jsonl",
            "prefix_candidates": self.out_dir / "prefix_candidates.jsonl",
            "mcsp_jobs": self.out_dir / "mcsp_search_jobs.jsonl",
            "prefix_jobs": self.out_dir / "prefix_search_jobs.jsonl",
            "frontier": self.out_dir / "frontier_theorems.jsonl",
            "attempts": self.out_dir / "attempts.jsonl",
            "report": self.out_dir / "run_report.json",
            "errors": self.out_dir / "errors.jsonl",
        }
        self.verifier = LeanVerifier(cfg.lean)
        self.search_enabled = bool(cfg.completion.enabled and cfg.full_generation.engine == "vllm")
        self.generator = make_generator(cfg) if (cfg.full_generation.enabled or self.search_enabled) else NoOpGenerator()
        self.ranker = EncoderRanker(cfg.encoder_ranker)
        self.report: Dict[str, Any] = {
            "experiment_name": cfg.experiment.experiment_name,
            "started_at": now_ts(),
            "num_theorems_processed": 0,
            "num_full_attempts": 0,
            "num_full_solved": 0,
            "num_failed_attempts": 0,
            "num_flat_tactic": 0,
            "num_shallow_structured": 0,
            "num_mcsp_candidates": 0,
            "num_strong_mcsp_candidates": 0,
            "num_mcsp_holes": 0,
            "num_mcsp_search_jobs": 0,
            "num_solved_holes": 0,
            "num_prefix_candidates": 0,
            "num_prefix_search_jobs": 0,
            "num_high_quality_prefixes": 0,
            "num_medium_prefixes": 0,
            "num_frontier_theorems": 0,
            "num_curriculum_items": 0,
            "total_lean_calls_approx": 0,
            "errors": 0,
        }
        self.seen_items: set[str] = set()
        self.processed_attempt_ids: set[str] = self.load_processed_attempt_ids()
        self.attempts_with_mcsp_job: set[str] = set()
        self.attempts_with_prefix_job: set[str] = set()
        self.frontier_theorem_ids: set[str] = set()

    def load_processed_attempt_ids(self) -> set[str]:
        if not self.cfg.experiment.resume or self.cfg.io.overwrite:
            return set()
        path = self.paths["attempts"]
        if not path.exists():
            return set()
        done: set[str] = set()
        for rec in read_jsonl(path):
            aid = rec.get("attempt_id")
            if aid:
                done.add(str(aid))
        return done

    def run(self) -> None:
        theorems = load_theorems(self.cfg.io.theorems_jsonl, self.cfg)
        if not theorems:
            raise RuntimeError(f"No theorems loaded from {self.cfg.io.theorems_jsonl}")
        attempts = load_attempts(self.cfg.io.attempts_jsonl, self.cfg) if self.cfg.io.attempts_jsonl else {}
        if self.cfg.full_generation.enabled:
            generated = generate_full_attempts(theorems, self.generator, self.cfg)
            for tid, arr in generated.items():
                attempts.setdefault(tid, []).extend(arr)
        if not attempts:
            raise RuntimeError("No proof attempts available.")
        for idx, tr in enumerate(theorems):
            if tr.theorem_id not in attempts:
                continue
            try:
                self.process_theorem(tr, attempts[tr.theorem_id])
            except Exception as e:
                self.report["errors"] += 1
                append_jsonl(self.paths["errors"], {
                    "theorem_id": tr.theorem_id,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
            if (idx + 1) % self.cfg.experiment.save_every == 0:
                self.save_report(partial=True)
        self.save_report(partial=False)

    def process_theorem(self, tr: TheoremRecord, attempts: List[ProofAttempt]) -> None:
        self.report["num_theorems_processed"] += 1
        for attempt in attempts:
            if attempt.attempt_id in self.processed_attempt_ids:
                continue
            self.report["num_full_attempts"] += 1
            if self.cfg.io.write_attempts:
                append_jsonl(self.paths["attempts"], dataclasses.asdict(attempt))
            self.processed_attempt_ids.add(attempt.attempt_id)
            body = strip_surrounding_theorem_if_present(strip_code_fences(attempt.proof_text)).strip()
            if not body:
                continue
            full_code = build_lean_code(tr.imports, tr.formal_statement, body)
            full_ver = self.verifier.verify_code(full_code, self.cfg.lean.timeout_full_proof_sec, allow_sorry=False)
            self.report["total_lean_calls_approx"] += 1
            if full_ver.ok:
                self.report["num_full_solved"] += 1
                item = self.full_solved_item(tr, attempt, full_ver)
                self.write_curriculum_item(item)
                continue
            self.report["num_failed_attempts"] += 1
            style = classify_proof_style(body, self.cfg.proof_style)
            self.bump_style(style.label)
            if self.cfg.mcsp.enabled and style.label in {"MCSP_CANDIDATE", "STRONG_MCSP_CANDIDATE", "SHALLOW_STRUCTURED"}:
                self.process_mcsp(tr, attempt, style)
            if self.cfg.prefix_fallback.enabled:
                self.process_prefix_fallback(tr, attempt, style)
            if attempt.attempt_id in self.attempts_with_mcsp_job or attempt.attempt_id in self.attempts_with_prefix_job:
                self.write_frontier_theorem(tr)

    def bump_style(self, label: str) -> None:
        if label == "FLAT_TACTIC":
            self.report["num_flat_tactic"] += 1
        elif label == "SHALLOW_STRUCTURED":
            self.report["num_shallow_structured"] += 1
        elif label == "MCSP_CANDIDATE":
            self.report["num_mcsp_candidates"] += 1
        elif label == "STRONG_MCSP_CANDIDATE":
            self.report["num_strong_mcsp_candidates"] += 1

    def full_solved_item(self, tr: TheoremRecord, attempt: ProofAttempt, ver: LeanResult) -> Dict[str, Any]:
        return {
            "item_id": stable_id(tr.theorem_id, attempt.proof_text, prefix="item"),
            "item_type": "full_solved_proof",
            "source_dataset": tr.raw.get("source_name") or self.cfg.dataset.source_name,
            "domain": tr.raw.get("domain") or self.cfg.dataset.domain,
            "parent_theorem_id": tr.theorem_id,
            "attempt_id": attempt.attempt_id,
            "formal_statement": tr.formal_statement,
            "natural_language_statement": tr.natural_language_statement,
            "input_context": {},
            "target_proof": strip_surrounding_theorem_if_present(strip_code_fences(attempt.proof_text)).strip(),
            "verified": True,
            "uses_sorry": False,
            "quality_class": "high",
            "quality_score": 1.0,
            "verification": lean_result_public(ver, self.cfg.lean.verbose_errors),
        }

    def process_mcsp(self, tr: TheoremRecord, attempt: ProofAttempt, style: ProofStyle) -> None:
        body = strip_surrounding_theorem_if_present(strip_code_fences(attempt.proof_text)).strip()
        holes = extract_hole_candidates(tr, attempt, style, self.cfg)
        for hole in holes:
            ratio = hole_ratio(hole, body)
            if self.cfg.mcsp.reject_single_giant_hole and ratio >= self.cfg.mcsp.reject_if_hole_ratio_above:
                status = "DEGENERATE_SKETCH"
                self.write_hole_record(tr, hole, status, None, None, [])
                continue
            semi_code = build_lean_code(tr.imports, tr.formal_statement, hole.semi_proof_body)
            semi_ver = self.verifier.verify_code(semi_code, self.cfg.lean.timeout_suffix_sec, allow_sorry=True)
            self.report["total_lean_calls_approx"] += 1
            if not semi_ver.ok:
                self.write_hole_record(tr, hole, "INVALID_SKETCH", semi_ver, None, [])
                continue
            self.report["num_mcsp_holes"] += 1
            self.attempts_with_mcsp_job.add(attempt.attempt_id)
            if self.cfg.io.write_search_jobs:
                self.write_mcsp_search_job(tr, hole, semi_ver)
            if not self.search_enabled:
                self.write_hole_record(tr, hole, "VALID_SKETCH_NO_GENERATOR", semi_ver, None, [])
                continue
            sols, stats = adaptive_search_hole(tr, hole, self.generator, self.verifier, self.ranker, self.cfg)
            self.report["total_lean_calls_approx"] += len(sols)
            status = "SOLVED" if stats.solved_count > 0 else ("PARTIAL_PROGRESS" if stats.valid_partial_count > 0 else "UNRESOLVED")
            self.write_hole_record(tr, hole, status, semi_ver, stats, sols)
            if stats.solved_count > 0:
                self.report["num_solved_holes"] += 1
                best = best_solution(sols)
                if best:
                    item = curriculum_item_from_hole(tr, hole, best, stats)
                    self.write_curriculum_item(item)

    def process_prefix_fallback(self, tr: TheoremRecord, attempt: ProofAttempt, style: ProofStyle) -> None:
        prefixes = extract_prefix_candidates(tr, attempt, self.verifier, self.cfg)
        self.report["total_lean_calls_approx"] += len(prefixes)
        for prefix in prefixes:
            self.report["num_prefix_candidates"] += 1
            self.attempts_with_prefix_job.add(attempt.attempt_id)
            if self.cfg.io.write_search_jobs:
                self.write_prefix_search_job(tr, prefix)
            if not self.search_enabled:
                self.write_prefix_record(tr, prefix, None, [], "VALID_PREFIX_NO_GENERATOR")
                continue
            sols, stats = adaptive_search_prefix(tr, prefix, self.generator, self.verifier, self.cfg)
            self.report["total_lean_calls_approx"] += len(sols)
            status = "SOLVED" if stats.solved_count > 0 else ("PARTIAL_PROGRESS" if stats.valid_partial_count > 0 else "UNRESOLVED")
            self.write_prefix_record(tr, prefix, stats, sols, status)
            if stats.quality_class == "high":
                self.report["num_high_quality_prefixes"] += 1
            elif stats.quality_class == "medium":
                self.report["num_medium_prefixes"] += 1
            if stats.solved_count > 0 and stats.quality_class in {"high", "medium"}:
                best = best_solution(sols)
                if best:
                    item = curriculum_item_from_prefix(tr, prefix, best, stats)
                    self.write_curriculum_item(item)

    def write_hole_record(self, tr: TheoremRecord, hole: HoleCandidate, status: str, semi_ver: Optional[LeanResult], stats: Optional[SearchStats], sols: List[CandidateSolution]) -> None:
        append_jsonl(self.paths["mcsp_holes"], {
            "hole_id": hole.hole_id,
            "parent_theorem_id": tr.theorem_id,
            "attempt_id": hole.attempt_id,
            "semi_proof_body": hole.semi_proof_body,
            "original_block": hole.original_block,
            "hole_kind": hole.hole_kind,
            "start_line": hole.start_line,
            "end_line": hole.end_line,
            "proof_style_label": hole.proof_style_label,
            "status": status,
            "semi_proof_verification": lean_result_public(semi_ver, self.cfg.lean.verbose_errors),
            "search_stats": stats_to_dict(stats) if stats else None,
            "best_solution": best_solution(sols).text if best_solution(sols) else None,
            "num_solutions_logged": min(5, len(sols)),
            "sample_solutions": [
                {"text": s.text, "ok": s.ok, "valid_prefix_steps": s.valid_prefix_steps, "verification": lean_result_public(s.verification, False)}
                for s in sols[:5]
            ],
        })

    def write_prefix_record(self, tr: TheoremRecord, prefix: PrefixCandidate, stats: Optional[SearchStats], sols: List[CandidateSolution], status: str) -> None:
        append_jsonl(self.paths["prefix_candidates"], {
            "prefix_id": prefix.prefix_id,
            "parent_theorem_id": tr.theorem_id,
            "attempt_id": prefix.attempt_id,
            "prefix_body": prefix.prefix_body,
            "reason": prefix.reason,
            "end_line": prefix.end_line,
            "status": status,
            "prefix_verification": lean_result_public(prefix.verification, self.cfg.lean.verbose_errors),
            "search_stats": stats_to_dict(stats) if stats else None,
            "best_suffix": best_solution(sols).text if best_solution(sols) else None,
            "num_solutions_logged": min(5, len(sols)),
            "sample_solutions": [
                {"text": s.text, "ok": s.ok, "valid_prefix_steps": s.valid_prefix_steps, "verification": lean_result_public(s.verification, False)}
                for s in sols[:5]
            ],
        })

    def write_mcsp_search_job(self, tr: TheoremRecord, hole: HoleCandidate, semi_ver: LeanResult) -> None:
        goal = infer_goal_from_original_block(hole.original_block)
        job = {
            "job_id": stable_id(hole.hole_id, "mcsp", prefix="job"),
            "job_type": "HOLE_FILLING",
            "parent_theorem_id": tr.theorem_id,
            "attempt_id": hole.attempt_id,
            "hole_id": hole.hole_id,
            "formal_statement": tr.formal_statement,
            "natural_language_statement": tr.natural_language_statement,
            "imports": tr.imports,
            "semi_proof_text": hole.semi_proof_body,
            "proof_with_sorry": hole.semi_proof_body,
            "target_hole_marker": "sorry",
            "original_block": hole.original_block,
            "hole_kind": hole.hole_kind,
            "hole_goal": goal,
            "proof_prefix": proof_prefix_before_line(hole.semi_proof_body, hole.start_line),
            "proof_style": hole.proof_style_label,
            "start_line": hole.start_line,
            "end_line": hole.end_line,
            "frontier_score": 1.0 if hole.proof_style_label == "STRONG_MCSP_CANDIDATE" else 0.6,
            "semi_proof_verification": lean_result_public(semi_ver, False),
            "source": "run_04_mine_failed_problems",
            "raw_metadata": {
                "source_dataset": tr.raw.get("source_dataset"),
                "domain": tr.raw.get("domain"),
            },
        }
        append_jsonl(self.paths["mcsp_jobs"], job)
        self.report["num_mcsp_search_jobs"] += 1

    def write_prefix_search_job(self, tr: TheoremRecord, prefix: PrefixCandidate) -> None:
        job = {
            "job_id": stable_id(prefix.prefix_id, "prefix", prefix.prefix_body, prefix="job"),
            "job_type": "PREFIX_COMPLETION",
            "parent_theorem_id": tr.theorem_id,
            "attempt_id": prefix.attempt_id,
            "prefix_id": prefix.prefix_id,
            "formal_statement": tr.formal_statement,
            "natural_language_statement": tr.natural_language_statement,
            "imports": tr.imports,
            "prefix_text": prefix.prefix_body,
            "proof_prefix": prefix.prefix_body,
            "state_after_prefix": f"Accepted prefix with {prefix.end_line} proof lines; continue to close the remaining goals.",
            "reason": prefix.reason,
            "end_line": prefix.end_line,
            "frontier_score": max(0.1, 1.0 - min(1.0, prefix.end_line / max(1, self.cfg.prefix_fallback.max_prefix_lines))),
            "prefix_verification": lean_result_public(prefix.verification, False),
            "source": "run_04_mine_failed_problems",
            "raw_metadata": {
                "source_dataset": tr.raw.get("source_dataset"),
                "domain": tr.raw.get("domain"),
            },
        }
        append_jsonl(self.paths["prefix_jobs"], job)
        self.report["num_prefix_search_jobs"] += 1

    def write_frontier_theorem(self, tr: TheoremRecord) -> None:
        if not self.cfg.io.write_frontier_theorems or tr.theorem_id in self.frontier_theorem_ids:
            return
        append_jsonl(self.paths["frontier"], {
            "parent_theorem_id": tr.theorem_id,
            "formal_statement": tr.formal_statement,
            "natural_language_statement": tr.natural_language_statement,
            "imports": tr.imports,
            "source_dataset": tr.raw.get("source_dataset"),
            "domain": tr.raw.get("domain"),
            "source": "run_04_mine_failed_problems",
        })
        self.frontier_theorem_ids.add(tr.theorem_id)
        self.report["num_frontier_theorems"] += 1

    def write_curriculum_item(self, item: Dict[str, Any]) -> None:
        if self.cfg.quality.deduplicate:
            key = sha1_text(normalize_ws(item.get("formal_statement", "")) + "\n" + normalize_ws(item.get("target_proof", "")))
            if key in self.seen_items:
                return
            self.seen_items.add(key)
        append_jsonl(self.paths["curriculum"], item)
        self.report["num_curriculum_items"] += 1

    def save_report(self, partial: bool) -> None:
        failed = max(1, int(self.report.get("num_failed_attempts", 0)))
        useful_attempts = self.attempts_with_mcsp_job | self.attempts_with_prefix_job
        self.report["num_attempts_with_mcsp_job"] = len(self.attempts_with_mcsp_job)
        self.report["num_attempts_with_prefix_job"] = len(self.attempts_with_prefix_job)
        self.report["num_attempts_with_any_useful_job"] = len(useful_attempts)
        self.report["num_no_useful_skeleton_or_prefix_attempts"] = max(0, int(self.report.get("num_failed_attempts", 0)) - len(useful_attempts))
        self.report["rates"] = {
            "structured_mcsp_candidate_attempt_rate": (self.report.get("num_mcsp_candidates", 0) + self.report.get("num_strong_mcsp_candidates", 0)) / failed,
            "compilable_mcsp_skeleton_attempt_rate": len(self.attempts_with_mcsp_job) / failed,
            "valid_prefix_attempt_rate": len(self.attempts_with_prefix_job) / failed,
            "any_useful_job_attempt_rate": len(useful_attempts) / failed,
            "no_useful_attempt_rate": self.report["num_no_useful_skeleton_or_prefix_attempts"] / failed,
        }
        self.report["partial"] = partial
        self.report["updated_at"] = now_ts()
        write_json(self.paths["report"], self.report)
        self.write_summary_csv()

    def write_summary_csv(self) -> None:
        rows = [
            ("num_failed_attempts", self.report.get("num_failed_attempts", 0)),
            ("num_mcsp_search_jobs", self.report.get("num_mcsp_search_jobs", 0)),
            ("num_prefix_search_jobs", self.report.get("num_prefix_search_jobs", 0)),
            ("num_attempts_with_mcsp_job", self.report.get("num_attempts_with_mcsp_job", 0)),
            ("num_attempts_with_prefix_job", self.report.get("num_attempts_with_prefix_job", 0)),
            ("num_attempts_with_any_useful_job", self.report.get("num_attempts_with_any_useful_job", 0)),
            ("num_no_useful_skeleton_or_prefix_attempts", self.report.get("num_no_useful_skeleton_or_prefix_attempts", 0)),
        ]
        rates = self.report.get("rates", {})
        for k, v in rates.items():
            rows.append((k, v))
        path = self.out_dir / "mining_summary.csv"
        with path.open("w", encoding="utf-8") as f:
            f.write("metric,value\n")
            for k, v in rows:
                f.write(f"{k},{v}\n")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--theorems-jsonl", type=str, default=None)
    p.add_argument("--attempts-jsonl", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max-theorems", type=int, default=None)
    p.add_argument("--enable-generation", action="store_true")
    p.add_argument("--no-mcsp", action="store_true")
    p.add_argument("--no-prefix", action="store_true")
    p.add_argument("--n-probe", type=int, default=None)
    p.add_argument("--n-main", type=int, default=None)
    p.add_argument("--n-confirm", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config-only", action="store_true")
    return p.parse_args()



def apply_cli_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    if args.theorems_jsonl:
        cfg.io.theorems_jsonl = args.theorems_jsonl
    if args.attempts_jsonl:
        cfg.io.attempts_jsonl = args.attempts_jsonl
    if args.out_dir:
        cfg.io.out_dir = args.out_dir
    if args.project_root:
        cfg.lean.project_root = args.project_root
    if args.model:
        cfg.full_generation.model_name = args.model
    if args.max_theorems is not None:
        cfg.dataset.max_theorems = args.max_theorems
    if args.enable_generation:
        cfg.full_generation.enabled = True
        cfg.full_generation.engine = "vllm"
        cfg.completion.enabled = True
    if args.no_mcsp:
        cfg.mcsp.enabled = False
    if args.no_prefix:
        cfg.prefix_fallback.enabled = False
    if args.n_probe is not None:
        cfg.completion.n_probe = args.n_probe
    if args.n_main is not None:
        cfg.completion.n_main = args.n_main
    if args.n_confirm is not None:
        cfg.completion.n_confirm = args.n_confirm
    if args.overwrite:
        cfg.io.overwrite = True
    if args.dry_run:
        cfg.experiment.dry_run = True
    return cfg


def validate_cfg(cfg: RunConfig) -> None:
    if cfg.completion.n_probe <= 0 or cfg.completion.n_main < cfg.completion.n_probe or cfg.completion.n_confirm < cfg.completion.n_main:
        raise ValueError("Require 0 < n_probe <= n_main <= n_confirm")
    if not cfg.mcsp.enabled and not cfg.prefix_fallback.enabled:
        raise ValueError("Both MCSP and prefix fallback are disabled.")
    if not Path(cfg.io.theorems_jsonl).exists():
        raise FileNotFoundError(f"Theorems JSONL not found: {cfg.io.theorems_jsonl}")
    if cfg.io.attempts_jsonl and not Path(cfg.io.attempts_jsonl).exists():
        raise FileNotFoundError(f"Attempts JSONL not found: {cfg.io.attempts_jsonl}")


def input_summary(cfg: RunConfig) -> Dict[str, Any]:
    theorems = load_theorems(cfg.io.theorems_jsonl, cfg)
    attempts = load_attempts(cfg.io.attempts_jsonl or cfg.io.theorems_jsonl, cfg)
    attempt_count = sum(len(v) for v in attempts.values())
    theorem_ids = {t.theorem_id for t in theorems}
    joinable_attempt_count = sum(len(v) for k, v in attempts.items() if k in theorem_ids)
    status_counts: Counter[str] = Counter()
    failure_role_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for rec in read_jsonl(cfg.io.attempts_jsonl or cfg.io.theorems_jsonl):
        status = rec.get("status")
        if status is not None:
            status_counts[str(status)] += 1
        failure = rec.get("failure") or {}
        if isinstance(failure, dict) and failure.get("failure_role"):
            failure_role_counts[str(failure["failure_role"])] += 1
        source = first_present_nested(rec, ["source", "problem.source", "source_dataset", "problem.source_dataset"], "unknown")
        source_counts[str(source)] += 1
    return {
        "theorems_jsonl": cfg.io.theorems_jsonl,
        "attempts_jsonl": cfg.io.attempts_jsonl,
        "theorems_loaded": len(theorems),
        "attempt_theorem_keys": len(attempts),
        "attempts_loaded_after_filter": attempt_count,
        "attempts_joinable_to_loaded_theorems": joinable_attempt_count,
        "attempt_status_filter": cfg.dataset.attempt_status_filter,
        "raw_status_counts": dict(status_counts),
        "raw_failure_role_counts": dict(failure_role_counts),
        "raw_source_counts": dict(source_counts),
        "out_dir": cfg.io.out_dir,
        "writes_run05_jobs": cfg.io.write_search_jobs,
        "completion_search_enabled": cfg.completion.enabled,
    }


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    validate_cfg(cfg)
    if cfg.experiment.dry_run or args.config_only:
        summary = {"config": dataclasses.asdict(cfg), "input_summary": input_summary(cfg)}
        out_dir = ensure_dir(cfg.io.out_dir)
        write_json(out_dir / "config_only_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return
    miner = Miner(cfg)
    miner.run()
    print(f"Done. Outputs written to: {cfg.io.out_dir}")


if __name__ == "__main__":
    main()

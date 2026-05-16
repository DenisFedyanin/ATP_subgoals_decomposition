#!/usr/bin/env python3
"""
run_05_search_sorry_and_prefixes.py

Encoder-guided Lean tree search for jobs produced by run_04_mine_failed_problems.py.

This stage does NOT mine new theorems and does NOT select frontier tasks. It consumes:
  - mcsp_search_jobs.jsonl   : `sorry`-hole filling jobs from MCSP/semi-proof skeletons
  - prefix_search_jobs.jsonl : prefix-completion jobs from failed/linear proof attempts

For each job, it runs verifier-in-the-loop best-first tree search:
  state/proof-fragment -> generator proposes one-step or short-block tactics
                       -> encoder scores (state, tactic) pairs and embeds them
                       -> greedy k-DPP selects a high-quality diverse subset
                       -> Lean verifies selected transitions
                       -> valid children re-enter the frontier

Important design choices:
  - For HOLE_FILLING, search is local w.r.t. the extracted hole goal/context whenever
    a local task template is available. If no local task template is available, the
    verifier falls back to checking the candidate inside the parent semi-proof, with
    other holes allowed as `sorry`. In all cases, final acceptance requires inserting
    the found replacement into the parent semi-proof.
  - For PREFIX_COMPLETION, search extends an already Lean-valid prefix and accepts only
    full theorem proofs with no `sorry`.
  - No unresolved job is treated as a negative example. It only means "not solved within
    this configured search budget".

The file is intentionally self-contained and provides adapter hooks for your real
vLLM generator and trained encoder. Without these adapters it can still run in a
"heuristic/dry" mode for I/O/schema testing, but real proof search requires a Lean
project and a generator.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import hashlib
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# =============================================================================
# Config
# =============================================================================


@dataclass
class ExperimentConfig:
    experiment_name: str = "search_sorry_and_prefixes_v1"
    seed: int = 42
    resume: bool = True
    save_every_jobs: int = 25
    log_level: str = "info"
    dry_run: bool = False
    max_jobs: Optional[int] = None
    job_type_filter: Optional[str] = None  # HOLE_FILLING | PREFIX_COMPLETION | None


@dataclass
class InputConfig:
    mcsp_jobs_path: str = "outputs/run_04_mine_failed_problems/mcsp_search_jobs.jsonl"
    prefix_jobs_path: str = "outputs/run_04_mine_failed_problems/prefix_search_jobs.jsonl"
    frontier_path: Optional[str] = "outputs/run_04_mine_failed_problems/frontier_theorems.jsonl"


@dataclass
class OutputConfig:
    out_dir: str = "outputs/run_05_search_sorry_and_prefixes"
    solved_holes_filename: str = "solved_hole_replacements.jsonl"
    solved_prefixes_filename: str = "solved_prefix_suffixes.jsonl"
    assembled_full_proofs_filename: str = "assembled_full_proofs.jsonl"
    search_traces_filename: str = "search_traces.jsonl"
    unresolved_filename: str = "unresolved_jobs.jsonl"
    curriculum_items_filename: str = "curriculum_items.jsonl"
    report_filename: str = "run_report.json"
    overwrite: bool = False
    write_search_traces: bool = True
    write_all_expanded_nodes: bool = False


@dataclass
class LeanConfig:
    project_root: str = "."
    use_lake_env: bool = True
    lean_cmd: str = "lean"
    timeout_per_check_sec: int = 10
    max_workers: int = 1  # reserved; current implementation is sequential for stability
    default_imports: List[str] = field(default_factory=lambda: ["Mathlib"])
    keep_temp_files: bool = False
    verbose_errors: bool = False
    allow_sorry_for_partial_checks: bool = True
    trace_tactic: str = "trace_state"
    try_capture_state: bool = True


@dataclass
class GeneratorConfig:
    model_name: str = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
    engine: str = "none"  # none | vllm | adapter
    adapter_path: Optional[str] = None  # e.g. "src.generator:generate_tactic_candidates"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    temperature: float = 0.8
    top_p: float = 0.95
    raw_candidates_per_node: int = 32
    max_new_tokens_per_candidate: int = 128
    candidate_unit: str = "tactic_or_short_block"
    max_block_lines_hole: int = 2
    max_block_lines_prefix: int = 3
    batch_prompts: bool = False
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\ntheorem ", "\n\nlemma ", "#check"])
    forbid_substrings: List[str] = field(default_factory=lambda: ["sorry", "admit", "axiom", "unsafe", "import ", "theorem ", "lemma "])


@dataclass
class EncoderConfig:
    checkpoint_path: Optional[str] = "outputs/run_02_train_encoder/encoder_best.pt"
    adapter_path: Optional[str] = None  # e.g. "src.encoder_search:score_and_embed"
    input_format: str = "state_tactic_pair"
    embedding_dim: int = 768
    batch_size: int = 64
    normalize_embeddings: bool = True
    heuristic_fallback: bool = True
    use_generator_logprob: bool = True
    w_encoder: float = 0.55
    w_generator_logprob: float = 0.35
    w_length_penalty: float = 0.05
    w_bad_pattern_penalty: float = 0.05


@dataclass
class KDPPConfig:
    enabled: bool = True
    raw_pool_size: int = 32
    prefilter_top_m: int = 24
    select_k: int = 8
    kernel: str = "quality_weighted_cosine"
    similarity_metric: str = "cosine"
    quality_temperature: float = 0.7
    similarity_temperature: float = 1.0
    min_quality_threshold: float = 0.15
    jitter: float = 1e-6
    algorithm: str = "greedy_map"


@dataclass
class SearchBudgetConfig:
    max_nodes_per_job: int
    max_depth: int
    timeout_sec: int
    max_block_lines: int


@dataclass
class SearchConfig:
    algorithm: str = "best_first_tree_search"
    node_selection: str = "priority_queue"
    beam_size: int = 32
    expansions_per_node: int = 8
    stop_on_first_solution: bool = False
    max_solutions_per_job: int = 3
    deduplicate_states: bool = True
    deduplicate_tactics_per_node: bool = True
    max_repeated_state_visits: int = 1
    hole_filling: SearchBudgetConfig = field(default_factory=lambda: SearchBudgetConfig(
        max_nodes_per_job=512, max_depth=16, timeout_sec=120, max_block_lines=2
    ))
    prefix_completion: SearchBudgetConfig = field(default_factory=lambda: SearchBudgetConfig(
        max_nodes_per_job=768, max_depth=24, timeout_sec=180, max_block_lines=3
    ))


@dataclass
class NodePriorityConfig:
    w_encoder: float = 1.0
    w_generator_logprob: float = 0.7
    w_verifier_progress: float = 0.8
    w_novelty: float = 0.4
    w_depth_penalty: float = 0.2
    w_repeated_state_penalty: float = 0.5
    w_runtime_penalty: float = 0.3


@dataclass
class AcceptanceConfig:
    require_lean_verified: bool = True
    require_no_sorry_final: bool = True
    save_local_hole_solution: bool = True
    require_contextual_verification_for_training: bool = True
    max_solution_lines_hole: int = 80
    max_solution_lines_prefix: int = 120


@dataclass
class RunConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    lean: LeanConfig = field(default_factory=LeanConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    k_dpp: KDPPConfig = field(default_factory=KDPPConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    node_priority: NodePriorityConfig = field(default_factory=NodePriorityConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)


# =============================================================================
# Utilities
# =============================================================================


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: str, prefix: str = "id") -> str:
    return f"{prefix}_{sha1_text('\n---\n'.join(parts))[:16]}"


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
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:lean|lean4)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def line_count(text: str) -> int:
    return len([x for x in (text or "").splitlines() if x.strip()])


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
            raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml or use JSON configs.") from e
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


def import_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(f"Adapter path must be of form module:function, got {path!r}")
    module_name, fn_name = path.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"Adapter {path} is not callable")
    return fn


def log(msg: str, level: str = "info", cfg_level: str = "info") -> None:
    levels = {"debug": 10, "info": 20, "warning": 30, "error": 40}
    if levels.get(level, 20) >= levels.get(cfg_level, 20):
        print(f"[{level.upper()}] {msg}", file=sys.stderr)


# =============================================================================
# Lean assembly and verification
# =============================================================================


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
    tactic_state: str = ""


class LeanVerifier:
    def __init__(self, cfg: LeanConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root).resolve()
        self.cmd_prefix = ["lake", "env", cfg.lean_cmd] if cfg.use_lake_env else [cfg.lean_cmd]

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
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            has_sorry_warning = "uses 'sorry'" in combined or "declaration uses 'sorry'" in combined
            ok = proc.returncode == 0
            if ok and not allow_sorry and has_sorry_warning:
                ok = False
            return LeanResult(
                ok=ok,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                returncode=proc.returncode,
                elapsed_sec=elapsed,
                file_path=tmp_path if self.cfg.keep_temp_files else None,
                has_sorry_warning=has_sorry_warning,
                error_summary=summarize_lean_error(combined),
                tactic_state=parse_tactic_state(combined),
            )
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            return LeanResult(False, e.stdout or "", e.stderr or "", 124, elapsed, tmp_path if self.cfg.keep_temp_files else None, False, "timeout", "")
        except Exception as e:
            elapsed = time.time() - start
            return LeanResult(False, "", repr(e), 1, elapsed, tmp_path if self.cfg.keep_temp_files else None, False, repr(e), "")
        finally:
            if not self.cfg.keep_temp_files:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def summarize_lean_error(text: str) -> str:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    keep = []
    for l in lines:
        low = l.lower()
        if "error:" in low or "warning:" in low or "unknown" in low or "failed" in low or "unsolved" in low or "timeout" in low:
            keep.append(l)
    return " | ".join(keep[:6])


def parse_tactic_state(text: str) -> str:
    """Best-effort extraction from Lean output. This is intentionally conservative."""
    if not text:
        return ""
    lines = text.splitlines()
    chunks: List[str] = []
    for i, line in enumerate(lines):
        if "⊢" in line or "goals" in line.lower() or "case " in line:
            lo = max(0, i - 8)
            hi = min(len(lines), i + 12)
            chunks.append("\n".join(lines[lo:hi]))
    return "\n---\n".join(chunks[-2:]).strip()


def extract_imports(job: Dict[str, Any], default_imports: Sequence[str]) -> List[str]:
    raw = first_present(job, ["imports", "header_imports", "lean_imports"])
    if isinstance(raw, list):
        imports = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str) and raw.strip():
        imports = [x.strip() for x in raw.splitlines() if x.strip()]
    else:
        imports = list(default_imports)
    out = []
    for imp in imports:
        out.append(imp if imp.startswith("import ") else f"import {imp}")
    return out


def first_present(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


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
    raise ValueError("Could not parse Lean theorem/lemma/example from formal_statement")


def strip_surrounding_theorem_if_present(text: str) -> str:
    s = strip_code_fences(text).strip()
    m = re.search(r"(?s)\b(theorem|lemma|example)\b.*?:=\s*by\s*(.*)$", s)
    if m:
        return m.group(2).strip()
    if s.startswith("by\n") or s == "by" or s.startswith("by "):
        return re.sub(r"^by\s*", "", s, count=1).strip()
    return s


def build_theorem_code(imports: Sequence[str], formal_statement: str, proof_body: str) -> str:
    header = theorem_header_from_formal(formal_statement)
    body = strip_surrounding_theorem_if_present(proof_body)
    return "\n".join(imports).strip() + "\n\n" + header + "\n" + indent(body, 2) + "\n"


def build_partial_theorem_code(imports: Sequence[str], formal_statement: str, proof_body: str) -> str:
    body = strip_surrounding_theorem_if_present(proof_body).rstrip()
    if body:
        body = body + "\n" + "sorry"
    else:
        body = "sorry"
    return build_theorem_code(imports, formal_statement, body)


def build_local_task_code_from_template(job: Dict[str, Any], proof_body: str, imports: Sequence[str]) -> Optional[str]:
    template = first_present(job, ["local_task_code", "hole_task_code", "local_theorem_template", "task_template"])
    if not isinstance(template, str) or not template.strip():
        return None
    proof = strip_surrounding_theorem_if_present(proof_body).strip()
    replacements = {
        "{{proof}}": indent(proof, 2),
        "{{PROOF}}": indent(proof, 2),
        "__PROOF__": indent(proof, 2),
        "by\n  sorry": "by\n" + indent(proof, 2),
        "by sorry": "by\n" + indent(proof, 2),
    }
    code = template
    replaced = False
    for k, v in replacements.items():
        if k in code:
            code = code.replace(k, v, 1)
            replaced = True
            break
    if not replaced:
        code = code.rstrip() + "\n" + indent(proof, 2) + "\n"
    if not re.search(r"^import\s+", code, flags=re.MULTILINE):
        code = "\n".join(imports).strip() + "\n\n" + code
    return code


def build_local_task_code_from_goal(job: Dict[str, Any], proof_body: str, imports: Sequence[str]) -> Optional[str]:
    """Fallback when run_04 stored explicit binder/context fields.

    Expected optional fields:
      - local_binders: Lean binder string, e.g. "(x y : ℝ) (h : x ≤ y)"
      - hole_goal: goal proposition
    """
    binders = first_present(job, ["local_binders", "binders"])
    goal = first_present(job, ["hole_goal", "goal", "target_goal"])
    if not isinstance(goal, str) or not goal.strip():
        return None
    if not isinstance(binders, str):
        binders = ""
    name = stable_id(str(job.get("job_id", "hole")), goal, prefix="hole_task")
    proof = strip_surrounding_theorem_if_present(proof_body).strip()
    return "\n".join(imports).strip() + f"\n\nexample {binders} : {goal.strip()} := by\n" + indent(proof, 2) + "\n"


def build_hole_code(job: Dict[str, Any], proof_body: str, imports: Sequence[str], allow_partial: bool) -> Tuple[str, str]:
    """Return (code, mode) for verifying a hole candidate.

    mode is one of: local_template, local_goal, parent_semiproof.
    """
    body = strip_surrounding_theorem_if_present(proof_body).strip()
    body_for_check = body + "\n" + "sorry" if allow_partial and body else ("sorry" if allow_partial else body)

    code = build_local_task_code_from_template(job, body_for_check, imports)
    if code:
        return code, "local_template"

    code = build_local_task_code_from_goal(job, body_for_check, imports)
    if code:
        return code, "local_goal"

    # Last-resort verifier backend: insert into the parent semi-proof.
    semi = first_present(job, ["semi_proof_text", "parent_semi_proof", "sketch_text", "proof_with_sorry"])
    formal = first_present(job, ["formal_statement", "theorem", "statement"])
    if not isinstance(semi, str) or not isinstance(formal, str):
        raise ValueError("HOLE_FILLING job lacks local task template/local goal and parent semi-proof/formal statement")
    filled = insert_replacement_into_semi_proof(job, semi, body_for_check)
    return build_theorem_code(imports, formal, filled), "parent_semiproof"


def insert_replacement_into_semi_proof(job: Dict[str, Any], semi_proof_text: str, replacement_body: str) -> str:
    semi = strip_surrounding_theorem_if_present(semi_proof_text)
    replacement = strip_surrounding_theorem_if_present(replacement_body).strip()
    marker = first_present(job, ["target_hole_marker", "hole_marker", "marker"])
    if isinstance(marker, str) and marker and marker in semi:
        return semi.replace(marker, replacement, 1)

    hole_id = str(first_present(job, ["hole_id", "target_hole_id"], ""))
    if hole_id:
        for pat in [f"{{{{HOLE:{hole_id}}}}}", f"{{{{{hole_id}}}}}", f"-- HOLE {hole_id}"]:
            if pat in semi:
                return semi.replace(pat, replacement, 1)

    idx = first_present(job, ["hole_index", "target_hole_index", "sorry_index"], 0)
    try:
        idx_int = int(idx)
    except Exception:
        idx_int = 0
    return replace_nth_sorry_line(semi, replacement, idx_int)


def replace_nth_sorry_line(text: str, replacement: str, n: int) -> str:
    lines = text.splitlines()
    seen = 0
    out: List[str] = []
    replaced = False
    for line in lines:
        if not replaced and re.match(r"^(\s*)sorry\b", line):
            if seen == n:
                m = re.match(r"^(\s*)", line)
                base_indent = len(m.group(1)) if m else 0
                out.append(indent_to_column(replacement, base_indent))
                replaced = True
            else:
                out.append(line)
            seen += 1
        else:
            out.append(line)
    if not replaced:
        # Fallback: replace first textual sorry occurrence.
        return re.sub(r"\bsorry\b", lambda _: replacement, text, count=1)
    return "\n".join(out)


def indent_to_column(text: str, col: int) -> str:
    pad = " " * col
    return "\n".join(pad + l if l.strip() else l for l in text.splitlines())


# =============================================================================
# Generator adapters
# =============================================================================


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
            "No generator configured. Use generator.engine='vllm' or generator.engine='adapter' "
            "with generator.adapter_path='module:function'."
        )


class AdapterGenerator(BaseGenerator):
    """Adapter wrapper.

    The callable can have either signature:
      fn(prompts: List[str], n: int, temperature: float, top_p: float, max_new_tokens: int, stop: Sequence[str])
    returning List[List[str]] or List[List[dict]], or:
      fn(prompt: str, n: int, **kwargs)
    returning List[str]/List[dict].
    """
    def __init__(self, path: str):
        self.fn = import_callable(path)

    def generate(self, prompts: List[str], n: int, temperature: float, top_p: float, max_new_tokens: int, stop: Sequence[str]) -> List[List[Generation]]:
        try:
            raw = self.fn(prompts=prompts, n=n, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, stop=stop)
        except TypeError:
            raw = [self.fn(prompt=p, n=n, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, stop=stop) for p in prompts]
        return [normalize_generation_list(x) for x in raw]


class VLLMGenerator(BaseGenerator):
    def __init__(self, cfg: GeneratorConfig):
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError("vLLM is not installed but generator.engine='vllm'.") from e
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
        outs = self.llm.generate(prompts, params)
        result: List[List[Generation]] = []
        for out in outs:
            gens = []
            for o in out.outputs:
                gens.append(Generation(text=o.text, logprob=None, metadata={"finish_reason": getattr(o, "finish_reason", None)}))
            result.append(gens)
        return result


def normalize_generation_list(raw: Any) -> List[Generation]:
    out: List[Generation] = []
    for x in raw or []:
        if isinstance(x, Generation):
            out.append(x)
        elif isinstance(x, dict):
            out.append(Generation(text=str(x.get("text", "")), logprob=x.get("logprob"), metadata={k: v for k, v in x.items() if k not in {"text", "logprob"}}))
        else:
            out.append(Generation(text=str(x)))
    return out


def build_generator(cfg: GeneratorConfig) -> BaseGenerator:
    if cfg.engine == "vllm":
        return VLLMGenerator(cfg)
    if cfg.engine == "adapter":
        if not cfg.adapter_path:
            raise ValueError("generator.engine='adapter' requires generator.adapter_path")
        return AdapterGenerator(cfg.adapter_path)
    return NoOpGenerator()


# =============================================================================
# Encoder adapter and heuristic fallback
# =============================================================================


@dataclass
class Candidate:
    text: str
    raw_text: str
    logprob: Optional[float] = None
    quality: float = 0.0
    encoder_score: float = 0.0
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EncoderScorer:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.fn: Optional[Callable[..., Any]] = import_callable(cfg.adapter_path) if cfg.adapter_path else None

    def score_and_embed(self, state: str, candidates: List[Candidate], job: Dict[str, Any]) -> List[Candidate]:
        if self.fn is not None:
            try:
                raw = self.fn(state=state, candidates=[c.text for c in candidates], job=job, checkpoint_path=self.cfg.checkpoint_path)
            except TypeError:
                raw = self.fn(state, [c.text for c in candidates])
            return self._apply_adapter_output(candidates, raw)
        if not self.cfg.heuristic_fallback:
            raise RuntimeError("No encoder adapter configured and heuristic_fallback=False")
        return self._heuristic_score_and_embed(state, candidates)

    def _apply_adapter_output(self, candidates: List[Candidate], raw: Any) -> List[Candidate]:
        # Expected formats:
        #   {"scores": [...], "embeddings": [[...], ...]}
        #   List[{"score": float, "embedding": [...]}, ...]
        if isinstance(raw, dict):
            scores = raw.get("scores") or raw.get("quality_scores") or []
            embs = raw.get("embeddings") or raw.get("pair_embeddings") or []
            for i, c in enumerate(candidates):
                if i < len(scores):
                    c.encoder_score = float(scores[i])
                if i < len(embs):
                    c.embedding = [float(x) for x in embs[i]]
        elif isinstance(raw, list):
            for i, item in enumerate(raw[:len(candidates)]):
                if isinstance(item, dict):
                    candidates[i].encoder_score = float(item.get("score", item.get("quality", 0.0)))
                    candidates[i].embedding = [float(x) for x in item.get("embedding", item.get("pair_embedding", []))]
                elif isinstance(item, (int, float)):
                    candidates[i].encoder_score = float(item)
        for c in candidates:
            if not c.embedding:
                c.embedding = hashing_embedding(c.text, self.cfg.embedding_dim)
            c.quality = squash_quality(c.encoder_score)
        return candidates

    def _heuristic_score_and_embed(self, state: str, candidates: List[Candidate]) -> List[Candidate]:
        for c in candidates:
            t = c.text.strip()
            score = 0.0
            # Tactics often useful in Lean Workbook-style algebra/number theory.
            useful = ["simp", "simp_all", "nlinarith", "linarith", "omega", "norm_num", "ring", "field_simp", "constructor", "intro", "cases", "induction", "have", "suffices", "exact", "apply", "rw", "rwa"]
            for u in useful:
                if re.search(rf"\b{re.escape(u)}\b", t):
                    score += 0.35
            if "sorry" in t or "admit" in t:
                score -= 5.0
            if line_count(t) > 3:
                score -= 0.2 * (line_count(t) - 3)
            if c.logprob is not None:
                score += 0.05 * float(c.logprob)
            # Penalize obvious text spillover.
            if "```" in t or "Here" in t or "proof" in t.lower() and not re.search(r"\bhave\b", t):
                score -= 0.5
            c.encoder_score = score
            c.embedding = hashing_embedding(state + "\n---TACTIC---\n" + t, self.cfg.embedding_dim)
            c.quality = squash_quality(score)
        return candidates


def hashing_embedding(text: str, dim: int) -> List[float]:
    # Deterministic lightweight fallback embedding. Not a replacement for the trained encoder.
    vals = [0.0] * dim
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_']*|[≤≥=<>+\-*/^]+|\d+|\S", text)
    for tok in toks:
        h = int(hashlib.sha1(tok.encode("utf-8", errors="ignore")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vals[idx] += sign
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def squash_quality(x: float) -> float:
    # Stable sigmoid mapped away from exact 0/1.
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# Candidate cleaning and k-DPP
# =============================================================================


def clean_candidate_text(raw: str, max_block_lines: int) -> str:
    s = strip_code_fences(raw).strip()
    s = re.sub(r"^by\s*", "", s).strip()
    # Drop explanatory prefaces if the model leaked prose.
    if "\n" in s:
        lines = [l.rstrip() for l in s.splitlines()]
    else:
        lines = [s]
    cleaned: List[str] = []
    for line in lines:
        l = line.rstrip()
        if not l.strip():
            if cleaned:
                break
            continue
        if re.match(r"^[-*]\s+", l):
            l = re.sub(r"^[-*]\s+", "", l)
        if l.strip().startswith(("```", "#check")):
            break
        # Stop if the model starts a new declaration.
        if re.match(r"^\s*(theorem|lemma|example|import|namespace|section)\b", l):
            break
        cleaned.append(l)
        if len([x for x in cleaned if x.strip()]) >= max_block_lines:
            break
    return "\n".join(cleaned).strip()


def filter_candidates(gens: List[Generation], cfg: GeneratorConfig, max_block_lines: int) -> List[Candidate]:
    out: List[Candidate] = []
    seen: set[str] = set()
    for g in gens:
        txt = clean_candidate_text(g.text, max_block_lines=max_block_lines)
        if not txt:
            continue
        low = txt.lower()
        if any(f.lower() in low for f in cfg.forbid_substrings):
            continue
        h = sha1_text(normalize_ws(txt))
        if h in seen:
            continue
        seen.add(h)
        out.append(Candidate(text=txt, raw_text=g.text, logprob=g.logprob, metadata=g.metadata))
    return out


def select_by_k_dpp(candidates: List[Candidate], cfg: KDPPConfig) -> List[Candidate]:
    if not candidates:
        return []
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("k-DPP selection requires numpy. Install numpy or provide it in the environment.") from e

    # Filter/prefilter by quality.
    filtered = [c for c in candidates if c.quality >= cfg.min_quality_threshold]
    if not filtered:
        filtered = sorted(candidates, key=lambda c: c.quality, reverse=True)[:max(1, cfg.select_k)]
    filtered = sorted(filtered, key=lambda c: c.quality, reverse=True)[:cfg.prefilter_top_m]
    k = min(cfg.select_k, len(filtered))
    if k <= 0:
        return []
    if k == 1:
        return [max(filtered, key=lambda c: c.quality)]

    E = np.asarray([c.embedding for c in filtered], dtype=float)
    # Normalize embeddings.
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E = E / norms
    S = E @ E.T
    S = np.maximum(S, 0.0)
    if cfg.similarity_temperature != 1.0:
        S = np.power(S, cfg.similarity_temperature)
    q = np.asarray([max(1e-6, c.quality) for c in filtered], dtype=float)
    if cfg.quality_temperature != 1.0:
        q = np.power(q, 1.0 / max(1e-6, cfg.quality_temperature))
    L = (q[:, None] * S * q[None, :]).astype(float)
    L += np.eye(len(filtered)) * cfg.jitter

    selected: List[int] = []
    remaining = set(range(len(filtered)))
    # Start from the best-quality candidate to stabilize greedy MAP.
    first = int(np.argmax(q))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        best_idx = None
        best_val = -float("inf")
        for i in remaining:
            idxs = selected + [i]
            sub = L[np.ix_(idxs, idxs)]
            sign, logdet = np.linalg.slogdet(sub)
            val = float(logdet if sign > 0 else -1e18)
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [filtered[i] for i in selected]


# =============================================================================
# Search data structures
# =============================================================================


@dataclass
class SearchNode:
    node_id: str
    job_id: str
    parent_node_id: Optional[str]
    depth: int
    proof_fragment: str
    current_state: str
    goals_hash: str
    last_tactic: str = ""
    cumulative_generator_logprob: float = 0.0
    cumulative_encoder_score: float = 0.0
    cumulative_quality: float = 0.0
    cumulative_runtime_sec: float = 0.0
    num_valid_steps: int = 0
    priority: float = 0.0


@dataclass
class SearchSolution:
    solution_id: str
    job_id: str
    job_type: str
    proof_fragment: str
    status: str
    local_ok: bool
    contextual_ok: bool
    final_ok: bool
    depth: int
    node_id: str
    lean_elapsed_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchStats:
    job_id: str
    job_type: str
    started_at: float = field(default_factory=time.time)
    nodes_expanded: int = 0
    candidates_generated: int = 0
    candidates_after_filter: int = 0
    candidates_selected: int = 0
    lean_checks: int = 0
    valid_transitions: int = 0
    invalid_tactics: int = 0
    duplicate_states: int = 0
    solutions_found: int = 0
    timeouts: int = 0
    errors: Dict[str, int] = field(default_factory=dict)

    def elapsed(self) -> float:
        return time.time() - self.started_at


# =============================================================================
# Search engine
# =============================================================================


class TreeSearcher:
    def __init__(self, cfg: RunConfig, verifier: LeanVerifier, generator: BaseGenerator, encoder: EncoderScorer):
        self.cfg = cfg
        self.verifier = verifier
        self.generator = generator
        self.encoder = encoder

    def search_job(self, job: Dict[str, Any]) -> Tuple[List[SearchSolution], SearchStats, List[Dict[str, Any]]]:
        job_type = normalize_job_type(job)
        job_id = str(first_present(job, ["job_id", "id"], stable_id(json.dumps(job, sort_keys=True), prefix="job")))
        job["job_id"] = job_id
        stats = SearchStats(job_id=job_id, job_type=job_type)
        budget = self.cfg.search.hole_filling if job_type == "HOLE_FILLING" else self.cfg.search.prefix_completion

        root_state = self.initial_state_text(job, job_type)
        root = SearchNode(
            node_id=stable_id(job_id, "root", prefix="node"),
            job_id=job_id,
            parent_node_id=None,
            depth=0,
            proof_fragment="",
            current_state=root_state,
            goals_hash=state_hash(root_state),
            priority=0.0,
        )
        heap: List[Tuple[float, int, SearchNode]] = []
        counter = 0
        self.push_node(heap, root, counter)
        counter += 1
        visited_states: Dict[str, int] = {root.goals_hash: 1}
        solutions: List[SearchSolution] = []
        traces: List[Dict[str, Any]] = []

        deadline = time.time() + budget.timeout_sec
        while heap and stats.nodes_expanded < budget.max_nodes_per_job and time.time() < deadline:
            _, _, node = heapq.heappop(heap)
            if node.depth >= budget.max_depth:
                continue
            stats.nodes_expanded += 1

            prompt = self.build_prompt(job, job_type, node)
            try:
                gen_batches = self.generator.generate(
                    [prompt],
                    n=self.cfg.generator.raw_candidates_per_node,
                    temperature=self.cfg.generator.temperature,
                    top_p=self.cfg.generator.top_p,
                    max_new_tokens=self.cfg.generator.max_new_tokens_per_candidate,
                    stop=self.cfg.generator.stop_sequences,
                )
                gens = gen_batches[0]
            except Exception as e:
                stats.errors[type(e).__name__] = stats.errors.get(type(e).__name__, 0) + 1
                if self.cfg.experiment.dry_run:
                    gens = dry_run_candidates(job_type)
                else:
                    raise

            stats.candidates_generated += len(gens)
            candidates = filter_candidates(gens, self.cfg.generator, max_block_lines=budget.max_block_lines)
            if self.cfg.search.deduplicate_tactics_per_node:
                candidates = dedup_candidates(candidates)
            stats.candidates_after_filter += len(candidates)
            if not candidates:
                continue

            candidates = self.encoder.score_and_embed(node.current_state, candidates, job)
            candidates = combine_quality_scores(candidates, self.cfg.encoder)
            selected = select_by_k_dpp(candidates, self.cfg.k_dpp)
            selected = selected[:self.cfg.search.expansions_per_node]
            stats.candidates_selected += len(selected)

            node_trace = {
                "job_id": job_id,
                "job_type": job_type,
                "node_id": node.node_id,
                "depth": node.depth,
                "state_hash": node.goals_hash,
                "selected_candidates": [candidate_to_record(c) for c in selected],
            }
            if self.cfg.outputs.write_all_expanded_nodes:
                node_trace["proof_fragment"] = node.proof_fragment
                node_trace["current_state"] = node.current_state
            traces.append(node_trace)

            for cand in selected:
                child_fragment = append_tactic_block(node.proof_fragment, cand.text)
                transition = self.check_transition(job, job_type, child_fragment, partial=True)
                stats.lean_checks += 1
                if not transition.ok:
                    stats.invalid_tactics += 1
                    err = transition.error_summary or "lean_error"
                    stats.errors[err[:120]] = stats.errors.get(err[:120], 0) + 1
                    continue

                stats.valid_transitions += 1
                solved_res = self.check_transition(job, job_type, child_fragment, partial=False)
                stats.lean_checks += 1
                is_solved = solved_res.ok

                if is_solved:
                    sol = self.make_solution(job, job_type, child_fragment, node, cand, solved_res)
                    solutions.append(sol)
                    stats.solutions_found += 1
                    if self.cfg.search.stop_on_first_solution or len(solutions) >= self.cfg.search.max_solutions_per_job:
                        return solutions, stats, traces

                new_state = transition.tactic_state or solved_res.tactic_state or self.synthetic_state_after(node.current_state, cand.text, transition)
                gh = state_hash(new_state or child_fragment)
                if self.cfg.search.deduplicate_states:
                    prev_visits = visited_states.get(gh, 0)
                    if prev_visits >= self.cfg.search.max_repeated_state_visits:
                        stats.duplicate_states += 1
                        continue
                    visited_states[gh] = prev_visits + 1

                child = SearchNode(
                    node_id=stable_id(job_id, node.node_id, cand.text, str(node.depth + 1), prefix="node"),
                    job_id=job_id,
                    parent_node_id=node.node_id,
                    depth=node.depth + 1,
                    proof_fragment=child_fragment,
                    current_state=new_state,
                    goals_hash=gh,
                    last_tactic=cand.text,
                    cumulative_generator_logprob=node.cumulative_generator_logprob + float(cand.logprob or 0.0),
                    cumulative_encoder_score=node.cumulative_encoder_score + cand.encoder_score,
                    cumulative_quality=node.cumulative_quality + cand.quality,
                    cumulative_runtime_sec=node.cumulative_runtime_sec + transition.elapsed_sec,
                    num_valid_steps=node.num_valid_steps + 1,
                )
                child.priority = self.compute_node_priority(child, cand, transition, visited_states.get(gh, 1))
                self.push_node(heap, child, counter)
                counter += 1
                if len(heap) > self.cfg.search.beam_size * 10:
                    heap = heapq.nsmallest(self.cfg.search.beam_size * 5, heap)
                    heapq.heapify(heap)

        if time.time() >= deadline:
            stats.timeouts += 1
        return solutions, stats, traces

    def push_node(self, heap: List[Tuple[float, int, SearchNode]], node: SearchNode, counter: int) -> None:
        heapq.heappush(heap, (-node.priority, counter, node))

    def initial_state_text(self, job: Dict[str, Any], job_type: str) -> str:
        if job_type == "HOLE_FILLING":
            ctx = first_present(job, ["hole_context", "local_context", "context"], "")
            goal = first_present(job, ["hole_goal", "goal", "target_goal"], "")
            return f"Local context:\n{ctx}\n\nGoal:\n{goal}".strip()
        return str(first_present(job, ["state_after_prefix", "current_state", "goal_state", "prefix_state"], ""))

    def build_prompt(self, job: Dict[str, Any], job_type: str, node: SearchNode) -> str:
        if job_type == "HOLE_FILLING":
            ctx = first_present(job, ["hole_context", "local_context", "context"], "")
            goal = first_present(job, ["hole_goal", "goal", "target_goal"], "")
            sketch = first_present(job, ["semi_proof_text", "parent_semi_proof", "sketch_text"], "")
            return (
                "You are filling one Lean proof hole. Generate exactly one next Lean tactic or a very short tactic block.\n"
                "Do not generate a full proof. Do not use sorry or admit. Return only Lean tactic code.\n\n"
                f"Local context:\n{ctx}\n\nGoal:\n{goal}\n\n"
                f"Proof fragment already found for this hole:\n{node.proof_fragment or '(empty)'}\n\n"
                f"Surrounding semi-proof sketch, for orientation only:\n{truncate_middle(str(sketch), 2500)}\n\n"
                "Next Lean tactic/block:"
            )
        formal = first_present(job, ["formal_statement", "theorem", "statement"], "")
        prefix = first_present(job, ["prefix_text", "prefix", "proof_prefix"], "")
        state = node.current_state or first_present(job, ["state_after_prefix", "current_state"], "")
        return (
            "The following Lean proof prefix is already accepted. Continue from the current tactic state.\n"
            "Generate exactly one next Lean tactic or a very short tactic block. Do not use sorry or admit.\n"
            "Return only Lean tactic code.\n\n"
            f"Theorem statement:\n{formal}\n\n"
            f"Accepted prefix:\n{prefix}\n\n"
            f"Suffix fragment already found:\n{node.proof_fragment or '(empty)'}\n\n"
            f"Current tactic state:\n{state}\n\n"
            "Next Lean tactic/block:"
        )

    def check_transition(self, job: Dict[str, Any], job_type: str, fragment: str, partial: bool) -> LeanResult:
        imports = extract_imports(job, self.cfg.lean.default_imports)
        if job_type == "HOLE_FILLING":
            code, _mode = build_hole_code(job, fragment, imports, allow_partial=partial)
            return self.verifier.verify_code(code, timeout_sec=self.cfg.lean.timeout_per_check_sec, allow_sorry=partial)
        formal = first_present(job, ["formal_statement", "theorem", "statement"])
        if not isinstance(formal, str) or not formal.strip():
            raise ValueError("PREFIX_COMPLETION job lacks formal_statement")
        prefix = str(first_present(job, ["prefix_text", "prefix", "proof_prefix"], ""))
        body = append_tactic_block(prefix, fragment)
        code = build_partial_theorem_code(imports, formal, body) if partial else build_theorem_code(imports, formal, body)
        return self.verifier.verify_code(code, timeout_sec=self.cfg.lean.timeout_per_check_sec, allow_sorry=partial)

    def make_solution(self, job: Dict[str, Any], job_type: str, fragment: str, parent: SearchNode, cand: Candidate, res: LeanResult) -> SearchSolution:
        job_id = str(job["job_id"])
        contextual_ok = False
        final_ok = False
        status = "SOLVED"
        metadata: Dict[str, Any] = {"last_tactic": cand.text, "encoder_score": cand.encoder_score, "quality": cand.quality}

        if job_type == "HOLE_FILLING":
            # Candidate has locally solved the target hole or passed the parent-semiproof verifier.
            contextual = self.contextual_hole_check(job, fragment)
            contextual_ok = contextual.ok
            final_ok = contextual_ok and not contextual.has_sorry_warning
            if contextual_ok and final_ok:
                status = "SOLVED_FULL_PARENT"
            elif contextual_ok:
                status = "SOLVED_CONTEXTUAL"
            else:
                status = "SOLVED_LOCAL_ONLY"
            metadata["contextual_error_summary"] = contextual.error_summary
            metadata["contextual_elapsed_sec"] = contextual.elapsed_sec
        else:
            contextual_ok = True
            final_ok = True
            status = "SOLVED_PREFIX_SUFFIX"

        return SearchSolution(
            solution_id=stable_id(job_id, fragment, status, prefix="solution"),
            job_id=job_id,
            job_type=job_type,
            proof_fragment=fragment,
            status=status,
            local_ok=True,
            contextual_ok=contextual_ok,
            final_ok=final_ok,
            depth=parent.depth + 1,
            node_id=parent.node_id,
            lean_elapsed_sec=res.elapsed_sec,
            metadata=metadata,
        )

    def contextual_hole_check(self, job: Dict[str, Any], fragment: str) -> LeanResult:
        imports = extract_imports(job, self.cfg.lean.default_imports)
        semi = first_present(job, ["semi_proof_text", "parent_semi_proof", "sketch_text", "proof_with_sorry"])
        formal = first_present(job, ["formal_statement", "theorem", "statement"])
        if not isinstance(semi, str) or not isinstance(formal, str):
            # If no parent sketch exists, local proof is all we can validate.
            return LeanResult(True, "", "", 0, 0.0, None, False, "no_parent_sketch_available", "")
        filled = insert_replacement_into_semi_proof(job, semi, fragment)
        code = build_theorem_code(imports, formal, filled)
        # Other holes may still be sorry; target replacement itself must not contain sorry.
        return self.verifier.verify_code(code, timeout_sec=self.cfg.lean.timeout_per_check_sec, allow_sorry=True)

    def compute_node_priority(self, node: SearchNode, cand: Candidate, transition: LeanResult, repeated_visits: int) -> float:
        c = self.cfg.node_priority
        avg_encoder = node.cumulative_encoder_score / max(1, node.num_valid_steps)
        avg_logprob = node.cumulative_generator_logprob / max(1, node.num_valid_steps)
        progress = verifier_progress_score(transition)
        novelty = 1.0 / max(1, repeated_visits)
        return (
            c.w_encoder * avg_encoder
            + c.w_generator_logprob * avg_logprob
            + c.w_verifier_progress * progress
            + c.w_novelty * novelty
            - c.w_depth_penalty * node.depth
            - c.w_repeated_state_penalty * max(0, repeated_visits - 1)
            - c.w_runtime_penalty * min(10.0, transition.elapsed_sec) / 10.0
        )

    def synthetic_state_after(self, prev_state: str, tactic: str, res: LeanResult) -> str:
        if res.tactic_state:
            return res.tactic_state
        return truncate_middle(prev_state, 1500) + "\n\nAfter accepted tactic:\n" + tactic


def verifier_progress_score(res: LeanResult) -> float:
    if not res.ok:
        return 0.0
    score = 0.25
    txt = (res.stdout + "\n" + res.stderr + "\n" + res.tactic_state).lower()
    if "no goals" in txt or "goals accomplished" in txt:
        score += 1.0
    if "unsolved goals" not in txt:
        score += 0.15
    if res.elapsed_sec < 2.0:
        score += 0.1
    return score


def dry_run_candidates(job_type: str) -> List[Generation]:
    base = ["simp", "simp_all", "nlinarith", "linarith", "norm_num", "omega", "ring", "aesop", "exact?", "assumption"]
    if job_type == "HOLE_FILLING":
        base += ["constructor", "intro h", "have h : True := by trivial"]
    return [Generation(text=x) for x in base]


# =============================================================================
# Search helpers and output conversion
# =============================================================================


def normalize_job_type(job: Dict[str, Any]) -> str:
    jt = str(first_present(job, ["job_type", "type", "item_type"], "")).upper()
    if jt in {"HOLE", "MCSP", "MCSP_HOLE", "HOLE_FILL", "HOLE_FILLING", "SORRY_HOLE"}:
        return "HOLE_FILLING"
    if jt in {"PREFIX", "PREFIX_COMPLETION", "PREFIX_SUFFIX", "PREFIX_SEARCH"}:
        return "PREFIX_COMPLETION"
    # Infer from fields.
    if any(k in job for k in ["hole_goal", "semi_proof_text", "target_hole_id", "hole_id"]):
        return "HOLE_FILLING"
    return "PREFIX_COMPLETION"


def state_hash(state: str) -> str:
    return sha1_text(normalize_ws(state))[:16]


def append_tactic_block(prefix: str, block: str) -> str:
    p = strip_surrounding_theorem_if_present(prefix).rstrip()
    b = strip_surrounding_theorem_if_present(block).strip()
    if not p:
        return b
    if not b:
        return p
    return p + "\n" + b


def dedup_candidates(cands: List[Candidate]) -> List[Candidate]:
    out: List[Candidate] = []
    seen: set[str] = set()
    for c in cands:
        h = sha1_text(normalize_ws(c.text))
        if h not in seen:
            out.append(c)
            seen.add(h)
    return out


def combine_quality_scores(candidates: List[Candidate], cfg: EncoderConfig) -> List[Candidate]:
    for c in candidates:
        logp = float(c.logprob or 0.0)
        len_pen = min(1.0, max(0.0, (line_count(c.text) - 1) / 5.0))
        bad_pen = 1.0 if re.search(r"\b(sorry|admit|theorem|lemma|import)\b", c.text, flags=re.I) else 0.0
        raw = cfg.w_encoder * c.encoder_score + cfg.w_generator_logprob * logp - cfg.w_length_penalty * len_pen - cfg.w_bad_pattern_penalty * bad_pen
        c.quality = squash_quality(raw)
        c.metadata.update({"combined_raw_quality": raw, "length_penalty": len_pen, "bad_pattern_penalty": bad_pen})
    return candidates


def candidate_to_record(c: Candidate) -> Dict[str, Any]:
    return {
        "text": c.text,
        "logprob": c.logprob,
        "quality": c.quality,
        "encoder_score": c.encoder_score,
        "metadata": c.metadata,
    }


def solution_to_record(sol: SearchSolution, job: Dict[str, Any]) -> Dict[str, Any]:
    rec = dataclasses.asdict(sol)
    rec.update({
        "parent_theorem_id": first_present(job, ["parent_theorem_id", "theorem_id", "id"]),
        "formal_statement": first_present(job, ["formal_statement", "theorem", "statement"]),
        "source_job": job,
    })
    return rec


def curriculum_item_from_solution(sol: SearchSolution, job: Dict[str, Any]) -> Dict[str, Any]:
    job_type = normalize_job_type(job)
    if job_type == "HOLE_FILLING":
        item_type = "solved_mcsp_hole_replacement"
        input_context = {
            "hole_context": first_present(job, ["hole_context", "local_context", "context"]),
            "hole_goal": first_present(job, ["hole_goal", "goal", "target_goal"]),
            "semi_proof_text": first_present(job, ["semi_proof_text", "parent_semi_proof", "sketch_text"]),
        }
    else:
        item_type = "solved_prefix_suffix"
        input_context = {
            "prefix_text": first_present(job, ["prefix_text", "prefix", "proof_prefix"]),
            "state_after_prefix": first_present(job, ["state_after_prefix", "current_state", "goal_state"]),
        }
    return {
        "item_id": stable_id(sol.solution_id, item_type, prefix="curriculum"),
        "item_type": item_type,
        "job_id": sol.job_id,
        "parent_theorem_id": first_present(job, ["parent_theorem_id", "theorem_id", "id"]),
        "formal_statement": first_present(job, ["formal_statement", "theorem", "statement"]),
        "input_context": input_context,
        "target_proof": sol.proof_fragment,
        "verified": sol.final_ok or sol.contextual_ok,
        "uses_sorry": False,
        "quality_class": solution_quality(sol),
        "metadata": sol.metadata,
    }


def solution_quality(sol: SearchSolution) -> str:
    if sol.final_ok and sol.depth <= 8:
        return "high"
    if sol.contextual_ok or sol.final_ok:
        return "medium"
    return "low"


def stats_to_record(stats: SearchStats) -> Dict[str, Any]:
    d = dataclasses.asdict(stats)
    d["elapsed_sec"] = stats.elapsed()
    return d


def truncate_middle(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... <truncated> ...\n" + text[-half:]


# =============================================================================
# Job loading, reporting, and main
# =============================================================================


def load_jobs(cfg: RunConfig) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    if Path(cfg.inputs.mcsp_jobs_path).exists():
        for j in read_jsonl(cfg.inputs.mcsp_jobs_path):
            j = dict(j)
            j["job_type"] = "HOLE_FILLING"
            jobs.append(j)
    if Path(cfg.inputs.prefix_jobs_path).exists():
        for j in read_jsonl(cfg.inputs.prefix_jobs_path):
            j = dict(j)
            j["job_type"] = "PREFIX_COMPLETION"
            jobs.append(j)
    if cfg.experiment.job_type_filter:
        wanted = cfg.experiment.job_type_filter.upper()
        jobs = [j for j in jobs if normalize_job_type(j) == wanted]
    jobs.sort(key=job_priority, reverse=True)
    if cfg.experiment.max_jobs is not None:
        jobs = jobs[:cfg.experiment.max_jobs]
    return jobs


def job_priority(job: Dict[str, Any]) -> float:
    jt = normalize_job_type(job)
    base = 2.0 if jt == "HOLE_FILLING" else 1.0
    frontier = safe_float(first_present(job, ["frontier_score", "quality_score", "run04_quality_score"], 0.0))
    structural = 1.0 if str(first_present(job, ["mcsp_quality_class", "proof_style"], "")).lower().startswith("strong") else 0.0
    context_size = len(str(first_present(job, ["hole_context", "state_after_prefix", "current_state"], "")))
    return base + 1.5 * frontier + 0.5 * structural - 0.00005 * context_size


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def existing_done_job_ids(cfg: RunConfig, out_dir: Path) -> set[str]:
    if not cfg.experiment.resume:
        return set()
    done: set[str] = set()
    for fn in [cfg.outputs.solved_holes_filename, cfg.outputs.solved_prefixes_filename, cfg.outputs.unresolved_filename]:
        p = out_dir / fn
        if not p.exists():
            continue
        for rec in read_jsonl(p):
            jid = first_present(rec, ["job_id", "source_job", "job", "id"])
            if isinstance(jid, dict):
                jid = first_present(jid, ["job_id", "id"])
            if jid:
                done.add(str(jid))
    return done


def prepare_outputs(cfg: RunConfig) -> Dict[str, Path]:
    out_dir = ensure_dir(cfg.outputs.out_dir)
    paths = {
        "solved_holes": out_dir / cfg.outputs.solved_holes_filename,
        "solved_prefixes": out_dir / cfg.outputs.solved_prefixes_filename,
        "assembled_full_proofs": out_dir / cfg.outputs.assembled_full_proofs_filename,
        "search_traces": out_dir / cfg.outputs.search_traces_filename,
        "unresolved": out_dir / cfg.outputs.unresolved_filename,
        "curriculum_items": out_dir / cfg.outputs.curriculum_items_filename,
        "report": out_dir / cfg.outputs.report_filename,
    }
    if cfg.outputs.overwrite:
        for p in paths.values():
            if p.exists() and p.is_file():
                p.unlink()
    return paths


def write_solution_outputs(sol: SearchSolution, job: Dict[str, Any], paths: Dict[str, Path], cfg: RunConfig) -> None:
    rec = solution_to_record(sol, job)
    if sol.job_type == "HOLE_FILLING":
        append_jsonl(paths["solved_holes"], rec)
        if sol.status == "SOLVED_FULL_PARENT":
            append_jsonl(paths["assembled_full_proofs"], rec)
    else:
        append_jsonl(paths["solved_prefixes"], rec)
    item = curriculum_item_from_solution(sol, job)
    # Only contextual/final verified items enter the training-positive corpus.
    if sol.job_type == "PREFIX_COMPLETION" or sol.contextual_ok or sol.final_ok:
        append_jsonl(paths["curriculum_items"], item)


def write_unresolved(job: Dict[str, Any], stats: SearchStats, paths: Dict[str, Path], reason: str = "UNRESOLVED_WITHIN_BUDGET") -> None:
    append_jsonl(paths["unresolved"], {
        "job_id": job.get("job_id"),
        "job_type": normalize_job_type(job),
        "reason": reason,
        "stats": stats_to_record(stats),
        "source_job": job,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mcsp-jobs", type=str, default=None)
    parser.add_argument("--prefix-jobs", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--generator-engine", type=str, default=None, choices=["none", "vllm", "adapter"])
    parser.add_argument("--generator-adapter", type=str, default=None)
    parser.add_argument("--encoder-adapter", type=str, default=None)
    parser.add_argument("--lean-project-root", type=str, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--job-type", type=str, default=None, choices=["HOLE_FILLING", "PREFIX_COMPLETION"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mcsp_jobs:
        cfg.inputs.mcsp_jobs_path = args.mcsp_jobs
    if args.prefix_jobs:
        cfg.inputs.prefix_jobs_path = args.prefix_jobs
    if args.out_dir:
        cfg.outputs.out_dir = args.out_dir
    if args.generator_engine:
        cfg.generator.engine = args.generator_engine
    if args.generator_adapter:
        cfg.generator.adapter_path = args.generator_adapter
        cfg.generator.engine = "adapter"
    if args.encoder_adapter:
        cfg.encoder.adapter_path = args.encoder_adapter
    if args.lean_project_root:
        cfg.lean.project_root = args.lean_project_root
    if args.max_jobs is not None:
        cfg.experiment.max_jobs = args.max_jobs
    if args.job_type:
        cfg.experiment.job_type_filter = args.job_type
    if args.dry_run:
        cfg.experiment.dry_run = True
    if args.overwrite:
        cfg.outputs.overwrite = True
        cfg.experiment.resume = False

    random.seed(cfg.experiment.seed)
    paths = prepare_outputs(cfg)
    jobs = load_jobs(cfg)
    log(f"Loaded {len(jobs)} jobs", cfg_level=cfg.experiment.log_level)

    if cfg.experiment.dry_run and cfg.generator.engine == "none":
        log("dry_run=True: generator errors will be replaced by heuristic dry-run candidates", "warning", cfg.experiment.log_level)

    done = existing_done_job_ids(cfg, ensure_dir(cfg.outputs.out_dir))
    if done:
        log(f"Resume enabled: skipping {len(done)} already finished jobs", cfg_level=cfg.experiment.log_level)

    verifier = LeanVerifier(cfg.lean)
    generator = build_generator(cfg.generator)
    encoder = EncoderScorer(cfg.encoder)
    searcher = TreeSearcher(cfg, verifier, generator, encoder)

    report = {
        "experiment_name": cfg.experiment.experiment_name,
        "started_at": now_ts(),
        "num_jobs_loaded": len(jobs),
        "num_jobs_processed": 0,
        "num_hole_jobs": 0,
        "num_prefix_jobs": 0,
        "num_jobs_solved": 0,
        "num_hole_solutions": 0,
        "num_prefix_solutions": 0,
        "num_unresolved": 0,
        "total_nodes_expanded": 0,
        "total_lean_checks": 0,
        "errors": {},
    }

    for idx, job in enumerate(jobs, start=1):
        job_type = normalize_job_type(job)
        job_id = str(first_present(job, ["job_id", "id"], stable_id(json.dumps(job, sort_keys=True), prefix="job")))
        job["job_id"] = job_id
        if job_id in done:
            continue
        log(f"[{idx}/{len(jobs)}] Searching {job_type} job {job_id}", cfg_level=cfg.experiment.log_level)
        try:
            sols, stats, traces = searcher.search_job(job)
            report["num_jobs_processed"] += 1
            report["num_hole_jobs"] += int(job_type == "HOLE_FILLING")
            report["num_prefix_jobs"] += int(job_type == "PREFIX_COMPLETION")
            report["total_nodes_expanded"] += stats.nodes_expanded
            report["total_lean_checks"] += stats.lean_checks

            if cfg.outputs.write_search_traces:
                append_jsonl(paths["search_traces"], {
                    "job_id": job_id,
                    "job_type": job_type,
                    "stats": stats_to_record(stats),
                    "traces": traces,
                })

            if sols:
                report["num_jobs_solved"] += 1
                for sol in sols:
                    write_solution_outputs(sol, job, paths, cfg)
                    if job_type == "HOLE_FILLING":
                        report["num_hole_solutions"] += 1
                    else:
                        report["num_prefix_solutions"] += 1
            else:
                report["num_unresolved"] += 1
                write_unresolved(job, stats, paths)

        except Exception as e:
            key = type(e).__name__
            report["errors"][key] = report["errors"].get(key, 0) + 1
            append_jsonl(paths["unresolved"], {
                "job_id": job_id,
                "job_type": job_type,
                "reason": "SCRIPT_ERROR",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "source_job": job,
            })
            log(f"Job {job_id} failed with {key}: {e}", "error", cfg.experiment.log_level)

        if idx % cfg.experiment.save_every_jobs == 0:
            write_json(paths["report"], report)

    report["finished_at"] = now_ts()
    write_json(paths["report"], report)
    log(f"Finished. Report written to {paths['report']}", cfg_level=cfg.experiment.log_level)


if __name__ == "__main__":
    main()

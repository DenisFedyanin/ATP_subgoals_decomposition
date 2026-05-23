#!/usr/bin/env python3
"""
Собираем verified proof trajectories из результатов run_05_search_sorry_and_prefixes.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# =============================================================================
# Config
# =============================================================================


@dataclass
class ExperimentConfig:
    experiment_name: str = "assemble_curriculum_v1"
    seed: int = 42
    resume: bool = False
    dry_run: bool = False
    config_only: bool = False
    max_records: Optional[int] = None
    log_level: str = "info"


@dataclass
class InputConfig:
    run05_dir: str = "outputs/run_05_search_sorry_and_prefixes"
    solved_prefixes_path: Optional[str] = None
    solved_holes_path: Optional[str] = None
    run05_assembled_full_proofs_path: Optional[str] = None


@dataclass
class OutputConfig:
    out_dir: str = "outputs/run_06_assemble_curriculum"
    overwrite: bool = False
    final_trajectories_filename: str = "final_trajectories.jsonl"
    final_curriculum_items_filename: str = "final_curriculum_items.jsonl"
    prefix_trajectories_filename: str = "assembled_prefix_trajectories.jsonl"
    mcsp_trajectories_filename: str = "assembled_mcsp_trajectories.jsonl"
    partial_mcsp_filename: str = "partial_mcsp_sketches.jsonl"
    rejected_filename: str = "rejected_assemblies.jsonl"
    report_filename: str = "run_report.json"


@dataclass
class LeanConfig:
    project_root: str = "."
    use_lake_env: bool = True
    lean_cmd: str = "lean"
    timeout_sec: int = 20
    default_imports: List[str] = field(default_factory=lambda: ["Mathlib"])
    verify_assembled: bool = True
    keep_temp_files: bool = False
    verbose_errors: bool = False


@dataclass
class AssemblyConfig:
    require_contextual_hole_solution: bool = True
    accept_local_only_hole_solution: bool = False
    require_no_sorry_final: bool = True
    allow_partial_mcsp_output: bool = True
    include_run05_already_assembled: bool = True
    choose_one_solution_per_hole: bool = True
    max_solution_candidates_per_hole: int = 5
    deduplicate_final_code: bool = True
    normalize_whitespace_for_hash: bool = True
    reject_if_remaining_sorry_for_final: bool = True
    require_prefix_solution_status: List[str] = field(default_factory=lambda: ["SOLVED_PREFIX_SUFFIX", "SOLVED"])
    accepted_hole_statuses: List[str] = field(default_factory=lambda: ["SOLVED_CONTEXTUAL", "SOLVED_FULL_PARENT"])


@dataclass
class RunConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    lean: LeanConfig = field(default_factory=LeanConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)


# =============================================================================
# Basic utilities
# =============================================================================


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: str, prefix: str = "id") -> str:
    joined = "\n---\n".join(str(p) for p in parts)
    return f"{prefix}_{sha1_text(joined)[:16]}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {e}") from e


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=False) + "\n")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_code_fences(text: str) -> str:
    s = str(text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def line_count(text: str) -> int:
    return len([l for l in str(text or "").splitlines() if l.strip()])


def first_present(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def nested_first(d: Dict[str, Any], paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        cur: Any = d
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur or cur[key] is None:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return default


def has_sorry_or_admit(text: str) -> bool:
    return bool(re.search(r"\b(sorry|admit)\b", str(text or "")))


def indent_block(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + l if l.strip() else l for l in str(text or "").splitlines())


def indent_like_line(text: str, reference_line: str) -> str:
    m = re.match(r"^(\s*)", reference_line or "")
    return indent_block_to_column(text, len(m.group(1)) if m else 0)


def indent_block_to_column(text: str, col: int) -> str:
    pad = " " * col
    return "\n".join(pad + l if l.strip() else l for l in str(text or "").splitlines())


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def log(msg: str, level: str = "info", cfg_level: str = "info") -> None:
    order = {"debug": 0, "info": 1, "warning": 2, "error": 3}
    if order.get(level, 1) >= order.get(cfg_level, 1):
        print(f"[{level.upper()}] {msg}", file=sys.stderr)


# =============================================================================
# Config loading
# =============================================================================


def deep_update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    for k, v in values.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            deep_update_dataclass(cur, v)
        else:
            setattr(obj, k, v)
    return obj


def load_config(path: Optional[str]) -> RunConfig:
    cfg = RunConfig()
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML. Use JSON config or install PyYAML.") from e
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config must be a dict/object")
    return deep_update_dataclass(cfg, data)


# =============================================================================
# Lean verification
# =============================================================================


@dataclass
class LeanResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    elapsed_sec: float
    path: Optional[str] = None
    has_sorry_warning: bool = False
    error_summary: str = ""


class LeanVerifier:
    def __init__(self, cfg: LeanConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.project_root).resolve()

    def verify_code(self, code: str, allow_sorry: bool = False) -> LeanResult:
        if not allow_sorry and has_sorry_or_admit(code):
            return LeanResult(
                ok=False,
                stdout="",
                stderr="Rejected before Lean: assembled code contains sorry/admit.",
                returncode=-1,
                elapsed_sec=0.0,
                has_sorry_warning=True,
                error_summary="contains_sorry_or_admit",
            )
        start = time.time()
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as f:
                tmp_path = f.name
                f.write(code)
            cmd = [self.cfg.lean_cmd, tmp_path]
            if self.cfg.use_lake_env:
                cmd = ["lake", "env"] + cmd
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.cfg.timeout_sec,
            )
            elapsed = time.time() - start
            out = proc.stdout or ""
            err = proc.stderr or ""
            has_sorry = bool(re.search(r"declaration uses 'sorry'|warning:.*sorry|sorryAx", out + "\n" + err, flags=re.I))
            ok = proc.returncode == 0 and (allow_sorry or not has_sorry)
            return LeanResult(
                ok=ok,
                stdout=out,
                stderr=err,
                returncode=proc.returncode,
                elapsed_sec=elapsed,
                path=tmp_path if self.cfg.keep_temp_files else None,
                has_sorry_warning=has_sorry,
                error_summary=summarize_lean_error(out, err),
            )
        except subprocess.TimeoutExpired as e:
            return LeanResult(
                ok=False,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                returncode=-9,
                elapsed_sec=time.time() - start,
                path=tmp_path,
                error_summary="timeout",
            )
        finally:
            if tmp_path and not self.cfg.keep_temp_files:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def summarize_lean_error(stdout: str, stderr: str) -> str:
    text = (stderr or stdout or "").strip()
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    useful = [l for l in lines if "error:" in l.lower() or "warning:" in l.lower()]
    if useful:
        return useful[0][:300]
    return lines[0][:300] if lines else ""


# =============================================================================
# Lean code assembly helpers
# =============================================================================


THEOREM_DECL_RE = re.compile(r"^\s*(theorem|lemma|example)\b", re.M)


def extract_imports_from_record(record: Dict[str, Any], default_imports: Sequence[str]) -> List[str]:
    imports = first_present(record, ["imports", "import_lines", "header_imports"], None)
    if imports is None:
        source = record.get("source_job") if isinstance(record.get("source_job"), dict) else {}
        imports = first_present(source, ["imports", "import_lines", "header_imports"], None)
    if imports is None:
        return list(default_imports)
    if isinstance(imports, str):
        lines = [l.strip() for l in imports.splitlines() if l.strip()]
        if lines and all(l.startswith("import ") for l in lines):
            return [l[len("import "):].strip() for l in lines]
        return [x.strip() for x in re.split(r"[,\n]", imports) if x.strip() and not x.strip().startswith("--")]
    if isinstance(imports, list):
        out = []
        for x in imports:
            s = str(x).strip()
            if s.startswith("import "):
                s = s[len("import "):].strip()
            if s:
                out.append(s)
        return out or list(default_imports)
    return list(default_imports)


def imports_to_text(imports: Sequence[str]) -> str:
    return "\n".join(f"import {i}" if not str(i).strip().startswith("import ") else str(i).strip() for i in imports if str(i).strip())


def strip_existing_proof_from_decl(formal: str) -> str:
    """Return a theorem/lemma/example declaration header without an existing proof body.

    Handles common forms:
      theorem foo : P := by ...
      theorem foo : P := ...
      theorem foo : P
    """
    s = strip_code_fences(str(formal or "")).strip()
    if not s:
        return s
    # Remove imports accidentally included in formal statement.
    lines = [l for l in s.splitlines() if not l.strip().startswith("import ")]
    s = "\n".join(lines).strip()
    # Prefer cut at first ':=' that belongs to the declaration.
    idx = s.find(":=")
    if idx >= 0:
        return s[:idx].rstrip()
    # If it already ends with ':= by', normalize.
    return s.rstrip()


def proof_body_from_record_field(text: str) -> str:
    s = strip_code_fences(str(text or "")).strip()
    # If the field itself contains a theorem declaration, keep only body after ':= by' when possible.
    m = re.search(r":=\s*by\s*\n?(.*)$", s, flags=re.S)
    if m and THEOREM_DECL_RE.search(s[:m.start()]):
        return m.group(1).strip()
    # If it starts with 'by', strip that wrapper.
    s = re.sub(r"^by\s*", "", s).strip()
    return s


def build_theorem_code(imports: Sequence[str], formal_statement: str, proof_body: str) -> str:
    proof_body = proof_body_from_record_field(proof_body)
    # If proof_body accidentally is a full Lean file/declaration, return it with imports if needed.
    if THEOREM_DECL_RE.search(proof_body) and ":=" in proof_body:
        body_has_import = any(l.strip().startswith("import ") for l in proof_body.splitlines())
        return proof_body if body_has_import else imports_to_text(imports) + "\n\n" + proof_body
    header = strip_existing_proof_from_decl(formal_statement)
    if not header:
        # Last-resort: proof_body may already be full theorem code.
        return imports_to_text(imports) + "\n\n" + proof_body
    return imports_to_text(imports) + "\n\n" + header.rstrip() + " := by\n" + indent_block(proof_body, 2) + "\n"


def append_tactic_block(prefix: str, block: str) -> str:
    p = proof_body_from_record_field(prefix).rstrip()
    b = proof_body_from_record_field(block).strip()
    if not p:
        return b
    if not b:
        return p
    return p + "\n" + b


def extract_prefix_text(rec: Dict[str, Any]) -> str:
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    return str(first_present(rec, ["prefix_text", "prefix", "proof_prefix"], first_present(source, ["prefix_text", "prefix", "proof_prefix"], "")))


def extract_suffix_text(rec: Dict[str, Any]) -> str:
    return str(first_present(rec, ["suffix_text", "proof_fragment", "target_proof", "solution_proof"], ""))


def extract_formal_statement(rec: Dict[str, Any]) -> str:
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    return str(first_present(rec, ["formal_statement", "theorem", "statement"], first_present(source, ["formal_statement", "theorem", "statement"], "")))


def extract_parent_theorem_id(rec: Dict[str, Any]) -> str:
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    return str(first_present(rec, ["parent_theorem_id", "theorem_id", "id"], first_present(source, ["parent_theorem_id", "theorem_id", "id"], "")))


def extract_semi_proof_text(rec: Dict[str, Any]) -> str:
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    return str(first_present(rec, ["semi_proof_text", "parent_semi_proof", "sketch_text", "proof_with_sorry"], first_present(source, ["semi_proof_text", "parent_semi_proof", "sketch_text", "proof_with_sorry"], "")))


def extract_hole_id(rec: Dict[str, Any]) -> str:
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    hid = first_present(rec, ["hole_id", "target_hole_id"], first_present(source, ["hole_id", "target_hole_id"], None))
    if hid is not None:
        return str(hid)
    marker = first_present(source, ["target_hole_marker", "hole_marker"], None)
    idx = first_present(source, ["hole_index", "target_hole_index", "sorry_index"], None)
    if marker is not None:
        return "marker:" + sha1_text(str(marker))[:12]
    if idx is not None:
        return f"sorry_index:{idx}"
    return stable_id(json.dumps(source, sort_keys=True, ensure_ascii=False), prefix="hole")


def extract_hole_index(rec: Dict[str, Any]) -> Optional[int]:
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    idx = first_present(rec, ["hole_index", "target_hole_index", "sorry_index"], first_present(source, ["hole_index", "target_hole_index", "sorry_index"], None))
    if idx is None:
        return None
    try:
        return int(idx)
    except Exception:
        return None


def extract_hole_marker(rec: Dict[str, Any]) -> Optional[str]:
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    marker = first_present(rec, ["target_hole_marker", "hole_marker"], first_present(source, ["target_hole_marker", "hole_marker"], None))
    return str(marker) if marker else None


def insert_replacement_into_semi_proof(rec: Dict[str, Any], semi: str, replacement: str) -> Tuple[str, str]:
    """Insert one replacement into a semi-proof.

    Returns (new_text, method). Supports explicit markers, hole ids, and nth-sorry fallback.
    """
    replacement = proof_body_from_record_field(replacement)
    marker = extract_hole_marker(rec)
    if marker and marker in semi:
        return semi.replace(marker, replacement, 1), "marker"

    hole_id = extract_hole_id(rec)
    if hole_id:
        for pat in [f"{{{{HOLE:{hole_id}}}}}", f"{{{{{hole_id}}}}}", f"-- HOLE {hole_id}", f"/- HOLE {hole_id} -/"]:
            if pat in semi:
                return semi.replace(pat, replacement, 1), "hole_id_marker"

    idx = extract_hole_index(rec)
    if idx is None:
        idx = 0
    return replace_nth_sorry_line(semi, replacement, idx), f"nth_sorry:{idx}"


def replace_nth_sorry_line(text: str, replacement: str, n: int) -> str:
    lines = str(text or "").splitlines()
    seen = 0
    out: List[str] = []
    replaced = False
    for line in lines:
        if not replaced and re.match(r"^(\s*)sorry\b", line):
            if seen == n:
                out.append(indent_like_line(replacement, line))
                replaced = True
            else:
                out.append(line)
            seen += 1
        else:
            out.append(line)
    if replaced:
        return "\n".join(out)
    # Fallback: replace nth textual sorry occurrence, preserving no indentation.
    count = 0
    def repl(_m: re.Match[str]) -> str:
        nonlocal count
        if count == n:
            count += 1
            return replacement
        count += 1
        return _m.group(0)
    return re.sub(r"\bsorry\b", repl, str(text or ""))


def apply_multiple_replacements(semi: str, chosen: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply multiple hole replacements.

    Explicit markers can be applied in any order. Plain nth-sorry replacements must be applied
    in descending index order so earlier substitutions do not shift later `sorry` positions.
    """
    def order_key(rec: Dict[str, Any]) -> Tuple[int, int]:
        marker = extract_hole_marker(rec)
        if marker:
            return (0, 0)
        idx = extract_hole_index(rec)
        return (1, -(idx if idx is not None else 0))

    text = semi
    applied: List[Dict[str, Any]] = []
    for rec in sorted(chosen, key=order_key):
        before = text
        replacement = str(first_present(rec, ["replacement_proof", "proof_fragment", "target_proof", "solution_proof"], ""))
        text, method = insert_replacement_into_semi_proof(rec, text, replacement)
        applied.append({
            "solution_id": rec.get("solution_id"),
            "job_id": rec.get("job_id"),
            "hole_id": extract_hole_id(rec),
            "hole_index": extract_hole_index(rec),
            "method": method,
            "changed": before != text,
        })
    return text, applied


# =============================================================================
# Assembly data and scoring
# =============================================================================


@dataclass
class AssemblyResult:
    record: Dict[str, Any]
    accepted: bool
    reason: str
    lean_result: Optional[LeanResult] = None


def solution_score(rec: Dict[str, Any]) -> float:
    status = str(rec.get("status", ""))
    status_bonus = {
        "SOLVED_FULL_PARENT": 5.0,
        "SOLVED_CONTEXTUAL": 4.0,
        "SOLVED_PREFIX_SUFFIX": 4.0,
        "SOLVED": 3.0,
        "SOLVED_LOCAL_ONLY": 1.0,
    }.get(status, 0.0)
    meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    quality = safe_float(meta.get("quality"), 0.0)
    enc = safe_float(meta.get("encoder_score"), 0.0)
    depth = safe_float(rec.get("depth"), 999.0)
    final_ok = 1.0 if rec.get("final_ok") else 0.0
    contextual_ok = 1.0 if rec.get("contextual_ok") else 0.0
    return status_bonus + 2.0 * final_ok + 1.0 * contextual_ok + quality + 0.1 * enc - 0.03 * depth


def sketch_key_from_record(rec: Dict[str, Any]) -> str:
    semi = extract_semi_proof_text(rec)
    formal = extract_formal_statement(rec)
    pid = extract_parent_theorem_id(rec)
    source = rec.get("source_job") if isinstance(rec.get("source_job"), dict) else {}
    explicit = first_present(source, ["sketch_id", "semi_proof_id", "parent_sketch_id"], None)
    if explicit:
        return str(explicit)
    return stable_id(pid, formal, semi, prefix="sketch")


def is_acceptable_hole_solution(rec: Dict[str, Any], cfg: AssemblyConfig) -> bool:
    status = str(rec.get("status", ""))
    if status == "SOLVED_LOCAL_ONLY" and not cfg.accept_local_only_hole_solution:
        return False
    if cfg.require_contextual_hole_solution and not (rec.get("contextual_ok") or rec.get("final_ok")):
        return False
    if cfg.accepted_hole_statuses and status not in cfg.accepted_hole_statuses:
        if not (status == "SOLVED_LOCAL_ONLY" and cfg.accept_local_only_hole_solution):
            return False
    repl = str(first_present(rec, ["replacement_proof", "proof_fragment", "target_proof", "solution_proof"], ""))
    if cfg.require_no_sorry_final and has_sorry_or_admit(repl):
        return False
    return bool(repl.strip())


def is_acceptable_prefix_solution(rec: Dict[str, Any], cfg: AssemblyConfig) -> bool:
    status = str(rec.get("status", ""))
    if cfg.require_prefix_solution_status and status not in cfg.require_prefix_solution_status:
        # Some older runs may use final_ok without status normalization.
        if not rec.get("final_ok"):
            return False
    suffix = extract_suffix_text(rec)
    if cfg.require_no_sorry_final and has_sorry_or_admit(suffix):
        return False
    return bool(suffix.strip())


# =============================================================================
# Assembler
# =============================================================================


class CurriculumAssembler:
    def __init__(self, cfg: RunConfig, verifier: Optional[LeanVerifier], paths: Dict[str, Path]):
        self.cfg = cfg
        self.verifier = verifier
        self.paths = paths
        self.seen_final_hashes: set[str] = set()
        self.report: Dict[str, Any] = {
            "experiment_name": cfg.experiment.experiment_name,
            "started_at": now_ts(),
            "num_prefix_records_read": 0,
            "num_hole_records_read": 0,
            "num_prefix_trajectories_accepted": 0,
            "num_mcsp_trajectories_accepted": 0,
            "num_partial_mcsp_sketches": 0,
            "num_rejected": 0,
            "num_lean_checks": 0,
            "num_lean_failures": 0,
            "num_duplicates": 0,
            "errors": {},
        }

    def assemble_prefix_records(self, records: Iterable[Dict[str, Any]]) -> None:
        for rec in records:
            self.report["num_prefix_records_read"] += 1
            if self.cfg.experiment.max_records and self.report["num_prefix_records_read"] > self.cfg.experiment.max_records:
                break
            try:
                result = self.assemble_one_prefix(rec)
                self.write_result(result)
            except Exception as e:
                self.report["errors"][type(e).__name__] = self.report["errors"].get(type(e).__name__, 0) + 1
                self.reject(rec, "SCRIPT_ERROR", error=repr(e), traceback_text=traceback.format_exc())

    def assemble_one_prefix(self, rec: Dict[str, Any]) -> AssemblyResult:
        if not is_acceptable_prefix_solution(rec, self.cfg.assembly):
            return AssemblyResult(rec, False, "prefix_solution_not_acceptable")
        formal = extract_formal_statement(rec)
        prefix = extract_prefix_text(rec)
        suffix = extract_suffix_text(rec)
        if not formal.strip():
            return AssemblyResult(rec, False, "missing_formal_statement")
        if not prefix.strip() and not suffix.strip():
            return AssemblyResult(rec, False, "missing_prefix_and_suffix")
        proof_body = append_tactic_block(prefix, suffix)
        imports = extract_imports_from_record(rec, self.cfg.lean.default_imports)
        full_code = build_theorem_code(imports, formal, proof_body)
        traj_id = stable_id("prefix", extract_parent_theorem_id(rec), full_code, prefix="traj")
        assembled = {
            "trajectory_id": traj_id,
            "trajectory_type": "PREFIX_SUFFIX",
            "parent_theorem_id": extract_parent_theorem_id(rec),
            "formal_statement": formal,
            "imports": imports,
            "proof_body": proof_body,
            "prefix_text": proof_body_from_record_field(prefix),
            "suffix_text": proof_body_from_record_field(suffix),
            "full_theorem_code": full_code,
            "verified": False,
            "uses_sorry": has_sorry_or_admit(full_code),
            "source_solution": rec,
            "metadata": {
                "source_job_id": rec.get("job_id"),
                "source_solution_id": rec.get("solution_id"),
                "assembly_stage": "run_06",
            },
        }
        return self.verify_and_accept(assembled, "prefix")

    def assemble_hole_records(self, records: Iterable[Dict[str, Any]]) -> None:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            self.report["num_hole_records_read"] += 1
            if not is_acceptable_hole_solution(rec, self.cfg.assembly):
                self.reject(rec, "hole_solution_not_acceptable")
                continue
            key = sketch_key_from_record(rec)
            groups.setdefault(key, []).append(rec)

        for key, group in groups.items():
            try:
                for result in self.assemble_one_mcsp_group(key, group):
                    self.write_result(result)
            except Exception as e:
                self.report["errors"][type(e).__name__] = self.report["errors"].get(type(e).__name__, 0) + 1
                self.reject({"sketch_key": key, "records": group[:3]}, "SCRIPT_ERROR", error=repr(e), traceback_text=traceback.format_exc())

    def assemble_one_mcsp_group(self, sketch_key: str, group: List[Dict[str, Any]]) -> List[AssemblyResult]:
        if not group:
            return []
        base = group[0]
        semi = extract_semi_proof_text(base)
        formal = extract_formal_statement(base)
        parent_id = extract_parent_theorem_id(base)
        if not semi.strip():
            return [AssemblyResult({"sketch_key": sketch_key, "records": group}, False, "missing_semi_proof_text")]
        if not formal.strip():
            return [AssemblyResult({"sketch_key": sketch_key, "records": group}, False, "missing_formal_statement")]

        # Group candidate replacements by hole id/index and choose the best per hole.
        by_hole: Dict[str, List[Dict[str, Any]]] = {}
        for rec in group:
            by_hole.setdefault(extract_hole_id(rec), []).append(rec)
        chosen: List[Dict[str, Any]] = []
        hole_candidates_summary: Dict[str, Any] = {}
        for hid, cands in by_hole.items():
            cands_sorted = sorted(cands, key=solution_score, reverse=True)[: self.cfg.assembly.max_solution_candidates_per_hole]
            if self.cfg.assembly.choose_one_solution_per_hole:
                chosen.append(cands_sorted[0])
            else:
                # MVP: still assemble the best combination only; full combinatorial assembly can be added later.
                chosen.append(cands_sorted[0])
            hole_candidates_summary[hid] = [
                {
                    "solution_id": c.get("solution_id"),
                    "job_id": c.get("job_id"),
                    "status": c.get("status"),
                    "score": solution_score(c),
                    "depth": c.get("depth"),
                }
                for c in cands_sorted
            ]

        filled_body, applied = apply_multiple_replacements(semi, chosen)
        remaining_sorry = has_sorry_or_admit(filled_body)
        imports = extract_imports_from_record(base, self.cfg.lean.default_imports)
        full_code = build_theorem_code(imports, formal, filled_body)
        traj_id = stable_id("mcsp", parent_id, full_code, prefix="traj")
        assembled = {
            "trajectory_id": traj_id,
            "trajectory_type": "MCSP_FILLED_SKETCH",
            "parent_theorem_id": parent_id,
            "sketch_key": sketch_key,
            "formal_statement": formal,
            "imports": imports,
            "original_semi_proof_text": semi,
            "filled_proof_body": filled_body,
            "proof_body": filled_body,
            "full_theorem_code": full_code,
            "verified": False,
            "uses_sorry": has_sorry_or_admit(full_code),
            "holes_filled": [extract_hole_id(c) for c in chosen],
            "num_holes_filled": len(chosen),
            "remaining_sorry": remaining_sorry,
            "applied_replacements": applied,
            "hole_candidates_summary": hole_candidates_summary,
            "source_solutions": chosen,
            "metadata": {
                "assembly_stage": "run_06",
                "num_solution_records_in_group": len(group),
            },
        }

        if remaining_sorry and self.cfg.assembly.allow_partial_mcsp_output:
            partial = dict(assembled)
            partial["trajectory_type"] = "PARTIAL_MCSP_FILLED_SKETCH"
            partial["accepted_for_training"] = False
            partial["reason"] = "remaining_sorry_after_available_replacements"
            return [AssemblyResult(partial, False, "partial_mcsp_remaining_sorry")]

        return [self.verify_and_accept(assembled, "mcsp")]

    def include_run05_already_assembled(self, records: Iterable[Dict[str, Any]]) -> None:
        if not self.cfg.assembly.include_run05_already_assembled:
            return
        for rec in records:
            try:
                full_code = str(first_present(rec, ["full_theorem_code", "assembled_full_code"], ""))
                formal = extract_formal_statement(rec)
                imports = extract_imports_from_record(rec, self.cfg.lean.default_imports)
                body = str(first_present(rec, ["proof_body", "filled_proof_body", "proof_fragment", "target_proof"], ""))
                if not full_code and formal and body:
                    full_code = build_theorem_code(imports, formal, body)
                if not full_code.strip():
                    self.reject(rec, "run05_assembled_missing_code")
                    continue
                traj = {
                    "trajectory_id": stable_id("run05_full", full_code, prefix="traj"),
                    "trajectory_type": "RUN05_ALREADY_ASSEMBLED_FULL_PROOF",
                    "parent_theorem_id": extract_parent_theorem_id(rec),
                    "formal_statement": formal,
                    "imports": imports,
                    "proof_body": body,
                    "full_theorem_code": full_code,
                    "verified": False,
                    "uses_sorry": has_sorry_or_admit(full_code),
                    "source_solution": rec,
                    "metadata": {"assembly_stage": "run_06", "source": "run05_assembled_full_proofs"},
                }
                result = self.verify_and_accept(traj, "mcsp")
                self.write_result(result)
            except Exception as e:
                self.reject(rec, "SCRIPT_ERROR", error=repr(e), traceback_text=traceback.format_exc())

    def verify_and_accept(self, assembled: Dict[str, Any], kind: str) -> AssemblyResult:
        if self.cfg.assembly.require_no_sorry_final and has_sorry_or_admit(str(assembled.get("full_theorem_code", ""))):
            return AssemblyResult(assembled, False, "assembled_code_contains_sorry")
        code_hash = sha1_text(normalize_ws(str(assembled.get("full_theorem_code", ""))) if self.cfg.assembly.normalize_whitespace_for_hash else str(assembled.get("full_theorem_code", "")))
        assembled["code_hash"] = code_hash
        if self.cfg.assembly.deduplicate_final_code and code_hash in self.seen_final_hashes:
            self.report["num_duplicates"] += 1
            return AssemblyResult(assembled, False, "duplicate_final_code")

        lean_res: Optional[LeanResult] = None
        if self.cfg.lean.verify_assembled and not self.cfg.experiment.dry_run:
            if self.verifier is None:
                raise RuntimeError("Lean verifier missing but verify_assembled=True")
            lean_res = self.verifier.verify_code(str(assembled.get("full_theorem_code", "")), allow_sorry=False)
            self.report["num_lean_checks"] += 1
            assembled["lean_verification"] = dataclasses.asdict(lean_res)
            if not lean_res.ok:
                self.report["num_lean_failures"] += 1
                return AssemblyResult(assembled, False, "lean_verification_failed", lean_res)
        else:
            assembled["lean_verification"] = {"skipped": True, "dry_run": self.cfg.experiment.dry_run}

        assembled["verified"] = True
        assembled["accepted_for_training"] = True
        self.seen_final_hashes.add(code_hash)
        return AssemblyResult(assembled, True, "accepted", lean_res)

    def write_result(self, result: AssemblyResult) -> None:
        rec = result.record
        rec["assembly_status"] = result.reason
        if result.accepted:
            traj_type = str(rec.get("trajectory_type", ""))
            append_jsonl(self.paths["final_trajectories"], rec)
            append_jsonl(self.paths["final_curriculum_items"], curriculum_item_from_trajectory(rec))
            if traj_type == "PREFIX_SUFFIX":
                append_jsonl(self.paths["prefix_trajectories"], rec)
                self.report["num_prefix_trajectories_accepted"] += 1
            else:
                append_jsonl(self.paths["mcsp_trajectories"], rec)
                self.report["num_mcsp_trajectories_accepted"] += 1
        else:
            if result.reason == "partial_mcsp_remaining_sorry":
                append_jsonl(self.paths["partial_mcsp"], rec)
                self.report["num_partial_mcsp_sketches"] += 1
            else:
                self.reject(rec, result.reason, lean_result=result.lean_result)

    def reject(self, rec: Dict[str, Any], reason: str, error: Optional[str] = None, traceback_text: Optional[str] = None, lean_result: Optional[LeanResult] = None) -> None:
        out = {
            "reason": reason,
            "record": rec,
        }
        if error:
            out["error"] = error
        if traceback_text:
            out["traceback"] = traceback_text
        if lean_result is not None:
            out["lean_result"] = dataclasses.asdict(lean_result)
        append_jsonl(self.paths["rejected"], out)
        self.report["num_rejected"] += 1


def curriculum_item_from_trajectory(traj: Dict[str, Any]) -> Dict[str, Any]:
    ttype = str(traj.get("trajectory_type", ""))
    if ttype == "PREFIX_SUFFIX":
        item_type = "assembled_prefix_suffix_trajectory"
        input_context = {
            "formal_statement": traj.get("formal_statement"),
            "prefix_text": traj.get("prefix_text"),
        }
        target = traj.get("suffix_text")
    else:
        item_type = "assembled_mcsp_full_trajectory"
        input_context = {
            "formal_statement": traj.get("formal_statement"),
            "original_semi_proof_text": traj.get("original_semi_proof_text"),
            "holes_filled": traj.get("holes_filled"),
        }
        target = traj.get("proof_body")
    return {
        "item_id": stable_id(str(traj.get("trajectory_id")), item_type, prefix="curriculum"),
        "item_type": item_type,
        "trajectory_id": traj.get("trajectory_id"),
        "parent_theorem_id": traj.get("parent_theorem_id"),
        "formal_statement": traj.get("formal_statement"),
        "input_context": input_context,
        "target_proof": target,
        "full_theorem_code": traj.get("full_theorem_code"),
        "verified": bool(traj.get("verified")),
        "uses_sorry": bool(traj.get("uses_sorry")),
        "accepted_for_training": bool(traj.get("accepted_for_training")),
        "metadata": traj.get("metadata", {}),
    }


# =============================================================================
# I/O setup and main
# =============================================================================


def default_input_paths(cfg: RunConfig) -> Tuple[Path, Path, Path]:
    run05 = Path(cfg.inputs.run05_dir)
    solved_prefixes = Path(cfg.inputs.solved_prefixes_path) if cfg.inputs.solved_prefixes_path else run05 / "solved_prefix_suffixes.jsonl"
    solved_holes = Path(cfg.inputs.solved_holes_path) if cfg.inputs.solved_holes_path else run05 / "solved_hole_replacements.jsonl"
    assembled = Path(cfg.inputs.run05_assembled_full_proofs_path) if cfg.inputs.run05_assembled_full_proofs_path else run05 / "assembled_full_proofs.jsonl"
    return solved_prefixes, solved_holes, assembled


def prepare_outputs(cfg: RunConfig) -> Dict[str, Path]:
    out_dir = ensure_dir(cfg.outputs.out_dir)
    paths = {
        "final_trajectories": out_dir / cfg.outputs.final_trajectories_filename,
        "final_curriculum_items": out_dir / cfg.outputs.final_curriculum_items_filename,
        "prefix_trajectories": out_dir / cfg.outputs.prefix_trajectories_filename,
        "mcsp_trajectories": out_dir / cfg.outputs.mcsp_trajectories_filename,
        "partial_mcsp": out_dir / cfg.outputs.partial_mcsp_filename,
        "rejected": out_dir / cfg.outputs.rejected_filename,
        "report": out_dir / cfg.outputs.report_filename,
    }
    if cfg.outputs.overwrite:
        for p in paths.values():
            if p.exists() and p.is_file():
                p.unlink()
    return paths


def count_jsonl(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def config_summary(cfg: RunConfig) -> Dict[str, Any]:
    solved_prefixes_path, solved_holes_path, run05_assembled_path = default_input_paths(cfg)
    warnings: List[str] = []
    if not solved_prefixes_path.exists():
        warnings.append(f"missing_solved_prefixes: {solved_prefixes_path}")
    if not solved_holes_path.exists():
        warnings.append(f"missing_solved_holes: {solved_holes_path}")
    if not run05_assembled_path.exists():
        warnings.append(f"missing_run05_assembled_full_proofs: {run05_assembled_path}")
    if not any(p.exists() for p in [solved_prefixes_path, solved_holes_path, run05_assembled_path]):
        warnings.append("no_run05_outputs_found")
    return {
        "experiment_name": cfg.experiment.experiment_name,
        "created_at": now_ts(),
        "inputs": {
            "run05_dir": cfg.inputs.run05_dir,
            "solved_prefixes_path": str(solved_prefixes_path),
            "solved_prefixes_exists": solved_prefixes_path.exists(),
            "solved_prefixes_count": count_jsonl(solved_prefixes_path),
            "solved_holes_path": str(solved_holes_path),
            "solved_holes_exists": solved_holes_path.exists(),
            "solved_holes_count": count_jsonl(solved_holes_path),
            "run05_assembled_path": str(run05_assembled_path),
            "run05_assembled_exists": run05_assembled_path.exists(),
            "run05_assembled_count": count_jsonl(run05_assembled_path),
        },
        "assembly": dataclasses.asdict(cfg.assembly),
        "lean": dataclasses.asdict(cfg.lean),
        "warnings": warnings,
        "config": dataclasses.asdict(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run05-dir", type=str, default=None)
    parser.add_argument("--solved-prefixes", type=str, default=None)
    parser.add_argument("--solved-holes", type=str, default=None)
    parser.add_argument("--run05-assembled", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--lean-project-root", type=str, default=None)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--accept-local-only-holes", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run05_dir:
        cfg.inputs.run05_dir = args.run05_dir
    if args.solved_prefixes:
        cfg.inputs.solved_prefixes_path = args.solved_prefixes
    if args.solved_holes:
        cfg.inputs.solved_holes_path = args.solved_holes
    if args.run05_assembled:
        cfg.inputs.run05_assembled_full_proofs_path = args.run05_assembled
    if args.out_dir:
        cfg.outputs.out_dir = args.out_dir
    if args.lean_project_root:
        cfg.lean.project_root = args.lean_project_root
    if args.no_verify:
        cfg.lean.verify_assembled = False
    if args.dry_run:
        cfg.experiment.dry_run = True
        cfg.lean.verify_assembled = False
    if args.config_only:
        cfg.experiment.config_only = True
    if args.overwrite:
        cfg.outputs.overwrite = True
    if args.max_records is not None:
        cfg.experiment.max_records = args.max_records
    if args.accept_local_only_holes:
        cfg.assembly.accept_local_only_hole_solution = True
        cfg.assembly.require_contextual_hole_solution = False
        if "SOLVED_LOCAL_ONLY" not in cfg.assembly.accepted_hole_statuses:
            cfg.assembly.accepted_hole_statuses.append("SOLVED_LOCAL_ONLY")

    solved_prefixes_path, solved_holes_path, run05_assembled_path = default_input_paths(cfg)
    if cfg.experiment.config_only:
        out_dir = ensure_dir(cfg.outputs.out_dir)
        summary = config_summary(cfg)
        summary_path = out_dir / "config_only_summary.json"
        write_json(summary_path, summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))
        log(f"Config-only summary written to {summary_path}", cfg_level=cfg.experiment.log_level)
        return
    if not any(p.exists() for p in [solved_prefixes_path, solved_holes_path, run05_assembled_path]):
        raise RuntimeError(
            "No run_05 outputs found for run_06 assembly. Run run_05_search_sorry_and_prefixes.py first, "
            "or pass --solved-prefixes/--solved-holes/--run05-assembled."
        )

    paths = prepare_outputs(cfg)
    verifier = LeanVerifier(cfg.lean) if cfg.lean.verify_assembled and not cfg.experiment.dry_run else None
    assembler = CurriculumAssembler(cfg, verifier, paths)

    log(f"Reading prefix solutions from {solved_prefixes_path}", cfg_level=cfg.experiment.log_level)
    assembler.assemble_prefix_records(read_jsonl(solved_prefixes_path))

    log(f"Reading hole replacements from {solved_holes_path}", cfg_level=cfg.experiment.log_level)
    assembler.assemble_hole_records(read_jsonl(solved_holes_path))

    if cfg.assembly.include_run05_already_assembled and run05_assembled_path.exists():
        log(f"Reading run05 already assembled proofs from {run05_assembled_path}", cfg_level=cfg.experiment.log_level)
        assembler.include_run05_already_assembled(read_jsonl(run05_assembled_path))

    assembler.report["finished_at"] = now_ts()
    assembler.report["inputs"] = {
        "solved_prefixes_path": str(solved_prefixes_path),
        "solved_prefixes_count": count_jsonl(solved_prefixes_path),
        "solved_holes_path": str(solved_holes_path),
        "solved_holes_count": count_jsonl(solved_holes_path),
        "run05_assembled_path": str(run05_assembled_path),
        "run05_assembled_count": count_jsonl(run05_assembled_path),
    }
    assembler.report["outputs"] = {k: str(v) for k, v in paths.items()}
    write_json(paths["report"], assembler.report)
    log(f"Finished assembly. Report written to {paths['report']}", cfg_level=cfg.experiment.log_level)


if __name__ == "__main__":
    main()

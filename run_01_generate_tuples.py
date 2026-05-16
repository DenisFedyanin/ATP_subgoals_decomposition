# run_01_generate_tuples.py

import os
import re
import gc
import sys
import json
import time
import random
import hashlib
import argparse
import datetime
import traceback
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import deque, Counter


CONFIG = {
    "run_name": "tuple_generation_v1_full",
    "seed": 17,

    "model": {
        "name": "ByteDance-Seed/BFS-Prover-V1-7B",
        "prompt_format": "{state_pp}:::",
        "max_new_tokens": 64,
        "precision": "bf16",
        "device_map": "auto",
        "trust_remote_code": True,
        "compute_tactic_scores": False,
    },

    "generation": {
        "passes": [
            {"name": "precision", "n": 8, "temperature": 0.4, "top_p": 0.90},
            {"name": "diversity", "n": 8, "temperature": 0.8, "top_p": 0.95},
        ],
        "deduplicate_tactics": True,
        "execute_all_unique_candidates": True,
        "fallback_if_unique_less_than": 4,
        "fallback": {"name": "fallback", "n": 8, "temperature": 1.0, "top_p": 0.98},
        "max_tactic_chars": 300,
    },

    "lean": {
        "backend": "LeanDojo",
        "tactic_timeout_sec": 5,
        "theorem_timeout_sec": 120,
        "memory_limit": "16g",
        "cpu_limit": "1",
        "load_used_packages_only": "1",
        "cache_dir": "lean_repos/.cache/lean_dojo",
        "tmp_dir": "outputs/tmp",
        "trace_repos_before_use": True,
    },

    "dataset_sources": {
        "synthetic_small_local": {
            "enabled": True,
            "weight": 0.10,
            "repo_dir": "lean_repos/synthetic_small_local",
            "lean_toolchain": "leanprover/lean4:v4.8.0-rc1",
            "mathlib_ref": "v4.8.0-rc1",
        },
        "lean_workbook_local": {
            "enabled": False,
            "weight": 0.45,
            "repo_dir": "lean_repos/lean_workbook_local",
            "dataset_name": "internlm/Lean-Workbook",
            "split": "train",
            "max_rows_to_materialize": 5000,
            "lean_toolchain": "leanprover/lean4:v4.8.0-rc1",
            "mathlib_ref": "v4.8.0-rc1",
        },
        "external_manifests": [
            {
                "enabled": False,
                "dataset_name": "mathlib_leandojo",
                "weight": 0.25,
                "path": "data/theorem_sets/mathlib_manifest.jsonl",
            },
            {
                "enabled": False,
                "dataset_name": "lean_github",
                "weight": 0.20,
                "path": "data/theorem_sets/lean_github_manifest.jsonl",
            },
        ],
    },

    "tuple_search": {
        "search_style": "bounded_bfs",
        "max_depth_per_theorem": 8,
        "max_states_expanded_per_theorem": 64,
        "max_attempt_tuples_per_theorem": 512,
        "max_children_enqueued_per_state": 4,
        "stop_policy": "finish_current_state_candidates_then_stop_if_proof_closed",
    },

    "dataset_size": {
        "target_total_attempt_tuples": 1_000_000,
        "target_valid_tuples_min": 300_000,
        "target_unique_states_min": 75_000,
        "target_unique_theorems_min": 5_000,
    },

    "filtering": {
        "drop_empty_tactic": True,
        "drop_non_tactic_text": True,
        "states_longer_than_chars": 12000,
        "dedup_same_state_tactic": True,
        "dedup_exact_tuple": True,
    },

    "output": {
        "root_dir": "data/tuples/full_v1",
        "primary_format": "parquet",
        "secondary_format": "jsonl",
        "shard_size_tuples": 50_000,
        "write_jsonl": True,
        "write_parquet": True,
        "metadata_file": "run_metadata.json",
        "stats_file": "stats.json",
        "theorem_index_file": "theorem_index.jsonl",
    },
}


@dataclass
class TheoremSpec:
    dataset_name: str
    source_split: str
    theorem_id: str
    theorem_name: str
    file_path: str
    repo_path: str
    repo_url: str = ""
    repo_commit: str = ""
    source_weight: float = 1.0
    original_name: str = ""


@dataclass
class StateItem:
    state: Any
    state_id: Optional[int]
    parent_state_id: Optional[int]
    depth: int
    proof_prefix: List[str]


def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha1_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cmd(cmd: str, cwd: Optional[str | Path] = None, check: bool = True) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{p.stdout}")
    return p.stdout


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


def set_leandojo_env() -> None:
    lean_cfg = CONFIG["lean"]
    os.environ["CACHE_DIR"] = str(Path(lean_cfg["cache_dir"]).resolve())
    os.environ["TMP_DIR"] = str(Path(lean_cfg["tmp_dir"]).resolve())
    os.environ["TACTIC_MEMORY_LIMIT"] = str(lean_cfg["memory_limit"])
    os.environ["TACTIC_CPU_LIMIT"] = str(lean_cfg["cpu_limit"])
    os.environ["LOAD_USED_PACKAGES_ONLY"] = str(lean_cfg["load_used_packages_only"])

    ensure_dir(os.environ["CACHE_DIR"])
    ensure_dir(os.environ["TMP_DIR"])


def check_cuda_or_fail() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script expects an A100/CUDA VM.")

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[cuda] GPU={name}, capability={cap}, torch={torch.__version__}")

    if cap[0] < 8:
        print("[warning] This script is configured for A100-class GPUs. It may still run, but BF16/throughput may be worse.")


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def get_git_commit(path: str | Path) -> str:
    try:
        return run_cmd("git rev-parse HEAD", cwd=path, check=True).strip()
    except Exception:
        return ""


def init_git_repo(path: str | Path) -> str:
    path = Path(path)
    run_cmd("git init", cwd=path, check=True)
    run_cmd('git config user.email "tuple-pilot@example.com"', cwd=path, check=True)
    run_cmd('git config user.name "Tuple Pilot"', cwd=path, check=True)

    write_text(
        path / ".gitignore",
        ".lake/\n"
        "state.db\n"
        "*.olean\n"
        "*.ilean\n"
        "*.trace\n"
        "__pycache__/\n"
        ".cache/\n",
    )

    run_cmd("git add .", cwd=path, check=True)
    run_cmd('git commit -m "init repo" || true', cwd=path, check=False)
    return get_git_commit(path)


def sanitize_lean_identifier(x: str, fallback: str) -> str:
    if not x:
        x = fallback
    x = re.sub(r"[^A-Za-z0-9_']", "_", x)
    x = re.sub(r"_+", "_", x)
    if not re.match(r"^[A-Za-z_]", x):
        x = "_" + x
    return x


def extract_theorem_name(stmt: str) -> Optional[str]:
    m = re.search(r"\btheorem\s+([A-Za-z0-9_'.]+)", stmt)
    if m:
        return m.group(1)

    m = re.search(r"\blemma\s+([A-Za-z0-9_'.]+)", stmt)
    if m:
        return m.group(1)

    return None


def force_theorem_by_sorry(stmt: str, new_name: str) -> Optional[str]:
    s = stmt.strip()
    s = re.sub(r"--.*", "", s)
    s = s.replace("admit", "sorry")

    if not re.search(r"\b(theorem|lemma)\s+", s):
        return None

    s = re.sub(r"\b(theorem|lemma)\s+[A-Za-z0-9_'.]+", f"theorem {new_name}", s, count=1)

    if ":= by" in s:
        head = s.split(":= by", 1)[0].strip()
        return head + " := by\n  sorry\n"

    if ":=" in s:
        head = s.split(":=", 1)[0].strip()
        return head + " := by\n  sorry\n"

    return s + " := by\n  sorry\n"


def tactic_head(tactic: str) -> str:
    t = tactic.strip()
    if not t:
        return ""

    t = t.replace("·", "").strip()
    m = re.match(r"([A-Za-z_][A-Za-z0-9_']*)", t)
    return m.group(1) if m else t.split()[0]


def clean_tactic(raw: str) -> str:
    s = raw.strip()
    s = s.replace("\r", "\n")
    s = s.split("\n")[0].strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("` ")

    if s.startswith("by "):
        s = s[3:].strip()

    return s


def looks_like_tactic(tactic: str) -> bool:
    if not tactic.strip():
        return False

    bad = [
        "Here is",
        "The tactic",
        "I will",
        "```",
        "import ",
        "theorem ",
        "lemma ",
        "sorry",
        "by sorry",
    ]
    return not any(b.lower() in tactic.lower() for b in bad)


def classify_error(msg: Optional[str]) -> str:
    if not msg:
        return "unknown"

    s = str(msg).lower()

    if "timeout" in s or "timed out" in s:
        return "timeout"
    if "unknown identifier" in s:
        return "unknown_identifier"
    if "unknown constant" in s:
        return "unknown_constant"
    if "type mismatch" in s:
        return "type_mismatch"
    if "application type mismatch" in s:
        return "application_type_mismatch"
    if "unsolved goals" in s:
        return "unsolved_goals"
    if "no goals to be solved" in s:
        return "no_goals"
    if "failed to synthesize" in s:
        return "failed_to_synthesize"
    if "unexpected token" in s:
        return "syntax_error"
    if "invalid" in s:
        return "invalid_syntax_or_command"
    if "maximum recursion" in s or "maximum heartbeats" in s or "resource" in s:
        return "resource_limit"
    if "declaration uses 'sorry'" in s:
        return "sorry_warning"

    return "lean_error"


def count_goals_from_pp(pp: Optional[str]) -> int:
    if not pp:
        return 0

    n = pp.count("⊢")
    return max(1, n) if pp.strip() else 0


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


class BFSProverGenerator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_name = cfg["name"]
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        dtype = torch.bfloat16 if self.cfg["precision"] == "bf16" else torch.float16

        print(f"[model] loading {self.model_name} with dtype={dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.cfg.get("trust_remote_code", True),
            token=os.environ.get("HF_TOKEN"),
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.cfg.get("device_map", "auto"),
            torch_dtype=dtype,
            trust_remote_code=self.cfg.get("trust_remote_code", True),
            low_cpu_mem_usage=True,
            token=os.environ.get("HF_TOKEN"),
        )

        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.model.device
        print("[model] loaded")

    def _decode_tactic(self, prompt: str, decoded: str) -> str:
        if ":::" in decoded:
            after = decoded.split(":::", 1)[1]
        elif decoded.startswith(prompt):
            after = decoded[len(prompt):]
        else:
            after = decoded

        return clean_tactic(after)

    def generate(self, state_pp: str) -> Tuple[List[Dict[str, Any]], str]:
        prompt = self.cfg["prompt_format"].format(state_pp=state_pp.strip())
        all_items = []
        seen = set()

        for gen_pass in CONFIG["generation"]["passes"]:
            items = self._generate_pass(prompt, gen_pass)
            for item in items:
                tactic = item["tactic"]
                if not tactic:
                    continue
                if CONFIG["generation"]["deduplicate_tactics"] and tactic in seen:
                    continue
                seen.add(tactic)
                all_items.append(item)

        threshold = CONFIG["generation"]["fallback_if_unique_less_than"]
        if len(all_items) < threshold:
            items = self._generate_pass(prompt, CONFIG["generation"]["fallback"])
            for item in items:
                tactic = item["tactic"]
                if not tactic:
                    continue
                if CONFIG["generation"]["deduplicate_tactics"] and tactic in seen:
                    continue
                seen.add(tactic)
                all_items.append(item)

        for i, item in enumerate(all_items):
            item["tactic_rank"] = i

        if self.cfg.get("compute_tactic_scores", False):
            for item in all_items:
                item["tactic_score"] = self.score_tactic(prompt, item["tactic"])

        return all_items, prompt

    def _generate_pass(self, prompt: str, gen_pass: Dict[str, Any]) -> List[Dict[str, Any]]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=gen_pass["n"],
                max_new_tokens=self.cfg["max_new_tokens"],
                temperature=gen_pass["temperature"],
                top_p=gen_pass["top_p"],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        items = []

        for x in decoded:
            tactic = self._decode_tactic(prompt, x)

            if len(tactic) > CONFIG["generation"]["max_tactic_chars"]:
                continue
            if CONFIG["filtering"]["drop_empty_tactic"] and not tactic:
                continue
            if CONFIG["filtering"]["drop_non_tactic_text"] and not looks_like_tactic(tactic):
                continue

            items.append({
                "tactic": tactic,
                "tactic_head": tactic_head(tactic),
                "generation_pass": gen_pass["name"],
                "generation_params": {
                    "temperature": gen_pass["temperature"],
                    "top_p": gen_pass["top_p"],
                    "max_new_tokens": self.cfg["max_new_tokens"],
                },
                "tactic_score": None,
            })

        return items

    def score_tactic(self, prompt: str, tactic: str) -> Optional[float]:
        import torch

        text = prompt + tactic
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        start = max(prompt_len - 1, 0)
        vals = token_lp[:, start:]

        if vals.numel() == 0:
            return None

        return float(vals.mean().detach().cpu())


class LeanRunner:
    def __init__(self):
        set_leandojo_env()

        from lean_dojo import LeanGitRepo, Theorem, Dojo, trace
        from lean_dojo.interaction.dojo import (
            TacticState,
            ProofFinished,
            LeanError,
            ProofGivenUp,
            DojoCrashError,
            DojoInitError,
            DojoTacticTimeoutError,
        )

        self.LeanGitRepo = LeanGitRepo
        self.Theorem = Theorem
        self.Dojo = Dojo
        self.trace = trace

        self.TacticState = TacticState
        self.ProofFinished = ProofFinished
        self.LeanError = LeanError
        self.ProofGivenUp = ProofGivenUp
        self.DojoCrashError = DojoCrashError
        self.DojoInitError = DojoInitError
        self.DojoTacticTimeoutError = DojoTacticTimeoutError

        self.repo_cache = {}
        self.traced_cache = set()

    def get_repo(self, repo_path: str):
        repo_path = str(Path(repo_path).resolve())

        if repo_path in self.repo_cache:
            return self.repo_cache[repo_path]

        repo = self.LeanGitRepo.from_path(Path(repo_path))
        self.repo_cache[repo_path] = repo

        if CONFIG["lean"]["trace_repos_before_use"] and repo_path not in self.traced_cache:
            print(f"[lean] tracing repo: {repo_path}")
            self.trace(repo)
            self.traced_cache.add(repo_path)

        return repo

    def theorem(self, spec: TheoremSpec):
        repo = self.get_repo(spec.repo_path)
        return self.Theorem(repo, spec.file_path, spec.theorem_name)

    def is_tactic_state(self, x: Any) -> bool:
        return isinstance(x, self.TacticState)

    def is_proof_finished(self, x: Any) -> bool:
        return isinstance(x, self.ProofFinished)

    def num_goals(self, state: Any) -> int:
        if isinstance(state, self.TacticState):
            try:
                return int(state.num_goals)
            except Exception:
                return count_goals_from_pp(getattr(state, "pp", None))

        return 0

    def state_pp(self, state: Any) -> Optional[str]:
        return getattr(state, "pp", None)

    def state_id(self, state: Any) -> Optional[int]:
        return safe_int(getattr(state, "id", None))

    def open_dojo(self, spec: TheoremSpec):
        thm = self.theorem(spec)
        return self.Dojo(thm, timeout=CONFIG["lean"]["theorem_timeout_sec"])

    def run_tactic(self, dojo: Any, state: Any, tactic: str) -> Tuple[Dict[str, Any], Any]:
        t0 = time.perf_counter()

        try:
            result = dojo.run_tac(state, tactic)
            tau_ms = int((time.perf_counter() - t0) * 1000)
            fields = self.result_to_fields(result)
            fields["tau_ms"] = tau_ms
            return fields, result

        except self.DojoTacticTimeoutError as e:
            tau_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "y": 0,
                "result_type": "Timeout",
                "tau_ms": tau_ms,
                "lean_output": str(e),
                "next_state_id": None,
                "next_state_pp": None,
                "next_state_hash": None,
                "n_goals_after": None,
                "m_closed": 0,
                "c_err": "timeout",
                "error_msg": str(e),
            }, None

        except Exception as e:
            tau_ms = int((time.perf_counter() - t0) * 1000)
            msg = f"{type(e).__name__}: {e}"
            return {
                "y": 0,
                "result_type": type(e).__name__,
                "tau_ms": tau_ms,
                "lean_output": msg,
                "next_state_id": None,
                "next_state_pp": None,
                "next_state_hash": None,
                "n_goals_after": None,
                "m_closed": 0,
                "c_err": classify_error(msg),
                "error_msg": msg,
            }, None

    def result_to_fields(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, self.TacticState):
            pp = getattr(result, "pp", None)
            return {
                "y": 1,
                "result_type": "TacticState",
                "lean_output": pp or "",
                "next_state_id": safe_int(getattr(result, "id", None)),
                "next_state_pp": pp,
                "next_state_hash": sha1_text(pp),
                "n_goals_after": self.num_goals(result),
                "m_closed": 0,
                "c_err": "none",
                "error_msg": None,
            }

        if isinstance(result, self.ProofFinished):
            return {
                "y": 1,
                "result_type": "ProofFinished",
                "lean_output": "no goals",
                "next_state_id": safe_int(getattr(result, "tactic_state_id", None)),
                "next_state_pp": None,
                "next_state_hash": None,
                "n_goals_after": 0,
                "m_closed": 1,
                "c_err": "none",
                "error_msg": None,
            }

        if isinstance(result, self.LeanError):
            err = getattr(result, "error", str(result))
            return {
                "y": 0,
                "result_type": "LeanError",
                "lean_output": err,
                "next_state_id": None,
                "next_state_pp": None,
                "next_state_hash": None,
                "n_goals_after": None,
                "m_closed": 0,
                "c_err": classify_error(err),
                "error_msg": err,
            }

        if isinstance(result, self.ProofGivenUp):
            return {
                "y": 0,
                "result_type": "ProofGivenUp",
                "lean_output": "proof given up",
                "next_state_id": None,
                "next_state_pp": None,
                "next_state_hash": None,
                "n_goals_after": None,
                "m_closed": 0,
                "c_err": "proof_given_up",
                "error_msg": "proof given up",
            }

        msg = str(result)
        return {
            "y": 0,
            "result_type": type(result).__name__,
            "lean_output": msg,
            "next_state_id": None,
            "next_state_pp": None,
            "next_state_hash": None,
            "n_goals_after": None,
            "m_closed": 0,
            "c_err": classify_error(msg),
            "error_msg": msg,
        }


class TupleWriter:
    def __init__(self, root_dir: str | Path, shard_size: int):
        self.root = ensure_dir(root_dir)
        self.shard_size = shard_size
        self.buffer = []
        self.shard_idx = 0
        self.total = 0
        self.stats = Counter()
        self.unique_states = set()
        self.unique_theorems = set()
        self.started_at = now_iso()

        self.parquet_dir = ensure_dir(self.root / "parquet")
        self.jsonl_dir = ensure_dir(self.root / "jsonl")

    def add(self, row: Dict[str, Any]) -> None:
        self.buffer.append(row)
        self.total += 1

        self.stats[f"result_type::{row.get('result_type')}"] += 1
        self.stats[f"y::{row.get('y')}"] += 1
        self.stats[f"c_err::{row.get('c_err')}"] += 1
        self.stats[f"tactic_head::{row.get('tactic_head')}"] += 1

        if row.get("state_hash"):
            self.unique_states.add(row["state_hash"])
        if row.get("theorem_id"):
            self.unique_theorems.add(row["theorem_id"])

        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        shard_name = f"tuples-{self.shard_idx:05d}"
        rows = self.buffer
        self.buffer = []

        if CONFIG["output"]["write_jsonl"]:
            append_jsonl(self.jsonl_dir / f"{shard_name}.jsonl", rows)

        if CONFIG["output"]["write_parquet"]:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_parquet(self.parquet_dir / f"{shard_name}.parquet", index=False)

        print(f"[writer] flushed shard={self.shard_idx} rows={len(rows)} total={self.total}")
        self.shard_idx += 1
        self.write_stats()

    def write_stats(self) -> None:
        stats = {
            "run_name": CONFIG["run_name"],
            "started_at": self.started_at,
            "updated_at": now_iso(),
            "total_tuples": self.total,
            "num_shards": self.shard_idx,
            "unique_states": len(self.unique_states),
            "unique_theorems": len(self.unique_theorems),
            "counters": dict(self.stats),
        }
        save_json(self.root / CONFIG["output"]["stats_file"], stats)

    def close(self) -> None:
        self.flush()
        self.write_stats()


class WeightedTheoremSampler:
    def __init__(self, pools: Dict[str, List[TheoremSpec]], weights: Dict[str, float], seed: int):
        self.rng = random.Random(seed)
        self.pools = {}

        for k, rows in pools.items():
            rows = list(rows)
            self.rng.shuffle(rows)
            self.pools[k] = deque(rows)

        self.weights = {k: float(weights.get(k, 1.0)) for k in self.pools if self.pools[k]}

    def next(self) -> Optional[TheoremSpec]:
        nonempty = [k for k, v in self.pools.items() if len(v) > 0]
        if not nonempty:
            return None

        weights = [self.weights.get(k, 1.0) for k in nonempty]
        total = sum(weights)
        r = self.rng.random() * total
        upto = 0.0
        chosen = nonempty[-1]

        for k, w in zip(nonempty, weights):
            upto += w
            if r <= upto:
                chosen = k
                break

        return self.pools[chosen].popleft()


def make_synthetic_theorems() -> List[str]:
    xs = []

    for i in range(1, 60):
        xs.append(f"theorem syn_nat_add_zero_{i} (n : Nat) : n + 0 = n := by\n  sorry\n")
        xs.append(f"theorem syn_nat_zero_add_{i} (n : Nat) : 0 + n = n := by\n  sorry\n")
        xs.append(f"theorem syn_nat_add_comm_{i} (a b : Nat) : a + b = b + a := by\n  sorry\n")
        xs.append(f"theorem syn_nat_add_assoc_{i} (a b c : Nat) : (a + b) + c = a + (b + c) := by\n  sorry\n")

    for i in range(1, 60):
        xs.append(f"theorem syn_eq_trans_{i} {{α : Type}} (a b c : α) (h1 : a = b) (h2 : b = c) : a = c := by\n  sorry\n")
        xs.append(f"theorem syn_eq_symm_{i} {{α : Type}} (a b : α) (h : a = b) : b = a := by\n  sorry\n")
        xs.append(f"theorem syn_and_left_{i} (p q : Prop) (h : p ∧ q) : p := by\n  sorry\n")
        xs.append(f"theorem syn_and_right_{i} (p q : Prop) (h : p ∧ q) : q := by\n  sorry\n")
        xs.append(f"theorem syn_and_comm_{i} (p q : Prop) : p ∧ q → q ∧ p := by\n  sorry\n")

    for i in range(1, 40):
        xs.append(f"theorem syn_imp_trans_{i} (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r := by\n  sorry\n")
        xs.append(f"theorem syn_or_comm_{i} (p q : Prop) : p ∨ q → q ∨ p := by\n  sorry\n")
        xs.append(f"theorem syn_exists_pair_{i} : ∃ n : Nat, n = n := by\n  sorry\n")

    return xs


def create_lean_project(repo_dir: Path, lean_toolchain: str, mathlib_ref: str, lean_file_text: str) -> str:
    ensure_dir(repo_dir)
    ensure_dir(repo_dir / "TupleGen")

    write_text(repo_dir / "lean-toolchain", lean_toolchain + "\n")
    write_text(
        repo_dir / "lakefile.lean",
        f"""
import Lake
open Lake DSL

package «tuple_gen_repo» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "{mathlib_ref}"

lean_lib TupleGen where
""".strip() + "\n",
    )
    write_text(repo_dir / "TupleGen" / "Generated.lean", lean_file_text)

    commit = init_git_repo(repo_dir)

    print(f"[repo] lake update/build for {repo_dir}")
    run_cmd("lake update", cwd=repo_dir, check=False)
    run_cmd("lake exe cache get", cwd=repo_dir, check=False)
    build_out = run_cmd("lake build TupleGen", cwd=repo_dir, check=False)

    if "error:" in build_out.lower():
        print("[warning] lake build reported errors. Repo may fail in LeanDojo.")
        print(build_out[-4000:])

    return commit


def materialize_synthetic_repo() -> List[TheoremSpec]:
    cfg = CONFIG["dataset_sources"]["synthetic_small_local"]
    repo_dir = Path(cfg["repo_dir"]).resolve()

    theorems = make_synthetic_theorems()
    lean_text = (
        "import Mathlib.Data.Nat.Basic\n"
        "import Mathlib.Tactic\n\n"
        "set_option maxHeartbeats 0\n"
        "set_option autoImplicit true\n\n"
        "namespace TupleGen\n\n"
        + "\n".join(theorems)
        + "\n\nend TupleGen\n"
    )

    commit = create_lean_project(repo_dir, cfg["lean_toolchain"], cfg["mathlib_ref"], lean_text)

    specs = []
    for i, thm in enumerate(theorems):
        name = extract_theorem_name(thm)
        if not name:
            continue

        specs.append(TheoremSpec(
            dataset_name="synthetic_small_local",
            source_split="generated",
            theorem_id=f"synthetic:{i}",
            theorem_name=f"TupleGen.{name}",
            original_name=name,
            file_path="TupleGen/Generated.lean",
            repo_path=str(repo_dir),
            repo_url=str(repo_dir),
            repo_commit=commit,
            source_weight=cfg["weight"],
        ))

    return specs


def materialize_lean_workbook_repo() -> List[TheoremSpec]:
    cfg = CONFIG["dataset_sources"]["lean_workbook_local"]
    if not cfg["enabled"]:
        return []

    from datasets import load_dataset

    repo_dir = Path(cfg["repo_dir"]).resolve()
    ds = load_dataset(cfg["dataset_name"], split=cfg["split"])

    statements = []
    meta = []

    for i, row in enumerate(ds):
        if len(statements) >= cfg["max_rows_to_materialize"]:
            break

        stmt = None
        for c in ["formal_statement", "formal", "statement", "lean_code", "code"]:
            if c in row and row[c]:
                stmt = row[c]
                break

        if not stmt or ("theorem " not in stmt and "lemma " not in stmt):
            continue

        original = extract_theorem_name(stmt) or f"workbook_{i}"
        new_name = sanitize_lean_identifier(f"workbook_{i}_{original}", f"workbook_{i}")
        lean_stmt = force_theorem_by_sorry(stmt, new_name)

        if not lean_stmt:
            continue

        statements.append(lean_stmt)
        meta.append((i, original, new_name))

    lean_text = (
        "import Mathlib\n\n"
        "set_option maxHeartbeats 0\n"
        "set_option autoImplicit true\n\n"
        "namespace TupleGen\n\n"
        + "\n".join(statements)
        + "\n\nend TupleGen\n"
    )

    commit = create_lean_project(repo_dir, cfg["lean_toolchain"], cfg["mathlib_ref"], lean_text)

    specs = []
    for source_idx, original, new_name in meta:
        specs.append(TheoremSpec(
            dataset_name="lean_workbook_local",
            source_split=cfg["split"],
            theorem_id=f"lean_workbook:{source_idx}",
            theorem_name=f"TupleGen.{new_name}",
            original_name=original,
            file_path="TupleGen/Generated.lean",
            repo_path=str(repo_dir),
            repo_url=str(repo_dir),
            repo_commit=commit,
            source_weight=cfg["weight"],
        ))

    return specs


def load_manifest(path: str | Path, dataset_name: str, weight: float) -> List[TheoremSpec]:
    path = Path(path)
    if not path.exists():
        print(f"[warning] manifest not found: {path}")
        return []

    rows = []

    for r in read_jsonl(path):
        repo_path = r.get("repo_path") or r.get("local_repo_path")
        theorem_name = r.get("theorem_name") or r.get("full_name") or r.get("name")
        file_path = r.get("file_path")

        if not repo_path or not theorem_name or not file_path:
            continue

        rows.append(TheoremSpec(
            dataset_name=r.get("dataset_name", dataset_name),
            source_split=r.get("source_split", r.get("split", "train")),
            theorem_id=str(r.get("theorem_id", f"{dataset_name}:{len(rows)}")),
            theorem_name=theorem_name,
            original_name=r.get("original_name", theorem_name),
            file_path=file_path,
            repo_path=str(Path(repo_path).resolve()),
            repo_url=r.get("repo_url", repo_path),
            repo_commit=r.get("repo_commit", get_git_commit(repo_path)),
            source_weight=float(r.get("source_weight", weight)),
        ))

    return rows


def load_theorem_pools() -> Tuple[Dict[str, List[TheoremSpec]], Dict[str, float]]:
    pools = {}
    weights = {}

    if CONFIG["dataset_sources"]["synthetic_small_local"]["enabled"]:
        rows = materialize_synthetic_repo()
        pools["synthetic_small_local"] = rows
        weights["synthetic_small_local"] = CONFIG["dataset_sources"]["synthetic_small_local"]["weight"]
        print(f"[data] synthetic_small_local: {len(rows)} theorems")

    if CONFIG["dataset_sources"]["lean_workbook_local"]["enabled"]:
        rows = materialize_lean_workbook_repo()
        pools["lean_workbook_local"] = rows
        weights["lean_workbook_local"] = CONFIG["dataset_sources"]["lean_workbook_local"]["weight"]
        print(f"[data] lean_workbook_local: {len(rows)} theorems")

    for m in CONFIG["dataset_sources"]["external_manifests"]:
        if not m["enabled"]:
            continue

        rows = load_manifest(m["path"], m["dataset_name"], m["weight"])
        pools[m["dataset_name"]] = rows
        weights[m["dataset_name"]] = m["weight"]
        print(f"[data] {m['dataset_name']}: {len(rows)} theorems")

    pools = {k: v for k, v in pools.items() if v}
    weights = {k: weights[k] for k in pools}

    if not pools:
        raise RuntimeError("No theorem pools loaded. Enable synthetic/local/manifest source in CONFIG.")

    return pools, weights


def save_theorem_index(root_dir: str | Path, pools: Dict[str, List[TheoremSpec]]) -> None:
    path = Path(root_dir) / CONFIG["output"]["theorem_index_file"]
    rows = []

    for _, specs in pools.items():
        for s in specs:
            rows.append(asdict(s))

    append_jsonl(path, rows)


def make_tuple_record(
    tuple_id: int,
    spec: TheoremSpec,
    state_item: StateItem,
    state_pp: str,
    state_hash: str,
    sibling_group_id: str,
    n_goals_before: int,
    candidate: Dict[str, Any],
    prompt: str,
    outcome: Dict[str, Any],
    is_duplicate_state_tactic: bool,
    is_duplicate_exact_tuple: bool,
) -> Dict[str, Any]:
    n_after = outcome.get("n_goals_after")
    delta = None if n_after is None else int(n_after - n_goals_before)

    y = int(outcome.get("y", 0))
    m_closed = int(outcome.get("m_closed", 0))
    m_new = int(delta is not None and delta > 0)
    m_decreased = int(delta is not None and delta < 0)

    is_positive_transition = int(y == 1)
    is_negative_error = int(y == 0)
    is_timeout = int(outcome.get("result_type") == "Timeout")
    is_strong_positive = int(m_closed == 1 or m_decreased == 1)
    is_weak_positive = int(y == 1 and m_new == 1)

    return {
        "schema_version": "tuple_v1",
        "tuple_id": tuple_id,
        "created_at": now_iso(),

        "dataset_name": spec.dataset_name,
        "source_split": spec.source_split,
        "theorem_id": spec.theorem_id,
        "theorem_name": spec.theorem_name,
        "original_theorem_name": spec.original_name,
        "file_path": spec.file_path,
        "repo_path": spec.repo_path,
        "repo_url": spec.repo_url,
        "repo_commit": spec.repo_commit,

        "state_id": state_item.state_id,
        "parent_state_id": state_item.parent_state_id,
        "depth": state_item.depth,
        "sibling_group_id": sibling_group_id,
        "proof_prefix": "\n".join(state_item.proof_prefix),
        "state_pp": state_pp,
        "state_hash": state_hash,

        "tactic": candidate["tactic"],
        "tactic_head": candidate["tactic_head"],
        "tactic_rank": candidate["tactic_rank"],
        "tactic_score": candidate.get("tactic_score"),
        "generation_pass": candidate.get("generation_pass"),
        "generation_params": json.dumps(candidate.get("generation_params", {}), ensure_ascii=False),
        "model_name": CONFIG["model"]["name"],
        "prompt": prompt,

        "y": y,
        "result_type": outcome.get("result_type"),
        "tau_ms": outcome.get("tau_ms"),
        "lean_output": outcome.get("lean_output"),
        "next_state_id": outcome.get("next_state_id"),
        "next_state_pp": outcome.get("next_state_pp"),
        "next_state_hash": outcome.get("next_state_hash"),
        "error_msg": outcome.get("error_msg"),
        "c_err": outcome.get("c_err"),

        "n_goals_before": n_goals_before,
        "n_goals_after": n_after,
        "delta_n_goals": delta,
        "m_closed": m_closed,
        "m_new": m_new,
        "m_decreased": m_decreased,

        "is_positive_transition": is_positive_transition,
        "is_negative_error": is_negative_error,
        "is_timeout": is_timeout,
        "is_strong_positive": is_strong_positive,
        "is_weak_positive": is_weak_positive,
        "is_on_closed_path": None,

        "is_duplicate_state_tactic": int(is_duplicate_state_tactic),
        "is_duplicate_exact_tuple": int(is_duplicate_exact_tuple),
    }


class TupleGenerationRun:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.target = args.target_tuples or CONFIG["dataset_size"]["target_total_attempt_tuples"]

        if args.output_dir:
            CONFIG["output"]["root_dir"] = args.output_dir

        if args.synthetic_only:
            CONFIG["dataset_sources"]["synthetic_small_local"]["enabled"] = True
            CONFIG["dataset_sources"]["lean_workbook_local"]["enabled"] = False
            for m in CONFIG["dataset_sources"]["external_manifests"]:
                m["enabled"] = False

        if args.enable_lean_workbook:
            CONFIG["dataset_sources"]["lean_workbook_local"]["enabled"] = True

        if args.manifest:
            CONFIG["dataset_sources"]["external_manifests"].append({
                "enabled": True,
                "dataset_name": args.manifest_dataset_name,
                "weight": args.manifest_weight,
                "path": args.manifest,
            })

        self.root = ensure_dir(CONFIG["output"]["root_dir"])
        self.writer = TupleWriter(self.root, CONFIG["output"]["shard_size_tuples"])
        self.generator = None
        self.lean = None

        self.seen_state_tactic = set()
        self.seen_exact_tuple = set()
        self.global_tuple_id = 0
        self.run_stats = Counter()
        self.failed_theorems = []

    def setup(self) -> None:
        save_json(self.root / CONFIG["output"]["metadata_file"], {
            "run_name": CONFIG["run_name"],
            "created_at": now_iso(),
            "config": CONFIG,
            "argv": sys.argv,
        })

        check_cuda_or_fail()
        self.generator = BFSProverGenerator(CONFIG["model"])
        self.lean = LeanRunner()

    def run(self) -> None:
        pools, weights = load_theorem_pools()
        save_theorem_index(self.root, pools)

        sampler = WeightedTheoremSampler(pools, weights, CONFIG["seed"])
        print(f"[run] theorem pools loaded: {sum(len(v) for v in pools.values())} theorem specs")
        print(f"[run] target tuples: {self.target}")

        while self.writer.total < self.target:
            spec = sampler.next()
            if spec is None:
                print("[run] theorem sampler exhausted")
                break

            try:
                self.process_theorem(spec)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"[error] theorem failed: {spec.theorem_name} | {msg}")
                self.failed_theorems.append({
                    "theorem_id": spec.theorem_id,
                    "theorem_name": spec.theorem_name,
                    "dataset_name": spec.dataset_name,
                    "error": msg,
                    "traceback": traceback.format_exc(),
                })
                self.run_stats["theorem_failures"] += 1

            if self.writer.total % 1000 < CONFIG["generation"]["passes"][0]["n"]:
                self.print_progress()

        self.writer.close()
        save_json(self.root / "failed_theorems.json", self.failed_theorems)
        self.print_progress(final=True)

    def print_progress(self, final: bool = False) -> None:
        stats = {
            "total_tuples": self.writer.total,
            "target": self.target,
            "unique_states": len(self.writer.unique_states),
            "unique_theorems": len(self.writer.unique_theorems),
            "valid": self.writer.stats.get("y::1", 0),
            "invalid": self.writer.stats.get("y::0", 0),
            "proof_finished": self.writer.stats.get("result_type::ProofFinished", 0),
            "theorem_failures": self.run_stats.get("theorem_failures", 0),
        }
        prefix = "[final]" if final else "[progress]"
        print(prefix, json.dumps(stats, ensure_ascii=False))

    def process_theorem(self, spec: TheoremSpec) -> None:
        cfg = CONFIG["tuple_search"]
        theorem_tuple_count = 0
        states_expanded = 0
        proof_closed = False

        with self.lean.open_dojo(spec) as pair:
            dojo, init_state = pair

            if not self.lean.is_tactic_state(init_state):
                self.run_stats["bad_initial_state"] += 1
                return

            queue = deque([
                StateItem(
                    state=init_state,
                    state_id=self.lean.state_id(init_state),
                    parent_state_id=None,
                    depth=0,
                    proof_prefix=[],
                )
            ])

            while queue:
                if self.writer.total >= self.target:
                    return
                if theorem_tuple_count >= cfg["max_attempt_tuples_per_theorem"]:
                    return
                if states_expanded >= cfg["max_states_expanded_per_theorem"]:
                    return
                if proof_closed and cfg["stop_policy"] == "finish_current_state_candidates_then_stop_if_proof_closed":
                    return

                item = queue.popleft()

                if item.depth > cfg["max_depth_per_theorem"]:
                    continue
                if not self.lean.is_tactic_state(item.state):
                    continue

                state_pp = self.lean.state_pp(item.state)
                if not state_pp:
                    continue
                if len(state_pp) > CONFIG["filtering"]["states_longer_than_chars"]:
                    self.run_stats["dropped_long_state"] += 1
                    continue

                state_hash = sha1_text(state_pp)
                sibling_group_id = sha1_text(f"{spec.theorem_id}:{state_hash}:{item.depth}")[:16]
                n_goals_before = self.lean.num_goals(item.state)
                candidates, prompt = self.generator.generate(state_pp)

                if not candidates:
                    self.run_stats["states_with_no_candidates"] += 1
                    states_expanded += 1
                    continue

                valid_children = []
                state_closed_here = False

                for cand in candidates:
                    if self.writer.total >= self.target:
                        break
                    if theorem_tuple_count >= cfg["max_attempt_tuples_per_theorem"]:
                        break

                    key_state_tactic = (state_hash, cand["tactic"])
                    is_dup_state_tactic = key_state_tactic in self.seen_state_tactic

                    if CONFIG["filtering"]["dedup_same_state_tactic"] and is_dup_state_tactic:
                        self.run_stats["dedup_state_tactic_skipped"] += 1
                        continue

                    self.seen_state_tactic.add(key_state_tactic)
                    outcome, result_obj = self.lean.run_tactic(dojo, item.state, cand["tactic"])

                    exact_key = (
                        state_hash,
                        cand["tactic"],
                        outcome.get("next_state_hash"),
                        outcome.get("c_err"),
                    )
                    is_dup_exact = exact_key in self.seen_exact_tuple

                    if CONFIG["filtering"]["dedup_exact_tuple"] and is_dup_exact:
                        self.run_stats["dedup_exact_tuple_skipped"] += 1
                        continue

                    self.seen_exact_tuple.add(exact_key)

                    rec = make_tuple_record(
                        tuple_id=self.global_tuple_id,
                        spec=spec,
                        state_item=item,
                        state_pp=state_pp,
                        state_hash=state_hash,
                        sibling_group_id=sibling_group_id,
                        n_goals_before=n_goals_before,
                        candidate=cand,
                        prompt=prompt,
                        outcome=outcome,
                        is_duplicate_state_tactic=is_dup_state_tactic,
                        is_duplicate_exact_tuple=is_dup_exact,
                    )

                    self.writer.add(rec)
                    self.global_tuple_id += 1
                    theorem_tuple_count += 1

                    if outcome.get("result_type") == "ProofFinished":
                        proof_closed = True
                        state_closed_here = True

                    if self.lean.is_tactic_state(result_obj):
                        child = StateItem(
                            state=result_obj,
                            state_id=self.lean.state_id(result_obj),
                            parent_state_id=item.state_id,
                            depth=item.depth + 1,
                            proof_prefix=item.proof_prefix + [cand["tactic"]],
                        )
                        valid_children.append((rec, child))

                states_expanded += 1

                if valid_children and not state_closed_here:
                    selected = self.select_children(valid_children, cfg["max_children_enqueued_per_state"])
                    for _, child in selected:
                        if child.depth <= cfg["max_depth_per_theorem"]:
                            queue.append(child)

                if theorem_tuple_count >= cfg["max_attempt_tuples_per_theorem"]:
                    return

    def select_children(self, valid_children: List[Tuple[Dict[str, Any], StateItem]], k: int):
        def score(x):
            rec, _ = x
            closed = rec.get("m_closed", 0)
            decreased = rec.get("m_decreased", 0)
            rank = rec.get("tactic_rank", 999)
            delta = rec.get("delta_n_goals")
            delta_score = 0 if delta is None else -delta
            return (-closed, -decreased, -delta_score, rank)

        valid_children = sorted(valid_children, key=score)
        return valid_children[:k]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-tuples", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--enable-lean-workbook", action="store_true")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--manifest-dataset-name", type=str, default="external_manifest")
    p.add_argument("--manifest-weight", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(CONFIG["seed"])
    set_leandojo_env()

    run = TupleGenerationRun(args)
    run.setup()
    run.run()


if __name__ == "__main__":
    main()

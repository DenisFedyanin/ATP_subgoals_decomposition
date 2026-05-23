#!/usr/bin/env python3
"""
оценка ATP-прувера:

1) Офлайн-валидация энкодера (на отложенных кортежах):
Точность предсказания статуса/типа результата
Оценка времени выполнения тактик
Прогноз эффекта от шага

2) Оценка в поиске доказательств (на новых теоремах):
Классический BFS по скорам генератора.
BFS с ранжированием энкодером и прунингом веток
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import importlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch


CONFIG: Dict[str, Any] = {
    "run_name": "search_eval_v1_limited_two_level",
    "seed": 17,

    "offline_encoder_eval": {
        "enabled": True,
        "tuple_parquet_dir": "data/tuples/full_v1/parquet",
        "max_rows": 200_000,
        "split_by": "theorem_id",
        "heldout_theorem_ratio": 0.05,
        "sample_seed": 17,
        "use_encoder_split_metadata": True,
        "split_metadata_path": "outputs/encoder_runs/transition_encoder_v1/split_metadata.json",
        "preferred_eval_splits": ["valid", "test"],

        "required_columns": [
            "dataset_name", "source_split", "theorem_id", "theorem_name",
            "state_pp", "tactic", "tactic_head", "sibling_group_id", "y", "result_type",
            "tau_ms", "n_goals_before", "n_goals_after", "delta_n_goals",
            "m_closed", "m_new", "m_decreased", "c_err", "tactic_rank",
            "tactic_score", "state_hash", "next_state_hash",
            "is_duplicate_state_tactic", "is_duplicate_exact_tuple",
        ],

        "status_task": "binary_y_and_result_type",
        "execution_time_target": "log1p_tau_ms",
        "output_task": "coarse_output_label",
        "evaluate_sibling_ranking": True,
        "max_sibling_groups_for_ranking": 50_000,

        "coarse_output_labels": [
            "invalid_error", "invalid_timeout", "closed", "valid_decreased",
            "valid_same", "valid_increased", "valid_other",
        ],
    },

    "inputs": {
        "generator_model": "ByteDance-Seed/BFS-Prover-V1-7B",
        "encoder_checkpoint": "outputs/encoder_runs/transition_encoder_v1/checkpoint_best.pt",
        "encoder_config_path": "outputs/encoder_runs/transition_encoder_v1/train_config.json",
        "lean_repos_root": "lean_repos",

        "eval_sets": [
            {
                "name": "lean_workbook_heldout_val",
                "path": "data/theorem_sets/eval/lean_workbook_heldout_val.jsonl",
                "enabled": True,
                "max_theorems": 300,
            },
            {
                "name": "mathlib_heldout_val",
                "path": "data/theorem_sets/eval/mathlib_heldout_val.jsonl",
                "enabled": True,
                "max_theorems": 200,
            },
        ],
    },

    "search_modes": [
        {
            "name": "vanilla_generator_rank",
            "use_encoder": False,
            "ranking": "generator_rank",
            "gentle_pruning": False,
            "enabled": True,
        },
        {
            "name": "encoder_rerank_gentle_prune",
            "use_encoder": True,
            "ranking": "encoder_utility",
            "gentle_pruning": True,
            "enabled": True,
        },
    ],

    "generation": {
        "prompt_format": "{state_pp}:::",
        "max_new_tokens": 64,
        "candidates_per_state": 16,
        "precision_pass": {"n": 8, "temperature": 0.4, "top_p": 0.90},
        "diversity_pass": {"n": 8, "temperature": 0.8, "top_p": 0.95},
        "deduplicate_tactics": True,
        "drop_empty_tactics": True,
        "drop_tactics_longer_than_chars": 300,
        "compute_generator_scores": False,
    },

    "encoder_scoring": {
        "device": "cuda",
        "precision": "bf16",
        "max_input_tokens": 3072,
        "fallback_to_run02_checkpoint": True,
        "input_format": "<STATE>\n{state_pp}\n\n<TACTIC>\n{tactic}\n",
        "normalize_scores_per_state": True,
        "score_components": {
            "utility_score": 1.0,
            "validity_prob": 0.6,
            "closure_prob": 0.8,
            "delta_decrease_prob": 0.3,
            "error_prob_penalty": -0.8,
            "timeout_prob_penalty": -0.5,
        },
    },

    "gentle_pruning": {
        "enabled": True,
        "execute_top_encoder_k": 8,
        "always_keep_generator_top_k": 2,
        "drop_only_if": {
            "pred_error_prob_gt": 0.90,
            "pred_validity_prob_lt": 0.05,
        },
        "never_drop_if_generator_rank_lte": 2,
    },

    "search_budget": {
        "max_nodes_expanded": 128,
        "max_tactics_executed": 1024,
        "max_depth": 24,
        "max_wall_time_sec_per_theorem": 120,
        "tactic_timeout_sec": 5,
        "max_children_per_state": 8,
        "stop_on_first_proof": True,
    },

    "queue_policy": {
        "type": "best_first",
        "node_score": "path_score",
        "path_score_discount": 0.95,
        "tie_breaker": "shallower_depth_first",
    },

    "lean_execution": {
        "backend": "LeanDojo",
        "memory_limit": "16g",
        "cpu_limit": "1",
        "capture_error_messages": True,
        "capture_next_states": True,
        "record_tactic_time_ms": True,
    },

    "metrics": {
        "budget_curve_points": [32, 64, 128, 256, 512, 1024],
        "primary": [
            "solve_rate", "solve_rate_at_budget", "median_nodes_to_solve",
            "median_tactics_to_solve", "median_time_to_solve",
        ],
    },

    "output": {
        "save_dir": "outputs/search_runs/search_eval_v1_limited_two_level",
        "attempt_shard_size": 50_000,
        "write_run_config": True,
        "write_offline_encoder_eval": True,
        "write_summary_metrics": True,
        "write_per_theorem_results": True,
        "write_tactic_attempts": True,
        "write_proofs_found": True,
        "write_comparison_report": True,
        "write_budget_curves": True,
        "write_solve_overlap": True,
        "write_plots": True,
        "fail_if_output_dir_nonempty": True,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_mean(xs: Sequence[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and not pd.isna(x)]
    return float(np.mean(xs)) if xs else None


def safe_median(xs: Sequence[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and not pd.isna(x)]
    return float(np.median(xs)) if xs else None


def tactic_head(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    t = t.replace("·", "").strip()
    m = re.match(r"([A-Za-z_][A-Za-z0-9_']*|rw|simp|exact|apply|constructor|intro|intros|cases|induction|rfl|omega|linarith|ring|norm_num)", t)
    return m.group(1) if m else t.split()[0]


def normalize_text(x: Optional[str], max_chars: Optional[int] = None) -> str:
    if not isinstance(x, str):
        return ""
    x = re.sub(r"\n{3,}", "\n\n", x.strip())
    x = re.sub(r"[ \t]+", " ", x)
    if max_chars and len(x) > max_chars:
        head = int(max_chars * 0.60)
        tail = max_chars - head
        x = x[:head] + "\n...[TRUNCATED]...\n" + x[-tail:]
    return x




def binary_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    pairs = [(int(y), float(s)) for y, s in zip(y_true, y_score) if y is not None and s is not None and not pd.isna(s)]
    if not pairs:
        return None
    pos = [(y, s) for y, s in pairs if y == 1]
    neg = [(y, s) for y, s in pairs if y == 0]
    if not pos or not neg:
        return None
    ranks = sorted([(s, y) for y, s in pairs])
    rank_sum_pos = 0.0
    for i, (_, y) in enumerate(ranks, start=1):
        if y == 1:
            rank_sum_pos += i
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if a is not None and b is not None]
    if not pairs:
        return None
    return float(sum(1 for a, b in pairs if a == b) / len(pairs))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    pairs = [(float(a), float(b)) for a, b in zip(y_true, y_pred) if a is not None and b is not None and not pd.isna(a) and not pd.isna(b)]
    if not pairs:
        return None
    return float(math.sqrt(np.mean([(a - b) ** 2 for a, b in pairs])))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    pairs = [(float(a), float(b)) for a, b in zip(y_true, y_pred) if a is not None and b is not None and not pd.isna(a) and not pd.isna(b)]
    if not pairs:
        return None
    return float(np.mean([abs(a - b) for a, b in pairs]))

# Data label construction

def delta_bucket(row: Dict[str, Any]) -> str:
    result_type = str(row.get("result_type", ""))
    if int(row.get("y", 0) or 0) == 0:
        return "invalid"
    if int(row.get("m_closed", 0) or 0) == 1 or result_type == "ProofFinished":
        return "closed"
    d = row.get("delta_n_goals", None)
    if d is None or pd.isna(d):
        return "invalid"
    d = int(d)
    if d < 0:
        return "decreased"
    if d == 0:
        return "same"
    if d == 1:
        return "increased_by_1"
    return "increased_by_2plus"


def coarse_output_label(row: Dict[str, Any]) -> str:
    result_type = str(row.get("result_type", ""))
    c_err = str(row.get("c_err", "") or "")
    if int(row.get("y", 0) or 0) == 0:
        if result_type.lower() == "timeout" or c_err == "timeout":
            return "invalid_timeout"
        return "invalid_error"
    if int(row.get("m_closed", 0) or 0) == 1 or result_type == "ProofFinished":
        return "closed"
    d = row.get("delta_n_goals", None)
    if d is None or pd.isna(d):
        return "valid_other"
    d = int(d)
    if d < 0:
        return "valid_decreased"
    if d == 0:
        return "valid_same"
    return "valid_increased"


def utility_target(row: Dict[str, Any]) -> float:
    result_type = str(row.get("result_type", ""))
    y = int(row.get("y", 0) or 0)
    if y == 1 and (result_type == "ProofFinished" or int(row.get("m_closed", 0) or 0) == 1):
        return 4.0
    if y == 1 and int(row.get("m_decreased", 0) or 0) == 1:
        return 3.0
    if y == 1 and int(row.get("m_new", 0) or 0) == 1:
        return 2.0
    if y == 1:
        return 1.5
    c_err = str(row.get("c_err", "") or "")
    return -0.5 if result_type == "Timeout" or c_err == "timeout" else 0.0

# Adapters


@dataclass
class TacticCandidate:
    tactic: str
    tactic_head: str
    generator_rank: int
    generator_score: Optional[float] = None
    generation_pass: str = "unknown"


@dataclass
class EncoderPrediction:
    utility_score: float = 0.0
    validity_prob: float = 0.5
    closure_prob: float = 0.0
    error_prob: float = 0.5
    timeout_prob: float = 0.0
    delta_decrease_prob: float = 0.0
    result_type_pred: str = "unknown"
    delta_bucket_pred: str = "unknown"
    output_label_pred: str = "unknown"
    tau_ms_pred: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def labels_from_mapping(label_mappings: Dict[str, Any], id_key: str, to_id_key: str, fallback: Sequence[str]) -> List[str]:
    id_map = label_mappings.get(id_key, {}) if isinstance(label_mappings, dict) else {}
    if isinstance(id_map, dict) and id_map:
        try:
            return [str(v) for _, v in sorted(id_map.items(), key=lambda kv: int(kv[0]))]
        except Exception:
            return [str(v) for _, v in sorted(id_map.items(), key=lambda kv: str(kv[0]))]
    to_id = label_mappings.get(to_id_key, {}) if isinstance(label_mappings, dict) else {}
    if isinstance(to_id, dict) and to_id:
        try:
            return [str(k) for k, _ in sorted(to_id.items(), key=lambda kv: int(kv[1]))]
        except Exception:
            return list(fallback)
    return list(fallback)


def output_label_from_prediction(
    validity_prob: float,
    result_type_pred: str,
    delta_bucket_pred: str,
    timeout_prob: float,
    error_prob: float,
) -> str:
    if validity_prob < 0.5:
        return "invalid_timeout" if timeout_prob > error_prob else "invalid_error"
    if result_type_pred == "ProofFinished" or delta_bucket_pred == "closed":
        return "closed"
    if delta_bucket_pred == "decreased":
        return "valid_decreased"
    if delta_bucket_pred == "same":
        return "valid_same"
    if delta_bucket_pred in {"increased_by_1", "increased_by_2plus"}:
        return "valid_increased"
    return "valid_other"


class Run02CheckpointBackend:
    """Inference shim for checkpoints produced by run_02_train_encoder_updated.py."""

    def __init__(self, checkpoint_path: str | Path, scoring_cfg: Dict[str, Any]):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {self.checkpoint_path}")
        requested_device = str(scoring_cfg.get("device", "cuda"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("[encoder] cuda requested but unavailable; falling back to cpu")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.scoring_cfg = scoring_cfg
        self.max_input_tokens = int(scoring_cfg.get("max_input_tokens", 2048))

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        train_config = checkpoint.get("config", {})
        if isinstance(train_config, dict) and "config" in train_config and "model" not in train_config:
            train_config = train_config["config"]
        model_cfg = dict(train_config.get("model", {}))
        if not model_cfg:
            raise ValueError("Checkpoint does not contain run_02 model config")
        model_cfg["gradient_checkpointing"] = False

        label_mappings = checkpoint.get("label_mappings", {})
        self.result_types = labels_from_mapping(
            label_mappings,
            "id_to_result_type",
            "result_type_to_id",
            ["TacticState", "ProofFinished", "LeanError", "Timeout", "ProofGivenUp", "Other"],
        )
        self.delta_buckets = labels_from_mapping(
            label_mappings,
            "id_to_delta_bucket",
            "delta_bucket_to_id",
            ["invalid", "closed", "decreased", "same", "increased_by_1", "increased_by_2plus"],
        )

        run02_path = Path(__file__).with_name("run_02_train_encoder_updated.py")
        spec = importlib.util.spec_from_file_location("_run02_encoder_module", run02_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import run_02 encoder module from {run02_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.model = module.TransitionEncoderV1(model_cfg, len(self.result_types), len(self.delta_buckets))
        missing, unexpected = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[encoder] checkpoint load warnings: missing={missing[:8]} unexpected={unexpected[:8]}")
        self.model.to(self.device)
        self.model.eval()

        from transformers import AutoTokenizer

        tokenizer_dir = self.checkpoint_path.parent / "tokenizer"
        tokenizer_name = str(tokenizer_dir) if tokenizer_dir.exists() else model_cfg.get("tokenizer") or model_cfg.get("backbone")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"[encoder] loaded run_02 checkpoint backend from {self.checkpoint_path}")

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        precision = str(self.scoring_cfg.get("precision", "bf16"))
        autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        use_autocast = self.device.type == "cuda" and precision in {"bf16", "fp16", "float16"}
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=use_autocast):
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        validity = torch.sigmoid(out["validity_logits"]).detach().float().cpu().numpy()
        closure = torch.sigmoid(out["closure_logits"]).detach().float().cpu().numpy()
        utility = out["utility_logits"].detach().float().cpu().numpy()
        result_probs_arr = torch.softmax(out["result_type_logits"], dim=-1).detach().float().cpu().numpy()
        delta_probs_arr = torch.softmax(out["delta_bucket_logits"], dim=-1).detach().float().cpu().numpy()

        rows: List[Dict[str, Any]] = []
        for i in range(len(texts)):
            result_probs = {label: float(result_probs_arr[i, j]) for j, label in enumerate(self.result_types)}
            delta_probs = {label: float(delta_probs_arr[i, j]) for j, label in enumerate(self.delta_buckets)}
            result_type_pred = argmax_dict(result_probs, "unknown")
            delta_bucket_pred = argmax_dict(delta_probs, "unknown")
            timeout_prob = float(result_probs.get("Timeout", 0.0))
            error_prob = float(result_probs.get("LeanError", 0.0) + result_probs.get("ProofGivenUp", 0.0))
            validity_prob = float(validity[i])
            rows.append({
                "utility_score": float(utility[i]),
                "validity_prob": validity_prob,
                "closure_prob": max(float(closure[i]), float(result_probs.get("ProofFinished", 0.0))),
                "error_prob": error_prob,
                "timeout_prob": timeout_prob,
                "delta_decrease_prob": float(delta_probs.get("decreased", 0.0)),
                "result_type_probs": result_probs,
                "delta_bucket_probs": delta_probs,
                "result_type_pred": result_type_pred,
                "delta_bucket_pred": delta_bucket_pred,
                "output_label_pred": output_label_from_prediction(
                    validity_prob,
                    result_type_pred,
                    delta_bucket_pred,
                    timeout_prob,
                    error_prob,
                ),
            })
        return rows


class GeneratorAdapter:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["inputs"]["generator_model"]
        self.gen_cfg = config["generation"]
        self.backend = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            mod = importlib.import_module("src.generator")
            if hasattr(mod, "BFSProverGenerator"):
                self.backend = mod.BFSProverGenerator(
                    model_name=self.model_name,
                    max_new_tokens=self.gen_cfg["max_new_tokens"],
                )
                print("[generator] using src.generator.BFSProverGenerator")
                return
            if hasattr(mod, "Generator"):
                self.backend = mod.Generator(self.model_name)
                print("[generator] using src.generator.Generator")
                return
        except Exception as e:
            print(f"[generator] src.generator not available, using HF fallback if possible: {e}")

        if torch is None:
            raise RuntimeError("torch is unavailable and src.generator backend was not found")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            dtype = torch.bfloat16 if self.config["encoder_scoring"].get("precision") == "bf16" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backend = "hf"
            print("[generator] using HF fallback")
        except Exception as e:
            raise RuntimeError(f"Could not load generator backend: {e}")

    def generate(self, state_pp: str) -> List[TacticCandidate]:
        if self.backend not in (None, "hf") and hasattr(self.backend, "generate_tactics"):
            out = self.backend.generate_tactics(state_pp=state_pp, config=self.gen_cfg)
            return self._normalize_backend_candidates(out)
        return self._hf_generate(state_pp)

    def _normalize_backend_candidates(self, out: Any) -> List[TacticCandidate]:
        rows = []
        seen = set()
        for i, item in enumerate(out):
            if isinstance(item, str):
                tactic, score, gen_pass = item, None, "backend"
            else:
                tactic = item.get("tactic") or item.get("text") or ""
                score = item.get("score") or item.get("logprob")
                gen_pass = item.get("generation_pass", "backend")
            tactic = clean_generated_tactic(tactic)
            if not tactic or tactic in seen:
                continue
            seen.add(tactic)
            rows.append(TacticCandidate(tactic, tactic_head(tactic), len(rows), score, gen_pass))
        return rows[: self.gen_cfg["candidates_per_state"]]

    def _hf_generate(self, state_pp: str) -> List[TacticCandidate]:
        prompt = self.gen_cfg["prompt_format"].format(state_pp=state_pp.strip())
        specs = [("precision", self.gen_cfg["precision_pass"]), ("diversity", self.gen_cfg["diversity_pass"])]
        candidates: List[TacticCandidate] = []
        seen = set()
        for pass_name, spec in specs:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    num_return_sequences=int(spec["n"]),
                    max_new_tokens=int(self.gen_cfg["max_new_tokens"]),
                    temperature=float(spec["temperature"]),
                    top_p=float(spec["top_p"]),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            decoded = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            for x in decoded:
                t = x.split(":::", 1)[1] if ":::" in x else x[len(prompt):]
                t = clean_generated_tactic(t)
                if not t or t in seen:
                    continue
                if self.gen_cfg.get("drop_tactics_longer_than_chars") and len(t) > self.gen_cfg["drop_tactics_longer_than_chars"]:
                    continue
                seen.add(t)
                candidates.append(TacticCandidate(t, tactic_head(t), len(candidates), None, pass_name))
                if len(candidates) >= int(self.gen_cfg["candidates_per_state"]):
                    return candidates
        return candidates


class EncoderAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ckpt = config["inputs"]["encoder_checkpoint"]
        self.backend = None
        self.device = config["encoder_scoring"].get("device", "cuda")
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            mod = importlib.import_module("src.encoder")
            if hasattr(mod, "load_encoder_for_inference"):
                self.backend = mod.load_encoder_for_inference(self.ckpt, self.config["encoder_scoring"])
                print("[encoder] using src.encoder.load_encoder_for_inference")
                return
            if hasattr(mod, "TransitionEncoderV1"):
                cls = mod.TransitionEncoderV1
                if hasattr(cls, "from_checkpoint"):
                    self.backend = cls.from_checkpoint(self.ckpt, self.config["encoder_scoring"])
                    print("[encoder] using TransitionEncoderV1.from_checkpoint")
                    return
        except Exception as e:
            print(f"[encoder] could not load src.encoder backend: {e}")
        if self.config["encoder_scoring"].get("fallback_to_run02_checkpoint", True):
            self.backend = Run02CheckpointBackend(self.ckpt, self.config["encoder_scoring"])
            return
        raise RuntimeError(
            "No encoder inference backend found. Provide src.encoder or a run_02 checkpoint."
        )

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[EncoderPrediction]:
        if not pairs:
            return []
        texts = [self.config["encoder_scoring"]["input_format"].format(state_pp=s, tactic=t) for s, t in pairs]
        if hasattr(self.backend, "predict_batch"):
            raw = self.backend.predict_batch(texts)
        elif hasattr(self.backend, "score_batch"):
            raw = self.backend.score_batch(pairs)
        else:
            raise RuntimeError("Encoder backend must implement predict_batch(texts) or score_batch(pairs)")
        return [self._normalize_prediction(x) for x in raw]

    def _normalize_prediction(self, x: Any) -> EncoderPrediction:
        if isinstance(x, EncoderPrediction):
            return x
        if not isinstance(x, dict):
            raise TypeError(f"Encoder prediction must be dict-like, got {type(x)}")
        result_probs = x.get("result_type_probs", {}) or {}
        delta_probs = x.get("delta_bucket_probs", {}) or {}
        pred = EncoderPrediction(
            utility_score=float(x.get("utility_score", x.get("score", 0.0)) or 0.0),
            validity_prob=float(x.get("validity_prob", x.get("p_valid", 0.5)) or 0.5),
            closure_prob=float(x.get("closure_prob", x.get("p_closed", 0.0)) or 0.0),
            error_prob=float(x.get("error_prob", result_probs.get("LeanError", 0.0)) or 0.0),
            timeout_prob=float(x.get("timeout_prob", result_probs.get("Timeout", 0.0)) or 0.0),
            delta_decrease_prob=float(x.get("delta_decrease_prob", delta_probs.get("decreased", 0.0)) or 0.0),
            result_type_pred=str(x.get("result_type_pred", argmax_dict(result_probs, "unknown"))),
            delta_bucket_pred=str(x.get("delta_bucket_pred", argmax_dict(delta_probs, "unknown"))),
            output_label_pred=str(x.get("output_label_pred", x.get("coarse_output_pred", "unknown"))),
            tau_ms_pred=x.get("tau_ms_pred", x.get("pred_tau_ms", None)),
            raw=x,
        )
        return pred


def argmax_dict(d: Dict[str, float], default: str = "unknown") -> str:
    if not d:
        return default
    return max(d.items(), key=lambda kv: kv[1])[0]


def clean_generated_tactic(s: str) -> str:
    s = (s or "").strip()
    s = s.split("\n")[0].strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("` ")
    if s.startswith("by "):
        s = s[3:].strip()
    return s


# Lean execution adapter

@dataclass
class LeanOutcome:
    y: int
    result_type: str
    tau_ms: int
    lean_output: str
    next_state: Any = None
    next_state_id: Optional[int] = None
    next_state_pp: Optional[str] = None
    n_goals_after: Optional[int] = None
    m_closed: int = 0
    c_err: str = "none"
    error_msg: Optional[str] = None


class LeanDojoSession:
    def __init__(self, theorem_spec: Dict[str, Any], config: Dict[str, Any]):
        self.theorem_spec = theorem_spec
        self.config = config
        self._ctx = None
        self.dojo = None
        self.init_state = None
        self._load_leandojo_symbols()

    def _load_leandojo_symbols(self) -> None:
        try:
            from lean_dojo import LeanGitRepo, Theorem, Dojo
            self.LeanGitRepo = LeanGitRepo
            self.Theorem = Theorem
            self.Dojo = Dojo
        except Exception as e:
            raise RuntimeError(f"LeanDojo import failed: {e}")

    def __enter__(self):
        repo = self._make_repo()
        thm = self.Theorem(repo, self.theorem_spec["file_path"], self.theorem_spec["theorem_name"])
        timeout = int(self.config["search_budget"]["max_wall_time_sec_per_theorem"])
        self._ctx = self.Dojo(thm, timeout=timeout)
        self.dojo, self.init_state = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False

    def _make_repo(self):
        spec = self.theorem_spec
        if spec.get("repo_path"):
            return self.LeanGitRepo.from_path(Path(spec["repo_path"]).resolve())
        repo_url = spec.get("repo_url") or spec.get("url")
        repo_commit = spec.get("repo_commit") or spec.get("commit") or "HEAD"
        if not repo_url:
            raise ValueError(f"Theorem spec needs repo_path or repo_url: {spec}")
        return self.LeanGitRepo(repo_url, repo_commit)

    def initial_state(self) -> Any:
        return self.init_state

    def run_tactic(self, state: Any, tactic: str) -> LeanOutcome:
        t0 = time.perf_counter()
        try:
            result = self.dojo.run_tac(state, tactic)
            tau_ms = int((time.perf_counter() - t0) * 1000)
            return normalize_lean_result(result, tau_ms)
        except Exception as e:
            tau_ms = int((time.perf_counter() - t0) * 1000)
            msg = str(e)
            return LeanOutcome(0, type(e).__name__, tau_ms, msg, c_err=classify_error(msg), error_msg=msg)


def state_pp(state: Any) -> str:
    return str(getattr(state, "pp", "") or "")


def state_id(state: Any) -> Optional[int]:
    sid = getattr(state, "id", None)
    try:
        return int(sid) if sid is not None else None
    except Exception:
        return None


def num_goals_from_state(state: Any) -> int:
    ng = getattr(state, "num_goals", None)
    try:
        return int(ng)
    except Exception:
        pp = state_pp(state)
        return pp.count("⊢") if pp else 0


def normalize_lean_result(result: Any, tau_ms: int) -> LeanOutcome:
    rt = type(result).__name__
    if rt == "TacticState":
        pp = state_pp(result)
        return LeanOutcome(
            y=1,
            result_type="TacticState",
            tau_ms=tau_ms,
            lean_output=pp,
            next_state=result,
            next_state_id=state_id(result),
            next_state_pp=pp,
            n_goals_after=num_goals_from_state(result),
            m_closed=0,
            c_err="none",
        )
    if rt == "ProofFinished":
        sid = getattr(result, "tactic_state_id", None)
        try:
            sid = int(sid) if sid is not None else None
        except Exception:
            sid = None
        return LeanOutcome(
            y=1,
            result_type="ProofFinished",
            tau_ms=tau_ms,
            lean_output="no goals",
            next_state=None,
            next_state_id=sid,
            next_state_pp=None,
            n_goals_after=0,
            m_closed=1,
            c_err="none",
        )
    if rt == "LeanError":
        msg = str(getattr(result, "error", result))
        return LeanOutcome(0, "LeanError", tau_ms, msg, c_err=classify_error(msg), error_msg=msg)
    if rt == "ProofGivenUp":
        msg = "proof given up"
        return LeanOutcome(0, "ProofGivenUp", tau_ms, msg, c_err="proof_given_up", error_msg=msg)
    msg = str(result)
    return LeanOutcome(0, rt, tau_ms, msg, c_err=classify_error(msg), error_msg=msg)


def classify_error(msg: str) -> str:
    s = (msg or "").lower()
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
    if "unexpected token" in s or "invalid" in s:
        return "syntax_or_invalid"
    if "maximum recursion" in s or "maximum heartbeats" in s:
        return "resource_limit"
    return "lean_error"


# Offline encoder evaluation

def load_tuple_dataframe(tuple_dir: str | Path, max_rows: Optional[int] = None, seed: int = 17) -> pd.DataFrame:
    files = sorted(Path(tuple_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet tuple files found in {tuple_dir}")
    frames = []
    n = 0
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)
        n += len(df)
        if max_rows is not None and n >= max_rows * 2:
            break
    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def theorem_level_holdout(df: pd.DataFrame, ratio: float, seed: int, col: str = "theorem_id") -> pd.DataFrame:
    if col not in df.columns:
        return df.sample(frac=ratio, random_state=seed).reset_index(drop=True)
    ids = sorted(df[col].dropna().unique().tolist())
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n = max(1, int(len(ids) * ratio))
    heldout = set(ids[:n])
    return df[df[col].isin(heldout)].reset_index(drop=True)


def split_key_for_eval_row(row: pd.Series, split_by: str) -> str:
    theorem_id = str(row.get("theorem_id", "unknown"))
    if split_by == "dataset_theorem_id":
        dataset = str(row.get("dataset_name", "unknown"))
        return f"{dataset}::{theorem_id}"
    return theorem_id


def select_offline_eval_rows(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta = {
        "offline_split_source": "random_theorem_holdout",
        "split_metadata_path": cfg.get("split_metadata_path"),
        "preferred_eval_splits": cfg.get("preferred_eval_splits", []),
    }
    meta_path = Path(str(cfg.get("split_metadata_path", "")))
    if cfg.get("use_encoder_split_metadata", True) and meta_path.exists():
        split_meta = read_json(meta_path)
        split_by = str(split_meta.get("split_by") or cfg.get("split_by", "theorem_id"))
        target_keys = set()
        split_keys = split_meta.get("split_keys", {})
        for split_name in cfg.get("preferred_eval_splits", ["valid", "test"]):
            target_keys.update(str(x) for x in split_keys.get(split_name, []))
        if target_keys:
            tmp = df.copy()
            tmp["_eval_split_key"] = tmp.apply(lambda row: split_key_for_eval_row(row, split_by), axis=1)
            selected = tmp[tmp["_eval_split_key"].isin(target_keys)].drop(columns=["_eval_split_key"]).reset_index(drop=True)
            if not selected.empty:
                meta.update({
                    "offline_split_source": "encoder_split_metadata",
                    "split_by": split_by,
                    "num_target_split_keys": len(target_keys),
                    "num_selected_rows": int(len(selected)),
                })
                return selected, meta
        meta["split_metadata_warning"] = "metadata_found_but_no_matching_rows"

    selected = theorem_level_holdout(df, cfg["heldout_theorem_ratio"], cfg["sample_seed"], cfg["split_by"])
    meta["num_selected_rows"] = int(len(selected))
    return selected, meta


def run_offline_encoder_eval(config: Dict[str, Any], encoder: EncoderAdapter, out_dir: Path) -> Dict[str, Any]:
    cfg = config["offline_encoder_eval"]
    tuple_dir = cfg["tuple_parquet_dir"]
    print(f"[offline] loading tuples from {tuple_dir}")
    df = load_tuple_dataframe(tuple_dir, cfg.get("max_rows"), cfg["sample_seed"])
    missing = [c for c in cfg["required_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"Tuple dataset missing required columns for offline eval: {missing}")
    df, split_meta = select_offline_eval_rows(df, cfg)
    df = df.dropna(subset=["state_pp", "tactic"]).reset_index(drop=True)
    print(f"[offline] evaluating {len(df)} held-out tuple rows")

    pairs = [(normalize_text(s, 12000), normalize_text(t, 300)) for s, t in zip(df["state_pp"], df["tactic"])]
    preds: List[EncoderPrediction] = []
    bs = 64
    for i in range(0, len(pairs), bs):
        preds.extend(encoder.score_pairs(pairs[i:i+bs]))
        if (i // bs) % 50 == 0:
            print(f"[offline] scored {min(i+bs, len(pairs))}/{len(pairs)}")

    pred_df = pd.DataFrame([asdict(p) for p in preds])
    target_rows = df.to_dict("records")
    y_true = [int(x.get("y", 0) or 0) for x in target_rows]
    y_score = pred_df["validity_prob"].tolist()
    result_true = [str(x.get("result_type", "")) for x in target_rows]
    result_pred = pred_df["result_type_pred"].tolist()
    delta_true = [delta_bucket(x) for x in target_rows]
    delta_pred = pred_df["delta_bucket_pred"].tolist()
    output_true = [coarse_output_label(x) for x in target_rows]
    output_pred = pred_df["output_label_pred"].tolist()
    tau_true_log = [math.log1p(float(x.get("tau_ms", 0) or 0)) for x in target_rows]
    tau_pred = pred_df["tau_ms_pred"].tolist()
    tau_pred_log = [math.log1p(float(x)) if x is not None and not pd.isna(x) else None for x in tau_pred]
    tau_prediction_available = any(x is not None and not pd.isna(x) for x in tau_pred)

    metrics: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_theorems": int(df["theorem_id"].nunique()) if "theorem_id" in df.columns else None,
        "offline_split_source": split_meta.get("offline_split_source"),
        "tau_prediction_available": bool(tau_prediction_available),
        "status_validity_auc": binary_auc(y_true, y_score),
        "status_validity_accuracy_at_0_5": accuracy(y_true, [1 if s >= 0.5 else 0 for s in y_score]),
        "result_type_accuracy": accuracy(result_true, result_pred),
        "delta_bucket_accuracy": accuracy(delta_true, delta_pred),
        "coarse_output_accuracy": accuracy(output_true, output_pred),
        "tau_log_rmse": rmse(tau_true_log, tau_pred_log) if tau_prediction_available else None,
        "tau_log_mae": mae(tau_true_log, tau_pred_log) if tau_prediction_available else None,
    }

    if cfg.get("evaluate_sibling_ranking", True):
        metrics.update(evaluate_sibling_ranking(df, pred_df, cfg.get("max_sibling_groups_for_ranking", 50_000)))

    eval_dir = ensure_dir(out_dir / "offline_encoder_eval")
    write_json(eval_dir / "metrics.json", metrics)
    write_json(eval_dir / "split_selection.json", split_meta)
    pred_out = pd.concat([
        df[[c for c in ["theorem_id", "theorem_name", "sibling_group_id", "state_hash", "tactic", "tactic_rank", "y", "result_type", "tau_ms", "c_err"] if c in df.columns]].reset_index(drop=True),
        pred_df.drop(columns=["raw"], errors="ignore").reset_index(drop=True),
    ], axis=1)
    pred_out.to_parquet(eval_dir / "predictions.parquet", index=False)
    print(f"[offline] metrics: {metrics}")
    return metrics


def evaluate_sibling_ranking(df: pd.DataFrame, pred_df: pd.DataFrame, max_groups: int) -> Dict[str, Any]:
    if "sibling_group_id" not in df.columns:
        return {"sibling_ranking_auc": None, "sibling_groups_evaluated": 0}
    tmp = df[["sibling_group_id", "tactic_rank"]].copy()
    tmp["target_utility"] = [utility_target(x) for x in df.to_dict("records")]
    tmp["pred_utility"] = pred_df["utility_score"].values
    groups = list(tmp.groupby("sibling_group_id"))
    if len(groups) > max_groups:
        rnd = random.Random(17)
        groups = rnd.sample(groups, max_groups)
    pair_correct = 0
    pair_total = 0
    top1_hits = 0
    top1_total = 0
    for _, g in groups:
        if len(g) < 2:
            continue
        rows = g.to_dict("records")
        best_true = max(rows, key=lambda r: r["target_utility"])
        best_pred = max(rows, key=lambda r: r["pred_utility"])
        if best_true["target_utility"] > min(r["target_utility"] for r in rows):
            top1_total += 1
            top1_hits += int(best_pred["target_utility"] == best_true["target_utility"])
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["target_utility"] == b["target_utility"]:
                    continue
                pair_total += 1
                true_order = a["target_utility"] > b["target_utility"]
                pred_order = a["pred_utility"] > b["pred_utility"]
                pair_correct += int(true_order == pred_order)
    return {
        "sibling_ranking_pair_accuracy": float(pair_correct / pair_total) if pair_total else None,
        "sibling_ranking_top1_hit_rate": float(top1_hits / top1_total) if top1_total else None,
        "sibling_groups_evaluated": int(len(groups)),
        "sibling_pairs_evaluated": int(pair_total),
    }


# Search evaluation

@dataclass(order=True)
class SearchNode:
    priority: float
    tie_depth: int
    node_id: int = field(compare=False)
    state: Any = field(compare=False)
    state_pp: str = field(compare=False)
    parent_node_id: Optional[int] = field(compare=False, default=None)
    depth: int = field(compare=False, default=0)
    path_score: float = field(compare=False, default=0.0)
    proof_prefix: List[str] = field(compare=False, default_factory=list)


def load_eval_theorems(config: Dict[str, Any], cli_max_theorems: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in config["inputs"]["eval_sets"]:
        if not spec.get("enabled", True):
            continue
        p = Path(spec["path"])
        if not p.exists():
            print(f"[search] eval set missing, skipping: {p}")
            continue
        items = read_jsonl(p)
        max_n = cli_max_theorems if cli_max_theorems is not None else spec.get("max_theorems")
        if max_n is not None:
            items = items[: int(max_n)]
        for x in items:
            x.setdefault("dataset_name", spec["name"])
            rows.append(x)
    print(f"[search] loaded {len(rows)} theorem specs")
    return rows


def candidate_search_score(
    cand: TacticCandidate,
    enc_pred: Optional[EncoderPrediction],
    mode: Dict[str, Any],
    config: Dict[str, Any],
) -> float:
    if not mode.get("use_encoder") or enc_pred is None:
        if cand.generator_score is not None:
            return float(cand.generator_score)
        return -float(cand.generator_rank)
    comp = config["encoder_scoring"]["score_components"]
    score = 0.0
    score += comp.get("utility_score", 1.0) * enc_pred.utility_score
    score += comp.get("validity_prob", 0.0) * enc_pred.validity_prob
    score += comp.get("closure_prob", 0.0) * enc_pred.closure_prob
    score += comp.get("delta_decrease_prob", 0.0) * enc_pred.delta_decrease_prob
    score += comp.get("error_prob_penalty", 0.0) * enc_pred.error_prob
    score += comp.get("timeout_prob_penalty", 0.0) * enc_pred.timeout_prob
    return float(score)


def apply_gentle_pruning(
    candidates: List[TacticCandidate],
    preds: List[Optional[EncoderPrediction]],
    scores: List[float],
    mode: Dict[str, Any],
    config: Dict[str, Any],
) -> List[int]:
    if not mode.get("gentle_pruning"):
        return list(range(len(candidates)))
    cfg = config["gentle_pruning"]
    keep = set()
    for i, c in enumerate(candidates):
        if c.generator_rank < int(cfg["always_keep_generator_top_k"]):
            keep.add(i)
    ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    keep.update(ranked[: int(cfg["execute_top_encoder_k"])])
    drop_if = cfg.get("drop_only_if", {})
    for i, p in enumerate(preds):
        if i in keep:
            continue
        if p is None:
            keep.add(i)
            continue
        if candidates[i].generator_rank <= int(cfg.get("never_drop_if_generator_rank_lte", -1)):
            keep.add(i)
            continue
        if p.error_prob > float(drop_if.get("pred_error_prob_gt", 2.0)):
            continue
        if p.validity_prob < float(drop_if.get("pred_validity_prob_lt", -1.0)):
            continue
        keep.add(i)
    return sorted(keep, key=lambda i: scores[i], reverse=True)


def run_search_for_theorem(
    theorem_spec: Dict[str, Any],
    mode: Dict[str, Any],
    generator: GeneratorAdapter,
    encoder: Optional[EncoderAdapter],
    config: Dict[str, Any],
    candidate_cache: Optional[Dict[str, List[TacticCandidate]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    budget = config["search_budget"]
    start_time = time.perf_counter()
    attempts: List[Dict[str, Any]] = []
    proof_rows: List[Dict[str, Any]] = []
    solved = False
    proof: List[str] = []
    failure_reason = None
    nodes_expanded = 0
    tactics_executed = 0
    max_depth_reached = 0
    valid_tactics = 0
    invalid_tactics = 0
    timeouts = 0
    node_counter = 0

    try:
        with LeanDojoSession(theorem_spec, config) as session:
            init = session.initial_state()
            init_pp = state_pp(init)
            queue: List[SearchNode] = []
            root = SearchNode(
                priority=0.0,
                tie_depth=0,
                node_id=node_counter,
                state=init,
                state_pp=init_pp,
                depth=0,
                path_score=0.0,
                proof_prefix=[],
            )
            heapq.heappush(queue, root)
            seen_states = {sha1_text(init_pp)}
            node_counter += 1

            while queue:
                elapsed = time.perf_counter() - start_time
                if elapsed > float(budget["max_wall_time_sec_per_theorem"]):
                    failure_reason = "wall_time_budget_exceeded"
                    break
                if nodes_expanded >= int(budget["max_nodes_expanded"]):
                    failure_reason = "node_budget_exceeded"
                    break
                if tactics_executed >= int(budget["max_tactics_executed"]):
                    failure_reason = "tactic_budget_exceeded"
                    break

                node = heapq.heappop(queue)
                if node.depth > int(budget["max_depth"]):
                    continue
                nodes_expanded += 1
                max_depth_reached = max(max_depth_reached, node.depth)

                cache_key = sha1_text(node.state_pp) or f"node:{node.node_id}"
                if candidate_cache is not None and cache_key in candidate_cache:
                    candidates = list(candidate_cache[cache_key])
                else:
                    candidates = generator.generate(node.state_pp)
                    if candidate_cache is not None:
                        candidate_cache[cache_key] = list(candidates)
                if not candidates:
                    continue

                enc_preds: List[Optional[EncoderPrediction]] = [None] * len(candidates)
                if mode.get("use_encoder"):
                    if encoder is None:
                        raise RuntimeError("search mode needs encoder but encoder is None")
                    pairs = [(node.state_pp, c.tactic) for c in candidates]
                    enc_preds = encoder.score_pairs(pairs)

                scores = [candidate_search_score(c, p, mode, config) for c, p in zip(candidates, enc_preds)]
                if mode.get("use_encoder") and config["encoder_scoring"].get("normalize_scores_per_state", True) and len(scores) > 1:
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    if std_score > 1e-6:
                        scores = [(s - mean_score) / std_score for s in scores]
                keep_indices = apply_gentle_pruning(candidates, enc_preds, scores, mode, config)
                children_added = 0

                for idx in keep_indices:
                    if tactics_executed >= int(budget["max_tactics_executed"]):
                        break
                    c = candidates[idx]
                    p = enc_preds[idx]
                    score = scores[idx]
                    before_goals = num_goals_from_state(node.state)
                    outcome = session.run_tactic(node.state, c.tactic)
                    tactics_executed += 1
                    valid_tactics += int(outcome.y == 1)
                    invalid_tactics += int(outcome.y == 0)
                    timeouts += int(outcome.c_err == "timeout" or outcome.result_type == "Timeout")
                    after_goals = outcome.n_goals_after
                    delta = None if after_goals is None else int(after_goals - before_goals)

                    attempt = {
                        "theorem_id": theorem_spec.get("theorem_id"),
                        "theorem_name": theorem_spec.get("theorem_name"),
                        "dataset_name": theorem_spec.get("dataset_name"),
                        "search_mode": mode["name"],
                        "node_id": node.node_id,
                        "parent_node_id": node.parent_node_id,
                        "depth": node.depth,
                        "state_hash": sha1_text(node.state_pp),
                        "tactic": c.tactic,
                        "tactic_head": c.tactic_head,
                        "generator_rank": c.generator_rank,
                        "generator_score": c.generator_score,
                        "encoder_score": p.utility_score if p else None,
                        "final_search_score": score,
                        "pred_validity_prob": p.validity_prob if p else None,
                        "pred_closure_prob": p.closure_prob if p else None,
                        "pred_error_prob": p.error_prob if p else None,
                        "pred_timeout_prob": p.timeout_prob if p else None,
                        "y": outcome.y,
                        "result_type": outcome.result_type,
                        "tau_ms": outcome.tau_ms,
                        "c_err": outcome.c_err,
                        "error_msg": outcome.error_msg,
                        "n_goals_before": before_goals,
                        "n_goals_after": after_goals,
                        "delta_n_goals": delta,
                        "m_closed": outcome.m_closed,
                        "next_state_hash": sha1_text(outcome.next_state_pp),
                        "tactics_executed_so_far": tactics_executed,
                        "nodes_expanded_so_far": nodes_expanded,
                    }
                    attempts.append(attempt)

                    if outcome.m_closed:
                        solved = True
                        proof = node.proof_prefix + [c.tactic]
                        proof_rows.append({
                            "theorem_id": theorem_spec.get("theorem_id"),
                            "theorem_name": theorem_spec.get("theorem_name"),
                            "dataset_name": theorem_spec.get("dataset_name"),
                            "search_mode": mode["name"],
                            "proof": proof,
                            "proof_length": len(proof),
                            "time_sec": time.perf_counter() - start_time,
                            "nodes_expanded": nodes_expanded,
                            "tactics_executed": tactics_executed,
                        })
                        if budget.get("stop_on_first_proof", True):
                            break

                    if outcome.y == 1 and outcome.next_state is not None and children_added < int(budget["max_children_per_state"]):
                        ns_hash = sha1_text(outcome.next_state_pp)
                        if ns_hash and ns_hash not in seen_states:
                            seen_states.add(ns_hash)
                            child_score = node.path_score * float(config["queue_policy"].get("path_score_discount", 0.95)) + score
                            child = SearchNode(
                                priority=-child_score,
                                tie_depth=node.depth + 1,
                                node_id=node_counter,
                                state=outcome.next_state,
                                state_pp=outcome.next_state_pp or "",
                                parent_node_id=node.node_id,
                                depth=node.depth + 1,
                                path_score=child_score,
                                proof_prefix=node.proof_prefix + [c.tactic],
                            )
                            heapq.heappush(queue, child)
                            node_counter += 1
                            children_added += 1

                if solved and budget.get("stop_on_first_proof", True):
                    failure_reason = None
                    break
    except Exception as e:
        failure_reason = f"search_crash:{type(e).__name__}:{e}"

    time_sec = time.perf_counter() - start_time
    result = {
        "theorem_id": theorem_spec.get("theorem_id"),
        "theorem_name": theorem_spec.get("theorem_name"),
        "dataset_name": theorem_spec.get("dataset_name"),
        "search_mode": mode["name"],
        "solved": bool(solved),
        "proof": proof if solved else None,
        "proof_length": len(proof) if solved else None,
        "time_sec": time_sec,
        "nodes_expanded": nodes_expanded,
        "tactics_executed": tactics_executed,
        "valid_tactics": valid_tactics,
        "invalid_tactics": invalid_tactics,
        "timeouts": timeouts,
        "max_depth_reached": max_depth_reached,
        "failure_reason": failure_reason,
    }
    return result, attempts, proof_rows


def run_search_eval(config: Dict[str, Any], generator: GeneratorAdapter, encoder: Optional[EncoderAdapter], out_dir: Path, cli_max_theorems: Optional[int]) -> Dict[str, Any]:
    theorems = load_eval_theorems(config, cli_max_theorems)
    if not theorems:
        print("[search] no theorem specs found; skipping search eval")
        return {"skipped": True, "reason": "no_theorems"}

    modes = [m for m in config["search_modes"] if m.get("enabled", True)]
    all_results: List[Dict[str, Any]] = []
    all_attempts: List[Dict[str, Any]] = []
    all_proofs: List[Dict[str, Any]] = []
    attempts_dir = ensure_dir(out_dir / "tactic_attempts")
    shard_size = int(config["output"]["attempt_shard_size"])
    attempt_shard_idx = 0

    for ti, thm in enumerate(theorems):
        print(f"[search] theorem {ti+1}/{len(theorems)}: {thm.get('theorem_name')}")
        candidate_cache: Dict[str, List[TacticCandidate]] = {}
        for mode in modes:
            result, attempts, proofs = run_search_for_theorem(
                thm,
                mode,
                generator,
                encoder,
                config,
                candidate_cache=candidate_cache,
            )
            all_results.append(result)
            all_attempts.extend(attempts)
            all_proofs.extend(proofs)
            print(f"  mode={mode['name']} solved={result['solved']} tactics={result['tactics_executed']} time={result['time_sec']:.1f}s")
            if len(all_attempts) >= shard_size:
                pd.DataFrame(all_attempts).to_parquet(attempts_dir / f"attempts-{attempt_shard_idx:05d}.parquet", index=False)
                all_attempts = []
                attempt_shard_idx += 1
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_attempts:
        pd.DataFrame(all_attempts).to_parquet(attempts_dir / f"attempts-{attempt_shard_idx:05d}.parquet", index=False)
    results_df = pd.DataFrame(all_results)
    proofs_df = pd.DataFrame(all_proofs)
    results_df.to_parquet(out_dir / "per_theorem_results.parquet", index=False)
    results_df.to_csv(out_dir / "per_theorem_results.csv", index=False)
    if not proofs_df.empty:
        proofs_df.to_parquet(out_dir / "proofs_found.parquet", index=False)
        append_jsonl(out_dir / "proofs_found.jsonl", proofs_df.to_dict("records"))
    summary = summarize_search_results(results_df, config)
    write_json(out_dir / "summary_metrics.json", summary)
    pd.DataFrame(summary.get("per_mode", [])).to_csv(out_dir / "summary_metrics.csv", index=False)
    if config["output"].get("write_budget_curves", True):
        pd.DataFrame(summary.get("budget_curve", [])).to_csv(out_dir / "budget_curve.csv", index=False)
    if config["output"].get("write_solve_overlap", True):
        pd.DataFrame(summary.get("solve_overlap", [])).to_csv(out_dir / "solve_overlap.csv", index=False)
    pd.DataFrame(summary.get("per_dataset", [])).to_csv(out_dir / "per_dataset_metrics.csv", index=False)
    write_comparison_report(out_dir / "comparison_report.md", summary, config)
    if config["output"].get("write_plots", True):
        write_search_plots(out_dir, summary)
    return summary


def summarize_search_results(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    if df.empty:
        return {"per_mode": [], "per_dataset": [], "budget_curve": [], "solve_overlap": [], "comparison": {}}
    per_mode = []
    for mode, g in df.groupby("search_mode"):
        solved = g[g["solved"] == True]
        tactic_den = g["tactics_executed"].replace(0, np.nan)
        per_mode.append({
            "search_mode": mode,
            "num_theorems": int(len(g)),
            "solve_rate": float(g["solved"].mean()),
            "num_solved": int(g["solved"].sum()),
            "mean_nodes_expanded": safe_mean(g["nodes_expanded"].tolist()),
            "mean_tactics_executed": safe_mean(g["tactics_executed"].tolist()),
            "mean_time_sec": safe_mean(g["time_sec"].tolist()),
            "median_nodes_to_solve": safe_median(solved["nodes_expanded"].tolist()) if len(solved) else None,
            "median_tactics_to_solve": safe_median(solved["tactics_executed"].tolist()) if len(solved) else None,
            "median_time_to_solve": safe_median(solved["time_sec"].tolist()) if len(solved) else None,
            "mean_valid_tactic_rate": safe_mean((g["valid_tactics"] / tactic_den).tolist()),
            "mean_invalid_tactic_rate": safe_mean((g["invalid_tactics"] / tactic_den).tolist()),
            "mean_timeout_rate": safe_mean((g["timeouts"] / tactic_den).tolist()),
        })

    per_dataset = []
    if "dataset_name" in df.columns:
        for (dataset, mode), g in df.groupby(["dataset_name", "search_mode"], dropna=False):
            tactic_den = g["tactics_executed"].replace(0, np.nan)
            solved = g[g["solved"] == True]
            per_dataset.append({
                "dataset_name": str(dataset),
                "search_mode": mode,
                "num_theorems": int(len(g)),
                "solve_rate": float(g["solved"].mean()),
                "num_solved": int(g["solved"].sum()),
                "mean_tactics_executed": safe_mean(g["tactics_executed"].tolist()),
                "median_tactics_to_solve": safe_median(solved["tactics_executed"].tolist()) if len(solved) else None,
                "mean_valid_tactic_rate": safe_mean((g["valid_tactics"] / tactic_den).tolist()),
            })

    budget_curve = []
    for mode, g in df.groupby("search_mode"):
        solved_mask = g["solved"] == True
        for point in config["metrics"].get("budget_curve_points", []):
            point = int(point)
            at_budget = solved_mask & (g["tactics_executed"] <= point)
            budget_curve.append({
                "search_mode": mode,
                "budget_tactics": point,
                "num_theorems": int(len(g)),
                "num_solved_at_budget": int(at_budget.sum()),
                "solve_rate_at_budget": float(at_budget.mean()),
            })

    solve_overlap = []
    mode_names = set(df["search_mode"].dropna().astype(str))
    if {"vanilla_generator_rank", "encoder_rerank_gentle_prune"}.issubset(mode_names):
        tmp = df.copy()
        theorem_key = tmp.get("theorem_id", pd.Series([None] * len(tmp))).fillna(tmp.get("theorem_name", "unknown")).astype(str)
        dataset_key = tmp.get("dataset_name", pd.Series(["unknown"] * len(tmp))).fillna("unknown").astype(str)
        tmp["_theorem_key"] = dataset_key + "::" + theorem_key
        pivot = tmp.pivot_table(index="_theorem_key", columns="search_mode", values="solved", aggfunc="max", fill_value=False)
        base = pivot["vanilla_generator_rank"].astype(bool)
        enc = pivot["encoder_rerank_gentle_prune"].astype(bool)
        solve_overlap.append({
            "baseline_mode": "vanilla_generator_rank",
            "encoder_mode": "encoder_rerank_gentle_prune",
            "num_theorems": int(len(pivot)),
            "solved_by_both": int((base & enc).sum()),
            "baseline_only": int((base & ~enc).sum()),
            "encoder_only": int((~base & enc).sum()),
            "unsolved_by_both": int((~base & ~enc).sum()),
            "net_new_solves": int((~base & enc).sum() - (base & ~enc).sum()),
        })

    comp = {}
    modes = {x["search_mode"]: x for x in per_mode}
    if "vanilla_generator_rank" in modes and "encoder_rerank_gentle_prune" in modes:
        base = modes["vanilla_generator_rank"]
        enc = modes["encoder_rerank_gentle_prune"]
        comp = {
            "delta_solve_rate_encoder_minus_baseline": none_safe_sub(enc.get("solve_rate"), base.get("solve_rate")),
            "relative_median_tactics_reduction": relative_reduction(base.get("median_tactics_to_solve"), enc.get("median_tactics_to_solve")),
            "relative_median_nodes_reduction": relative_reduction(base.get("median_nodes_to_solve"), enc.get("median_nodes_to_solve")),
            "relative_median_time_reduction": relative_reduction(base.get("median_time_to_solve"), enc.get("median_time_to_solve")),
            "relative_mean_tactics_reduction": relative_reduction(base.get("mean_tactics_executed"), enc.get("mean_tactics_executed")),
            "relative_mean_nodes_reduction": relative_reduction(base.get("mean_nodes_expanded"), enc.get("mean_nodes_expanded")),
        }
    return {
        "per_mode": per_mode,
        "per_dataset": per_dataset,
        "budget_curve": budget_curve,
        "solve_overlap": solve_overlap,
        "comparison": comp,
    }


def none_safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def relative_reduction(base: Optional[float], new: Optional[float]) -> Optional[float]:
    if base is None or new is None or base == 0:
        return None
    return float((base - new) / base)


def write_search_plots(out_dir: Path, summary: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable, skipping plots: {e}")
        return

    per_mode = pd.DataFrame(summary.get("per_mode", []))
    if not per_mode.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
        ax.bar(per_mode["search_mode"], per_mode["solve_rate"], color=[palette[i % len(palette)] for i in range(len(per_mode))])
        ax.set_ylabel("solve rate")
        ax.set_ylim(0, max(0.25, float(per_mode["solve_rate"].max()) * 1.25))
        ax.set_title("Solve rate by search mode")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(out_dir / "solve_rate_by_mode.png", dpi=160)
        plt.close(fig)

    curve = pd.DataFrame(summary.get("budget_curve", []))
    if not curve.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode, g in curve.groupby("search_mode"):
            g = g.sort_values("budget_tactics")
            ax.plot(g["budget_tactics"], g["solve_rate_at_budget"], marker="o", label=mode)
        ax.set_xlabel("executed tactic budget")
        ax.set_ylabel("solve rate at budget")
        ax.set_title("Budget curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "budget_curve.png", dpi=160)
        plt.close(fig)

    per_dataset = pd.DataFrame(summary.get("per_dataset", []))
    if not per_dataset.empty:
        pivot = per_dataset.pivot(index="dataset_name", columns="search_mode", values="solve_rate")
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("solve rate")
        ax.set_title("Solve rate by dataset")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(out_dir / "solve_rate_by_dataset.png", dpi=160)
        plt.close(fig)


def write_comparison_report(path: Path, summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    lines = [
        f"# Search evaluation report: {config['run_name']}",
        "",
        "## Per-mode metrics",
        "",
    ]
    for row in summary.get("per_mode", []):
        lines.append(f"### {row['search_mode']}")
        for k, v in row.items():
            if k != "search_mode":
                lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("## Comparison")
    lines.append("")
    for k, v in summary.get("comparison", {}).items():
        lines.append(f"- {k}: {v}")
    if summary.get("solve_overlap"):
        lines.extend(["", "## Solve overlap", ""])
        for row in summary["solve_overlap"]:
            lines.append(
                "- solved_by_both={solved_by_both}, baseline_only={baseline_only}, "
                "encoder_only={encoder_only}, net_new_solves={net_new_solves}".format(**row)
            )
    if summary.get("per_dataset"):
        lines.extend(["", "## Per-dataset metrics", ""])
        for row in summary["per_dataset"]:
            lines.append(
                f"- {row['dataset_name']} / {row['search_mode']}: "
                f"solve_rate={row['solve_rate']}, num_solved={row['num_solved']}/{row['num_theorems']}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--offline-only", action="store_true")
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--config-only", action="store_true")
    p.add_argument("--max-theorems", type=int, default=None)
    p.add_argument("--encoder-checkpoint", type=str, default=None)
    p.add_argument("--tuple-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--encoder-device", type=str, choices=["cuda", "cpu"], default=None)
    p.add_argument("--max-offline-rows", type=int, default=None)
    p.add_argument("--search-budget-tactics", type=int, default=None)
    p.add_argument("--search-budget-nodes", type=int, default=None)
    p.add_argument("--overwrite-output-dir", action="store_true")
    p.add_argument("--skip-encoder", action="store_true")
    return p.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.encoder_checkpoint:
        cfg["inputs"]["encoder_checkpoint"] = args.encoder_checkpoint
    if args.tuple_dir:
        cfg["offline_encoder_eval"]["tuple_parquet_dir"] = args.tuple_dir
    if args.output_dir:
        cfg["output"]["save_dir"] = args.output_dir
    if args.encoder_device:
        cfg["encoder_scoring"]["device"] = args.encoder_device
    if args.max_offline_rows is not None:
        cfg["offline_encoder_eval"]["max_rows"] = int(args.max_offline_rows)
    if args.search_budget_tactics is not None:
        cfg["search_budget"]["max_tactics_executed"] = int(args.search_budget_tactics)
    if args.search_budget_nodes is not None:
        cfg["search_budget"]["max_nodes_expanded"] = int(args.search_budget_nodes)
    if args.skip_encoder:
        cfg["offline_encoder_eval"]["enabled"] = False
        for m in cfg["search_modes"]:
            if m.get("use_encoder"):
                m["enabled"] = False
    return cfg


def prepare_output_dir(save_dir: str | Path, overwrite: bool, config_only: bool) -> Path:
    out_dir = Path(save_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not config_only:
        if overwrite:
            shutil.rmtree(out_dir)
        elif CONFIG["output"].get("fail_if_output_dir_nonempty", True):
            raise FileExistsError(
                f"Output directory is not empty: {out_dir}. "
                "Pass --overwrite-output-dir to replace old search artifacts."
            )
    return ensure_dir(out_dir)


def config_only_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    eval_sets = []
    for spec in config["inputs"]["eval_sets"]:
        p = Path(spec["path"])
        eval_sets.append({
            "name": spec["name"],
            "path": str(p),
            "exists": p.exists(),
            "enabled": spec.get("enabled", True),
            "max_theorems": spec.get("max_theorems"),
        })
    tuple_dir = Path(config["offline_encoder_eval"]["tuple_parquet_dir"])
    checkpoint = Path(config["inputs"]["encoder_checkpoint"])
    split_meta = Path(config["offline_encoder_eval"]["split_metadata_path"])
    return {
        "run_name": config["run_name"],
        "seed": config["seed"],
        "tuple_parquet_dir": str(tuple_dir),
        "tuple_parquet_dir_exists": tuple_dir.exists(),
        "encoder_checkpoint": str(checkpoint),
        "encoder_checkpoint_exists": checkpoint.exists(),
        "encoder_split_metadata": str(split_meta),
        "encoder_split_metadata_exists": split_meta.exists(),
        "eval_sets": eval_sets,
        "enabled_search_modes": [m["name"] for m in config["search_modes"] if m.get("enabled", True)],
        "search_budget": config["search_budget"],
        "budget_curve_points": config["metrics"].get("budget_curve_points", []),
    }


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(CONFIG, args)
    set_seed(int(config["seed"]))
    out_dir = prepare_output_dir(config["output"]["save_dir"], args.overwrite_output_dir, args.config_only)
    write_json(out_dir / "run_config.json", config)

    if args.config_only:
        summary = config_only_summary(config)
        write_json(out_dir / "config_only_summary.json", summary)
        print("[config_only]", json.dumps(summary, ensure_ascii=False, indent=2))
        return

    encoder = None
    if not args.skip_encoder and not args.search_only and config["offline_encoder_eval"].get("enabled", True):
        encoder = EncoderAdapter(config)
        run_offline_encoder_eval(config, encoder, out_dir)

    if args.offline_only:
        return

    generator = GeneratorAdapter(config)
    if not args.skip_encoder and encoder is None:
        encoder = EncoderAdapter(config)

    summary = run_search_eval(config, generator, encoder, out_dir, args.max_theorems)
    write_json(out_dir / "run_summary.json", summary)
    print(f"[done] outputs written to {out_dir}")


if __name__ == "__main__":
    main()

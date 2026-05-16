import os
import re
import gc
import sys
import json
import math
import time
import random
import hashlib
import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel, get_linear_schedule_with_warmup


CONFIG = {
    "run_name": "transition_encoder_v1_no_outcome_contrastive",
    "seed": 17,

    "data": {
        "tuple_dir": "data/tuples/full_v1/parquet",
        "output_dir": "outputs/encoder_runs/transition_encoder_v1",
        "split_by": "theorem_id",
        "train_ratio": 0.90,
        "valid_ratio": 0.05,
        "test_ratio": 0.05,
        "limit_rows": None,
        "drop_duplicate_state_tactic": True,
        "drop_duplicate_exact_tuple": True,
        "keep_result_types": ["TacticState", "ProofFinished", "LeanError", "Timeout", "ProofGivenUp"],
        "required_columns": [
            "theorem_id", "theorem_name", "state_pp", "tactic", "tactic_head", "sibling_group_id",
            "y", "result_type", "n_goals_before", "n_goals_after", "delta_n_goals", "m_closed",
            "m_new", "m_decreased", "c_err", "tactic_rank", "tactic_score", "state_hash",
            "next_state_hash", "is_duplicate_state_tactic", "is_duplicate_exact_tuple",
        ],
    },

    "preprocessing": {
        "main_input": "raw_state_plus_tactic",
        "input_format": "<STATE>\n{state_pp}\n\n<TACTIC>\n{tactic}\n",
        "normalize_whitespace": True,
        "max_state_chars": 12000,
        "max_tactic_chars": 300,
        "truncate_strategy": "head_tail",
        "state_head_ratio": 0.60,
        "state_tail_ratio": 0.40,
        "build_sibling_groups": True,
        "sibling_group_key": "sibling_group_id",
        "delta_bucket_scheme": [
            "invalid", "closed", "decreased", "same", "increased_by_1", "increased_by_2plus"
        ],
        "canonicalization": {
            "replace_main_input": False,
            "use_only_as_augmentation": True,
            "alpha_rename_variables": True,
            "rename_hypotheses": True,
            "rename_only_local_context": True,
            "rename_consistently_in_state_and_tactic": True,
            "do_not_rename_tactic_head": True,
            "do_not_rename_global_constants": True,
            "do_not_rename_namespaces": True,
            "do_not_rename_theorem_names": True,
        },
    },

    "model": {
        "name": "TransitionEncoderV1",
        "backbone": "google/byt5-small",
        "tokenizer": "google/byt5-small",
        "max_input_tokens": 2048,
        "pooling": "attention_pooling",
        "embedding_dim": 384,
        "normalize_embedding": True,
        "gradient_checkpointing": True,
        "projection_mlp": {"hidden_dim": 768, "num_layers": 2, "dropout": 0.10},
        "heads": {
            "validity": True,
            "result_type": True,
            "delta_bucket": True,
            "closure": True,
            "decomposition": True,
            "utility": True,
            "outcome_projection": False,
            "next_state_prediction": False,
        },
    },

    "batching": {
        "batch_unit": "sibling_group",
        "groups_per_batch": 2,
        "max_candidates_per_group": 12,
        "min_candidates_per_group": 2,
        "balance_valid_invalid": True,
        "ensure_positive_when_possible": True,
        "sample_high_utility_per_group": 2,
        "sample_positive_per_group": 2,
        "sample_negative_per_group": 2,
        "include_renamed_pairs": False,
        "renamed_pair_probability": 0.0,
        "num_workers": 0,
        "pin_memory": True,
    },

    "losses": {
        "validity_bce_weight": 1.0,
        "result_type_ce_weight": 0.5,
        "delta_bucket_ce_weight": 0.5,
        "closure_bce_weight": 0.3,
        "decomposition_bce_weight": 0.3,
        "sibling_margin_ranking_weight": 1.0,
        "ranking_margin": 0.25,
        "renaming_invariance_weight": 0.0,
        "outcome_infonce_weight": 0.0,
        "use_outcome_contrastive_loss": False,
        "max_ranking_pairs_per_group": 64,
    },

    "training": {
        "precision": "bf16",
        "device": "cuda",
        "epochs_total": 5,
        "optimizer": "adamw",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "gradient_clip_norm": 1.0,
        "gradient_accumulation_steps": 4,
        "eval_every_steps": 1000,
        "save_every_steps": 2000,
        "max_eval_batches": 200,
        "early_stopping_metric": "sibling_pairwise_accuracy",
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
    },

    "phases": [
        {
            "name": "supervised_warmup",
            "epochs": 1,
            "enabled_losses": [
                "validity_bce", "result_type_ce", "delta_bucket_ce", "closure_bce", "decomposition_bce"
            ],
            "include_renamed_pairs": False,
            "renamed_pair_probability": 0.0,
            "renaming_invariance_weight": 0.0,
        },
        {
            "name": "sibling_ranking",
            "epochs": 2,
            "enabled_losses": [
                "validity_bce", "result_type_ce", "delta_bucket_ce", "sibling_margin_ranking"
            ],
            "include_renamed_pairs": False,
            "renamed_pair_probability": 0.0,
            "renaming_invariance_weight": 0.0,
        },
        {
            "name": "renaming_invariance",
            "epochs": 2,
            "enabled_losses": [
                "validity_bce", "delta_bucket_ce", "sibling_margin_ranking", "renaming_invariance"
            ],
            "include_renamed_pairs": True,
            "renamed_pair_probability": 0.35,
            "renaming_invariance_weight": 0.2,
        },
    ],

    "evaluation": {
        "metrics": [
            "validity_auc", "result_type_accuracy", "delta_bucket_accuracy", "sibling_pairwise_accuracy",
            "proof_finished_top1_rate", "valid_tactic_topk_recall", "renaming_embedding_stability"
        ],
        "topk": [1, 3, 5, 8],
        "compare_against_generator_rank": True,
        "use_tactic_rank_baseline": True,
        "renaming_stability_sample_size": 512,
    },

    "output": {
        "save_dir": "outputs/encoder_runs/transition_encoder_v1",
        "save_best_checkpoint": True,
        "save_last_checkpoint": True,
        "save_tokenizer": True,
        "save_config": True,
        "save_metrics": True,
        "save_embedding_eval": True,
        "metrics_file": "metrics.json",
        "embedding_eval_file": "embedding_eval.json",
        "label_mappings_file": "label_mappings.json",
        "split_file": "split_index.json",
    },
}


RESULT_TYPES = ["TacticState", "ProofFinished", "LeanError", "Timeout", "ProofGivenUp", "Other"]
DELTA_BUCKETS = CONFIG["preprocessing"]["delta_bucket_scheme"]


def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha1_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()


def list_parquet_files(tuple_dir: str | Path) -> List[Path]:
    root = Path(tuple_dir)
    files = sorted(root.glob("*.parquet"))
    if not files and (root / "parquet").exists():
        files = sorted((root / "parquet").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {root} or {root / 'parquet'}")
    return files


def normalize_ws(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in s.split("\n")]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def truncate_head_tail(s: str, max_chars: int, head_ratio: float = 0.6, tail_ratio: float = 0.4) -> str:
    if len(s) <= max_chars:
        return s
    h = int(max_chars * head_ratio)
    t = max_chars - h
    return s[:h].rstrip() + "\n... <TRUNCATED> ...\n" + s[-t:].lstrip()


def clean_tactic(x: Any, max_chars: int) -> str:
    s = normalize_ws(x)
    s = s.split("\n")[0].strip()
    if s.startswith("by "):
        s = s[3:].strip()
    return s[:max_chars]


def make_input_text(state_pp: str, tactic: str) -> str:
    cfg = CONFIG["preprocessing"]
    return cfg["input_format"].format(state_pp=state_pp, tactic=tactic)


def result_type_to_label(x: Any) -> str:
    s = str(x) if x is not None else "Other"
    return s if s in RESULT_TYPES else "Other"


def delta_bucket(row: pd.Series) -> str:
    y = int(row.get("y", 0) or 0)
    if y == 0:
        return "invalid"
    if int(row.get("m_closed", 0) or 0) == 1 or result_type_to_label(row.get("result_type")) == "ProofFinished":
        return "closed"
    d = row.get("delta_n_goals")
    if pd.isna(d):
        return "invalid"
    d = int(d)
    if d < 0:
        return "decreased"
    if d == 0:
        return "same"
    if d == 1:
        return "increased_by_1"
    return "increased_by_2plus"


def utility_target(row: pd.Series) -> float:
    y = int(row.get("y", 0) or 0)
    result_type = result_type_to_label(row.get("result_type"))
    if y == 1 and result_type == "ProofFinished":
        return 4.0
    if y == 1 and int(row.get("m_decreased", 0) or 0) == 1:
        return 3.0
    if y == 1 and int(row.get("m_new", 0) or 0) == 1:
        return 2.0
    if y == 1:
        return 1.5
    if result_type == "Timeout":
        return -0.5
    return 0.0


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores)) + 1
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def check_cuda_or_fail() -> torch.device:
    if CONFIG["training"]["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this run is configured for A100/CUDA VM.")
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"[cuda] GPU={name}, capability={cap}, torch={torch.__version__}")
        return torch.device("cuda")
    return torch.device("cpu")


LEAN_KEYWORDS = {
    "theorem", "lemma", "def", "by", "let", "have", "show", "fun", "match", "with", "if", "then", "else",
    "Type", "Prop", "Sort", "where", "case", "of", "in", "namespace", "section", "variable", "variables",
}


def build_local_rename_map(state_pp: str) -> Dict[str, str]:
    before_goal = state_pp.split("⊢", 1)[0]
    mapping = {}
    var_i = 1
    hyp_i = 1

    for line in before_goal.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        left = line.split(":", 1)[0].strip()
        if left.startswith("case ") or "." in left:
            continue
        names = re.findall(r"\b[A-Za-z_][A-Za-z0-9_']*\b", left)
        for name in names:
            if name in LEAN_KEYWORDS or name in mapping:
                continue
            if name[0].isupper():
                continue
            if name.startswith("h") or name.startswith("H"):
                mapping[name] = f"HYP_{hyp_i}"
                hyp_i += 1
            else:
                mapping[name] = f"VAR_{var_i}"
                var_i += 1
    return mapping


def replace_identifiers(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    out = text
    for old in sorted(mapping, key=len, reverse=True):
        new = mapping[old]
        pat = rf"(?<![A-Za-z0-9_']){re.escape(old)}(?![A-Za-z0-9_'])"
        out = re.sub(pat, new, out)
    return out


def make_renamed_input(state_pp: str, tactic: str) -> str:
    mapping = build_local_rename_map(state_pp)
    renamed_state = replace_identifiers(state_pp, mapping)
    renamed_tactic = replace_identifiers(tactic, mapping)
    return make_input_text(renamed_state, renamed_tactic)


def load_and_prepare_dataframe(tuple_dir: str | Path, limit_rows: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    files = list_parquet_files(tuple_dir)
    print(f"[data] loading {len(files)} parquet shards from {tuple_dir}")

    need = CONFIG["data"]["required_columns"]
    dfs = []
    remaining = limit_rows

    for f in files:
        df = pd.read_parquet(f)
        for c in need:
            if c not in df.columns:
                df[c] = None
        df = df[need]
        dfs.append(df)
        if remaining is not None:
            remaining -= len(df)
            if remaining <= 0:
                break

    df = pd.concat(dfs, ignore_index=True)
    if limit_rows is not None and len(df) > limit_rows:
        df = df.sample(n=limit_rows, random_state=CONFIG["seed"]).reset_index(drop=True)

    before = len(df)
    df["state_pp"] = df["state_pp"].map(normalize_ws)
    df["tactic"] = df["tactic"].map(lambda x: clean_tactic(x, CONFIG["preprocessing"]["max_tactic_chars"]))
    df = df[(df["state_pp"].str.len() > 0) & (df["tactic"].str.len() > 0)].copy()

    if CONFIG["data"]["keep_result_types"]:
        keep = set(CONFIG["data"]["keep_result_types"])
        df["result_type"] = df["result_type"].map(result_type_to_label)
        df = df[df["result_type"].isin(keep | {"Other"})].copy()

    if CONFIG["data"]["drop_duplicate_state_tactic"]:
        df = df.drop_duplicates(subset=["state_hash", "tactic"], keep="first")

    if CONFIG["data"]["drop_duplicate_exact_tuple"]:
        df = df.drop_duplicates(subset=["state_hash", "tactic", "next_state_hash", "c_err"], keep="first")

    p = CONFIG["preprocessing"]
    df["state_pp"] = df["state_pp"].map(
        lambda s: truncate_head_tail(s, p["max_state_chars"], p["state_head_ratio"], p["state_tail_ratio"])
    )
    df["input_text"] = [make_input_text(s, t) for s, t in zip(df["state_pp"], df["tactic"])]

    df["y"] = df["y"].fillna(0).astype(int).clip(0, 1)
    df["m_closed"] = df["m_closed"].fillna(0).astype(int).clip(0, 1)
    df["m_new"] = df["m_new"].fillna(0).astype(int).clip(0, 1)
    df["m_decreased"] = df["m_decreased"].fillna(0).astype(int).clip(0, 1)
    df["tactic_rank"] = pd.to_numeric(df["tactic_rank"], errors="coerce").fillna(999).astype(int)
    df["delta_bucket"] = df.apply(delta_bucket, axis=1)
    df["utility_target"] = df.apply(utility_target, axis=1)
    df["result_type_label"] = df["result_type"].map({x: i for i, x in enumerate(RESULT_TYPES)}).fillna(len(RESULT_TYPES) - 1).astype(int)
    df["delta_bucket_label"] = df["delta_bucket"].map({x: i for i, x in enumerate(DELTA_BUCKETS)}).astype(int)

    if df["sibling_group_id"].isna().any():
        missing = df["sibling_group_id"].isna()
        df.loc[missing, "sibling_group_id"] = [
            sha1_text(f"{a}:{b}")[:16] for a, b in zip(df.loc[missing, "theorem_id"], df.loc[missing, "state_hash"])
        ]

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    stats = {
        "loaded_rows_before_filter": before,
        "rows_after_filter": len(df),
        "num_theorems": int(df["theorem_id"].nunique()),
        "num_sibling_groups": int(df["sibling_group_id"].nunique()),
        "result_type_counts": df["result_type"].value_counts().to_dict(),
        "delta_bucket_counts": df["delta_bucket"].value_counts().to_dict(),
        "valid_rate": float(df["y"].mean()) if len(df) else 0.0,
    }
    print("[data]", json.dumps(stats, ensure_ascii=False, indent=2))
    return df, stats


def split_by_theorem(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    rng = random.Random(CONFIG["seed"])
    theorem_ids = list(df["theorem_id"].dropna().unique())
    rng.shuffle(theorem_ids)
    n = len(theorem_ids)
    n_train = int(n * CONFIG["data"]["train_ratio"])
    n_valid = int(n * CONFIG["data"]["valid_ratio"])
    train_ids = set(theorem_ids[:n_train])
    valid_ids = set(theorem_ids[n_train:n_train + n_valid])
    test_ids = set(theorem_ids[n_train + n_valid:])

    splits = {
        "train": df.index[df["theorem_id"].isin(train_ids)].to_numpy(),
        "valid": df.index[df["theorem_id"].isin(valid_ids)].to_numpy(),
        "test": df.index[df["theorem_id"].isin(test_ids)].to_numpy(),
    }
    print("[split]", {k: int(len(v)) for k, v in splits.items()})
    return splits


class SiblingGroupDataset(Dataset):
    def __init__(self, df: pd.DataFrame, indices: np.ndarray, min_candidates_per_group: int = 2):
        sub = df.iloc[indices]
        groups = defaultdict(list)
        for idx, gid in zip(sub.index, sub["sibling_group_id"]):
            groups[str(gid)].append(int(idx))
        self.groups = [v for v in groups.values() if len(v) >= min_candidates_per_group]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> List[int]:
        return self.groups[idx]


class SiblingCollator:
    def __init__(self, df: pd.DataFrame, tokenizer: Any, train: bool, phase_cfg: Dict[str, Any]):
        self.df = df
        self.tokenizer = tokenizer
        self.train = train
        self.phase_cfg = phase_cfg
        self.max_candidates = CONFIG["batching"]["max_candidates_per_group"]
        self.max_input_tokens = CONFIG["model"]["max_input_tokens"]
        self.rng = random.Random(CONFIG["seed"] + (13 if train else 29))

    def _sample_group(self, indices: List[int]) -> List[int]:
        if len(indices) <= self.max_candidates:
            return indices

        rows = self.df.loc[indices]
        selected: List[int] = []

        def add_many(xs: List[int]) -> None:
            for x in xs:
                if x not in selected and len(selected) < self.max_candidates:
                    selected.append(int(x))

        high_n = int(CONFIG["batching"].get("sample_high_utility_per_group", 2))
        pos_n = int(CONFIG["batching"].get("sample_positive_per_group", 2))
        neg_n = int(CONFIG["batching"].get("sample_negative_per_group", 2))

        high = rows.sort_values(["utility_target", "y", "tactic_rank"], ascending=[False, False, True]).index[:high_n].tolist()
        add_many(high)

        if CONFIG["batching"].get("ensure_positive_when_possible", True):
            positives = rows[rows["y"] == 1].sort_values(["utility_target", "tactic_rank"], ascending=[False, True]).index[:pos_n].tolist()
            add_many(positives)

        if CONFIG["batching"].get("balance_valid_invalid", True):
            negatives = rows[rows["y"] == 0].sort_values(["tactic_rank"], ascending=[True]).index[:neg_n].tolist()
            add_many(negatives)

        rest = [int(i) for i in indices if int(i) not in set(selected)]
        self.rng.shuffle(rest)
        add_many(rest)
        return selected

    def __call__(self, batch_groups: List[List[int]]) -> Dict[str, Any]:
        selected = []
        group_ids = []
        for gidx, indices in enumerate(batch_groups):
            chosen = self._sample_group(indices)
            selected.extend(chosen)
            group_ids.extend([gidx] * len(chosen))

        rows = self.df.loc[selected]
        texts = rows["input_text"].tolist()
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )

        batch = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "group_ids": torch.tensor(group_ids, dtype=torch.long),
            "row_ids": torch.tensor(rows["row_id"].to_numpy(), dtype=torch.long),
            "y": torch.tensor(rows["y"].to_numpy(), dtype=torch.float32),
            "result_type_label": torch.tensor(rows["result_type_label"].to_numpy(), dtype=torch.long),
            "delta_bucket_label": torch.tensor(rows["delta_bucket_label"].to_numpy(), dtype=torch.long),
            "m_closed": torch.tensor(rows["m_closed"].to_numpy(), dtype=torch.float32),
            "m_new": torch.tensor(rows["m_new"].to_numpy(), dtype=torch.float32),
            "utility_target": torch.tensor(rows["utility_target"].to_numpy(), dtype=torch.float32),
            "tactic_rank": torch.tensor(rows["tactic_rank"].to_numpy(), dtype=torch.long),
            "result_type_str": rows["result_type"].tolist(),
            "sibling_group_id": rows["sibling_group_id"].tolist(),
        }

        include_aug = self.phase_cfg.get("include_renamed_pairs", False)
        prob = float(self.phase_cfg.get("renamed_pair_probability", 0.0))
        aug_texts = []
        aug_source_positions = []
        if self.train and include_aug and prob > 0:
            for pos, (_, r) in enumerate(rows.iterrows()):
                if self.rng.random() <= prob:
                    aug_texts.append(make_renamed_input(r["state_pp"], r["tactic"]))
                    aug_source_positions.append(pos)

        if aug_texts:
            aug = self.tokenizer(
                aug_texts,
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            batch["aug_input_ids"] = aug["input_ids"]
            batch["aug_attention_mask"] = aug["attention_mask"]
            batch["aug_source_positions"] = torch.tensor(aug_source_positions, dtype=torch.long)
        else:
            batch["aug_input_ids"] = None
            batch["aug_attention_mask"] = None
            batch["aug_source_positions"] = None

        return batch


class TransitionEncoderV1(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_result_types: int, num_delta_buckets: int):
        super().__init__()
        self.cfg = cfg
        self.backbone = T5EncoderModel.from_pretrained(cfg["backbone"])
        if cfg.get("gradient_checkpointing", False):
            self.backbone.gradient_checkpointing_enable()
        hidden = self.backbone.config.d_model
        emb_dim = cfg["embedding_dim"]
        mlp_hidden = cfg["projection_mlp"]["hidden_dim"]
        dropout = cfg["projection_mlp"]["dropout"]

        self.attn_pool = nn.Linear(hidden, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, emb_dim),
        )
        self.validity_head = nn.Linear(emb_dim, 1)
        self.result_type_head = nn.Linear(emb_dim, num_result_types)
        self.delta_bucket_head = nn.Linear(emb_dim, num_delta_buckets)
        self.closure_head = nn.Linear(emb_dim, 1)
        self.decomposition_head = nn.Linear(emb_dim, 1)
        self.utility_head = nn.Linear(emb_dim, 1)

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn_pool(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        z = self.proj(pooled)
        if self.cfg.get("normalize_embedding", True):
            z = F.normalize(z, p=2, dim=-1)
        return {
            "embedding": z,
            "validity_logits": self.validity_head(z).squeeze(-1),
            "result_type_logits": self.result_type_head(z),
            "delta_bucket_logits": self.delta_bucket_head(z),
            "closure_logits": self.closure_head(z).squeeze(-1),
            "decomposition_logits": self.decomposition_head(z).squeeze(-1),
            "utility_logits": self.utility_head(z).squeeze(-1),
        }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def sibling_ranking_loss(pred: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor, margin: float) -> torch.Tensor:
    losses = []
    unique = torch.unique(group_ids)
    max_pairs = CONFIG["losses"].get("max_ranking_pairs_per_group", 64)
    for gid in unique:
        idx = torch.where(group_ids == gid)[0]
        if idx.numel() < 2:
            continue
        p = pred[idx]
        t = target[idx]
        better = (t.unsqueeze(1) - t.unsqueeze(0)) > 0.25
        pairs = torch.nonzero(better, as_tuple=False)
        if pairs.numel() == 0:
            continue
        if pairs.size(0) > max_pairs:
            perm = torch.randperm(pairs.size(0), device=pairs.device)[:max_pairs]
            pairs = pairs[perm]
        i = pairs[:, 0]
        j = pairs[:, 1]
        losses.append(F.relu(margin - (p[i] - p[j])).mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def compute_losses(
    model: TransitionEncoderV1,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    enabled_losses: List[str],
    phase_cfg: Dict[str, Any],
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = CONFIG["losses"].copy()
    if "renaming_invariance_weight" in phase_cfg:
        weights["renaming_invariance_weight"] = phase_cfg["renaming_invariance_weight"]

    total = outputs["utility_logits"].sum() * 0.0
    logs = {}

    if "validity_bce" in enabled_losses:
        loss = F.binary_cross_entropy_with_logits(outputs["validity_logits"], batch["y"])
        total = total + weights["validity_bce_weight"] * loss
        logs["loss_validity_bce"] = float(loss.detach().cpu())

    if "result_type_ce" in enabled_losses:
        loss = F.cross_entropy(outputs["result_type_logits"], batch["result_type_label"])
        total = total + weights["result_type_ce_weight"] * loss
        logs["loss_result_type_ce"] = float(loss.detach().cpu())

    if "delta_bucket_ce" in enabled_losses:
        loss = F.cross_entropy(outputs["delta_bucket_logits"], batch["delta_bucket_label"])
        total = total + weights["delta_bucket_ce_weight"] * loss
        logs["loss_delta_bucket_ce"] = float(loss.detach().cpu())

    if "closure_bce" in enabled_losses:
        loss = F.binary_cross_entropy_with_logits(outputs["closure_logits"], batch["m_closed"])
        total = total + weights["closure_bce_weight"] * loss
        logs["loss_closure_bce"] = float(loss.detach().cpu())

    if "decomposition_bce" in enabled_losses:
        loss = F.binary_cross_entropy_with_logits(outputs["decomposition_logits"], batch["m_new"])
        total = total + weights["decomposition_bce_weight"] * loss
        logs["loss_decomposition_bce"] = float(loss.detach().cpu())

    if "sibling_margin_ranking" in enabled_losses:
        loss = sibling_ranking_loss(
            outputs["utility_logits"],
            batch["utility_target"],
            batch["group_ids"],
            CONFIG["losses"]["ranking_margin"],
        )
        total = total + weights["sibling_margin_ranking_weight"] * loss
        logs["loss_sibling_margin_ranking"] = float(loss.detach().cpu())

    if "renaming_invariance" in enabled_losses and batch.get("aug_input_ids") is not None:
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            aug_outputs = model(batch["aug_input_ids"], batch["aug_attention_mask"])
        src = batch["aug_source_positions"]
        raw_z = outputs["embedding"][src]
        aug_z = aug_outputs["embedding"]
        loss = (1.0 - F.cosine_similarity(raw_z, aug_z, dim=-1)).mean()
        total = total + weights["renaming_invariance_weight"] * loss
        logs["loss_renaming_invariance"] = float(loss.detach().cpu())

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs


def pairwise_group_accuracy(pred: np.ndarray, target: np.ndarray, group_ids: np.ndarray) -> Optional[float]:
    correct = 0
    total = 0
    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        if len(idx) < 2:
            continue
        p = pred[idx]
        t = target[idx]
        for i in range(len(idx)):
            for j in range(len(idx)):
                if t[i] > t[j] + 0.25:
                    correct += int(p[i] > p[j])
                    total += 1
    return None if total == 0 else correct / total


def topk_recall_by_group(pred: np.ndarray, labels: np.ndarray, group_ids: np.ndarray, k: int) -> Optional[float]:
    hits = 0
    total = 0
    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        if labels[idx].sum() <= 0:
            continue
        order = idx[np.argsort(-pred[idx])]
        hits += int(labels[order[:k]].sum() > 0)
        total += 1
    return None if total == 0 else hits / total


def proof_finished_top1(pred: np.ndarray, result_type: List[str], group_ids: np.ndarray) -> Optional[float]:
    result_type = np.asarray(result_type)
    hits = 0
    total = 0
    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        if not np.any(result_type[idx] == "ProofFinished"):
            continue
        best = idx[np.argmax(pred[idx])]
        hits += int(result_type[best] == "ProofFinished")
        total += 1
    return None if total == 0 else hits / total


def generator_rank_topk_recall(labels: np.ndarray, ranks: np.ndarray, group_ids: np.ndarray, k: int) -> Optional[float]:
    hits = 0
    total = 0
    for gid in np.unique(group_ids):
        idx = np.where(group_ids == gid)[0]
        if labels[idx].sum() <= 0:
            continue
        order = idx[np.argsort(ranks[idx])]
        hits += int(labels[order[:k]].sum() > 0)
        total += 1
    return None if total == 0 else hits / total


@torch.no_grad()
def evaluate(
    model: TransitionEncoderV1,
    loader: DataLoader,
    device: torch.device,
    phase_name: str,
    autocast_dtype: torch.dtype,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    all_valid_scores, all_y = [], []
    all_result_pred, all_result_true = [], []
    all_delta_pred, all_delta_true = [], []
    all_utility, all_util_target, all_group_ids, all_ranks = [], [], [], []
    all_result_type_str = []
    loss_vals = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            out = model(batch["input_ids"], batch["attention_mask"])
            loss, _ = compute_losses(model, out, batch, ["validity_bce", "result_type_ce", "delta_bucket_ce", "sibling_margin_ranking"], {}, device, autocast_dtype)
        loss_vals.append(float(loss.detach().cpu()))

        all_valid_scores.append(torch.sigmoid(out["validity_logits"]).detach().float().cpu().numpy())
        all_y.append(batch["y"].detach().float().cpu().numpy())
        all_result_pred.append(out["result_type_logits"].argmax(-1).detach().cpu().numpy())
        all_result_true.append(batch["result_type_label"].detach().cpu().numpy())
        all_delta_pred.append(out["delta_bucket_logits"].argmax(-1).detach().cpu().numpy())
        all_delta_true.append(batch["delta_bucket_label"].detach().cpu().numpy())
        all_utility.append(out["utility_logits"].detach().float().cpu().numpy())
        all_util_target.append(batch["utility_target"].detach().float().cpu().numpy())
        all_group_ids.append(batch["group_ids"].detach().cpu().numpy() + bi * 1000000)
        all_ranks.append(batch["tactic_rank"].detach().cpu().numpy())
        all_result_type_str.extend(batch["result_type_str"])

    if not all_y:
        return {"phase": phase_name, "error": "empty_eval"}

    y = np.concatenate(all_y)
    valid_scores = np.concatenate(all_valid_scores)
    result_pred = np.concatenate(all_result_pred)
    result_true = np.concatenate(all_result_true)
    delta_pred = np.concatenate(all_delta_pred)
    delta_true = np.concatenate(all_delta_true)
    utility = np.concatenate(all_utility)
    util_target = np.concatenate(all_util_target)
    group_ids = np.concatenate(all_group_ids)
    ranks = np.concatenate(all_ranks)

    metrics = {
        "phase": phase_name,
        "eval_loss": float(np.mean(loss_vals)),
        "validity_auc": binary_auc(y, valid_scores),
        "result_type_accuracy": float((result_pred == result_true).mean()),
        "delta_bucket_accuracy": float((delta_pred == delta_true).mean()),
        "sibling_pairwise_accuracy": pairwise_group_accuracy(utility, util_target, group_ids),
        "proof_finished_top1_rate": proof_finished_top1(utility, all_result_type_str, group_ids),
    }

    for k in CONFIG["evaluation"]["topk"]:
        metrics[f"valid_tactic_recall_at_{k}"] = topk_recall_by_group(utility, y, group_ids, k)
        if CONFIG["evaluation"].get("use_tactic_rank_baseline", True):
            metrics[f"generator_rank_valid_recall_at_{k}"] = generator_rank_topk_recall(y, ranks, group_ids, k)

    return metrics


@torch.no_grad()
def compute_renaming_embedding_stability(
    model: TransitionEncoderV1,
    df: pd.DataFrame,
    indices: np.ndarray,
    tokenizer: Any,
    device: torch.device,
    autocast_dtype: torch.dtype,
    sample_size: Optional[int] = 512,
    batch_size: int = 16,
) -> Dict[str, Any]:
    model.eval()
    idxs = [int(i) for i in indices if int(i) in df.index]
    if not idxs:
        return {"renaming_embedding_stability": None, "renaming_embedding_stability_n": 0}

    rng = np.random.default_rng(CONFIG["seed"] + 101)
    if sample_size is not None and len(idxs) > sample_size:
        idxs = rng.choice(idxs, size=sample_size, replace=False).tolist()

    sims = []
    for start in range(0, len(idxs), batch_size):
        batch_ids = idxs[start:start + batch_size]
        rows = df.loc[batch_ids]
        raw_texts = rows["input_text"].tolist()
        renamed_texts = [make_renamed_input(r["state_pp"], r["tactic"]) for _, r in rows.iterrows()]

        raw = tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            max_length=CONFIG["model"]["max_input_tokens"],
            return_tensors="pt",
        )
        renamed = tokenizer(
            renamed_texts,
            padding=True,
            truncation=True,
            max_length=CONFIG["model"]["max_input_tokens"],
            return_tensors="pt",
        )
        raw = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
        renamed = {k: v.to(device, non_blocking=True) for k, v in renamed.items()}

        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            z_raw = model(raw["input_ids"], raw["attention_mask"])["embedding"]
            z_renamed = model(renamed["input_ids"], renamed["attention_mask"])["embedding"]
        sim = F.cosine_similarity(z_raw, z_renamed, dim=-1).detach().float().cpu().numpy()
        sims.extend(sim.tolist())

    if not sims:
        return {"renaming_embedding_stability": None, "renaming_embedding_stability_n": 0}
    arr = np.asarray(sims, dtype=np.float64)
    return {
        "renaming_embedding_stability": float(arr.mean()),
        "renaming_embedding_stability_std": float(arr.std()),
        "renaming_embedding_stability_n": int(arr.size),
    }


def save_checkpoint(
    path: str | Path,
    model: TransitionEncoderV1,
    optimizer: torch.optim.Optimizer,
    label_mappings: Dict[str, Any],
    global_step: int,
    phase_name: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": CONFIG,
        "label_mappings": label_mappings,
        "global_step": global_step,
        "phase_name": phase_name,
        "metrics": metrics or {},
        "saved_at": now_iso(),
    }, path)


def train_phase(
    model: TransitionEncoderV1,
    tokenizer: Any,
    df: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    label_mappings: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    global_step: int,
    metrics_log: List[Dict[str, Any]],
    best_metric: float,
) -> Tuple[int, float]:
    phase_name = phase_cfg["name"]
    epochs = int(phase_cfg["epochs"])
    enabled_losses = phase_cfg["enabled_losses"]
    autocast_dtype = torch.bfloat16 if CONFIG["training"]["precision"] == "bf16" else torch.float16

    train_ds = SiblingGroupDataset(df, train_indices, CONFIG["batching"]["min_candidates_per_group"])
    valid_ds = SiblingGroupDataset(df, valid_indices, CONFIG["batching"]["min_candidates_per_group"])

    train_collator = SiblingCollator(df, tokenizer, train=True, phase_cfg=phase_cfg)
    valid_collator = SiblingCollator(df, tokenizer, train=False, phase_cfg={})

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batching"]["groups_per_batch"],
        shuffle=True,
        num_workers=CONFIG["batching"]["num_workers"],
        pin_memory=CONFIG["batching"]["pin_memory"],
        collate_fn=train_collator,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CONFIG["batching"]["groups_per_batch"],
        shuffle=False,
        num_workers=0,
        pin_memory=CONFIG["batching"]["pin_memory"],
        collate_fn=valid_collator,
        drop_last=False,
    )

    num_update_steps = max(1, math.ceil(len(train_loader) * epochs / CONFIG["training"]["gradient_accumulation_steps"]))
    num_warmup_steps = int(num_update_steps * CONFIG["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_update_steps)

    print(f"[phase] {phase_name}: groups={len(train_ds)}, batches={len(train_loader)}, epochs={epochs}, losses={enabled_losses}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum = CONFIG["training"]["gradient_accumulation_steps"]
    running = Counter()
    running_n = 0
    metric_name = CONFIG["training"].get("early_stopping_metric", "sibling_pairwise_accuracy")
    patience = CONFIG["training"].get("early_stopping_patience")
    min_delta = float(CONFIG["training"].get("early_stopping_min_delta", 0.0))
    phase_best_metric = -float("inf")
    bad_evals = 0
    stop_phase = False

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss, loss_logs = compute_losses(model, outputs, batch, enabled_losses, phase_cfg, device, autocast_dtype)
                loss = loss / accum

            loss.backward()
            running_n += 1
            for k, v in loss_logs.items():
                running[k] += v

            should_update = ((step + 1) % accum == 0) or ((step + 1) == len(train_loader))
            if should_update:
                if CONFIG["training"]["gradient_clip_norm"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["training"]["gradient_clip_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    msg = {k: round(v / max(running_n, 1), 4) for k, v in running.items()}
                    print(f"[train] phase={phase_name} epoch={epoch} step={global_step} {json.dumps(msg)}")
                    running = Counter()
                    running_n = 0

                if global_step % CONFIG["training"]["eval_every_steps"] == 0:
                    metrics = evaluate(
                        model, valid_loader, device, phase_name, autocast_dtype,
                        max_batches=CONFIG["training"]["max_eval_batches"],
                    )
                    if "renaming_embedding_stability" in CONFIG["evaluation"].get("metrics", []):
                        metrics.update(compute_renaming_embedding_stability(
                            model, df, valid_indices, tokenizer, device, autocast_dtype,
                            sample_size=CONFIG["evaluation"].get("renaming_stability_sample_size", 512),
                        ))
                    metrics["global_step"] = global_step
                    metrics["created_at"] = now_iso()
                    metrics_log.append(metrics)
                    save_json(output_dir / CONFIG["output"]["metrics_file"], metrics_log)
                    print("[eval]", json.dumps(metrics, ensure_ascii=False, indent=2))

                    metric = metrics.get(metric_name)
                    if metric is not None and metric > best_metric:
                        best_metric = float(metric)
                        if CONFIG["output"]["save_best_checkpoint"]:
                            save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, label_mappings, global_step, phase_name, metrics)
                            print(f"[save] new best checkpoint, {metric_name}={best_metric:.4f}")

                    if metric is not None:
                        if metric > phase_best_metric + min_delta:
                            phase_best_metric = float(metric)
                            bad_evals = 0
                        else:
                            bad_evals += 1
                            if patience is not None and bad_evals >= int(patience):
                                print(f"[early_stop] phase={phase_name} metric={metric_name} patience={patience}")
                                stop_phase = True
                    model.train()
                    if stop_phase:
                        break

                if global_step % CONFIG["training"]["save_every_steps"] == 0:
                    save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, label_mappings, global_step, phase_name, None)

            if stop_phase:
                break
        if stop_phase:
            break

    metrics = evaluate(
        model, valid_loader, device, phase_name, autocast_dtype,
        max_batches=CONFIG["training"]["max_eval_batches"],
    )
    if "renaming_embedding_stability" in CONFIG["evaluation"].get("metrics", []):
        metrics.update(compute_renaming_embedding_stability(
            model, df, valid_indices, tokenizer, device, autocast_dtype,
            sample_size=CONFIG["evaluation"].get("renaming_stability_sample_size", 512),
        ))
    metrics["global_step"] = global_step
    metrics["created_at"] = now_iso()
    metrics_log.append(metrics)
    save_json(output_dir / CONFIG["output"]["metrics_file"], metrics_log)
    print("[eval:phase_end]", json.dumps(metrics, ensure_ascii=False, indent=2))

    metric = metrics.get(metric_name)
    if metric is not None and metric > best_metric:
        best_metric = float(metric)
        save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, label_mappings, global_step, phase_name, metrics)

    save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, label_mappings, global_step, phase_name, metrics)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return global_step, best_metric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tuple-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--groups-per-batch", type=int, default=None)
    p.add_argument("--max-candidates-per-group", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    if args.tuple_dir:
        CONFIG["data"]["tuple_dir"] = args.tuple_dir
    if args.output_dir:
        CONFIG["data"]["output_dir"] = args.output_dir
        CONFIG["output"]["save_dir"] = args.output_dir
    if args.limit_rows is not None:
        CONFIG["data"]["limit_rows"] = args.limit_rows
    if args.backbone:
        CONFIG["model"]["backbone"] = args.backbone
        CONFIG["model"]["tokenizer"] = args.backbone
    if args.max_input_tokens:
        CONFIG["model"]["max_input_tokens"] = args.max_input_tokens
    if args.groups_per_batch:
        CONFIG["batching"]["groups_per_batch"] = args.groups_per_batch
    if args.max_candidates_per_group:
        CONFIG["batching"]["max_candidates_per_group"] = args.max_candidates_per_group
    if args.learning_rate:
        CONFIG["training"]["learning_rate"] = args.learning_rate


def main() -> None:
    args = parse_args()
    apply_args(args)
    set_seed(CONFIG["seed"])
    device = check_cuda_or_fail()

    output_dir = ensure_dir(CONFIG["output"]["save_dir"])
    save_json(output_dir / "train_config.json", {"config": CONFIG, "argv": sys.argv, "created_at": now_iso()})

    df, data_stats = load_and_prepare_dataframe(CONFIG["data"]["tuple_dir"], CONFIG["data"].get("limit_rows"))
    splits = split_by_theorem(df)
    split_payload = {k: df.iloc[v]["row_id"].astype(int).tolist() for k, v in splits.items()}
    save_json(output_dir / CONFIG["output"]["split_file"], split_payload)
    save_json(output_dir / "processed_stats.json", data_stats)

    if args.dry_run:
        print("[dry_run] stopping after preprocessing")
        return

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["tokenizer"])
    model = TransitionEncoderV1(CONFIG["model"], len(RESULT_TYPES), len(DELTA_BUCKETS)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )

    label_mappings = {
        "result_type_to_id": {x: i for i, x in enumerate(RESULT_TYPES)},
        "id_to_result_type": {i: x for i, x in enumerate(RESULT_TYPES)},
        "delta_bucket_to_id": {x: i for i, x in enumerate(DELTA_BUCKETS)},
        "id_to_delta_bucket": {i: x for i, x in enumerate(DELTA_BUCKETS)},
        "utility_target_semantics": {
            "4.0": "proof finished",
            "3.0": "valid and decreased number of goals",
            "2.0": "valid and created new subgoals / decomposition",
            "1.5": "valid but non-closed/non-decreased transition",
            "0.0": "Lean error / invalid tactic",
            "-0.5": "timeout",
        },
    }
    save_json(output_dir / CONFIG["output"]["label_mappings_file"], label_mappings)

    if CONFIG["output"]["save_tokenizer"]:
        tokenizer.save_pretrained(output_dir / "tokenizer")

    metrics_log = []
    global_step = 0
    best_metric = -1.0

    for phase in CONFIG["phases"]:
        global_step, best_metric = train_phase(
            model=model,
            tokenizer=tokenizer,
            df=df,
            train_indices=splits["train"],
            valid_indices=splits["valid"],
            optimizer=optimizer,
            label_mappings=label_mappings,
            phase_cfg=phase,
            device=device,
            output_dir=output_dir,
            global_step=global_step,
            metrics_log=metrics_log,
            best_metric=best_metric,
        )

    test_ds = SiblingGroupDataset(df, splits["test"], CONFIG["batching"]["min_candidates_per_group"])
    tokenizer = AutoTokenizer.from_pretrained(output_dir / "tokenizer") if (output_dir / "tokenizer").exists() else tokenizer
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batching"]["groups_per_batch"],
        shuffle=False,
        num_workers=0,
        pin_memory=CONFIG["batching"]["pin_memory"],
        collate_fn=SiblingCollator(df, tokenizer, train=False, phase_cfg={}),
    )
    autocast_dtype = torch.bfloat16 if CONFIG["training"]["precision"] == "bf16" else torch.float16
    test_metrics = evaluate(model, test_loader, device, "test", autocast_dtype, max_batches=None)
    if "renaming_embedding_stability" in CONFIG["evaluation"].get("metrics", []):
        test_metrics.update(compute_renaming_embedding_stability(
            model, df, splits["test"], tokenizer, device, autocast_dtype,
            sample_size=CONFIG["evaluation"].get("renaming_stability_sample_size", 512),
        ))
    test_metrics["global_step"] = global_step
    test_metrics["created_at"] = now_iso()
    save_json(output_dir / CONFIG["output"]["embedding_eval_file"], test_metrics)
    save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, label_mappings, global_step, "final", test_metrics)
    print("[final:test]", json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

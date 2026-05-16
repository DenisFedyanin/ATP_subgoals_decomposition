### Lean theorem

```lean
theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by
  sorry
```

###  proof trajectory

```lean
rw [hx]
norm_num
```


## Proof states

```lean
x : ℕ
hx : x = 2 ^ 9 + 1
⊢ x = 513
```

```lean
x : ℕ
hx : x = 2 ^ 9 + 1
⊢ 2 ^ 9 + 1 = 513
```

```lean
no goals
```

---
Все tuples ниже принадлежат одной sibling-группе:

```json
{
  "tuple_id": "run01_lw101_s0_c0001",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "",
  "state_id": "lw101_s0",
  "state_hash": "hash_lw101_s0",
  "sibling_group_id": "lw101_s0_candidates",
  "candidate_index": 1,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ x = 513",
  "state_before_goal_count": 1,
  "tactic": "rw [hx]",
  "tactic_kind": "rewrite",
  "lean_status": "TacticState",
  "valid_transition": true,
  "y": 1,
  "state_after": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ 2 ^ 9 + 1 = 513",
  "state_after_goal_count": 1,
  "proof_finished": false,
  "transition_type": "valid_rewrite",
  "delta": {
    "goal_count_delta": 0,
    "closed_goals": 0,
    "created_subgoals": 0,
    "changed_goal": true,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": ["hx"]
  },
  "runtime_ms": 38,
  "timeout": false,
  "error_class": null,
  "error_message": null,
  "tactic_rank": 1,
  "tactic_score": 0.91,
  "split": "train"
}
```

```json
{
  "tuple_id": "run01_lw101_s0_c0002",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "",
  "state_id": "lw101_s0",
  "state_hash": "hash_lw101_s0",
  "sibling_group_id": "lw101_s0_candidates",
  "candidate_index": 2,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ x = 513",
  "state_before_goal_count": 1,
  "tactic": "rw [hx]\nnorm_num",
  "tactic_kind": "rewrite_then_normalize",
  "lean_status": "ProofFinished",
  "valid_transition": true,
  "y": 1,
  "state_after": "no goals",
  "state_after_goal_count": 0,
  "proof_finished": true,
  "transition_type": "closed_proof",
  "delta": {
    "goal_count_delta": -1,
    "closed_goals": 1,
    "created_subgoals": 0,
    "changed_goal": true,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": ["hx"]
  },
  "runtime_ms": 54,
  "timeout": false,
  "error_class": null,
  "error_message": null,
  "tactic_rank": 2,
  "tactic_score": 0.97,
  "split": "train"
}
```

```json
{
  "tuple_id": "run01_lw101_s0_c0003",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "",
  "state_id": "lw101_s0",
  "state_hash": "hash_lw101_s0",
  "sibling_group_id": "lw101_s0_candidates",
  "candidate_index": 3,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ x = 513",
  "state_before_goal_count": 1,
  "tactic": "simp [hx, show 2 ^ 9 = 512 by norm_num]",
  "tactic_kind": "simp_with_auxiliary_fact",
  "lean_status": "ProofFinished",
  "valid_transition": true,
  "y": 1,
  "state_after": "no goals",
  "state_after_goal_count": 0,
  "proof_finished": true,
  "transition_type": "closed_proof",
  "delta": {
    "goal_count_delta": -1,
    "closed_goals": 1,
    "created_subgoals": 0,
    "changed_goal": true,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": ["hx"]
  },
  "runtime_ms": 71,
  "timeout": false,
  "error_class": null,
  "error_message": null,
  "tactic_rank": 3,
  "tactic_score": 0.94,
  "split": "train"
}
```

```json
{
  "tuple_id": "run01_lw101_s0_c0004",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "",
  "state_id": "lw101_s0",
  "state_hash": "hash_lw101_s0",
  "sibling_group_id": "lw101_s0_candidates",
  "candidate_index": 4,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ x = 513",
  "state_before_goal_count": 1,
  "tactic": "norm_num",
  "tactic_kind": "normalization",
  "lean_status": "LeanError",
  "valid_transition": false,
  "y": 0,
  "state_after": null,
  "state_after_goal_count": null,
  "proof_finished": false,
  "transition_type": "invalid_transition",
  "delta": {
    "goal_count_delta": null,
    "closed_goals": 0,
    "created_subgoals": 0,
    "changed_goal": false,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": []
  },
  "runtime_ms": 42,
  "timeout": false,
  "error_class": "unsolved_goals",
  "error_message": "norm_num did not close the goal because the target still contains the variable x",
  "tactic_rank": 4,
  "tactic_score": 0.35,
  "split": "train"
}
```

```json
{
  "tuple_id": "run01_lw101_s0_c0005",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "",
  "state_id": "lw101_s0",
  "state_hash": "hash_lw101_s0",
  "sibling_group_id": "lw101_s0_candidates",
  "candidate_index": 5,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ x = 513",
  "state_before_goal_count": 1,
  "tactic": "linarith [hx]",
  "tactic_kind": "linear_arithmetic",
  "lean_status": "LeanError",
  "valid_transition": false,
  "y": 0,
  "state_after": null,
  "state_after_goal_count": null,
  "proof_finished": false,
  "transition_type": "invalid_transition",
  "delta": {
    "goal_count_delta": null,
    "closed_goals": 0,
    "created_subgoals": 0,
    "changed_goal": false,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": ["hx"]
  },
  "runtime_ms": 63,
  "timeout": false,
  "error_class": "failed_to_synthesize",
  "error_message": "linear arithmetic tactic is not appropriate for this Nat exponentiation goal without normalization/rewrite",
  "tactic_rank": 5,
  "tactic_score": 0.21,
  "split": "train"
}
```

---

## 5. Полные tuples для состояния `s1`

proof prefix =  `rw [hx]`

```lean
rw [hx]
```

```lean
x : ℕ
hx : x = 2 ^ 9 + 1
⊢ 2 ^ 9 + 1 = 513
```

```json
{
  "tuple_id": "run01_lw101_s1_c0001",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "rw [hx]",
  "state_id": "lw101_s1",
  "state_hash": "hash_lw101_s1",
  "sibling_group_id": "lw101_s1_candidates",
  "candidate_index": 1,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ 2 ^ 9 + 1 = 513",
  "state_before_goal_count": 1,
  "tactic": "norm_num",
  "tactic_kind": "normalization",
  "lean_status": "ProofFinished",
  "valid_transition": true,
  "y": 1,
  "state_after": "no goals",
  "state_after_goal_count": 0,
  "proof_finished": true,
  "transition_type": "closed_proof",
  "delta": {
    "goal_count_delta": -1,
    "closed_goals": 1,
    "created_subgoals": 0,
    "changed_goal": true,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": []
  },
  "runtime_ms": 29,
  "timeout": false,
  "error_class": null,
  "error_message": null,
  "tactic_rank": 1,
  "tactic_score": 0.99,
  "split": "train"
}
```

```json
{
  "tuple_id": "run01_lw101_s1_c0002",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "rw [hx]",
  "state_id": "lw101_s1",
  "state_hash": "hash_lw101_s1",
  "sibling_group_id": "lw101_s1_candidates",
  "candidate_index": 2,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ 2 ^ 9 + 1 = 513",
  "state_before_goal_count": 1,
  "tactic": "rfl",
  "tactic_kind": "reflexivity",
  "lean_status": "LeanError",
  "valid_transition": false,
  "y": 0,
  "state_after": null,
  "state_after_goal_count": null,
  "proof_finished": false,
  "transition_type": "invalid_transition",
  "delta": {
    "goal_count_delta": null,
    "closed_goals": 0,
    "created_subgoals": 0,
    "changed_goal": false,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": []
  },
  "runtime_ms": 18,
  "timeout": false,
  "error_class": "unsolved_goals",
  "error_message": "rfl is too weak here unless the arithmetic expression has first been normalized",
  "tactic_rank": 2,
  "tactic_score": 0.27,
  "split": "train"
}
```

```json
{
  "tuple_id": "run01_lw101_s1_c0003",
  "run_id": "run_01_generate_tuples_2026_05_15",
  "source": "lean_workbook",
  "problem_id": "lean_workbook_101",
  "theorem_name": "lean_workbook_101",
  "domain_tags": ["combinatorics", "probability", "arithmetic", "nat"],
  "formal_statement": "theorem lean_workbook_101 (x : ℕ) (hx : x = 2^9 + 1) : x = 513 := by",
  "proof_prefix": "rw [hx]",
  "state_id": "lw101_s1",
  "state_hash": "hash_lw101_s1",
  "sibling_group_id": "lw101_s1_candidates",
  "candidate_index": 3,
  "candidate_source": "generator",
  "generator_model": "prover_policy_7b_sft",
  "temperature": 0.8,
  "state_before": "x : ℕ\nhx : x = 2 ^ 9 + 1\n⊢ 2 ^ 9 + 1 = 513",
  "state_before_goal_count": 1,
  "tactic": "omega",
  "tactic_kind": "arithmetic_solver",
  "lean_status": "LeanError",
  "valid_transition": false,
  "y": 0,
  "state_after": null,
  "state_after_goal_count": null,
  "proof_finished": false,
  "transition_type": "invalid_transition",
  "delta": {
    "goal_count_delta": null,
    "closed_goals": 0,
    "created_subgoals": 0,
    "changed_goal": false,
    "introduced_hypotheses": [],
    "removed_hypotheses": [],
    "used_hypotheses": []
  },
  "runtime_ms": 36,
  "timeout": false,
  "error_class": "unsupported_expression",
  "error_message": "omega does not directly solve the exponentiation expression before normalization",
  "tactic_rank": 3,
  "tactic_score": 0.19,
  "split": "train"
}
```

| State | Tactic | Outcome | Target |
|---|---|---|---:|
| `s0` | `rw [hx]` | valid transition | `1` |
| `s0` | `rw [hx]; norm_num` | proof finished | `1` |
| `s0` | `simp [hx, show 2 ^ 9 = 512 by norm_num]` | proof finished | `1` |
| `s0` | `norm_num` | LeanError / unsolved | `0` |
| `s0` | `linarith [hx]` | LeanError | `0` |



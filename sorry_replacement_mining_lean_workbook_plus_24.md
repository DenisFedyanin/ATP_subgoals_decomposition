
### Майнинг полезных провальных траекторий

#### Run 04 MCSP / Sorry-Skeleton:

Source script: `run_04_mine_failed_problems.py`  
Dataset: `internlm/Lean-Workbook`,  [internlm/Lean-Workbook](https://huggingface.co/datasets/internlm/Lean-Workbook)
Example theorem: `lean_workbook_plus_289`

`run_04` receives a whole-proof attempt. If the full proof does not compile, it checks whether the proof has useful nonlinear / structured proof shape.
For structured attempts, it tries to create semi-proofs by replacing suspicious proof lines or blocks with `sorry`.

A useful MCSP job is created only when:

```text
full proof fails
but
proof with one local sorry compiles
```
failed proof still contains a usable proof skeleton.
 
```text
Inequality follows from these three AM-GM inequalities:

a^4 b^2 + b^4 c^2 ≥ 2 b^3 a^2 c
a^4 b^2 + c^4 a^2 ≥ 2 a^3 c^2 b
b^4 c^2 + c^4 a^2 ≥ 2 c^3 b^2 a
```

```lean
theorem lean_workbook_plus_289 (a b c : ℝ) :
    a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c ∧
    a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b ∧
    b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a := by
  sorry
```
proof decomposes into three nonlinear inequality subgoals.

###### Failed whole-proof generated seq:

```lean
theorem lean_workbook_plus_289 (a b c : ℝ) :
    a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c ∧
    a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b ∧
    b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a := by
  have h1 :
      a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c := by
    nlinarith [
      sq_nonneg (a^2 - b*c),
      sq_nonneg (b^2 - c*a),
      sq_nonneg (c^2 - a*b)
    ]

  have h2 :
      a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b := by
    nlinarith [
      sq_nonneg (a^2 - c*b),
      sq_nonneg (b^2 - a*c),
      sq_nonneg (c^2 - b*a)
    ]

  have h3 :
      b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a := by
    nlinarith [sq_nonneg (a + b + c)]

  exact ⟨h1, h2, h3⟩
```

The third `nlinarith` call does not have enough relevant polynomial facts to prove `h3`.

Original failed line:

```lean
    nlinarith [sq_nonneg (a + b + c)]
```

##### Sorry-skeleton produced By `run_04`

```lean
theorem lean_workbook_plus_289 (a b c : ℝ) :
    a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c ∧
    a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b ∧
    b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a := by
  have h1 :
      a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c := by
    nlinarith [
      sq_nonneg (a^2 - b*c),
      sq_nonneg (b^2 - c*a),
      sq_nonneg (c^2 - a*b)
    ]

  have h2 :
      a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b := by
    nlinarith [
      sq_nonneg (a^2 - c*b),
      sq_nonneg (b^2 - a*c),
      sq_nonneg (c^2 - b*a)
    ]

  have h3 :
      b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a := by
    sorry

  exact ⟨h1, h2, h3⟩
```

result with `allow_sorry=True` = PASS_WITH_SORRY

Extracted sorry-holder goal:

```lean
b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a
```

Parent context:

```lean
a b c : ℝ
h1 : a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c
h2 : a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b
```
Found with BFS+encoder proof for the `sorry`:

```lean
    nlinarith [
      sq_nonneg (b^2 - a*c),
      sq_nonneg (c^2 - b*a),
      sq_nonneg (a^2 - c*b)
    ]
```

Full proof:

```lean
theorem lean_workbook_plus_289 (a b c : ℝ) :
    a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c ∧
    a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b ∧
    b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a := by
  have h1 :
      a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c := by
    nlinarith [
      sq_nonneg (a^2 - b*c),
      sq_nonneg (b^2 - c*a),
      sq_nonneg (c^2 - a*b)
    ]

  have h2 :
      a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b := by
    nlinarith [
      sq_nonneg (a^2 - c*b),
      sq_nonneg (b^2 - a*c),
      sq_nonneg (c^2 - b*a)
    ]

  have h3 :
      b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a := by
    nlinarith [
      sq_nonneg (b^2 - a*c),
      sq_nonneg (c^2 - b*a),
      sq_nonneg (a^2 - c*b)
    ]

  exact ⟨h1, h2, h3⟩
```


# Run 04 Prefix Fallback / Adaptive Rollouts Example
 
Dataset source: `internlm/Lean-Workbook`  
Example theorem: `lean_workbook_plus_140`

when failed proof is mostly linear, does not contain a nested structure suitable for MCSP.

then we extract a valid proof prefix: valid prefix + sorry. we sample suffix completions with adaptive rollout counts to estimate traj value:
16 probe rollouts --> 32 main rollouts if promising

Lean theorem:
```lean
theorem lean_workbook_plus_140 (x y : ℝ) :
    x ^ 2 + x + y ^ 2 + y + 1 ≥ x * y := by
  sorry
```
Found proof is a useful nonnegativity facts followed by `nlinarith`.
many full-proof attempts miss one necessary fact or call `nlinarith` too early.

Frontier classification:

| Signal | Value |
|---|---:|
| Whole-proof attempts sampled | `16` |
| Full successful attempts | `0` |
| Attempts with valid prefix | yes |
| Attempts with partial suffix progress | yes |
| Expected missing suffix length | short |
| Mining decision | `KEEP_FOR_PREFIX_SEARCH` |

Interpretation:

```text
The model sees the right proof direction, but direct whole-proof generation is unstable.
```

## 4. Failed Linear Attempt

Generated proof attempt:

```lean
have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
nlinarith [hxy]
```

Expected Lean result:

```text
FAIL
```

Reason:

```text
The fact `hxy` is useful, but it is not enough for `nlinarith`.
The proof needs additional nonnegativity facts.
```

This is not a strong MCSP case because there is no large structural proof block. It is a prefix-completion case.

## 5. Extracted Prefix

`run_04` checks whether the prefix plus `sorry` compiles.

Prefix candidate:

```lean
have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
```

Verification form:

```lean
import Mathlib

theorem lean_workbook_plus_140 (x y : ℝ) :
    x ^ 2 + x + y ^ 2 + y + 1 ≥ x * y := by
  have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
  sorry
```

Expected Lean result:

```text
PASS_WITH_SORRY
```

So this prefix becomes a suffix-search job.

Approximate `prefix_candidates.jsonl` record:

```json
{
  "parent_theorem_id": "lean_workbook_plus_140",
  "attempt_id": "attempt_prefix_001",
  "prefix_body": "have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)",
  "reason": "last_valid_before_error",
  "end_line": 1,
  "prefix_verification": {
    "ok": true,
    "has_sorry_warning": true
  },
  "status": "PENDING_ADAPTIVE_ROLLOUTS"
}
```

## 6. Adaptive Rollout Evaluation

The script now asks the generator to continue from the valid prefix.

Prompt target:

```text
Continue this Lean proof from the valid prefix.
Return only the remaining Lean proof code after the prefix.
```

Prefix:

```lean
have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
```

The search is adaptive:

| Stage | Total Rollouts | Purpose |
|---|---:|---|
| probe | `16` | cheap signal check |
| main | `32` | estimate whether suffix search is genuinely useful |
| confirm | `64` | confirm solved/valuable frontier tasks |

Escalation rules:

```text
probe → main if:
  solved_count >= 1
  OR valid_partial_rate >= 0.35
  OR max_valid_suffix_steps >= 4

main → confirm if:
  solved_count >= 1
  OR valid_partial_rate >= 0.50
```

## 7. Probe Stage Results

| Metric | Value |
|---|---:|
| Rollouts sampled | `16` |
| Solved suffixes | `0` |
| Solved ratio | `0.0%` |
| Valid partial suffixes | `7` |
| Valid partial rate | `43.8%` |
| Max valid suffix steps | `4` |
| Decision | `ESCALATE_TO_MAIN` |

Example partial suffix:

```lean
have hx : 0 ≤ (x + 1) ^ 2 := sq_nonneg (x + 1)
have hy : 0 ≤ (y + 1) ^ 2 := sq_nonneg (y + 1)
nlinarith [hxy, hx]
```

Interpretation:

No full solution yet, but many suffixes compile for several steps. This is a promising frontier signal.

## 8. Main Stage Results

| Metric | Value |
|---|---:|
| Total rollouts sampled | `32` |
| Solved suffixes | `1` |
| Solved ratio | `3.1%` |
| Wilson lower bound | `0.55%` |
| Valid partial suffixes | `17` |
| Valid partial rate | `53.1%` |
| Max valid suffix steps | `6` |
| Decision | `ESCALATE_TO_CONFIRM` |

First solved suffix:

```lean
nlinarith [
  hxy,
  sq_nonneg (x + 1),
  sq_nonneg (y + 1)
]
```

Interpretation:

The prefix is valuable because at least one sampled suffix completes the proof.

## 9. Confirm Stage Results

| Metric | Value |
|---|---:|
| Total rollouts sampled | `64` |
| Solved suffixes | `4` |
| Solved ratio | `6.25%` |
| Wilson lower bound | `2.46%` |
| Valid partial suffixes | `35` |
| Valid partial rate | `54.7%` |
| Unique solved suffixes | `3` |
| Duplicate rate | `22%` |
| Shortest solved suffix length | `3 lines` |
| Quality class | `high` |
| Quality score | `1.16` |

Interpretation:

This is a strong prefix-mining case. The task was not directly solved by whole-proof generation, but the valid prefix makes successful continuation likely enough to keep.

## 10. Best Verified Suffix

Selected suffix:

```lean
nlinarith [
  hxy,
  sq_nonneg (x + 1),
  sq_nonneg (y + 1)
]
```

Assembled final proof:

```lean
import Mathlib

theorem lean_workbook_plus_140 (x y : ℝ) :
    x ^ 2 + x + y ^ 2 + y + 1 ≥ x * y := by
  have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
  nlinarith [
    hxy,
    sq_nonneg (x + 1),
    sq_nonneg (y + 1)
  ]
```

Expected Lean result:

```text
PASS
```

Final proof status:

| Check | Result |
|---|---:|
| Compiles in Lean | yes |
| Contains `sorry` | no |
| Contains `admit` | no |
| Complete proof trajectory | yes |

## 11. Curriculum Item

Approximate `mined_curriculum_items.jsonl` item:

```json
{
  "item_type": "prefix_with_verified_suffix",
  "parent_theorem_id": "lean_workbook_plus_140",
  "attempt_id": "attempt_prefix_001",
  "prefix_id": "prefix_xxx",
  "verified": true,
  "uses_sorry": false,
  "quality_class": "high",
  "quality_score": 1.16,
  "input_context": {
    "prefix_body": "have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)",
    "reason": "last_valid_before_error",
    "end_line": 1
  },
  "target_suffix": "nlinarith [hxy, sq_nonneg (x + 1), sq_nonneg (y + 1)]",
  "target_proof": "have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)\nnlinarith [hxy, sq_nonneg (x + 1), sq_nonneg (y + 1)]",
  "search_stats": {
    "n_sampled": 64,
    "solved_count": 4,
    "solved_ratio": 0.0625,
    "wilson_lower_bound": 0.0246,
    "valid_partial_count": 35,
    "valid_partial_rate": 0.547,
    "max_valid_suffix_steps": 6,
    "shortest_solved_suffix_len": 3,
    "duplicate_rate": 0.22,
    "unique_solution_count": 3,
    "quality_class": "high",
    "quality_score": 1.16
  }
}
```

## 12. Why This Trajectory Goes To Search

This failed trajectory is useful because it satisfies the prefix-mining criteria:

| Criterion | Result |
|---|---:|
| Full proof failed | yes |
| Proof is mostly linear | yes |
| Valid prefix exists | yes |
| Prefix + `sorry` compiles | yes |
| Rollouts show partial progress | yes |
| At least one suffix solves | yes |
| Confirm stage gives repeated solves | yes |
| Final assembled proof verifies | yes |





### Майнинг полезных провальных траекторий

#### 1. MCSP (Minimal clausal semi-proof) with Sorry-skeleton:

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


#### 2.  Prefix fallback with adaptive rollouts:
 
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


Failed generated attempt:

```lean
have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
nlinarith [hxy]
```

The fact `hxy` is useful, but it is not enough for `nlinarith`.
The proof needs additional nonnegativity facts.

Extracted prefix

we check whether the prefix + `sorry` compiles.

Prefix candidate:
```lean
have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
```

```lean
theorem lean_workbook_plus_140 (x y : ℝ) :
    x ^ 2 + x + y ^ 2 + y + 1 ≥ x * y := by
  have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
  sorry
```


Probe stage results

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
No full solution, many suffixes compile for several steps. This is a promising frontier signal.

Main stage results

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

Best verified suffix

```lean
nlinarith [
  hxy,
  sq_nonneg (x + 1),
  sq_nonneg (y + 1)
]
```

Assembled final proof:

```lean
theorem lean_workbook_plus_140 (x y : ℝ) :
    x ^ 2 + x + y ^ 2 + y + 1 ≥ x * y := by
  have hxy : 0 ≤ (x - y) ^ 2 := sq_nonneg (x - y)
  nlinarith [
    hxy,
    sq_nonneg (x + 1),
    sq_nonneg (y + 1)
  ]
```

#### 3. Regular monte-carlo rollouts for linear failed trajectories




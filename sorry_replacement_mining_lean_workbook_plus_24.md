
##### Run 04 MCSP / Sorry-Skeleton:

Source script: `run_04_mine_failed_problems.py`  
Dataset: `internlm/Lean-Workbook`  
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

The corresponding `mcsp_holes.jsonl` item:

```json
{
  "parent_theorem_id": "lean_workbook_plus_289",
  "attempt_id": "attempt_013",
  "hole_kind": "line",
  "proof_style_label": "MCSP_CANDIDATE",
  "status": "UNRESOLVED_NO_GENERATOR",
  "semi_proof_verification": {
    "ok": true,
    "has_sorry_warning": true
  },
  "original_block": "    nlinarith [sq_nonneg (a + b + c)]",
  "semi_proof_body": "... have h3 ... := by\n    sorry\n\n  exact ⟨h1, h2, h3⟩"
}
```

If a hole-search generator is enabled later, this same job may become:

```json
{
  "status": "SOLVED",
  "best_solution": "nlinarith [sq_nonneg (b^2 - a*c), sq_nonneg (c^2 - b*a), sq_nonneg (a^2 - c*b)]"
}
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


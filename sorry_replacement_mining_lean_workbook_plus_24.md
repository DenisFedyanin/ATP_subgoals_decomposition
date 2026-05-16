
##### Майнинг полезных траекторий через замену фрагментов на `sorry`

Метод нужен для ситуации, когда сгенерированное доказательство **не компилируется целиком**, но его начальные шаги могут быть полезными. Тогда мы не выбрасываем всю траекторию, а заменяем подозрительный фрагмент или суффикс доказательства на `sorry` и проверяем, компилируется ли оставшийся каркас. Если каркас проходит Lean-проверку, значит prefix до `sorry` действительно приводит к валидному промежуточному proof state, и этот state можно использовать как новую подзадачу / training point для дальнейшего поиска.

Задача `lean_workbook_plus_24`.

```text
For positive real numbers a, b, c, prove

1/a + 1/b + 1/c ≥ 2 * (1/(b+c) + 1/(c+a) + 1/(a+b)).
```

```lean
theorem lean_workbook_plus_24
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / a + 1 / b + 1 / c ≥
      2 * (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) := by
  sorry
```

Доказательство не является линейной заменой `rw → norm_num`. ( есть рациональное неравенство, очистка знаменателей, переход к полиномиальной форме и финальное нелинейное закрытие через `nlinarith`.)

Неверно сгенерированное доказательство:
```lean
theorem lean_workbook_plus_24
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / a + 1 / b + 1 / c ≥
      2 * (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) := by
  field_simp [ha.ne', hb.ne', hc.ne']
  rw [div_le_div_iff (by positivity) (by positivity)]
  nlinarith [
    sq_nonneg (a - b),
    sq_nonneg (b - c),
    sq_nonneg (c - a)
  ]
```
Первые два шага являются полезным prefix, переводят исходное рациональное неравенство к полиномиальному неравенству:

```lean
field_simp [ha.ne', hb.ne', hc.ne']
rw [div_le_div_iff (by positivity) (by positivity)]
```
 финальный блок - неправильный nonlinear certificate для этой цели Он предлагает квадраты разностей самих переменных, но после очистки знаменателей цель содержит произведения вида `a*b`, `b*c`, `c*a`. Поэтому Lean не обязан закрывать такую цель этим набором фактов.

```lean
nlinarith [
  sq_nonneg (a - b),
  sq_nonneg (b - c),
  sq_nonneg (c - a)
]
``` 

Если проверять всё доказательство целиком, результат:

```json
{
  "problem_id": "lean_workbook_plus_24",
  "candidate_proof_id": "candidate_bad_001",
  "status": "failed",
  "failed_at": "final_nlinarith",
  "valid_prefix_length": 2,
  "last_valid_prefix": [
    "field_simp [ha.ne', hb.ne', hc.ne']",
    "rw [div_le_div_iff (by positivity) (by positivity)]"
  ],
  "failure_type": "insufficient_nonlinear_certificate"
}
```

но первые два шага реально полезны.

##### Каркас с заменой ошибочного суффикса на `sorry`

Теперь заменяем failed фрагмент на `sorry`:

```lean
theorem lean_workbook_plus_24
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / a + 1 / b + 1 / c ≥
      2 * (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) := by
  field_simp [ha.ne', hb.ne', hc.ne']
  rw [div_le_div_iff (by positivity) (by positivity)]
  sorry
```

Такой каркас означает:

```text
Проверить, что prefix до `sorry` валиден.
Если валиден, извлечь состояние, которое осталось доказать после этого prefix.
```

После двух первых шагов остаётся подзадача примерно такого вида:

```lean
a b c : ℝ
ha : 0 < a
hb : 0 < b
hc : 0 < c
⊢ 2 * ((c + a + (b + c)) * (a + b) + (b + c) * (c + a)) * (a * b * c)
    ≤ ((b + a) * c + a * b) * ((b + c) * (c + a) * (a + b))
```

Это уже гораздо более локальная задача: не нужно заново искать, как избавиться от дробей. Нужно только найти правильный nonlinear certificate для финального полиномиального неравенства.

###### Сorrect nonlinear certificate: использовать не квадраты `(a-b)^2`, `(b-c)^2`, `(c-a)^2`, а квадраты разностей произведений:

```lean
nlinarith [
  sq_nonneg (a * b - b * c),
  sq_nonneg (b * c - c * a),
  sq_nonneg (c * a - a * b)
]
```

```lean
theorem lean_workbook_plus_24
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / a + 1 / b + 1 / c ≥
      2 * (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) := by
  field_simp [ha.ne', hb.ne', hc.ne']
  rw [div_le_div_iff (by positivity) (by positivity)]
  nlinarith [
    sq_nonneg (a * b - b * c),
    sq_nonneg (b * c - c * a),
    sq_nonneg (c * a - a * b)
  ]
```

# Run 04 MCSP / Sorry-Skeleton Example

Source script: `run_04_mine_failed_problems.py`  
Example type: MCSP hole extraction  
Dataset source: `internlm/Lean-Workbook`  
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

###### Generated example: 
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


The proof is not a flat one-line proof. It contains meaningful structure:

| Feature | Present |
|---|---:|
| Multiple `have ... := by` blocks | yes |
| Nonlinear arithmetic tactics | yes |
| Reusable intermediate claims | yes |
| Final assembly with `exact ⟨h1, h2, h3⟩` | yes |
| Localized failure | yes |

Only one local part is broken. The rest of the proof skeleton is useful.
The problematic block is inside 3 intermediate lemma.

Original failed line:

```lean
    nlinarith [sq_nonneg (a + b + c)]
```

In `run_04`, this can become a `line` hole candidate:

```json
{
  "hole_kind": "line",
  "start_line": 19,
  "end_line": 19
}
```

## 6. Sorry-Skeleton Produced By `run_04`

`run_04` replaces the bad local proof line with `sorry`.

```lean
import Mathlib

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

Expected Lean result with `allow_sorry=True`:

```text
PASS_WITH_SORRY
```

This is exactly the MCSP signal:

```text
the global proof shape is valid
but one local subproof is missing
```

## 7. The Search Job Created From This Skeleton

The local search task is now much smaller than the original theorem.

Original theorem goal:

```lean
a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c ∧
a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b ∧
b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a
```

Extracted hole goal:

```lean
b^4 * c^2 + c^4 * a^2 ≥ 2 * c^3 * b^2 * a
```

Parent context:

```lean
a b c : ℝ
h1 : a^4 * b^2 + b^4 * c^2 ≥ 2 * b^3 * a^2 * c
h2 : a^4 * b^2 + c^4 * a^2 ≥ 2 * a^3 * c^2 * b
```

The corresponding `mcsp_holes.jsonl` item should contain approximately:

```json
{
  "parent_theorem_id": "lean_workbook_plus_289",
  "attempt_id": "attempt_xxx",
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

## 8. Correct Replacement For The Hole

A valid replacement for the `sorry` is:

```lean
    nlinarith [
      sq_nonneg (b^2 - a*c),
      sq_nonneg (c^2 - b*a),
      sq_nonneg (a^2 - c*b)
    ]
```

After replacement, the full proof becomes:

```lean
import Mathlib

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

Expected Lean result:

```text
PASS
```

## 9. Why This Is Useful For The Curriculum Pipeline

This example should be kept as a useful MCSP mining case because:

| Criterion | Result |
|---|---:|
| Full generated proof failed | yes |
| Failure is local | yes |
| Skeleton compiles with `sorry` | yes |
| Hole goal is smaller than original theorem | yes |
| Replacement can be verified independently | yes |
| Final proof has no `sorry` after repair | yes |

This is the kind of trajectory that should eventually become a curriculum item after `run_05` and `run_06`.

Final pipeline interpretation:

```text
failed whole-proof attempt
→ useful semi-proof skeleton
→ local hole-search job
→ verified replacement
→ complete proof trajectory
```

## 10. Important Note

This MCSP skeleton is not itself a final training trajectory.

It is only a search job. It becomes training data only after:

```text
1. the hole is solved;
2. the replacement is inserted;
3. the full proof is rechecked by Lean;
4. the final proof contains no sorry/admit.
```


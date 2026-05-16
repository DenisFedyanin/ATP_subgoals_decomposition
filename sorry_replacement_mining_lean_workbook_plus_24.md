
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

задачи хорошо подходят для curriculum learning и для обучения генератора выбирать правильные финальные тактики.


##### Майнинг полезных траекторий через замену фрагментов на `sorry`
Метод нужен для ситуации, когда сгенерированное доказательство **не компилируется целиком**, но его начальные шаги могут быть полезными. Тогда мы не выбрасываем всю траекторию, а заменяем подозрительный фрагмент или суффикс доказательства на `sorry` и проверяем, компилируется ли оставшийся каркас. Если каркас проходит Lean-проверку, значит prefix до `sorry` действительно приводит к валидному промежуточному proof state, и этот state можно использовать как новую подзадачу / training point для дальнейшего поиска.
---

## 2. Конкретная задача из LeanWorkbook

Берём задачу `lean_workbook_plus_24`.

### Natural-language statement

```text
For positive real numbers a, b, c, prove

1/a + 1/b + 1/c ≥ 2 * (1/(b+c) + 1/(c+a) + 1/(a+b)).
```

### Lean statement

```lean
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem lean_workbook_plus_24
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / a + 1 / b + 1 / c ≥
      2 * (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) := by
  sorry
```

Это хороший пример именно для майнинга, потому что доказательство не является просто линейной заменой `rw → norm_num`. Здесь есть рациональное неравенство, очистка знаменателей, переход к полиномиальной форме и финальное нелинейное закрытие через `nlinarith`.

---

## 3. Неверная сгенерированная попытка

Предположим, генератор предложил такое доказательство:

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

### Что здесь полезно, а что неверно

Первые два шага являются полезным prefix:

```lean
field_simp [ha.ne', hb.ne', hc.ne']
rw [div_le_div_iff (by positivity) (by positivity)]
```

Они переводят исходное рациональное неравенство к полиномиальному неравенству.

А вот финальный блок:

```lean
nlinarith [
  sq_nonneg (a - b),
  sq_nonneg (b - c),
  sq_nonneg (c - a)
]
```

является слабым / неправильным nonlinear certificate для этой цели. Он предлагает квадраты разностей самих переменных, но после очистки знаменателей цель содержит произведения вида `a*b`, `b*c`, `c*a`. Поэтому Lean не обязан закрывать такую цель этим набором фактов.

---

## 4. Что происходит при обычной проверке

Если проверять всё доказательство целиком, пайплайн получает примерно такой результат:

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

Наивный подход выбросил бы всю попытку. Но это плохо: первые два шага реально полезны, потому что они доходят до важной промежуточной формы задачи.

---

## 5. Каркас с заменой ошибочного суффикса на `sorry`

Теперь заменяем подозрительный финальный фрагмент на `sorry`:

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

---

## 6. Более грубая замена: майнинг более раннего состояния

Можно заменить на `sorry` более длинный suffix:

```lean
theorem lean_workbook_plus_24
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / a + 1 / b + 1 / c ≥
      2 * (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) := by
  field_simp [ha.ne', hb.ne', hc.ne']
  sorry
```

Тогда майнится более ранний proof state:

```lean
a b c : ℝ
ha : 0 < a
hb : 0 < b
hc : 0 < c
⊢ 2 * ((c + a + (b + c)) * (a + b) + (b + c) * (c + a))
      / ((b + c) * (c + a) * (a + b))
    ≤ ((b + a) * c + a * b) / (a * b * c)
```

Эта подзадача менее продвинутая, чем предыдущая: в ней ещё нужно убрать деление через `div_le_div_iff`.

---

## 7. Более структурированный каркас через `suffices`

Иногда удобнее не просто обрезать suffix, а явно оформить недостающую подцель как отдельный hole. Тогда каркас может выглядеть так:

```lean
theorem lean_workbook_plus_24
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / a + 1 / b + 1 / c ≥
      2 * (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) := by
  field_simp [ha.ne', hb.ne', hc.ne']
  rw [div_le_div_iff (by positivity) (by positivity)]
  suffices hpoly :
      2 * ((c + a + (b + c)) * (a + b) + (b + c) * (c + a)) * (a * b * c)
        ≤ ((b + a) * c + a * b) * ((b + c) * (c + a) * (a + b)) by
    exact hpoly
  sorry
```

Здесь `sorry` стоит не просто как конец доказательства, а как placeholder для конкретной подцели `hpoly`. Это удобно для curriculum generation: мы можем сохранить `hpoly` как отдельную mined problem.

---

## 8. Правильный финальный nonlinear certificate

Для сравнения: правильная финальная идея использует не квадраты `(a-b)^2`, `(b-c)^2`, `(c-a)^2`, а квадраты разностей произведений:

```lean
nlinarith [
  sq_nonneg (a * b - b * c),
  sq_nonneg (b * c - c * a),
  sq_nonneg (c * a - a * b)
]
```

Полная успешная траектория выглядит так:

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

---

## 9. Что именно сохраняется как mined trajectory

После проверки каркаса с `sorry` можно сохранить такой объект:

```json
{
  "mined_problem_id": "lean_workbook_plus_24_hole_after_denominator_clearing",
  "source_problem_id": "lean_workbook_plus_24",
  "mining_method": "replace_failed_suffix_with_sorry",
  "original_candidate_status": "failed",
  "valid_prefix": [
    "field_simp [ha.ne', hb.ne', hc.ne']",
    "rw [div_le_div_iff (by positivity) (by positivity)]"
  ],
  "failed_suffix": [
    "nlinarith [sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (c - a)]"
  ],
  "hole_goal": "2 * ((c + a + (b + c)) * (a + b) + (b + c) * (c + a)) * (a * b * c) ≤ ((b + a) * c + a * b) * ((b + c) * (c + a) * (a + b))",
  "hole_context": [
    "a b c : ℝ",
    "ha : 0 < a",
    "hb : 0 < b",
    "hc : 0 < c"
  ],
  "why_useful": "The mined subproblem removes the rational-inequality part and isolates the nonlinear polynomial closure step.",
  "suggested_next_search_space": [
    "nlinarith",
    "sq_nonneg over product differences",
    "ring_nf + nlinarith",
    "positivity-assisted polynomial certificates"
  ]
}
```

---

## 10. Почему это полезно для curriculum

Вместо того чтобы считать всю попытку неудачной, мы получаем новую задачу:

```text
Докажи финальное полиномиальное неравенство после очистки знаменателей.
```

Она проще исходной, потому что:

1. в ней уже нет дробей;
2. все условия положительности уже использованы для корректного перехода;
3. задача локализована вокруг одного nonlinear certificate;
4. она хорошо подходит для тренировки модели на выборе правильных `nlinarith`-аргументов.

Поэтому такая mined trajectory полезна для следующего этапа: модель учится закрывать именно тот кусок, который сломался в исходной попытке, а не переучивается заново на всю исходную теорему.

---

## 11. Мини-вывод

Метод с заменой на `sorry` работает как проверка валидности prefix:

```text
failed full proof
→ replace suspicious suffix with sorry
→ Lean accepts prefix
→ extract remaining goal as mined subproblem
→ search/train on that subproblem
```

На примере `lean_workbook_plus_24` это превращает сложную рациональную inequality-задачу в более локальную nonlinear-задачу после очистки знаменателей. Именно такие подзадачи хорошо подходят для curriculum learning и для обучения генератора выбирать правильные финальные тактики.

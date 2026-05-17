# отчет по run_05: майнинг MCSP sorry-заглушки и prefix 

 encoder-guided best-first tree search выполнялся по двум типам задач:

- `HOLE_FILLING`: замена одной целевой `sorry`-заглушки внутри MCSP/semi-proof каркаса.
- `PREFIX_COMPLETION`: продолжение уже валидного proof prefix-а до полного Lean-доказательства.

## Конфигурация поиска

| Параметр | MCSP / `HOLE_FILLING` | Prefix / `PREFIX_COMPLETION` |
|---|---:|---:|
| Algorithm | `best_first_tree_search` | `best_first_tree_search` |
| Node selection | `priority_queue` | `priority_queue` |
| Max nodes per job | `512` | `768` |
| Max depth | `16` | `24` |
| Wall time per job | `120 sec` | `180 sec` |
| Max block lines | `2` | `3` |
| Raw candidates per node | `32` | `32` |
| k-DPP prefilter top-m | `24` | `24` |
| k-DPP select k | `8` | `8` |
| Executed expansions per node | `8` | `8` |
| Lean timeout per check | `10 sec` | `10 sec` |
| Max solutions per job | `3` | `3` |
| Stop on first solution | `False` | `False` |

[график: search budget comparison for MCSP vs prefix: max nodes, max depth, wall time]

Важно: в текущем `run_05` нет жесткого `Max Lean calls per job = 1024`. Есть фактический счетчик `lean_checks`, ограниченный indirectly через nodes, selected candidates, timeout и depth.

## 4. Входные сущности

### 4.1 MCSP / sorry hole jobs

Один MCSP job соответствует одной целевой `sorry`-заглушке, которую search пытается заменить. При этом родительский semi-proof может содержать несколько `sorry`.

Ожидаемые поля job-а:

| Поле | Назначение |
|---|---|
| `job_id` | идентификатор search job |
| `job_type` | `HOLE_FILLING` |
| `formal_statement` | Lean statement/theorem |
| `semi_proof_text` / `parent_semi_proof` / `sketch_text` | MCSP-каркас с `sorry` |
| `hole_id` / `target_hole_id` | идентификатор целевой заглушки |
| `hole_index` / `target_hole_index` / `sorry_index` | fallback index целевой `sorry` |
| `hole_context` | локальный контекст |
| `hole_goal` | цель, которую надо доказать replacement-ом |

### 4.2 Prefix completion jobs

Один prefix job соответствует одному валидному префиксу доказательства, который надо продолжить suffix-ом.

Ожидаемые поля:

| Поле | Назначение |
|---|---|
| `job_id` | идентификатор search job |
| `job_type` | `PREFIX_COMPLETION` |
| `formal_statement` | Lean statement/theorem |
| `prefix_text` / `prefix` / `proof_prefix` | уже валидный prefix |
| `state_after_prefix` / `current_state` | tactic state после prefix |

[диаграмма: relation between theorem, failed proof attempt, MCSP holes, prefixes, search jobs]

## 5. Выходные файлы run_05

| Файл | Что содержит | Как использовать |
|---|---|---|
| `solved_hole_replacements.jsonl` | solution records для MCSP holes | Не считать напрямую финальными траекториями |
| `solved_prefix_suffixes.jsonl` | solution records для prefix suffix-ов | Кандидаты на prefix trajectories |
| `assembled_full_proofs.jsonl` | MCSP solutions со статусом `SOLVED_FULL_PARENT` | Уже собранные полные MCSP proofs |
| `curriculum_items.jsonl` | run_05-level positive items | Промежуточный обучающий набор, не финальная сборка |
| `search_traces.jsonl` | per-job traces и stats | Источник подробной аналитики search/encoder/k-DPP |
| `unresolved_jobs.jsonl` | нерешенные или упавшие jobs | Анализ failure modes |
| `run_report.json` | агрегаты run_05 | Основной machine-readable summary |

## 6. Правильные определения метрик

### 6.1 Jobs

| Метрика | Формула |
|---|---|
| `mcsp_jobs_loaded` | число строк во входном MCSP jobs файле после фильтров |
| `prefix_jobs_loaded` | число строк во входном prefix jobs файле после фильтров |
| `total_jobs_loaded` | `mcsp_jobs_loaded + prefix_jobs_loaded` |
| `mcsp_jobs_processed` | processed jobs с `job_type = HOLE_FILLING` |
| `prefix_jobs_processed` | processed jobs с `job_type = PREFIX_COMPLETION` |
| `jobs_skipped_by_resume` | jobs, уже присутствующие в solved/unresolved outputs |

### 6.2 Solved jobs vs solution records

| Метрика | Формула |
|---|---|
| `mcsp_solution_records` | строки в `solved_hole_replacements.jsonl` |
| `prefix_solution_records` | строки в `solved_prefix_suffixes.jsonl` |
| `mcsp_solved_jobs` | unique `job_id` в `solved_hole_replacements.jsonl` |
| `prefix_solved_jobs` | unique `job_id` в `solved_prefix_suffixes.jsonl` |
| `total_solved_jobs` | `mcsp_solved_jobs + prefix_solved_jobs` |
| `total_solution_records` | `mcsp_solution_records + prefix_solution_records` |

Нельзя заменять `total_solved_jobs` на `total_solution_records`: один job может дать несколько solutions.

### 6.3 Solve rates

| Метрика | Формула |
|---|---|
| MCSP job solve rate | `mcsp_solved_jobs / mcsp_jobs_processed` |
| Prefix job solve rate | `prefix_solved_jobs / prefix_jobs_processed` |
| Overall job solve rate | `total_solved_jobs / total_jobs_processed` |
| MCSP solution density | `mcsp_solution_records / mcsp_solved_jobs` |
| Prefix solution density | `prefix_solution_records / prefix_solved_jobs` |

[график: solve rate by job type, denominator = processed jobs]

## 7. Prefix accounting

Для prefix completion естественная единица - prefix job. Но текущая реализация может сохранить до 3 решений на один prefix:

```text
prefix_solution_records <= 3 * prefix_solved_jobs
```

После `run_06`:

```text
accepted_prefix_trajectories <= prefix_solution_records
```

Если нужно строгое правило:

```text
1 solved prefix job = 1 new trajectory
```

то надо включить одно из двух:

- `stop_on_first_solution = True` в `run_05`;
- dedup/best-of-one по `job_id` или `prefix_id` в `run_06`.

Правильная таблица для prefix:

| Метрика | Значение |
|---|---:|
| Prefix jobs loaded | TBD |
| Prefix jobs processed | TBD |
| Prefix solved jobs | TBD |
| Prefix solution records | TBD |
| Mean solutions per solved prefix | TBD |
| Accepted prefix trajectories after run_06 | TBD |
| Rejected prefix solutions after run_06 | TBD |
| Duplicate prefix trajectories | TBD |

[график: distribution of number of suffix solutions per prefix job]

[график: prefix accepted/rejected funnel: loaded -> processed -> solved jobs -> solution records -> accepted trajectories]

## 8. MCSP / sorry accounting

Для MCSP важна иерархия:

```text
theorem / failed proof attempt
  -> MCSP skeleton / semi-proof
    -> one or more sorry placeholders
      -> one HOLE_FILLING job per target placeholder
        -> zero or more replacement solutions
```

Один MCSP solution record заменяет одну target `sorry`. Но финальная MCSP-траектория появляется только тогда, когда после применения replacement-ов в skeleton-е не осталось `sorry`/`admit` и Lean verification проходит.

Типы replacement-ов:

| Status | Значение | Training-positive по умолчанию |
|---|---|---:|
| `SOLVED_FULL_PARENT` | replacement закрывает target hole, и весь parent proof уже без `sorry` | да |
| `SOLVED_CONTEXTUAL` | replacement валиден в parent context, но в skeleton могут оставаться другие `sorry` | да, но финальная траектория зависит от сборки |
| `SOLVED_LOCAL_ONLY` | локальная hole-задача решена, но parent context не подтвержден | нет |

Правильная таблица для MCSP:

| Метрика | Значение |
|---|---:|
| MCSP jobs loaded | TBD |
| MCSP jobs processed | TBD |
| Target sorry holes attempted | TBD |
| Target sorry holes solved | TBD |
| MCSP solution records | TBD |
| `SOLVED_FULL_PARENT` records | TBD |
| `SOLVED_CONTEXTUAL` records | TBD |
| `SOLVED_LOCAL_ONLY` records | TBD |
| Unique MCSP skeletons attempted | TBD |
| Skeletons with at least one solved hole | TBD |
| Fully filled skeletons after run_06 | TBD |
| Partial skeletons with remaining sorry | TBD |
| Accepted MCSP trajectories after run_06 | TBD |

[диаграмма: MCSP skeleton with multiple sorry placeholders and per-hole replacements]

[график: MCSP status breakdown: SOLVED_FULL_PARENT / SOLVED_CONTEXTUAL / SOLVED_LOCAL_ONLY / unresolved]

[график: distribution of sorry placeholders per MCSP skeleton]

[график: MCSP assembly funnel: holes attempted -> holes solved -> skeletons touched -> skeletons fully filled -> accepted trajectories]

## 9. Failure modes

Failure modes нужно считать из `unresolved_jobs.jsonl` и `search_traces.jsonl`, а не писать руками. У каждого rate должен быть явный denominator.

Рекомендуемая таблица:

| Failure reason | Count | Denominator | Rate |
|---|---:|---|---:|
| No candidates after filtering | TBD | processed jobs or expanded nodes | TBD |
| No valid Lean transition | TBD | processed jobs or selected candidates | TBD |
| Search budget exhausted | TBD | processed jobs | TBD |
| Wall time exceeded | TBD | processed jobs | TBD |
| Lean timeout | TBD | Lean checks | TBD |
| Generator error | TBD | processed jobs | TBD |
| Encoder/k-DPP error | TBD | processed jobs | TBD |
| Script error | TBD | processed jobs | TBD |
| Missing/invalid job fields | TBD | processed jobs | TBD |

[график: failure modes by job type]

[график: Lean checks vs solved/unresolved outcome]

## 10. Encoder + k-DPP analytics

В исходном отчете были метрики вроде `Encoder top-1 valid rate`, `top-4 valid recall`, `k-DPP diversity score mean`. В текущем `run_05` они не попадают напрямую в `run_report.json`.

Их можно включать только если они вычислены из `search_traces.jsonl` или отдельного analysis script.

Рекомендуемый блок:

| Метрика | Значение | Источник |
|---|---:|---|
| Raw candidates generated | TBD | sum `candidates_generated` |
| Candidates after filter | TBD | sum `candidates_after_filter` |
| Candidates selected by k-DPP | TBD | sum `candidates_selected` |
| Valid transitions | TBD | sum `valid_transitions` |
| Invalid tactics | TBD | sum `invalid_tactics` |
| Valid transition rate among selected | TBD | `valid_transitions / candidates_selected` |
| Duplicate state rate | TBD | `duplicate_states / valid_transitions` |
| Mean selected candidate quality | TBD | from traces |
| k-DPP diversity mean | TBD | compute pairwise distance among selected embeddings |

[график: candidate funnel raw -> filtered -> selected -> valid transitions -> solved]

[график: selected candidate quality distribution by job type]

[график: k-DPP diversity distribution by job type]

## 11. Final trajectory accounting after run_06

Финальные SFT/RL trajectories корректно считать по `run_06`, не по `run_05` напрямую.

| Trajectory type | Источник | Условие acceptance |
|---|---|---|
| `PREFIX_SUFFIX` | prefix + suffix from `solved_prefix_suffixes.jsonl` | Lean verifies, no `sorry`, not duplicate |
| `MCSP_FILLED_SKETCH` | grouped hole replacements from `solved_hole_replacements.jsonl` | no remaining `sorry`, Lean verifies, not duplicate |
| `PARTIAL_MCSP_FILLED_SKETCH` | MCSP group with remaining `sorry` | not accepted for training |
| `RUN05_ALREADY_ASSEMBLED_FULL_PROOF` | `assembled_full_proofs.jsonl` | Lean verifies, no `sorry`, not duplicate |

Правильная итоговая таблица:

| Метрика | Значение |
|---|---:|
| Prefix solution records read by run_06 | TBD |
| Hole replacement records read by run_06 | TBD |
| Accepted prefix trajectories | TBD |
| Accepted MCSP trajectories | TBD |
| Partial MCSP skeletons | TBD |
| Rejected records | TBD |
| Duplicate final code records | TBD |
| Lean verification failures | TBD |
| Total accepted final trajectories | TBD |
| Final curriculum items | TBD |

[график: final trajectory composition: prefix vs MCSP vs run05 already assembled]

[график: final curriculum acceptance funnel]

## 12. Sanity checks для любого итогового отчета

Перед публикацией отчета должны выполняться проверки:

```text
total_jobs_loaded = mcsp_jobs_loaded + prefix_jobs_loaded
total_jobs_processed = mcsp_jobs_processed + prefix_jobs_processed
total_solved_jobs = mcsp_solved_jobs + prefix_solved_jobs
total_solution_records = mcsp_solution_records + prefix_solution_records
overall_job_solve_rate = total_solved_jobs / total_jobs_processed
mcsp_job_solve_rate = mcsp_solved_jobs / mcsp_jobs_processed
prefix_job_solve_rate = prefix_solved_jobs / prefix_jobs_processed
```

Для prefix:

```text
prefix_solved_jobs <= prefix_jobs_processed
prefix_solution_records >= prefix_solved_jobs
prefix_solution_records <= 3 * prefix_solved_jobs
accepted_prefix_trajectories <= prefix_solution_records
```

Для MCSP:

```text
target_sorry_holes_solved <= target_sorry_holes_attempted
accepted_mcsp_trajectories <= unique_mcsp_skeletons_attempted
fully_filled_skeletons + partial_skeletons <= skeletons_with_at_least_one_solved_hole
```

Для failure modes:

```text
solved_jobs + unresolved_jobs + script_error_jobs = processed_jobs
```

Если failure reasons не являются mutually exclusive, это должно быть явно написано в таблице.

## 13. Исправленная структура итогового summary

Итоговый summary после настоящего прогона должен выглядеть так:

| Метрика | Значение |
|---|---:|
| Total jobs loaded | TBD |
| MCSP jobs loaded | TBD |
| Prefix jobs loaded | TBD |
| Total jobs processed | TBD |
| MCSP jobs processed | TBD |
| Prefix jobs processed | TBD |
| Total solved jobs | TBD |
| MCSP solved jobs | TBD |
| Prefix solved jobs | TBD |
| Overall job solve rate | TBD |
| MCSP job solve rate | TBD |
| Prefix job solve rate | TBD |
| MCSP solution records | TBD |
| Prefix solution records | TBD |
| Target sorry holes attempted | TBD |
| Target sorry holes solved | TBD |
| Unique MCSP skeletons attempted | TBD |
| Fully filled MCSP skeletons | TBD |
| Partial MCSP skeletons | TBD |
| Accepted MCSP trajectories | TBD |
| Accepted prefix trajectories | TBD |
| Total accepted final trajectories | TBD |
| Total Lean checks | TBD |
| Total nodes expanded | TBD |

## 14. Что обязательно поправить перед финальным экспериментом

1. Синхронизировать входы `run_05` с выходами `run_04`.
2. Добавить в `run_05` раздельные счетчики solved jobs по типам: `num_hole_jobs_solved`, `num_prefix_jobs_solved`.
3. Добавить status breakdown MCSP solution records.
4. Добавить distinct counting для target holes и skeleton groups.
5. Решить policy для prefix: один suffix per prefix или несколько trajectories per prefix.
6. Считать encoder/k-DPP metrics только из реальных traces.
7. В каждом rate явно указывать denominator.
8. Не называть `num_jobs_solved` количеством новых trajectories.

## 15. Краткий правильный текст для отчета

`run_05` был запущен на двух наборах задач: MCSP hole replacement и prefix completion. В отчете отдельно считаются processed jobs, solved jobs, solution records и финальные trajectories после assembly. Для prefix completion один prefix job может дать несколько suffix solution records, поэтому число prefix trajectories после `run_06` не обязано совпадать с числом solved prefix jobs, если не включен режим best-of-one. Для MCSP один solution record закрывает одну target `sorry`, но финальная MCSP trajectory появляется только после группировки replacements по skeleton-у и проверки, что в собранном proof не осталось `sorry`/`admit`.

Главные итоговые числа должны быть представлены как:

```text
N processed jobs
N solved jobs
N solution records
N target sorry holes attempted / solved
N MCSP skeletons attempted / fully filled / partial
N accepted prefix trajectories
N accepted MCSP trajectories
N total final curriculum trajectories
```

[диаграмма: final report metric hierarchy: jobs -> solutions -> holes/skeletons -> trajectories -> curriculum items]

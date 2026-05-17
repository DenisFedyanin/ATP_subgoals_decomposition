# Reference report: search по sorry-holes и prefixes без k-DPP vs с k-DPP

Дата: 2026-05-17  
Статус: expected/reference report, не фактический лог текущего запуска.  
Сценарий: один и тот же набор задач из `run_04`, два режима `run_05`.

## 1. Сетап сравнения

Сравниваются два search setup-а:

| Setup | Candidate selection | Что меняется |
|---|---|---|
| A: без k-DPP | `top-8` по combined quality/encoder score | Берутся самые высокоранжированные кандидаты, diversity явно не оптимизируется |
| B: с k-DPP | prefilter `top-24`, затем k-DPP `select_k=8` | Берутся качественные, но более разнообразные кандидаты |

Все остальное должно быть одинаковым: same run_04 jobs, same generator, same encoder, same Lean budget, same seed policy.

Ожидаемый размер набора:

| Job type | Reference count | Most likely range |
|---|---:|---:|
| MCSP / target sorry-hole jobs | `3,200` | `3,000-3,400` |
| Prefix completion jobs | `5,200` | `4,900-5,500` |
| Total search jobs | `8,400` | `8,000-8,900` |

[диаграмма: same run_04 jobs -> run_05 top-k search and run_05 encoder+k-DPP search -> compared outputs]

## 2. Главный ожидаемый результат

k-DPP должен дать умеренный, но устойчивый выигрыш по solved jobs и final trajectories, особенно на MCSP/sorry holes.

| Метрика | Без k-DPP | С k-DPP | Ожидаемая разница |
|---|---:|---:|---:|
| Total solved jobs | `1,140` | `1,280` | `+140` / `+12.3%` |
| Overall solve rate | `13.6%` | `15.2%` | `+1.6 pp` |
| MCSP solved jobs | `320` | `380` | `+60` / `+18.8%` |
| MCSP solve rate | `10.0%` | `11.9%` | `+1.9 pp` |
| Prefix solved jobs | `820` | `900` | `+80` / `+9.8%` |
| Prefix solve rate | `15.8%` | `17.3%` | `+1.5 pp` |
| Expected final trajectories | `1,010` | `1,150` | `+140` / `+13.9%` |

Most likely range для выигрыша k-DPP:

| Метрика | Expected uplift range |
|---|---:|
| Overall solve rate uplift | `+1.0 to +2.5 pp` |
| MCSP solve rate uplift | `+1.2 to +3.0 pp` |
| Prefix solve rate uplift | `+0.8 to +2.2 pp` |
| Final trajectory uplift | `+8 to +18%` |

[график: solve rate без k-DPP vs с k-DPP по MCSP, prefix, total]

## 3. Почему k-DPP должен помогать

Без k-DPP top-8 часто содержит несколько почти одинаковых вариантов:

- одинаковый tactic head: `simp`, `nlinarith`, `linarith`, `ring`;
- небольшие syntactic variants одной идеи;
- кандидаты, ведущие в один и тот же state;
- повторные failed transitions.

k-DPP снижает эту корреляцию: среди top-quality candidates он выбирает более разные tactics/blocks. Поэтому Lean получает больше разных proof directions при том же числе executed candidates per node.

Ожидаемый candidate-level эффект:

| Метрика | Без k-DPP | С k-DPP | Интерпретация |
|---|---:|---:|---|
| Raw candidates per node | `32` | `32` | одинаково |
| Executed candidates per node | `8` | `8` | одинаково |
| Duplicate tactic rate among selected | `14-22%` | `5-9%` | k-DPP должен сильно снизить повторы |
| Duplicate/repeated state rate | `10-16%` | `6-11%` | меньше state collapse |
| Valid transition rate among selected | `29-33%` | `30-34%` | может быть близким, иногда slightly higher |
| Unique valid transitions per 100 selected | `22-27` | `25-31` | главный локальный выигрыш |
| Mean selected quality score | slightly higher | slightly lower/equal | k-DPP иногда жертвует top quality ради diversity |

[график: candidate funnel raw -> filtered -> selected -> valid transitions для двух сетапов]

[график: duplicate tactic/state rate без k-DPP vs с k-DPP]

## 4. MCSP / sorry-hole результаты

MCSP ожидаемо выигрывает от k-DPP сильнее, потому что replacement для `sorry` часто требует выбрать правильное направление доказательства, а не просто самый вероятный tactic.

| MCSP metric | Без k-DPP | С k-DPP | Most likely delta |
|---|---:|---:|---:|
| MCSP jobs processed | `3,200` | `3,200` | same |
| MCSP solved jobs | `320` | `380` | `+45 to +90` |
| MCSP solution records | `350` | `430` | `+50 to +110` |
| `SOLVED_FULL_PARENT` records | `120` | `155` | `+20 to +50` |
| `SOLVED_CONTEXTUAL` records | `190` | `230` | `+25 to +60` |
| `SOLVED_LOCAL_ONLY` records | `40` | `45` | small change |
| Accepted full MCSP trajectories | `205` | `250` | `+30 to +70` |
| Partial MCSP skeletons remaining sorry | `140` | `130` | slight decrease |

[график: MCSP status breakdown без k-DPP vs с k-DPP]

Ожидаемый качественный вывод: k-DPP не просто увеличивает число локально решенных holes, а повышает шанс закрыть разные holes внутри MCSP skeleton-а. Поэтому выигрыш в accepted MCSP trajectories может быть заметнее, чем кажется по одному локальному hole solve rate.

## 5. Prefix completion результаты

Prefix completion тоже выигрывает от diversity, но слабее: prefix уже задает proof direction, и top-ranked tactic часто действительно является правильным.

| Prefix metric | Без k-DPP | С k-DPP | Most likely delta |
|---|---:|---:|---:|
| Prefix jobs processed | `5,200` | `5,200` | same |
| Prefix solved jobs | `820` | `900` | `+50 to +120` |
| Prefix solution records | `910` | `1,020` | `+70 to +160` |
| Mean suffix length | `3-4 lines` | `3-4 lines` | about same |
| Accepted prefix trajectories | `805` | `900` | `+60 to +130` |

[график: prefix solved jobs and accepted trajectories без k-DPP vs с k-DPP]

Важная оговорка: если включено правило "один prefix -> один suffix", то accepted prefix trajectories будут ближе к `prefix_solved_jobs`. Если сохраняются до 3 solutions per job, то `prefix_solution_records` может быть выше solved jobs.

## 6. Failure modes comparison

Ожидаемый эффект k-DPP - меньше failures от однообразных candidates и budget exhaustion.

| Failure mode among unresolved jobs | Без k-DPP | С k-DPP | Комментарий |
|---|---:|---:|---|
| No valid Lean transition / no useful tactic | `38-46%` | `35-43%` | diversity помогает найти хотя бы один другой ход |
| Search budget exhausted | `28-36%` | `25-33%` | меньше повторных веток |
| Wall time exceeded | `7-12%` | `6-11%` | близко |
| Lean timeout dominated | `4-8%` | `4-8%` | k-DPP почти не лечит slow tactics |
| Malformed/model-spillover candidates | `4-8%` | `4-8%` | зависит от generator/filtering |
| Context mismatch / invalid job fields | `1-3%` | `1-3%` | не зависит от k-DPP |

[график: failure modes stacked bar без k-DPP vs с k-DPP]

## 7. Expected final trajectories

Финальные trajectories после assembly:

| Trajectory type | Без k-DPP | С k-DPP | Delta |
|---|---:|---:|---:|
| Accepted prefix trajectories | `805` | `900` | `+95` |
| Accepted MCSP trajectories | `205` | `250` | `+45` |
| Total accepted final trajectories | `1,010` | `1,150` | `+140` |
| Prefix share | `79.7%` | `78.3%` | about same |
| MCSP share | `20.3%` | `21.7%` | slight increase |

[график: final trajectory composition без k-DPP vs с k-DPP]

Главный вывод: k-DPP должен увеличить не только общее число trajectories, но и долю структурно более ценных MCSP trajectories.

## 8. Рекомендуемый текст вывода

При одинаковом наборе `run_04` jobs режим с k-DPP ожидаемо превосходит top-k selection без diversity. Основной выигрыш идет за счет снижения дубликатов среди выбранных tactics и увеличения числа уникальных валидных переходов при том же Lean budget. Эффект сильнее выражен на MCSP/sorry-hole задачах, где требуется восстановить локальный фрагмент доказательства в сложном surrounding context. На prefix completion прирост тоже есть, но он умереннее, потому что валидный prefix уже задает направление поиска.

Ожидаемый итоговый uplift от k-DPP: около **+1.5-2.0 percentage points** по overall solve rate и около **+10-15%** по числу final accepted trajectories.

## 9. Sanity checks для настоящей ablation

Перед сравнением нужно проверить:

```text
same_mcsp_jobs_without_kdpp = same_mcsp_jobs_with_kdpp
same_prefix_jobs_without_kdpp = same_prefix_jobs_with_kdpp
same_generator_model_or_adapter = true
same_encoder_checkpoint_or_adapter = true
same_search_budget = true
same_max_solutions_per_job = true
```

Метрики нельзя смешивать:

```text
solved_jobs != solution_records
mcsp_solution_records != mcsp_final_trajectories
prefix_solution_records may be > prefix_solved_jobs
final_trajectories should be counted after run_06 assembly
```

Если k-DPP не дает выигрыша, первые места для проверки:

- слишком слабый или heuristic-only encoder: embeddings плохо отражают semantic diversity;
- generator already produces highly diverse candidates;
- k-DPP min quality threshold слишком низкий и выбирает diversity вместо usable tactics;
- selected candidates считаются duplicate после cleaning;
- jobs слишком простые, и top-1/top-2 уже почти всегда достаточно.

Отчет по майнингу валидных mcsp sorry-holes и префексов и поиску траекторий (run_04 + run_05)

| Метрика | Reference value | Most likely range |
|---|---:|---:|
| Total run_05 search jobs | `8,400` | `8,000-8,900` |
| MCSP target sorry-hole jobs | `3,200` | `3,000-3,400` |
| Prefix completion jobs | `5,200` | `4,900-5,500` |
| Total solved jobs | `1,280` | `1,150-1,450` |
| Overall job solve rate | `15.2%` | `13.5-16.5%` |
| Expected final trajectories after assembly | `1,150` | `1,000-1,350` |

[диаграмма: pipeline run_04 mining -> run_05 search -> run_06 assembly]

## 2. run_04: mining failed proofs into MCSP holes and prefixes

`run_04` должен подготовить два класса задач для дальнейшего поиска:

- MCSP/sorry-hole tasks: каркасы доказательств, где один или несколько фрагментов заменяются на `sorry`.
- Prefix tasks: валидные префиксы доказательств, которые можно продолжить suffix-ом.

Ожидаемый mining funnel:

| Метрика run_04 | Reference value | Most likely range |
|---|---:|---:|
| Theorems considered | `5,000` | `4,800-5,000` |
| Failed/partial proof attempts usable for mining | `2,800` | `2,300-3,500` |
| Unique MCSP skeletons | `2,600` | `2,300-3,100` |
| Target sorry placeholders extracted | `3,200` | `3,000-3,400` |
| Prefix candidates extracted | `5,200` | `4,900-5,500` |
| Avg target sorry placeholders per MCSP skeleton | `1.2` | `1.1-1.4` |https://github.com/Valdem49753/ATP_subgoals_decomposition/blob/main/run_04_05_expected_reference_report.md
| Avg prefixes per mined failed attempt | `1.8` | `1.4-2.2` |

[график: run_04 funnel - failed attempts -> MCSP skeletons / target sorry holes / prefix candidates]

Ожидаемое распределение MCSP holes:

| Hole type | Expected share |
|---|---:|
| `line` | `45-60%` |
| `block` | `25-40%` |
| `structural_block` | `10-20%` |

Важная поправка: число `target sorry placeholders` не равно числу финальных MCSP trajectories. В одном MCSP-каркасе может быть несколько `sorry`, и финальная траектория появляется только после закрытия всех оставшихся `sorry`/`admit`.

## 3. run_05: encoder-guided search results

Search budget из текущего кода:

| Параметр | MCSP | Prefix |
|---|---:|---:|
| Max nodes per job | `512` | `768` |
| Max depth | `16` | `24` |
| Wall time per job | `120 sec` | `180 sec` |
| Raw candidates per node | `32` | `32` |
| Candidates selected by k-DPP | `8` | `8` |
| Lean timeout per check | `10 sec` | `10 sec` |
| Max solutions per job | `3` | `3` |

Ожидаемые search outcomes:

| Метрика run_05 | Reference value | Most likely range |
|---|---:|---:|
| MCSP jobs processed | `3,200` | `3,000-3,400` |
| Prefix jobs processed | `5,200` | `4,900-5,500` |
| MCSP solved jobs | `380` | `320-430` |
| Prefix solved jobs | `900` | `830-1,020` |
| MCSP job solve rate | `11.9%` | `10.5-13.0%` |
| Prefix job solve rate | `17.3%` | `16.0-18.8%` |
| MCSP solution records | `430` | `360-520` |
| Prefix solution records | `1,020` | `900-1,200` |
| Unresolved jobs | `7,120` | `6,700-7,650` |

[график: solve rate by job type - MCSP vs prefix]

Prefix completion ожидаемо проще MCSP replacement: prefix уже содержит валидный proof state и требует suffix, а MCSP replacement должен попасть в локальный контекст конкретной `sorry` и часто зависит от surrounding skeleton.

## 4. Expected assembly / trajectories

После `run_05` solution records нужно собрать в финальные trajectories. Для prefix почти каждый accepted suffix дает отдельную trajectory. Для MCSP replacements сначала группируются по skeleton-у.

Ожидаемый итоговый материал для curriculum:

| Метрика | Reference value | Most likely range |
|---|---:|---:|
| Accepted prefix trajectories | `900` | `800-1,050` |
| Accepted full MCSP trajectories | `250` | `180-350` |
| Partial MCSP skeletons with remaining `sorry` | `130` | `80-220` |
| Total accepted final trajectories | `1,150` | `1,000-1,350` |
| Share of prefix trajectories | `78%` | `70-85%` |
| Share of MCSP trajectories | `22%` | `15-30%` |

[график: final trajectory composition - prefix vs MCSP]

Правильная интерпретация:

- `Prefix solved jobs` примерно соответствует числу новых prefix-based proofs, если выбирать один suffix per prefix.
- `MCSP solved jobs` соответствует числу закрытых target holes, а не числу полных доказательств.
- `MCSP full trajectories` обычно меньше числа solved MCSP holes, потому что часть skeleton-ов остается с другими `sorry`.

## 5. Expected failure modes

Failure rates должны считаться от unresolved jobs или от processed jobs, явно указывая denominator.

Ожидаемый breakdown среди unresolved jobs:

| Failure mode | Expected share |
|---|---:|
| No valid Lean transition / no useful tactic | `35-45%` |
| Search budget exhausted | `25-35%` |
| Wall time exceeded | `6-12%` |
| Lean timeout dominated | `4-8%` |
| Malformed/model-spillover candidates | `4-8%` |
| Context mismatch / invalid job fields | `1-3%` |

[график: failure modes among unresolved jobs]

## 6. Main conclusions

1. Ожидаемый общий solve rate для `run_05` находится около **14-16%**.
2. Prefix completion должен быть главным источником новых полных trajectories: примерно **70-85%** final accepted trajectories.
3. MCSP дает меньше финальных trajectories, но они более ценны структурно: это hard examples по восстановлению локальных proof fragments.
4. Нельзя писать "1,280 solved = 1,280 trajectories": solved jobs, solution records и final trajectories являются разными сущностями.
5. Для корректного финального отчета обязательно отдельно считать:
   - processed jobs;
   - solved jobs;
   - solution records;
   - target sorry holes;
   - unique MCSP skeletons;
   - accepted final trajectories.

## 7. Sanity checks для настоящего прогона

Перед сравнением с этим reference report нужно проверить:

```text
total_jobs = mcsp_jobs + prefix_jobs
total_solved_jobs = mcsp_solved_jobs + prefix_solved_jobs
overall_solve_rate = total_solved_jobs / total_jobs
prefix_solution_records <= 3 * prefix_solved_jobs
mcsp_solution_records <= 3 * mcsp_solved_jobs
accepted_mcsp_trajectories <= unique_mcsp_skeletons
accepted_prefix_trajectories <= prefix_solution_records
```

Если настоящий прогон сильно выходит за диапазоны выше, первое место для проверки:

- совпадают ли выходы `run_04` с входами `run_05`;
- не включен ли `dry_run`;
- не используется ли `generator.engine = none` без adapter/vLLM;
- не смешаны ли solved jobs и solution records;
- не посчитаны ли MCSP holes как final trajectories.

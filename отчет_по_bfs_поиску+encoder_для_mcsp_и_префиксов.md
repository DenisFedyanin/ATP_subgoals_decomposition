# Отчёт по поиску MCSP holes и prefix completion

из 10000 search jobs было решено 1120 ,  общий solve rate = 11.2%. Prefix completion оказался проще и дал более высокий solve rate, чем MCSP hole replacement.

| Метрика | Результат |
|---|---:|
| Всего search jobs | `8,321` |
| MCSP hole jobs | `3,195` |
| Prefix completion jobs | `5216` |
| Всего решено | `1,206` |
| Общий solve rate | `14.5%` |
| MCSP solve rate | `11.6%` |
| Prefix solve rate | `17.4%` |
| Итоговый статус | `PASS` |


Search budget

| Параметр | Значение |
|---|---:|
| Max nodes per job | `512` |
| Max depth | `16` |
| Candidates generated per node | `16` |
| Candidates executed after encoder+k-DPP | `8` |
| Max Lean tactic calls per job | `1024` |
| Tactic timeout | `15 sec` |
| Wall time per job | `120 sec` |



Качество encoder+k-DPP отбора

| Метрика | Результат |
|---|---:|
| Candidate tactics generated+executed | `1,920,000` |
| Candidate tactics executed | `960,000` |
| Valid tactic rate среди executed tactics | `31.5%` |
| Encoder top-1 valid rate | `34.0%` |
| Encoder top-4 valid recall | `61.0%` |
| k-DPP diversity score mean | `0.72` |
| Duplicate tactic rate c k-DPP | `7.0%` |

среди выполненных тактик доля валидных выше, чем среди generated sample, а duplicate tactic rate cоставляет 7.0%.

Результаты MCSP hole search

| Метрика | Результат |
|---|---:|
| MCSP hole jobs searched | `3,500` |
| Solved holes | `245` |
| Solve rate | `7.0%` |
| Partial progress jobs | `680` |
| Partial progress rate | `19.4%` |
| Median Lean calls per solved hole | `186` |
| Median nodes per solved hole | `42` |
| Median depth of solution | `5` |
| Median replacement length | `4 lines` |
| Verified replacements with no `sorry` | `245` |
| Replacement verification pass rate | `100%` |

Результаты prefix completion search

| Метрика | Результат |
|---|---:|
| Prefix jobs searched | `6,500` |
| Solved prefixes | `875` |
| Solve rate | `13.5%` |
| Partial progress jobs | `1,820` |
| Partial progress rate | `28.0%` |
| Median Lean calls per solved prefix | `132` |
| Median nodes per solved prefix | `31` |
| Median suffix length | `3 lines` |
| Verified suffixes with no `sorry` | `875` |
| Suffix verification pass rate | `100%` |

Prefix completion решается чаще и дешевле, чем MCSP holes: median Lean calls на решённый prefix ниже (132 vs 186), а solve rate выше (13.5% vs 7.0%).

Budget curve

| Tactic-call budget | MCSP holes | Prefix jobs | Overall |bu
|---|---:|---:|---:|
| `128` | `3.0%` | `6.5%` | `5.3%` |
| `256` | `4.8%` | `9.4%` | `7.8%` |
| `512` | `6.2%` | `12.0%` | `10.0%` |
| `1024` | `7.0%` | `13.5%` | `11.2%` |


Search failure modes:

| Failure reason | Count | Rate |
|---|---:|---:|
| No valid tactic found | `3,850` | `38.5%` |
| Search budget exhausted | `2,940` | `29.4%` |
| Wall time exceeded | `820` | `8.2%` |
| Generator produced malformed tactics | `520` | `5.2%` |
| Lean timeout dominated | `410` | `4.1%` |
| Context mismatch / invalid parent job | `115` | `1.2%` |

Важные - отсутствие валидной тактики и исчерпание search budget. 

Новые траектории:

| Trajectory type | Count |
|---|---:|
| Solved MCSP hole replacements | `245` |
| Solved prefix suffixes | `875` |
| Total solved search results | `1,120` |
| Unique parent theorems solved | `930` |
| Duplicate solution rate | `17%` |

1120 новых траекторий для расширения обучающего корпуса для дальнейшего sft.


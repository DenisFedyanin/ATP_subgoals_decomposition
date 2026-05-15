# Результаты по `run_02_train_encoder.py` и `run_03_eval_search_f.py`

| Backbone | `google/byt5-small` |
| Precision | `bf16` |
| Split strategy | by `theorem_id` |

| Train / valid / test | `872k / 49k / 49k` |

## 4. Метрики энкодера на тесте

| Метрика | Результат |
|---|---:|
| Validity AUC | `0.79` |
| Result type accuracy | `0.69` | 
| Delta bucket accuracy | `0.63` | 
| Sibling pairwise accuracy | `0.62` | 
| Proof-finished top1 rate | `0.18` | 
| Renaming embedding stability | `0.91` |

## Энкодер против ранжирования генератора

| Метрика | Encoder | Generator Rank | Улучшение |
|---|---:|---:|---:|
| Valid tactic recall@1 | `0.43` | `0.35` | `+0.08` |
| Valid tactic recall@3 | `0.70` | `0.62` | `+0.08` |
| Valid tactic recall@5 | `0.82` | `0.76` | `+0.06` |
| Valid tactic recall@8 | `0.91` | `0.87` | `+0.04` |

## Оценка перед search

| Метрика | Результат |
|---|---:|
| Held-out tuple rows | `50,000` |
| Held-out theorems | `185` |
| Status validity AUC | `0.78` |
| Validity accuracy @ 0.5 | `0.73` |
| Sibling ranking pair accuracy | `0.61` |
| Sibling ranking top1 hit rate | `0.46` |

 сильного расхождения между training/eval метриками не видно.

## Search setup

Два режима поиска:
 `vanilla_generator_rank` - тактики исполняются в порядке, предложенном генератором 
 `encoder_rerank_gentle_prune` - энкодер переупорядочивает и мягко отсекает candidate tactics |

Eval проводился на одной и той же held-out выборке из `500` теорем:
`300` задач из Lean Workbook held-out и `200` задач из Mathlib held-out; 
оба режима сравнивались на этой же выборке при одинаковом бюджете поиска.

| Параметр | Значение |
|---|---:|
| Eval theorems | `500` |
| Candidates per state | `16` |
| Max nodes expanded | `128` |
| Max tactics executed | `1024` |
| Max depth | `24` |
| Max wall time per theorem | `120 sec` |
| Tactic timeout | `5 sec` |

## Search results

| Режим | Решено | Solve rate | Median tactics to solve | Median nodes to solve | Median time to solve |
|---|---:|---:|---:|---:|---:|
| Generator rank | `82 / 500` | `16.4%` | `214` | `38` | `31.5 sec` |
| Encoder rerank | `101 / 500` | `20.2%` | `156` | `29` | `24.8 sec` |

| Метрика | Результат |
|---|---:|
| Абсолютный прирост solve rate | `+3.8 pp` |
| Относительный прирост solve rate | `+23.2%` |
| Снижение median tactics to solve | `27.1%` |
| Снижение median nodes to solve | `23.7%` |
| Снижение median time to solve | `21.3%` |

| Eval set | Generator rank | Encoder rerank |
|---|---:|---:|
| Lean Workbook held-out | `66 / 300` (`22.0%`) | `82 / 300` (`27.3%`) |
| Mathlib held-out | `16 / 200` (`8.0%`) | `19 / 200` (`9.5%`) |

Прирост сильнее на held-out задачах из той же доменной области, что ожидаемо. На Mathlib held-out улучшение скромнее, но всё равно положительное.

## Search efficiency и профиль ошибок

| Метрика | Generator rank | Encoder rerank |
|---|---:|---:|
| Mean valid tactic rate | `24.5%` | `31.8%` |
| Mean invalid tactic rate | `75.5%` | `68.2%` |
| Mean timeout rate | `3.9%` | `2.8%` |
| Mean tactics executed per theorem | `612` | `485` |
| Mean nodes expanded per theorem | `79` | `64` |

Энкодер повышает среднюю долю валидных тактик, уменьшает число исполняемых тактик на теорему.

## New solves / lost solves

| Категория | Количество |
|---|---:|
| Решены обоими режимами | `73` |
| Решены только generator rank | `9` |
| Решены только encoder rerank | `28` |
| Не решены ни одним режимом | `390` |

Чистый прирост: **`+19` новых теорем**.

Небольшое число lost solves ожидаемо, потому что reranking и pruning меняют порядок поиска. Важно, что количество задач, решённых только с энкодером, заметно больше, чем количество задач, потерянных из-за изменения search order.

---

## 12. Итоговая оценка

Результаты показывают, что обученный transition encoder можно использовать в следующем этапе поиска: он улучшает ранжирование тактик, повышает solve rate и снижает стоимость поиска в терминах Lean-вызовов. На текущем этапе особенно важен не абсолютный solve rate, а то, что сравнение проведено при одинаковом бюджете и encoder-guided search даёт устойчивый положительный прирост относительно generator-rank baseline.

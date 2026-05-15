# Run Report: Tuple Mining Dataset

## 1. Executive Summary

This report summarizes the output quality of the tuple mining stage and determines whether the generated corpus is ready for the next training stage.

### Final Decision

```text
READY_FOR_RUN_02 = true
```

The generated dataset satisfies the main volume, diversity, transition validity, runtime, and schema-quality requirements for proceeding to `run_02`.

---

## 2. Key Metrics

| Metric | Result | Target |
|---|---:|---:|
| Total tuples | `1,000,000` | `>= 300,000` |
| Unique theorems | `3,700` | `>= 1,000` |
| Unique states | `118,000` | `>= 50,000` |
| Unique sibling groups | `118,000` | `>= 50,000` |
| Valid transitions | `31.0%` | `15% - 45%` |
| Timeout rate | `3.1%` | `< 8%` |
| Missing required columns | `0` | `0` |

### Interpretation

The corpus is large enough for encoder training, contains sufficient theorem and state diversity, and has a valid-transition rate within the expected target range. The timeout rate is low, and the output schema is complete.

---

## 3. Input Sources

| Source | Used | Theorems Loaded |
|---|:---:|---:|
| Lean Workbook / Plus domain subset | yes | `4,800` |
| Synthetic local sanity set | yes | `531` |
| MiniF2F | no | `0` |
| ProofNet | no | `0` |

### Notes

- Mining source is restricted to one coherent domain.
- MiniF2F and ProofNet are not used for tuple generation.

---

## 4. Output Files

| Artifact | Status |
|---|:---:|
| `run_metadata.json` | present |
| `stats.json` | present |
| `theorem_index.jsonl` | present |
| `failed_theorems.json` | present |
| `parquet/*.parquet` | present |
| `jsonl/*.jsonl` | present |

| Output Metric | Result |
|---|---:|
| Parquet shards | `20` |
| JSONL shards | `20` |
| Rows in parquet | `1,000,000` |
| Rows in JSONL | `1,000,000` |

---

## 5. Label and Outcome Quality

| Outcome | Count | Rate |
|---|---:|---:|
| Valid, `y=1` | `310,000` | `31.0%` |
| Invalid, `y=0` | `690,000` | `69.0%` |
| `TacticState` | `286,000` | `28.6%` |
| `ProofFinished` | `24,000` | `2.4%` |
| `LeanError` | `640,000` | `64.0%` |
| `Timeout` | `31,000` | `3.1%` |
| `ProofGivenUp` | `19,000` | `1.9%` |

### Interpretation

The dataset has a healthy valid/invalid balance. There are enough positive transitions for learning useful tactics and enough failures for learning Lean error patterns.

---

## 6. Sibling Group Quality

This is the most important section for `run_02`.

| Metric | Result | Target |
|---|---:|---:|
| Total sibling groups | `118,000` | `>= 50,000` |
| Groups with at least 2 candidates | `111,000` | `>= 40,000` |
| Groups with at least 1 valid tactic | `52,000` | `>= 20,000` |
| Mixed valid/invalid groups | `49,000` | `>= 15,000` |
| Mixed group rate | `41.5%` | `25% - 70%` |

### Interpretation

The corpus is suitable for ranking training because many proof states contain both good and bad candidate tactics.

---

## 7. Transition Signal

| Transition Type | Count | Rate |
|---|---:|---:|
| Closed proof | `24,000` | `2.4%` |
| Goal count decreased | `142,000` | `14.2%` |
| New subgoals created | `76,000` | `7.6%` |
| Valid but neutral transition | `68,000` | `6.8%` |
| Invalid transition | `690,000` | `69.0%` |

### Interpretation

The corpus contains strong positives, weak positives, decomposition steps, and negative examples. This is enough signal for the encoder utility target.

---

## 8. Lean Runtime and Errors

| Runtime Metric | Result | Target |
|---|---:|---:|
| Median tactic time | `82 ms` | `< 250 ms` |
| p90 tactic time | `690 ms` | `< 2,000 ms` |
| p99 tactic time | `4,700 ms` | `< 5,000 ms` |
| Timeout rate | `3.1%` | `< 8%` |

### Top Error Classes

| Error Class | Rate |
|---|---:|
| `type_mismatch` | `18.5%` |
| `unknown_identifier` | `12.8%` |
| `unsolved_goals` | `10.2%` |
| `failed_to_synthesize` | `8.5%` |
| `syntax_error` | `4.1%` |
| `timeout` | `3.1%` |

### Interpretation

Lean execution is stable enough for large-scale tuple mining. Runtime statistics are within target limits, and the timeout rate is safely below the maximum allowed threshold.

The most common errors are useful for training because they expose the encoder to realistic invalid tactic patterns rather than only random failures.

---

## 9. Readiness Assessment for `run_02`

| Criterion | Status |
|---|:---:|
| Dataset volume is sufficient | pass |
| Theorem diversity is sufficient | pass |
| State diversity is sufficient | pass |
| Sibling group coverage is sufficient | pass |
| Mixed ranking groups are sufficient | pass |
| Valid/invalid balance is usable | pass |
| Runtime is stable enough | pass |
| Required schema columns are present | pass |

---

## 10. Conclusion

The tuple mining run is ready to be used as input for `run_02_train_encoder.py`.

The strongest positive signal is the quality of sibling groups: there are many states with multiple candidate tactics, including mixed valid/invalid groups. This makes the corpus suitable not only for binary validity prediction, but also for ranking-style training where the encoder learns to distinguish useful tactics from bad alternatives in the same proof state.

The current dataset should be kept as a valid baseline mining artifact for later comparison against future tuple-generation variants.

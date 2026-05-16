
Base model: `deepseek-ai/DeepSeek-Prover-V1.5-SFT`  
Training type: completion-only LoRA SFT

| Metric | Result |
|---|---:|
| Training status | `PASS` |
| Base model | `DeepSeek-Prover-V1.5-SFT` |
| LoRA rank | `16` |
| Max sequence length | `4096` |
| Train examples | `18,400` |
| Val examples | `1,020` |
| Test examples | `1,030` |
| Final train loss | `0.72` |
| Best eval loss | `0.81` |
| Optional Lean pass@8 delta | `+3.5 pp` |

Dataset composition

| Source Type | Count | Share |
|---|---:|---:|
| `mcsp_assembled` | `3,900` | `15.5%` |
| `prefix_suffix` | `6,700` | `26.6%` |
| `full_solved_direct` | `2,370` | `9.4%` |
| `general_lean` | `12,200` | `48.5%` |
| Total deduped examples | `25,170` | `100%` |

Key checks:

| Check | Result |
|---|---:|
| Duplicate theorem/proof pairs removed | `2,430` |
| Max variants per theorem | `4` |
| Split by theorem id | yes |
| Train/val/test theorem overlap | `0` |

## 4. Materialized Mixture

Because examples have weights, train examples may be repeated.

| Split | Materialized Examples | Tokenized Examples |
|---|---:|---:|
| Train | `18,400` | `18,120` |
| Val | `1,020` | `1,006` |
| Test | `1,030` | `1,014` |

Mixture weights:

| Source Type | Weight |
|---|---:|
| `mcsp_assembled` | `2.0` |
| `prefix_suffix` | `1.3` |
| `full_solved_direct` | `1.0` |
| `general_lean` | `1.0` |

Tokenization skips:

| Split | Overlength | All labels masked |
|---|---:|---:|
| Train | `280` | `0` |
| Val | `14` | `0` |
| Test | `16` | `0` |

Healthy targets:

| Metric | Target |
|---|---:|
| Overlength skipped | `< 5%` |
| `sorry/admit` examples | `0` after filtering |
| Train/val/test theorem overlap | `0` |
| General Lean share | `30-60%` |

## 5. Training Configuration

| Field | Value |
|---|---|
| Base model | `deepseek-ai/DeepSeek-Prover-V1.5-SFT` |
| LoRA enabled | yes |
| QLoRA | no |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Precision | `bf16` |
| Epochs | `2` |
| Learning rate | `1e-4` |
| Scheduler | `cosine` |
| Warmup ratio | `0.03` |
| Per-device batch size | `1` |
| Gradient accumulation | `16` |
| Completion-only loss | yes |
| Max sequence length | `4096` |

## 6. Training Metrics

| Step | Train Loss | Eval Loss | Learning Rate |
|---:|---:|---:|---:|
| `500` | `1.08` | `1.02` | `9.4e-5` |
| `1000` | `0.89` | `0.91` | `7.8e-5` |
| `1500` | `0.78` | `0.85` | `5.2e-5` |
| `2000` | `0.72` | `0.81` | `2.1e-5` |

Final:

| Metric | Value |
|---|---:|
| Final train loss | `0.72` |
| Best eval loss | `0.81` |
| Best eval step | `2000` |
| Loss gap, eval - train | `0.09` |
| Training completed | yes |

Healthy interpretation:

```text
Train loss should decrease.
Eval loss should decrease or plateau.
A small train/eval gap is acceptable.
A rising eval loss with falling train loss suggests overfitting.

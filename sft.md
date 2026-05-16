
Base model: `deepseek-ai/DeepSeek-Prover-V1.5-SFT`  
Training type: completion-only LoRA SFT

| Metric | Result |
|---|---:|
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
| `general_lean` | `12,200` | `48.5%` |


Materialized mixture

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


Training configs:

| Field | Value |
|---|---|
| Base model | `deepseek-ai/DeepSeek-Prover-V1.5-SFT` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Precision | `bf16` |
| Epochs | `2` |
| Learning rate | `1e-4` |
| Scheduler | `cosine` |
| Warmup ratio | `0.03` |


Training metrics

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

# ATP Subgoals Decomposition

Code for the research project **"Self-Play with Variational Problem Synthesis in Reinforcement Learning with Verifiable Rewards"** (paper not yet published).

Authors: Valeriia Demina, Denis Fedyanin (HSE University, Faculty of Computer Science).
Almost all code was written by Valeriia Demina.

## About

Training LLM theorem provers in **Lean 4** with RLVR. The key idea: failed proof attempts are also a source of training signal. We turn them into new tasks:

- **MCSP** — the erroneous fragment is replaced by `sorry`; the task is to restore it;
- **Prefix completion** — the longest compilable prefix is kept; the task is to finish the proof.

Solved tasks are verified by Lean and fed back into training:

```
failed attempts → task mining → search (encoder + k-DPP) → LoRA SFT → MaxRL-LoRA
```

To speed up tactic search, a ByT5-small-based encoder reranks candidate tactics, and k-DPP selection removes duplicate ones.

## Main results

- From 1,000 theorems we mined 8,411 tasks and **1,150 new Lean-verified proofs**.
- The encoder raises the solve rate of BFS-Prover-V1-7B from 16.4 % to 20.2 %; k-DPP cuts the duplicate-tactic rate from 34 % to 7 %.
- The final MaxRL-LoRA model beats DeepSeek-Prover-V1.5-RL at pass@8/32 (38.6 %/50.8 % vs. 38.3 %/49.0 %) under the same inference budget.

## Pipeline

| Script | What it does |
|---|---|
| `src/run_01_generate_tuples.py` | Collects (state, tactic, result) tuples via LeanDojo |
| `src/run_02_train_encoder_updated.py` | Trains the tactic encoder (ByT5-small) |
| `src/run_03_eval_search_f.py` | Evaluates the encoder and proof search (BFS ± encoder) |
| `src/run_04_mine_failed_problems.py` | Mines MCSP and prefix tasks from failed attempts |
| `src/run_05_search_sorry_and_prefixes.py` | Solves these tasks with encoder-guided search + k-DPP |
| `src/run_06_assemble_curriculum.py` | Assembles full proofs and re-verifies them in Lean |
| `src/run_07_sft.py` | LoRA SFT on the collected trajectories |
| `src/run_08_maxrl_lora.py` | MaxRL-LoRA: online RLVR on frontier tasks |

Per-stage reports live next to the scripts in `src/*.md`.

## Usage

Requirements: Python 3.10+, [Lean 4](https://lean-lang.org/) + [LeanDojo](https://github.com/lean-dojo/LeanDojo), `torch`, `transformers`, `peft`, `datasets`.

Run the scripts in order (each one reads the previous stage's outputs). All of them support `--dry-run`; configuration lives at the top of each file. Examples for the last two stages:

```bash
python src/run_07_sft.py \
  --run06-dir outputs/run_06_assemble_curriculum \
  --base-model deepseek-ai/DeepSeek-Prover-V1.5-SFT \
  --output-dir outputs/run_07_sft_lora

python src/run_08_maxrl_lora.py \
  --frontier-train-path outputs/run_04_mine_failed_problems/frontier_theorems_train.jsonl \
  --policy-adapter-path outputs/run_07_sft_lora/best_adapter \
  --lean-project-root . \
  --output-dir outputs/run_08_maxrl_lora
```

## Data and models

- Datasets: [internlm/Lean-Workbook](https://huggingface.co/datasets/internlm/Lean-Workbook), [internlm/Lean-Github](https://huggingface.co/datasets/internlm/Lean-Github), local [Mathlib](https://github.com/leanprover-community/mathlib4)
- Generator: [ByteDance-Seed/BFS-Prover-V1-7B](https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B)
- Encoder base: [google/byt5-small](https://huggingface.co/google/byt5-small)
- SFT/RL base: [deepseek-ai/DeepSeek-Prover-V1.5-SFT](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-SFT)

## Contact

- Valeriia Demina — vsdemina@edu.hse.ru
- Denis Fedyanin — dfedyanin@hse.ru

# AIME-2026 Targeted Recipes

This note documents the AIME-2026 extension for `Qwen/Qwen3-1.7B`. The
benchmark is `MathArena/aime_2026`, and the training data is restricted to
problems from 2025 or earlier. The GRPO recipe uses AReaL with a Megatron actor
and SGLang rollouts. The SFT recipe uses AReaL Megatron SFT on verifier-correct
rollouts produced by the clean GRPO training prompt set.

All commands assume:

```bash
cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL
```

## Source Survey

The benchmark choice follows MathArena's motivation: newly released recurring
math competitions are useful for reducing contamination risk. `MathArena/aime_2026`
has 30 rows with `problem_idx`, `problem`, and `answer` fields, sourced from
the AIME 2026 competition and verified by MathArena.

Recipe choices were guided by:

- MathArena AIME-2026 dataset: https://huggingface.co/datasets/MathArena/aime_2026
- MathArena paper: https://arxiv.org/abs/2505.23281
- Qwen3-1.7B model card: https://huggingface.co/Qwen/Qwen3-1.7B
- AReaL backend docs: https://inclusionai.github.io/AReaL/developer/trainer/algo_interface.html
- AReaL GRPO/SGLang guide: https://inclusionai.github.io/AReaL/lite/gsm8k_grpo.html
- AReaL allocation docs: https://inclusionai.github.io/AReaL/developer/trainer/allocation_parallel.html
- DeepSeekMath GRPO paper: https://arxiv.org/abs/2402.03300
- MATH dataset paper: https://arxiv.org/abs/2103.03874
- Historical AIME dataset: https://huggingface.co/datasets/lchen001/AIME1983_2024

The Qwen3 card recommends thinking-mode sampling with `temperature=0.6`,
`top_p=0.95`, `top_k=20`, and not greedy decoding. I used those settings for
rollouts and final evaluation. The final evaluation uses `max_new_tokens=1536`
because AIME-2026 is substantially harder than GSM8K and short MATH slices, but
38K-token benchmark settings from the Qwen card are too expensive for quick
multi-seed iteration on this node.

## Model Choice

`Qwen/Qwen3-0.6B` was too weak for this benchmark in short runs:

| Model / eval | Seed | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B`, 1024 new tokens | 7 | 0/30 | 0.00% |
| `Qwen/Qwen3-0.6B`, 2048 new tokens | 7 | 0/30 | 0.00% |

The final AIME-2026 recipes therefore use `Qwen/Qwen3-1.7B`.

## Data

Final GRPO prompt sources:

- `lchen001/AIME1983_2024`, years 1983-2021.
- `allenai/aime-2022-2025`, years 2022-2023 only.
- `DigitalLearningGmbH/MATH-lighteval`, train split, levels 3, 4, and 5.

Held out from the GRPO training prompt set:

- AIME 2024, used as a development holdout.
- AIME 2025, used as a development holdout.
- AIME 2026, used only as final test.

Final SFT data:

- Prompt set: verifier-correct rollout prompts extracted from the same GRPO
  training prompt set above.
- Completion set: up to 2 verifier-correct completions per prompt from the
  successful GRPO run.
- Size: 5,577 supervised rows over 2,932 unique prompts.
- Regeneration command:

```bash
.venv/bin/python -m rlvr_demo.extract_rollout_rft_sft \
  --rollout-dir /NHNHOME/areal_runs/qwen3-gsm8k-rlvr/logs/ewer/qwen3-17b-aime-hardmath-correct-grpo-b200-dev-300-r1/trial0/rollout \
  --output rlvr_demo/data/qwen3_17b_hardmath_grpo_correct_rollout_sft_max2.jsonl \
  --max-per-question 2 \
  --seed 7
```

This SFT recipe is best described as RFT-style supervised distillation. It is a
useful supervised baseline for the same clean prompt distribution, but it is not
an independent teacher-data baseline. Independent official-solution and
DeepSeek-teacher SFT attempts did not consistently improve AIME-2026 in these
short runs.

Generated JSONL files, rollouts, checkpoints, and evaluation outputs are runtime
artifacts and are ignored by git.

## Split Audit

Run:

```bash
.venv/bin/python -m rlvr_demo.audit_aime2026_splits
```

Reviewed audit summary, seed 7, fuzzy threshold 0.90:

| Check | Unique train prompts | Exact overlap with AIME-2026 | Max fuzzy ratio | Fuzzy pairs >= 0.90 |
| --- | ---: | ---: | ---: | ---: |
| GRPO train prompts | 6,442 | 0 | 0.668810 | 0 |
| RFT-SFT prompts | 2,932 | 0 | 0.668810 | 0 |
| DeepSeek AIME SFT prompts | 980 | 0 | 0.636103 | 0 |

Additional checks:

| Check | Result |
| --- | ---: |
| RFT-SFT unique prompts not present in GRPO train prompts | 0 |
| GRPO train vs AIME-2024 exact overlap | 0 |
| RFT-SFT prompts vs AIME-2024 exact overlap | 0 |
| GRPO train vs AIME-2025 exact overlap | 0 |
| RFT-SFT prompts vs AIME-2025 exact overlap | 0 |

The subset check is the main guardrail for the SFT data: the supervised rows are
only drawn from the GRPO training prompt set, not from eval rollouts.

## GRPO Recipe

Config:

```text
rlvr_demo/configs/qwen3_17b_aime_hardmath_correct_grpo_b200_dev_300.yaml
```

Run:

```bash
bash rlvr_demo/scripts/run_multi_math_grpo_b200.sh \
  rlvr_demo/configs/qwen3_17b_aime_hardmath_correct_grpo_b200_dev_300.yaml \
  experiment_name=qwen3-17b-aime-hardmath-correct-grpo-b200-dev-300-r1
```

Important settings:

| Setting | Value |
| --- | --- |
| Actor backend | `megatron:d2p1t1` |
| Rollout backend | `sglang:d2p1t1` |
| GPU topology | 2 Megatron actor GPUs + 2 SGLang rollout GPUs |
| Steps | 300 |
| Train batch | 16 prompts |
| GRPO samples | 8 |
| Max prompt length | 2,048 tokens |
| Max new tokens during training | 1,024 |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Reward | correctness only, no format bonus |
| Actor LR | `1e-6`, constant |
| Reward norm | group mean/std, group size 8 |
| PPO clip | `0.25` |
| KL | `0.0` |
| SGLang max running requests | 128 |
| SGLang static memory fraction | 0.60 |

The reviewed run completed in about 1,425 seconds. The best final checkpoint was
global step 299:

```text
/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-17b-aime-hardmath-correct-grpo-b200-dev-300-r1/trial0/default/epoch0epochstep299globalstep299
```

## SFT Recipe

Generate the RFT-SFT JSONL from the GRPO rollout logs with the extraction command
in the data section, then train:

```bash
bash rlvr_demo/scripts/run_multi_math_deepseek_sft_b200.sh \
  rlvr_demo/configs/qwen3_17b_aime_rollout_rft_sft_b200_300.yaml \
  experiment_name=qwen3-17b-aime-rollout-rft-sft-b200-300-r1
```

Important settings:

| Setting | Value |
| --- | --- |
| Backend | `megatron:d4p1t1` |
| GPU topology | all 4 B200 GPUs for Megatron SFT |
| Steps | 300 |
| Train batch | 16 examples |
| Max total length | 4,096 tokens |
| LR | `5e-7` |
| LR schedule | cosine |
| Warmup | 5% |
| Adam betas | `0.9`, `0.95` |
| Weight decay | `0.01` |
| Gradient clip | `1.0` |
| Checkpoint cadence | every 50 steps |

The reviewed run completed in about 87 seconds. The selected checkpoint is
global step 199:

```text
/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-17b-aime-rollout-rft-sft-b200-300-r1/trial0/default/epoch1epochstep16globalstep199
```

## Evaluation

Use the HF evaluator for final generated-answer scoring:

```bash
CUDA_VISIBLE_DEVICES=0 bash rlvr_demo/scripts/eval_multi_math_hf.sh \
  <MODEL_OR_CHECKPOINT> \
  rlvr_demo/results/<OUTPUT_DIR> \
  0 8 \
  --benchmarks aime_2026 \
  --seed <SEED> \
  --max-new-tokens 1536 \
  --max-prompt-length 2048 \
  --write-predictions
```

Run seeds `7`, `13`, and `21` for base, GRPO, and SFT.

Reviewed AIME-2026 results:

| Model / checkpoint | Seed 7 | Seed 13 | Seed 21 | Total |
| --- | ---: | ---: | ---: | ---: |
| Base `Qwen/Qwen3-1.7B` | 2/30 | 2/30 | 0/30 | 4/90 = 4.44% |
| GRPO step 299 | 3/30 | 3/30 | 4/30 | 10/90 = 11.11% |
| RFT-SFT step 199 | 2/30 | 2/30 | 2/30 | 6/90 = 6.67% |

Both final recipes improve the three-seed aggregate over the base model. GRPO is
the stronger recipe. RFT-SFT is weaker but has no seed-level regression against
the base seeds tested here.

Post-commit reproducibility rerun:

| Rerun | Seed 7 | Seed 13 | Seed 21 | Total |
| --- | ---: | ---: | ---: | ---: |
| GRPO `qwen3-17b-aime-hardmath-correct-grpo-b200-dev-300-repro1`, step 299 | 3/30 | 2/30 | 2/30 | 7/90 = 7.78% |
| RFT-SFT `qwen3-17b-aime-rollout-rft-sft-b200-300-repro1`, step 199 | 2/30 | 3/30 | 2/30 | 7/90 = 7.78% |

The GRPO rerun completed in 1,475.21 seconds. Extracting max-2 correct rollouts
from that rerun produced 5,326 supervised rows over 2,815 unique training
prompts. The SFT rerun trained on that freshly extracted JSONL and completed in
89.84 seconds. Both reruns remain positive against the same base aggregate of
4/90.

Discarded checks:

- Qwen3-0.6B base remained 0/30 at 1024 and 2048 new tokens.
- AIME-only and official-solution SFT variants did not consistently improve.
- DeepSeek-teacher AIME SFT tied or regressed on the three-seed aggregate.
- A max-4 rollout-SFT dataset overfit more quickly than max-2 and was worse on
  seed 7.

## Reproducibility Notes

- The benchmark is small, so report the exact seed list with any result.
- The AIME-2026 test set is never used for scheduled checkpoint selection.
- Use AIME-2024/2025 only as development holdouts for recipe selection.
- Keep `rlvr_demo/data/*.jsonl`, `rlvr_demo/results/`, `/NHNHOME/areal_runs`,
  rollouts, and checkpoints out of git.

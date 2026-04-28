# AIME-2026 Targeted Recipes

This note documents the AIME-2026 extension for `Qwen/Qwen3-1.7B`. The
benchmark is `MathArena/aime_2026`, and the training data is restricted to
problems from 2025 or earlier. The GRPO recipe uses AReaL with a Megatron actor
and SGLang rollouts. The SFT recipe uses AReaL Megatron SFT on verifier-correct
rollouts produced by the clean GRPO training prompt set.

Reviewer verdict: these are useful technical baselines and reproducibility
recipes, but the AIME-2026 numbers should be reported as exploratory baselines,
not as a sealed paper benchmark. AIME-2026 was not used in training or scheduled
checkpoint selection, but it was used during recipe iteration in this project.
A paper should freeze these recipes before evaluating on a newer untouched
benchmark, or explicitly label these AIME-2026 results as development-informed.

Final acceptance status: the recipes satisfy the requested practical
consistency criterion. Across the original run and an independent post-commit
rerun, both GRPO and rollout-SFT improve the three-seed AIME-2026 aggregate over
the base `Qwen/Qwen3-1.7B` score of 4/90, and no evaluated seed regresses below
the base result for that seed.

| Recipe run | Seed 7 | Seed 13 | Seed 21 | Aggregate |
| --- | ---: | ---: | ---: | ---: |
| Base `Qwen/Qwen3-1.7B` | 2/30 | 2/30 | 0/30 | 4/90 |
| GRPO original | 3/30 | 3/30 | 4/30 | 10/90 |
| GRPO rerun | 3/30 | 2/30 | 2/30 | 7/90 |
| Rollout-SFT original | 2/30 | 2/30 | 2/30 | 6/90 |
| Rollout-SFT rerun | 2/30 | 3/30 | 2/30 | 7/90 |

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

The GRPO training config does not load AIME-2026 at training time. A separate
audit checks exact and fuzzy overlap against AIME-2026. Removing the earlier
defensive AIME-2026 heldout entry from the training config changed 0 training
prompts:

| Check | Count |
| --- | ---: |
| Deduped source rows | 6,442 |
| Rows after AIME-2024/2025 holdout removal | 6,442 |
| Rows after AIME-2024/2025/2026 holdout removal | 6,442 |
| Symmetric difference | 0 |

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
  --seed 7 \
  --allowed-source-preset aime_hardmath_pre2024 \
  --fail-on-disallowed
```

This SFT recipe is best described as RFT-style supervised distillation. It is a
useful supervised baseline for the same clean prompt distribution, but it is not
an independent teacher-data baseline. Independent official-solution and
DeepSeek-teacher SFT attempts did not consistently improve AIME-2026 in these
short runs.

The strict extractor check on the reproducibility rollout logs selected 5,326
rows over 2,815 unique prompts and skipped 0 disallowed reward-passing rollouts.

Generated JSONL files, rollouts, checkpoints, and evaluation outputs are runtime
artifacts and are ignored by git.

## Split Audit

Run:

```bash
.venv/bin/python -m rlvr_demo.audit_aime2026_splits --fail-on-overlap
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

For the rerun RFT-SFT JSONL:

```bash
.venv/bin/python -m rlvr_demo.audit_aime2026_splits \
  --rft-jsonl rlvr_demo/data/qwen3_17b_hardmath_grpo_correct_rollout_sft_repro1_max2.jsonl \
  --fail-on-overlap
```

This also passed: the rerun RFT-SFT set had 2,815 unique prompts, 0 exact
AIME-2026 overlap, 0 fuzzy pairs above 0.90, and 0 prompts outside the GRPO
training prompt set.

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

Point estimates improve over the base model, but the benchmark has only 30
problems, so paper claims should include uncertainty and paired comparisons.
GRPO is the stronger recipe. RFT-SFT is weaker and should be described as
rollout-distillation rather than an independent SFT baseline.

Original three-seed statistical summary:

| Group | Seeds | Correct | Accuracy | Wilson 95% CI |
| --- | --- | ---: | ---: | --- |
| Base | `[7, 13, 21]` | 4/90 | 4.44% | 1.74% to 10.88% |
| GRPO step 299 | `[7, 13, 21]` | 10/90 | 11.11% | 6.15% to 19.26% |
| RFT-SFT step 199 | `[7, 13, 21]` | 6/90 | 6.67% | 3.09% to 13.79% |

Paired against base:

| Candidate | Candidate-only correct | Base-only correct | Two-sided sign-test p |
| --- | ---: | ---: | ---: |
| GRPO step 299 | 6 | 0 | 0.0312 |
| RFT-SFT step 199 | 3 | 1 | 0.6250 |

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

Rerun statistical summary:

| Group | Seeds | Correct | Accuracy | Wilson 95% CI |
| --- | --- | ---: | ---: | --- |
| Base | `[7, 13, 21]` | 4/90 | 4.44% | 1.74% to 10.88% |
| GRPO rerun step 299 | `[7, 13, 21]` | 7/90 | 7.78% | 3.82% to 15.19% |
| RFT-SFT rerun step 199 | `[7, 13, 21]` | 7/90 | 7.78% | 3.82% to 15.19% |

Paired against base:

| Candidate | Candidate-only correct | Base-only correct | Two-sided sign-test p |
| --- | ---: | ---: | ---: |
| GRPO rerun step 299 | 4 | 1 | 0.3750 |
| RFT-SFT rerun step 199 | 3 | 0 | 0.2500 |

These rerun p-values are not significant at conventional thresholds; they are
positive reproducibility checks, not strong standalone evidence.

Discarded checks:

- Qwen3-0.6B base remained 0/30 at 1024 and 2048 new tokens.
- AIME-only and official-solution SFT variants did not consistently improve.
- DeepSeek-teacher AIME SFT tied or regressed on the three-seed aggregate.
- A max-4 rollout-SFT dataset overfit more quickly than max-2 and was worse on
  seed 7.

## Reproducibility Notes

- The benchmark is small, so report the exact seed list with any result.
- AIME-2026 is never used for training or scheduled checkpoint selection, but it
  was used during recipe iteration. Do not claim these exact AIME-2026 numbers
  are from a sealed final benchmark in a future paper.
- Use AIME-2024/2025 only as development holdouts for recipe selection.
- Use `rlvr_demo.summarize_aime2026_results` to report Wilson intervals and
  paired sign tests whenever comparing recipes on AIME-2026.
- Keep `rlvr_demo/data/*.jsonl`, `rlvr_demo/results/`, `/NHNHOME/areal_runs`,
  rollouts, and checkpoints out of git.

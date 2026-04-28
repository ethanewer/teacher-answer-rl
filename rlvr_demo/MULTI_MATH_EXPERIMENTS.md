# Qwen3-0.6B Mixed-Math RLVR Baselines

This note documents the reviewed mixed-difficulty math baselines for
`Qwen/Qwen3-0.6B` on this 4x B200 node. The final recipes use AReaL with the
Megatron backend; the GRPO recipe uses SGLang rollouts.

The current reviewed baselines fix an earlier experimental flaw: older mixed
math runs used official test examples for scheduled GRPO validation. Those old
numbers are useful as exploration only. The results below select checkpoints on
a deterministic train-split validation holdout, then evaluate the selected
checkpoints once on the official test splits.

All commands assume:

```bash
cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL
```

## Research Survey

Recipe choices were guided by:

- Qwen3-0.6B model card:
  https://huggingface.co/Qwen/Qwen3-0.6B
- AReaL GSM8K GRPO tutorial:
  https://inclusionai.github.io/AReaL/lite/gsm8k_grpo.html
- AReaL allocation and parallelism docs:
  https://inclusionai.github.io/AReaL/developer/trainer/allocation_parallel.html
- DeepSeekMath GRPO paper:
  https://arxiv.org/abs/2402.03300
- DeepSeek-R1 Nature article:
  https://www.nature.com/articles/s41586-025-09422-z
- DeepScaleR paper:
  https://openreview.net/forum?id=I6GzDCne7U
- DeepScaleR preview data card:
  https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset
- OpenR1-Math-220k data card:
  https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- NuminaMath-CoT data card:
  https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
- GSM8K paper:
  https://arxiv.org/abs/2110.14168
- MATH paper:
  https://arxiv.org/abs/2103.03874

The important choices are:

- Use GRPO because math answers can be verified with a rule-based reward, so no
  critic model is needed.
- Use Qwen3 thinking-mode sampling with `temperature=0.6`, `top_p=0.95`, and
  `top_k=20`. The Qwen card discourages greedy decoding for thinking mode.
- Keep `max_new_tokens=512` for a short B200 recipe. Harder benchmarks would
  likely benefit from longer generations, but this setting keeps 250-step
  reruns practical.
- Use official GSM8K and MATH train/test splits for the final comparison so
  split hygiene is auditable. Large synthetic sets such as OpenR1, NuminaMath,
  and DeepScaleR are useful recipe references, but they were not used in the
  final clean baseline.

## Split Hygiene

Training sources:

- `openai/gsm8k`, subset `main`, split `train`
- `DigitalLearningGmbH/MATH-lighteval`, split `train`

Official test sources:

- `openai/gsm8k`, subset `main`, split `test`
- `DigitalLearningGmbH/MATH-lighteval`, split `test`

Every question is normalized by whitespace collapse and case folding, then
hashed with SHA-256. Duplicate train questions are dropped. Any train question
whose normalized hash appears in either official test split is removed.

Audit command:

```bash
.venv/bin/python -m rlvr_demo.audit_multi_math_splits \
  --deepseek-jsonl rlvr_demo/data/deepseek_v4_pro_multi_math_balanced_sft.jsonl
```

Reviewed audit, seed 7:

| Quantity | Count |
| --- | ---: |
| Raw train rows | 14,973 |
| Unique train rows | 14,972 |
| Official test rows | 6,319 |
| Unique train/test exact overlaps | 1 |
| Clean unique train rows | 14,971 |

Bucket counts:

| Split | GSM8K | MATH L1/2 | MATH L3 | MATH L4/5 | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean train pool | 7,473 | 1,912 | 1,592 | 3,992 | 2 |
| Shared validation holdout | 128 | 64 | 64 | 64 | 0 |
| GRPO train | 7,345 | 1,848 | 1,528 | 3,928 | 2 |
| Official-solution SFT train | 7,345 | 1,848 | 1,528 | 3,928 | 2 |
| Official test | 1,319 | 1,331 | 1,131 | 2,538 | 0 |

DeepSeek SFT teacher data:

| Split | GSM8K | MATH L1/2 | MATH L3 | MATH L4/5 |
| --- | ---: | ---: | ---: | ---: |
| Verified JSONL rows before shared-validation filter | 959 | 781 | 726 | 739 |
| Verified rows after shared-validation filter | 834 | 766 | 701 | 734 |
| DeepSeek SFT train | 802 | 734 | 669 | 702 |
| DeepSeek SFT validation | 32 | 32 | 32 | 32 |

Overlap checks from the reviewed audit:

| Check | Overlap |
| --- | ---: |
| GRPO train vs shared validation | 0 |
| GRPO train vs official test | 0 |
| Shared validation vs official test | 0 |
| Official-solution SFT train vs shared validation | 0 |
| Official-solution SFT train vs official test | 0 |
| DeepSeek SFT train vs shared validation | 0 |
| DeepSeek SFT train vs official test | 0 |
| DeepSeek SFT validation vs shared validation | 0 |
| DeepSeek SFT validation vs official test | 0 |

The raw DeepSeek JSONL still contains 170 verified rows that overlap the shared
validation holdout because they were generated during an earlier exploratory
pass. The SFT dataset loader filters them out, and the current generator samples
from the GRPO training pool so fresh regeneration avoids this overlap.

## Prompt And Reward

Both GRPO and SFT use the same user prompt:

```text
Please solve the math problem step by step. Use <think> tags to show your reasoning process, then provide the final answer.

Problem:
{question}

IMPORTANT: Put the final answer after `Final answer:`. For exact values, use a simplified exact expression or LaTeX. Do not include units or explanatory text after the final answer.

Format:
<think>
Your step-by-step reasoning here...
</think>
Final answer: [answer only]
```

The GRPO reward is:

```text
1.0 * math_verify(final_answer, gold_answer) + 0.1 * strict_format
```

`strict_format` requires a `<think>...</think>` block followed by
`Final answer:`. The reward verifies the extracted final answer first and falls
back to verifying the full completion.

## GRPO Recipe

Config:

```text
rlvr_demo/configs/qwen3_06b_multi_math_grpo_b200_250.yaml
```

Run:

```bash
bash rlvr_demo/scripts/run_multi_math_grpo_b200.sh \
  rlvr_demo/configs/qwen3_06b_multi_math_grpo_b200_250.yaml \
  experiment_name=qwen3-06b-multi-math-grpo-b200-250-reviewed-v2
```

Important settings:

| Setting | Value |
| --- | --- |
| Actor backend | `megatron:d2p1t1` |
| Rollout backend | `sglang:d2p1t1` |
| GPU topology | 2 Megatron actor GPUs + 2 SGLang rollout GPUs |
| Steps | 250 |
| Train batch | 32 prompts |
| GRPO samples | 8 |
| Max prompt length | 1,536 tokens |
| Max new tokens | 512 |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Actor LR | `5e-6`, constant |
| Reward norm | group mean/std, group size 8 |
| PPO clip | `0.4` |
| KL | `0.0` |
| SGLang context | 3,072 |
| SGLang max running requests | 192 |
| SGLang static memory fraction | 0.60 |

The reviewed 250-step rerun completed in 988.22 seconds after initialization.

AReaL train-holdout validation:

| Step | Correct | Accuracy | Mean reward | Strict format | Avg gen len | No EOS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 109/320 | 34.06% | 0.3566 | 15.94% | 487.1 | 75.62% |
| 50 | 159/320 | 49.69% | 0.5931 | 96.25% | 214.1 | 4.69% |
| 100 | 171/320 | 53.44% | 0.6319 | 97.50% | 249.1 | 2.81% |
| 150 | 166/320 | 51.88% | 0.6181 | 99.38% | 206.2 | 0.63% |
| 200 | 167/320 | 52.19% | 0.6194 | 97.50% | 248.3 | 2.81% |
| 250 | 102/320 | 31.87% | 0.3778 | 59.06% | 337.7 | 40.31% |

HF generated-answer validation on the same shared train holdout:

| Global step | Correct | Accuracy | Mean reward | Strict format |
| ---: | ---: | ---: | ---: | ---: |
| 49 | 167/320 | 52.19% | 0.6194 | 97.50% |
| 99 | 178/320 | 55.62% | 0.6522 | 95.94% |
| 149 | 162/320 | 50.62% | 0.6059 | 99.69% |
| 199 | 167/320 | 52.19% | 0.6197 | 97.81% |
| 249 | 99/320 | 30.94% | 0.3700 | 60.62% |

Selection rule: choose the scheduled checkpoint with the best HF generated-answer
accuracy on `mixed_train_validation`; break ties by mean reward, then earlier
global step. The reviewed run selects global step 99:

```text
/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-06b-multi-math-grpo-b200-250-reviewed-v2/trial0/default/epoch0epochstep99globalstep99
```

## DeepSeek SFT Recipe

Generate teacher data:

```bash
bash rlvr_demo/scripts/generate_multi_math_deepseek_sft.sh
```

The generator reads `DEEPSEEK_API_KEY` from
`/NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/.env`, uses `deepseek-v4-pro`, sets
`reasoning_effort=high`, enables thinking mode, and uses 128-way concurrency.
The JSONL is intentionally ignored by git:

```text
rlvr_demo/data/deepseek_v4_pro_multi_math_balanced_sft.jsonl
```

Train:

```bash
bash rlvr_demo/scripts/run_multi_math_deepseek_sft_b200.sh \
  rlvr_demo/configs/qwen3_06b_multi_math_deepseek_sft_b200_250.yaml \
  experiment_name=qwen3-06b-multi-math-deepseek-sft-b200-250-reviewed-v2
```

Important settings:

| Setting | Value |
| --- | --- |
| Backend | `megatron:d4p1t1` |
| GPU topology | all 4 B200 GPUs for Megatron SFT |
| Steps | 250 |
| Train batch | 32 examples |
| Max total length | 4,096 tokens |
| LR | `6e-6` |
| LR schedule | cosine |
| Warmup | 3% |
| Adam betas | `0.9`, `0.95` |
| Weight decay | `0.01` |
| Gradient clip | `1.0` |
| Checkpoint cadence | every 50 steps |

The reviewed SFT rerun completed in 69.95 seconds after initialization.

HF generated-answer validation on the shared train holdout:

| Global step | Correct | Accuracy | Mean reward | Strict format |
| ---: | ---: | ---: | ---: | ---: |
| 49 | 158/320 | 49.38% | 0.5819 | 88.12% |
| 99 | 145/320 | 45.31% | 0.5428 | 89.69% |
| 149 | 153/320 | 47.81% | 0.5653 | 87.19% |
| 199 | 154/320 | 48.12% | 0.5700 | 88.75% |
| 249 | 150/320 | 46.88% | 0.5556 | 86.88% |

Selection rule is the same as GRPO. The reviewed run selects global step 49:

```text
/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-06b-multi-math-deepseek-sft-b200-250-reviewed-v2/trial0/default/epoch0epochstep49globalstep49
```

## Evaluation Commands

Checkpoint validation sweep:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash rlvr_demo/scripts/eval_multi_math_validation_sweep.sh \
  qwen3-06b-multi-math-deepseek-sft-b200-250-reviewed-v2 \
  rlvr_demo/results/reviewed_v2_sft_validation \
  0 128

CUDA_VISIBLE_DEVICES=1 \
bash rlvr_demo/scripts/eval_multi_math_validation_sweep.sh \
  qwen3-06b-multi-math-grpo-b200-250-reviewed-v2 \
  rlvr_demo/results/reviewed_v2_grpo_validation_hf \
  0 128
```

Final official-test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash rlvr_demo/scripts/eval_multi_math_hf.sh \
  Qwen/Qwen3-0.6B \
  rlvr_demo/results/reviewed_v2_base_full \
  0 128

CUDA_VISIBLE_DEVICES=1 \
bash rlvr_demo/scripts/eval_multi_math_hf.sh \
  /NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-06b-multi-math-grpo-b200-250-reviewed-v2/trial0/default/epoch0epochstep99globalstep99 \
  rlvr_demo/results/reviewed_v2_grpo_step100_full \
  0 128

CUDA_VISIBLE_DEVICES=2 \
bash rlvr_demo/scripts/eval_multi_math_hf.sh \
  /NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-06b-multi-math-deepseek-sft-b200-250-reviewed-v2/trial0/default/epoch0epochstep49globalstep49 \
  rlvr_demo/results/reviewed_v2_deepseek_sft_step50_full \
  0 128
```

`--limit 0` means full benchmark slices. The evaluator uses one sampled
generation per example with seed 7 and the same Qwen3 sampling settings as
training.

## Reviewed Full-Test Results

The final table uses all 6,319 official test examples:

| Model | GSM8K | MATH L1/2 | MATH L3 | MATH L4/5 | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base `Qwen/Qwen3-0.6B` | 696/1319 = 52.77% | 506/1331 = 38.02% | 257/1131 = 22.72% | 244/2538 = 9.61% | 1703/6319 = 26.95% |
| GRPO step 99 | 879/1319 = 66.64% | 835/1331 = 62.73% | 508/1131 = 44.92% | 547/2538 = 21.55% | 2769/6319 = 43.82% |
| DeepSeek SFT step 49 | 763/1319 = 57.85% | 772/1331 = 58.00% | 481/1131 = 42.53% | 496/2538 = 19.54% | 2512/6319 = 39.75% |

Format rates on the same full test:

| Model | Overall format rate | Overall mean reward |
| --- | ---: | ---: |
| Base `Qwen/Qwen3-0.6B` | 10.56% | 0.2801 |
| GRPO step 99 | 96.77% | 0.5350 |
| DeepSeek SFT step 49 | 82.32% | 0.4799 |

Conclusion:

- Both reviewed recipes improve every difficulty bucket over the base model.
- GRPO is the strongest baseline overall: +16.87 percentage points over base
  on the full mixed official test.
- DeepSeek SFT is also positive: +12.80 percentage points over base, with much
  lower training wall time but additional teacher-generation cost.
- The final checkpoint is not safe for GRPO; step 249 collapsed on validation.
  Always select from scheduled checkpoints using the shared train-holdout
  validation sweep.

## Limitations

These are acceptable single-seed baselines for future work on this node, but a
paper making strong superiority claims should add:

- At least three training seeds for GRPO and SFT.
- Multiple sampled evaluation seeds or a deterministic decoding protocol.
- A stronger near-duplicate contamination audit; the current audit checks exact
  normalized question overlap.
- Longer-context variants for very hard MATH problems if wall time permits.

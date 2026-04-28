# Qwen3-0.6B Mixed-Math RLVR Experiments

This note documents the harder math extension of the Qwen3-0.6B RLVR demo.
The goal is a short, reproducible comparison between one GRPO recipe and one
SFT recipe on a range from grade-school arithmetic through MATH Level 5
competition problems.

All commands assume:

```bash
cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL
```

## Research Survey

The recipe choices are based on these references:

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

The main takeaways I used:

- GRPO is appropriate here because math answers can be verified by a rule-based
  reward, avoiding a critic model.
- Qwen3 thinking-mode sampling should use `temperature=0.6`, `top_p=0.95`, and
  `top_k=20`; the Qwen card explicitly discourages greedy decoding for thinking
  mode.
- Recent math-RL recipes such as DeepScaleR rely on verified math problems and
  longer context windows as difficulty rises. For this short B200 demo I kept
  `max_new_tokens=512` so 250-step runs finish quickly, but the dataset and
  configs are structured so the same recipe can be lengthened later.
- I did not train on OpenR1, NuminaMath, or DeepScaleR preview data in the final
  clean comparison because those datasets are large synthetic/curated mixtures
  with no official held-out split in the same format. They are useful recipe
  references, but the final experiment uses official GSM8K and MATH train/test
  splits so contamination control is explicit.
- The final SFT comparison uses DeepSeek V4 Pro high-reasoning solutions for
  questions sampled only from those cleaned training splits. Rows are admitted
  only when the teacher final answer verifies against the official train answer.

## Data

Training data:

- `openai/gsm8k`, subset `main`, split `train`
- `DigitalLearningGmbH/MATH-lighteval`, split `train`

Held-out test data:

- `openai/gsm8k`, subset `main`, split `test`
- `DigitalLearningGmbH/MATH-lighteval`, split `test`

Counts before token filtering:

| Split | GSM8K | MATH | Total |
| --- | ---: | ---: | ---: |
| Train | 7,473 | 7,500 | 14,973 |
| Test | 1,319 | 5,000 | 6,319 |

Contamination controls:

- Every question is normalized with whitespace collapse and case folding, then
  hashed with SHA-256.
- Duplicate train questions are dropped.
- Any train question whose normalized hash appears in either official test split
  is removed.
- This removed 1 train row in the mixed GSM8K+MATH setup.
- SFT validation is a 512-example deterministic holdout from the cleaned train
  questions, not from either official test split.

Usable tokenized rows:

| Dataset | Max length | Rows |
| --- | ---: | ---: |
| GRPO train | 1,536 prompt tokens | 14,969 |
| SFT train | 3,072 total tokens | 14,459 |
| SFT validation | 3,072 total tokens | 512 |

DeepSeek teacher SFT data for the final recipe:

| Bucket | Verified rows |
| --- | ---: |
| GSM8K train | 501 |
| MATH Level 1/2 train | 461 |
| MATH Level 3 train | 435 |
| MATH Level 4/5 train | 408 |
| Total unique verified rows | 1,805 |

The teacher generator also wrote 244 `wrong` rows. They are kept in the ignored
JSONL for auditability but are not loaded by the SFT dataset helper. At
`max_length=4096`, the final DeepSeek SFT split has 1,659 train rows and a
128-row deterministic train-split validation holdout.

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
`Final answer:`. The reward tries the extracted final answer first and falls
back to verifying the full completion.

## GRPO Recipe

Config:

```text
rlvr_demo/configs/qwen3_06b_multi_math_grpo_b200_250.yaml
```

Run:

```bash
bash rlvr_demo/scripts/run_multi_math_grpo_b200.sh
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
| Max new tokens | 512 |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Actor LR | `5e-6`, constant |
| Reward norm | group mean/std, group size 8 |
| PPO clip | `0.4` |
| KL | `0.0` |
| SGLang context | 3,072 |
| SGLang max running requests | 192 |
| SGLang static memory fraction | 0.60 |

The config writes checkpoints and aggregate validation rollouts every 50 steps.
Validation uses 320 total examples: 128 GSM8K test, 64 MATH Level 1/2, 64 MATH
Level 3, and 64 MATH Level 4/5.

Summarize the AReaL validation dumps:

```bash
.venv/bin/python -m rlvr_demo.summarize_eval_rollouts \
  /NHNHOME/areal_runs/qwen3-gsm8k-rlvr/logs/ewer/qwen3-06b-multi-math-grpo-b200-250/trial0/eval-rollout
```

## SFT Recipes

### Official-Solution Ablation

Config:

```text
rlvr_demo/configs/qwen3_06b_multi_math_sft_b200_250.yaml
```

Run:

```bash
bash rlvr_demo/scripts/run_multi_math_sft_b200.sh
```

Important settings:

| Setting | Value |
| --- | --- |
| Backend | `megatron:d4p1t1` |
| GPU topology | all 4 B200 GPUs for Megatron SFT |
| Steps | 250 |
| Train batch | 32 examples |
| Max total length | 3,072 tokens |
| LR | `1.5e-5` |
| LR schedule | cosine |
| Warmup | 3% |
| Adam betas | `0.9`, `0.95` |
| Weight decay | `0.01` |
| Gradient clip | `1.0` |

This ablation uses the same cleaned train questions as GRPO, with the official
GSM8K and MATH train-split solutions as supervised traces. It is matched at the
question level and avoids test answers, but it underperformed: step 100 reached
only 161/512 average generated-answer correctness on the four 128-example eval
buckets, worse than the base model. I kept it as a negative control, not as the
recommended SFT recipe.

### Final DeepSeek SFT Recipe

Generate the teacher data:

```bash
bash rlvr_demo/scripts/generate_multi_math_deepseek_sft.sh
```

The generator loads `DEEPSEEK_API_KEY` from
`/NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/.env`, uses model
`deepseek-v4-pro`, enables thinking mode, sets `reasoning_effort=high`, and runs
with 128-way concurrency. The data file is ignored by git:

```text
rlvr_demo/data/deepseek_v4_pro_multi_math_balanced_sft.jsonl
```

Normal reruns are idempotent and skip questions already present in the JSONL.
Pass `--retry-failed` to `rlvr_demo.generate_deepseek_multi_math_sft` only when
you intentionally want to retry existing `wrong` or `error` rows.

Train:

```bash
bash rlvr_demo/scripts/run_multi_math_deepseek_sft_b200.sh
```

Config:

```text
rlvr_demo/configs/qwen3_06b_multi_math_deepseek_sft_b200_250.yaml
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

The 250-step SFT run completed in 71.05 seconds after initialization on this
node. Generated-answer accuracy peaked around steps 100-150; for the recorded
run, the recommended checkpoint is step 100 because it tied step 150 on average
while keeping better GSM8K accuracy and better Level 4/5 formatting.

## Final Evaluation

The final generated-answer comparison uses `rlvr_demo.eval_hf_multi_math` with
the same prompt, answer extraction, `math_verify` scoring, and Qwen3 sampling
settings as training. Results are written under ignored `rlvr_demo/results/`.

Example for a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
HF_HOME=/NHNHOME/areal_cache/huggingface \
TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0 \
.venv/bin/python -m rlvr_demo.eval_hf_multi_math \
  --model /path/to/checkpoint \
  --output-dir rlvr_demo/results/example_eval_l128 \
  --limit 128 \
  --batch-size 64
```

Benchmarks:

- `gsm8k_test`: first 128 official GSM8K test examples
- `math_test_l12`: first 128 official MATH test Level 1/2 examples
- `math_test_l3`: first 128 official MATH test Level 3 examples
- `math_test_l45`: first 128 official MATH test Level 4/5 examples

## Results

### GRPO Training Validation

AReaL validation used 320 held-out prompts: 128 GSM8K test, 64 MATH Level 1/2,
64 MATH Level 3, and 64 MATH Level 4/5.

| Step | Correct | Accuracy | Mean reward | Strict format | Avg gen len | No EOS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 123/320 | 38.44% | 0.4016 | 17.19% | 486.0 | 75.00% |
| 50 | 194/320 | 60.62% | 0.7009 | 94.69% | 250.2 | 5.31% |
| 100 | 201/320 | 62.81% | 0.7272 | 99.06% | 228.9 | 0.94% |
| 150 | 187/320 | 58.44% | 0.6841 | 99.69% | 259.8 | 0.31% |
| 200 | 182/320 | 56.87% | 0.6681 | 99.38% | 269.9 | 0.63% |
| 250 | 174/320 | 54.37% | 0.6438 | 100.00% | 201.8 | 0.00% |

Recommended GRPO checkpoint:

```text
/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-06b-multi-math-grpo-b200-250/trial0/default/epoch0epochstep99globalstep99
```

### Generated-Answer Evaluation

Each cell below is accuracy on the first 128 examples of that official held-out
benchmark slice. The average column is over all 512 evaluated examples.

| Model / checkpoint | GSM8K | MATH L1/2 | MATH L3 | MATH L4/5 | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base `Qwen/Qwen3-0.6B` | 57.03% | 56.25% | 28.12% | 13.28% | 38.67% |
| GRPO step 100 | 68.75% | 82.81% | 64.84% | 39.84% | 64.06% |
| DeepSeek SFT step 50 | 59.38% | 81.25% | 60.16% | 34.38% | 58.79% |
| DeepSeek SFT step 100 | 62.50% | 80.47% | 64.84% | 42.97% | 62.70% |
| DeepSeek SFT step 150 | 59.38% | 82.03% | 67.19% | 42.19% | 62.70% |
| DeepSeek SFT step 200 | 59.38% | 80.47% | 67.19% | 40.62% | 61.91% |
| DeepSeek SFT step 250 | 63.28% | 78.91% | 60.16% | 38.28% | 60.16% |
| Official-solution SFT step 100 | 34.38% | 51.56% | 22.66% | 17.19% | 31.45% |

Final choices:

- GRPO recipe: train 250 steps, then select the best scheduled validation
  checkpoint from steps 50/100/150/200. The recorded run's best checkpoint was
  step 100; a reproducibility rerun peaked at step 200 and then collapsed at
  step 250, so the selection rule matters.
- SFT recipe: train the DeepSeek teacher recipe for 250 steps, select the
  step-100 checkpoint. It improves every difficulty bucket over base and is
  strongest on Level 4/5 in this evaluation, but it remains slightly behind GRPO
  on the overall average.

Recommended SFT checkpoint:

```text
/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-06b-multi-math-deepseek-sft-b200-250/trial0/default/epoch1epochstep48globalstep99
```

## Reproducibility Rerun

After the final scripts were in place, I reran the final recipes under separate
experiment names so the original artifacts were left intact.

DeepSeek data generation:

```bash
bash rlvr_demo/scripts/generate_multi_math_deepseek_sft.sh
```

Result: idempotent rerun found no remaining questions because the ignored JSONL
already contained all 2,048 selected train questions.

GRPO rerun:

```bash
bash rlvr_demo/scripts/run_multi_math_grpo_b200.sh \
  rlvr_demo/configs/qwen3_06b_multi_math_grpo_b200_250.yaml \
  experiment_name=qwen3-06b-multi-math-grpo-b200-250-repro1
```

This completed 250 steps in 1042.21 seconds after initialization. AReaL
validation summary:

| Step | Correct | Accuracy | Mean reward | Strict format | Avg gen len | No EOS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 137/320 | 42.81% | 0.4447 | 16.56% | 481.4 | 74.38% |
| 50 | 187/320 | 58.44% | 0.6834 | 99.06% | 193.4 | 0.94% |
| 100 | 180/320 | 56.25% | 0.6591 | 96.56% | 276.8 | 5.31% |
| 150 | 181/320 | 56.56% | 0.6634 | 97.81% | 281.0 | 3.44% |
| 200 | 200/320 | 62.50% | 0.7216 | 96.56% | 274.3 | 15.31% |
| 250 | 0/320 | 0.00% | 0.0000 | 0.00% | 510.4 | 100.00% |

This reproduced the intended short-run gain and the late-collapse risk. Use the
best validation checkpoint, not the final checkpoint.

DeepSeek SFT rerun:

```bash
bash rlvr_demo/scripts/run_multi_math_deepseek_sft_b200.sh \
  rlvr_demo/configs/qwen3_06b_multi_math_deepseek_sft_b200_250.yaml \
  experiment_name=qwen3-06b-multi-math-deepseek-sft-b200-250-repro1
```

This completed 250 steps in 70.51 seconds after initialization. The final
train loss was 0.0660 and the held-out teacher-forcing eval loss was 0.6286,
matching the original run closely enough for recipe reproducibility. As with
the recorded run, generated-answer checkpoint selection should be done on saved
checkpoints rather than by final SFT loss alone.

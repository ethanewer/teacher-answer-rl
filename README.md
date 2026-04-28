# Qwen3-0.6B RLVR Math Demo on AReaL

This project builds a short-running math RLVR demo for `Qwen/Qwen3-0.6B` using
[AReaL](https://github.com/inclusionAI/AReaL), the Megatron backend, and SGLang
rollouts on the 4x NVIDIA B200 node.

The working code lives under `AReaL/rlvr_demo`. Runtime logs, rollout dumps, and
checkpoints are written to `/NHNHOME/areal_runs/qwen3-gsm8k-rlvr` and are not
committed.

## AIME-2026 Targeted Result

The AIME-2026 extension is documented in
`AReaL/rlvr_demo/AIME_2026_EXPERIMENTS.md`. It adds
`MathArena/aime_2026` as a final benchmark and uses only training prompts from
2025 or earlier. `Qwen/Qwen3-0.6B` scored 0/30 on AIME-2026 in short baseline
checks, so the final targeted recipes use `Qwen/Qwen3-1.7B`.

Final AIME-2026 generated-answer results over seeds 7, 13, and 21:

| Model / checkpoint | Seed 7 | Seed 13 | Seed 21 | Total |
| --- | ---: | ---: | ---: | ---: |
| Base `Qwen/Qwen3-1.7B` | 2/30 | 2/30 | 0/30 | 4/90 = 4.44% |
| GRPO step 299 | 3/30 | 3/30 | 4/30 | 10/90 = 11.11% |
| RFT-SFT step 199 | 2/30 | 2/30 | 2/30 | 6/90 = 6.67% |

Post-commit reruns were also positive: GRPO step 299 reran at 7/90, and
RFT-SFT step 199 reran at 7/90 after regenerating the supervised rollout JSONL
from the rerun GRPO logs.

Recommended AIME-2026 recipes:

- GRPO: `AReaL/rlvr_demo/configs/qwen3_17b_aime_hardmath_correct_grpo_b200_dev_300.yaml`.
  This uses AReaL with a Megatron actor and SGLang rollouts on a 2+2 B200 split.
- SFT: `AReaL/rlvr_demo/configs/qwen3_17b_aime_rollout_rft_sft_b200_300.yaml`.
  This is supervised distillation from verifier-correct GRPO rollouts over the
  same clean pre-2026 prompt set.

Run the split audit with:

```bash
cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL
.venv/bin/python -m rlvr_demo.audit_aime2026_splits
```

The reviewed audit found 0 exact AIME-2026 train/test overlaps, 0 fuzzy matches
above 0.90, and 0 RFT-SFT prompts outside the GRPO training prompt set.

## Reviewed Mixed-Difficulty Math Result

The harder mixed-math extension is documented in
`AReaL/rlvr_demo/MULTI_MATH_EXPERIMENTS.md`. It trains on cleaned official
GSM8K train plus MATH train, removes any normalized question overlap with the
official test splits, and evaluates on held-out GSM8K plus MATH Level 1/2,
Level 3, and Level 4/5 buckets.

The reviewed baseline fixed an earlier exploratory flaw where scheduled GRPO
validation used official test examples. Current checkpoint selection uses only a
deterministic train-split validation holdout; official test examples are used
only for the final table below.

Full official-test generated-answer results, 6,319 examples total:

| Model / checkpoint | GSM8K | MATH L1/2 | MATH L3 | MATH L4/5 | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base `Qwen/Qwen3-0.6B` | 696/1319 = 52.77% | 506/1331 = 38.02% | 257/1131 = 22.72% | 244/2538 = 9.61% | 1703/6319 = 26.95% |
| GRPO step 99 | 879/1319 = 66.64% | 835/1331 = 62.73% | 508/1131 = 44.92% | 547/2538 = 21.55% | 2769/6319 = 43.82% |
| DeepSeek SFT step 49 | 763/1319 = 57.85% | 772/1331 = 58.00% | 481/1131 = 42.53% | 496/2538 = 19.54% | 2512/6319 = 39.75% |

The recommended final recipes are:

- GRPO: `AReaL/rlvr_demo/configs/qwen3_06b_multi_math_grpo_b200_250.yaml`,
  run with `AReaL/rlvr_demo/scripts/run_multi_math_grpo_b200.sh`. This uses
  AReaL with a Megatron actor and SGLang rollouts on a 2+2 B200 split. Train
  250 steps and select the best scheduled checkpoint on
  `mixed_train_validation`. The reviewed run selected global step 99 and the
  final checkpoint degraded, so validation selection is required.
- SFT: generate DeepSeek V4 Pro high-reasoning teacher data with
  `AReaL/rlvr_demo/scripts/generate_multi_math_deepseek_sft.sh`, then train
  `AReaL/rlvr_demo/configs/qwen3_06b_multi_math_deepseek_sft_b200_250.yaml`
  with `AReaL/rlvr_demo/scripts/run_multi_math_deepseek_sft_b200.sh`. Train
  250 steps and select the best scheduled checkpoint on
  `mixed_train_validation`. The reviewed run selected global step 49.

The generated DeepSeek JSONL, HF eval outputs, rollouts, and checkpoints are
runtime artifacts and are intentionally not committed.

The older GSM8K-only section below is preserved as setup history and a simpler
smoke baseline. For new research baselines, use the reviewed mixed-math recipes
above.

## GSM8K Result

All accuracy numbers below use the same first 256 GSM8K test examples unless
noted otherwise.

| Recipe | Engine used for score | Train wall time | Correct | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Base Qwen3-0.6B | HF evaluator | n/a | 107/256 | 41.80% |
| GRPO fast, seed 7, step 10 | AReaL + SGLang eval | 64.23s after init | 176/256 | 68.75% |
| GRPO fast, seed 7, checkpoint | HF evaluator | 64.23s after init | 162/256 | 63.28% |
| SFT DeepSeek matched, step 20 | HF evaluator | part of 60-step run | 149/256 | 58.20% |
| SFT DeepSeek matched, step 30 | HF evaluator | part of 60-step run | 155/256 | 60.55% |
| SFT DeepSeek matched, step 40 | HF evaluator | part of 60-step run | 152/256 | 59.38% |
| SFT DeepSeek matched, step 50 | HF evaluator | part of 60-step run | 176/256 | 68.75% |
| SFT DeepSeek matched, step 60 | HF evaluator | 46.54s after init | 150/256 | 58.59% |

Takeaways:

- The optimized GRPO recipe is reproducibly positive: three seeds improved from
  43.23% mean initial accuracy to 65.50% mean final accuracy in about one minute.
- The longer report-style AReaL run reproduced and exceeded the original report
  target: GSM8K full-test accuracy improved from 550/1319 (41.70%) to 945/1319
  (71.65%) after 180 steps.
- Matched SFT can also improve the model. Its best checkpoint reached 176/256
  (68.75%), but generated-answer accuracy peaked before the 60-step wall-clock
  endpoint while teacher-forcing loss kept dropping. Select SFT checkpoints by
  generated GSM8K accuracy, not SFT loss alone.
- GRPO remains the most consistent short recipe because it reached large gains
  across seeds without the extra DeepSeek teacher-generation cost. The SFT recipe
  also reran successfully, but its exact peak generated score moved across
  checkpoints and reruns.

Reproducibility rerun after finalizing the scripts:

| Recipe rerun | Engine used for score | Result |
| --- | --- | --- |
| DeepSeek generator | local generator | idempotent rerun reported no remaining items after adding `--retry-failed` gating |
| GRPO fast rerun, seed 7 | AReaL + SGLang eval | 97/256 = 37.89% to 146/256 = 57.03% in 61.51s after init |
| SFT 60-step rerun | AReaL Megatron train | completed in 46.47s after init; SFT eval loss 0.0342 at step 60 |
| SFT rerun checkpoint eval | HF evaluator | step 40: 158/256 = 61.72%; step 50: 157/256 = 61.33%; step 60: 152/256 = 59.38% |

## Node-Specific Setup

The successful setup uses the local venv in `AReaL/.venv` and B200-friendly
environment values in the wrapper scripts:

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `NCCL_SOCKET_IFNAME=eth0`
- `GLOO_SOCKET_IFNAME=eth0`
- `NCCL_CUMEM_ENABLE=0`
- `NCCL_NVLS_ENABLE=0`
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `HF_HOME=/NHNHOME/areal_cache/huggingface`
- `TRITON_CACHE_DIR=/NHNHOME/areal_cache/triton`

One AReaL local patch was required: `AReaL/areal/api/cli_args.py` launches worker
subprocesses with the active `sys.executable` instead of hard-coded `python3`.
Without this, worker processes can miss the venv CUDA/Python packages.

## Online Recipe Sources

I used current public guidance for the key hyperparameters:

- Qwen3-0.6B model card recommends thinking mode with `temperature=0.6`,
  `top_p=0.95`, `top_k=20`, and not greedy decoding:
  https://huggingface.co/Qwen/Qwen3-0.6B
- AReaL's GSM8K GRPO docs describe the AReaL separation between training and
  SGLang inference and the `RemoteSGLangEngine` flow:
  https://inclusionai.github.io/AReaL/lite/gsm8k_grpo.html
- AReaL's allocation docs describe Megatron-style data/tensor/pipeline
  parallelism and SGLang inference allocation:
  https://inclusionai.github.io/AReaL/developer/trainer/allocation_parallel.html
- DeepSeek V4 API docs list `deepseek-v4-pro`, OpenAI ChatCompletions support,
  and thinking/non-thinking mode:
  https://api-docs.deepseek.com/news/news260424
- DeepSeek thinking-mode docs specify `reasoning_effort="high"` and
  `thinking.enabled` for OpenAI-format requests:
  https://api-docs.deepseek.com/guides/thinking_mode

## RL Recipe

Fast optimized config:

```bash
cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL
bash rlvr_demo/scripts/run_b200_fast.sh \
  rlvr_demo/configs/qwen3_06b_gsm8k_areal_megatron_b200_fast.yaml
```

Important settings:

- Actor backend: `megatron:d2p1t1`
- Rollout backend: `sglang:d2p1t1`
- GPU split: 2 actor GPUs + 2 SGLang rollout GPUs
- Train batch: 32 prompts
- GRPO samples: 8
- Sampling: `temperature=0.6`, `top_p=0.95`, `top_k=20`
- SGLang: FlashInfer attention, `max_running_requests=192`,
  `mem_fraction_static=0.55`
- Actor optimizer: Adam, `lr=8e-6`, constant LR, `eps_clip=0.4`,
  group reward normalization

Seed sweep:

| Seed | Initial | Final | Gain |
| ---: | ---: | ---: | ---: |
| 7 | 113/256 = 44.14% | 176/256 = 68.75% | +24.61 pp |
| 8 | 108/256 = 42.19% | 169/256 = 66.02% | +23.83 pp |
| 9 | 111/256 = 43.36% | 158/256 = 61.72% | +18.36 pp |

Report-style full eval config:

```bash
bash rlvr_demo/scripts/run_b200_fast.sh \
  rlvr_demo/configs/qwen3_06b_gsm8k_areal_megatron_report.yaml
```

This ran 180 GRPO steps with full GSM8K test eval and reached 71.65%.

## DeepSeek Matched SFT Data

The SFT generator reads the same unique GSM8K train questions that appeared in
the GRPO rollout dump, then asks DeepSeek V4 Pro to solve each item.

```bash
cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL
.venv/bin/python -m rlvr_demo.generate_deepseek_sft \
  --concurrency 128 \
  --max-tokens 2048 \
  --log-every 16
```

The API key is loaded from `/NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/.env`.
Do not print or commit it.

Generation details:

- Model: `deepseek-v4-pro`
- Thinking: enabled
- Reasoning effort: `high`
- Concurrency: 128
- Source data: 391 unique questions from the fast GRPO training rollouts
- First full generation statuses: 379 `ok`, 12 `wrong`, 0 `error`
- Repro rerun retried the 12 wrong questions once; they remained wrong. The
  local JSONL now has 403 lines, but still only 379 successful teacher rows.
- Tokenized usable SFT rows at max length 2048: 374
- Generation wall time: 512.4s

The generated JSONL is intentionally ignored by git. Regenerate it with the
command above when needed. Normal reruns are idempotent; pass `--retry-failed`
if you intentionally want to retry rows already marked `wrong` or `error`.

## SFT Recipe

```bash
cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL
bash rlvr_demo/scripts/run_sft_b200.sh \
  rlvr_demo/configs/qwen3_06b_gsm8k_sft_megatron_b200_matched.yaml \
  experiment_name=qwen3-06b-gsm8k-sft-deepseek-matched-60 \
  total_train_steps=60
```

Important settings:

- Actor backend: `megatron:d4p1t1`
- GPU use: all 4 B200s for data parallel SFT
- Batch size: 32
- Max length: 2048
- Optimizer: Adam, `lr=2e-5`, cosine schedule, `beta2=0.95`,
  `weight_decay=0.01`, grad clip 1.0, 3% warmup
- Checkpoint/eval cadence: every 10 steps

SFT train loss and held-out teacher-forcing eval loss fell through step 60:
`0.7027 -> 0.0362`. Generated GSM8K accuracy peaked at step 50 and dropped at
step 60, which is the first clear overfitting/misalignment signal for this small
matched dataset.

## Evaluation

The AReaL/SGLang training evaluator is used during GRPO. A standalone AReaL
SGLang eval run for SFT stalled at 96/256 outputs with idle GPUs, so SFT
checkpoint comparison uses `rlvr_demo.eval_hf_gsm8k`, a simple batched
Transformers evaluator with the same prompt, sampling values, and reward parser.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
HF_HOME=/NHNHOME/areal_cache/huggingface \
TRANSFORMERS_NO_TF=1 USE_TF=0 USE_FLAX=0 \
.venv/bin/python -m rlvr_demo.eval_hf_gsm8k \
  --model /NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/ewer/qwen3-06b-gsm8k-sft-deepseek-matched-60/trial0/default/epoch4epochstep5globalstep49 \
  --output rlvr_demo/results/sft_step50_hf_eval.json \
  --predictions rlvr_demo/results/sft_step50_hf_predictions.jsonl \
  --limit 256 \
  --batch-size 64
```

`rlvr_demo/results/` is ignored by git.

## Files Added

- `AReaL/rlvr_demo/train_qwen3_gsm8k.py`: AReaL GRPO entrypoint.
- `AReaL/rlvr_demo/train_qwen3_gsm8k_sft.py`: AReaL SFT entrypoint.
- `AReaL/rlvr_demo/qwen3_gsm8k_data.py`: GSM8K prompt/data loader.
- `AReaL/rlvr_demo/qwen3_gsm8k_reward.py`: numeric reward and format parser.
- `AReaL/rlvr_demo/generate_deepseek_sft.py`: async DeepSeek teacher data generation.
- `AReaL/rlvr_demo/sft_data.py`: matched rollout question extraction and SFT tokenization.
- `AReaL/rlvr_demo/eval_qwen3_gsm8k.py`: standalone AReaL/SGLang eval path.
- `AReaL/rlvr_demo/eval_hf_gsm8k.py`: robust batched HF eval path.
- `AReaL/rlvr_demo/configs/*.yaml`: GRPO and SFT configs.
- `AReaL/rlvr_demo/scripts/*.sh`: B200 launcher wrappers.

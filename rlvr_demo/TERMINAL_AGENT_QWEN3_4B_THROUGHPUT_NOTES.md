# Qwen3-4B Terminal-Agent Throughput Notes

Date: 2026-05-03.

Hardware: one local H200 node, 8 GPUs.

## Completed Qwen3-4B-Instruct Medium SFT Run

Scope: SFT only on `nvidia/Nemotron-Terminal-Corpus`, config
`skill_based_medium`, split `train`, using `Qwen/Qwen3-4B-Instruct-2507`.
Teacher-answer-RL was not run in this Instruct-model pass.

The Qwen3-4B-Instruct tokenizer has no `<think>` tags and no
`enable_thinking` branch in its chat template. Training therefore strips the
released `<think>...</think>` block from assistant targets and trains directly
on the remaining Terminus JSON. Because the target has no reasoning, this run
uses full trajectory rows rather than one row per assistant turn.

Reproduction command:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL
AREAL_VENV=/wbl-fast/usrs/ee/teacher-answer-rl/AReaL/.venv-megatron \
HF_HOME=/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash rlvr_demo/scripts/run_terminal_sft_h200.sh \
  rlvr_demo/configs/qwen3_4b_instruct_terminal_sft_h200.yaml
```

Training recipe:

- Megatron backend: `megatron:d8p1t1` on the local 8 H200 GPUs.
- Global batch size: 512 trajectories.
- Packing: FFD sequence packing, `max_tokens_per_mb=32768`,
  `pad_to_maximum=true`.
- Context limit: 32768 tokens; `truncate_long=true`.
- Optimizer: Adam, lr `8e-6`, cosine schedule, 3% warmup, weight decay `0.01`,
  grad clip `1.0`.
- Checkpoint cadence: every 25 optimizer steps plus final.

Retained data manifest:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/data/skill_based_medium.stripthink_valid_terminus_json_v1.jsonl
```

Retention summary:

| metric | value |
| --- | ---: |
| rows seen | 89,343 |
| rows retained | 89,172 |
| row retention | 99.81% |
| assistant turns seen | 715,780 |
| assistant turns retained | 674,205 |
| assistant-turn retention | 94.19% |

Completed run:

| item | value |
| --- | --- |
| experiment | `qwen3-4b-instruct-terminal-sft-medium-stripthink-retained-trajectory32768-h200` |
| completed | 2026-05-04 02:42:54 UTC |
| checkpoint elapsed wall clock | 22,875.30s / 6.35h |
| AReaL training elapsed | 22,532.91s / 6.26h |
| final optimizer/global step | 174 / 173 |
| examples seen | 89,088 |
| final loss avg | 0.355384 |
| final step total tokens | 7,453,350 |
| final step trained tokens | 3,259,086 |

Final checkpoint:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-sft-medium-stripthink-retained-trajectory32768-h200/trial0/default/epoch0epochstep173globalstep173
```

Checkpoint/event logs:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-sft-medium-stripthink-retained-trajectory32768-h200/trial0/checkpoint_events.jsonl
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/results/qwen3-4b-instruct-medium-sft/checkpoint_log.csv
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/results/qwen3-4b-instruct-medium-sft/checkpoint_log.jsonl
```

Saved checkpoints:

| global step | optimizer step | elapsed wall clock | timestamp UTC |
| ---: | ---: | ---: | --- |
| 24 | 25 | 3,956.38s | 2026-05-03T21:27:35Z |
| 49 | 50 | 7,405.56s | 2026-05-03T22:25:04Z |
| 74 | 75 | 10,150.99s | 2026-05-03T23:10:49Z |
| 99 | 100 | 13,513.47s | 2026-05-04T00:06:52Z |
| 124 | 125 | 16,670.92s | 2026-05-04T00:59:29Z |
| 149 | 150 | 19,917.16s | 2026-05-04T01:53:36Z |
| 173 | 174 | 22,875.30s | 2026-05-04T02:42:54Z |

## Evaluation Results

Generation settings for the final evals are from the local
`generation_config.json`: temperature `0.7`, top-p `0.8`, top-k `20`.

Docker-backed Terminal-Bench did not run on this node. A one-task Slurm canary
(`tb-canary-base-fix-git`, job `9150`) stayed in `CONFIGURING` on
`m7i-cpu-dy-m7i-cpu-cr-0-1` for 5m26s with no batch log and was cancelled.
Earlier CPU Slurm canaries showed the same behavior, so the completed
comparison uses non-Docker evals.

Prompt-only Terminal-Bench fallback used the real Terminal-Bench task
instructions and Terminus JSON prompt for 10 tasks, 5 samples per task:

```text
fix-git
git-leak-recovery
log-summary-date-ranges
multi-source-data-merger
nginx-request-logging
vulnerable-secret
constraints-scheduling
regex-log
sqlite-db-truncate
modernize-scientific-stack
```

This is not an execution score. It only measures first-response JSON and command
schema behavior on Terminal-Bench prompts.

| model | samples | JSON valid | command schema valid | nonempty commands | task_complete valid | avg commands |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base `Qwen/Qwen3-4B-Instruct-2507` | 50 | 84.00% | 84.00% | 84.00% | 84.00% | 3.60 |
| medium SFT final | 50 | 98.00% | 98.00% | 98.00% | 98.00% | 3.86 |

Prompt-eval outputs:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/prompt_eval/base_easy10_5_genconfig.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/prompt_eval/sft_medium_easy10_5_genconfig.json
```

Reference offline eval used 64 sampled reference turns from an 8,192-row source
window of `skill_based_medium`. Because the full medium split was used for
training, this is an imitation/reference replay metric, not a held-out
generalization metric.

| model | examples | JSON valid | command schema valid | task_complete accuracy | normalized command similarity | command exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base `Qwen/Qwen3-4B-Instruct-2507` | 64 | 68.75% | 68.75% | 50.00% | 0.1785 | 1.56% |
| medium SFT final | 64 | 92.19% | 92.19% | 92.19% | 0.3768 | 1.56% |

Reference-eval outputs:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/eval_results/offline/base_skill_medium_rows8192_holdout64_genconfig.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/eval_results/offline/sft_medium_rows8192_holdout64_genconfig.json
```

Limitations:

- No Docker-backed Terminal-Bench pass/fail score was produced because the CPU
  Slurm Docker path did not reach a runnable batch shell.
- The prompt-only Terminal-Bench fallback does not execute commands or run task
  verifiers.
- The reference eval is not held out from the completed full-medium SFT run.

## Qwen3-4B-Thinking SFT Results

Model: `Qwen/Qwen3-4B-Thinking-2507`.

Format: one row per assistant turn, prior assistant thinking stripped from
history, current assistant turn trained with released thinking. Sequence packing
uses FFD packing.

| max packed length | experiment | total tok/s | trained tok/s | mean step time | H200 memory | medium ETA | skill all ETA | full mix ETA |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 40960 | `qwen3-4b-terminal-sft-throughput-turnrow-packed40960-20260503T182939Z` | 61,411.98 | 11,372.26 | 145.05s | ~133/140 GB | 28.15h | 40.79h | 112.95h |
| 32768 | `qwen3-4b-terminal-sft-throughput-turnrow-packed32768-20260503T184742Z` | 64,355.95 | 11,917.42 | 138.42s | ~110-114/140 GB | 26.86h | 38.93h | 107.78h |

Token length sample with the Qwen3 tokenizer:

| split | sampled turns | >32768 | >40960 | p99 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `skill_based_easy` | 2,173 | 0.3221% | 0.0000% | 24,366 | 38,109 | 6,756.88 |
| `skill_based_medium` | 4,099 | 0.1952% | 0.0976% | 23,756 | 46,395 | 9,170.04 |
| `skill_based_mixed` | 1,537 | 0.0000% | 0.0000% | 17,913 | 22,672 | 6,609.06 |

The active 4B Thinking config uses `max_tokens_per_mb=32768` because it was
faster and left substantially more memory headroom than 40960 on this node.

## Qwen3-4B-Instruct Template Check

Model: `Qwen/Qwen3-4B-Instruct-2507`.

Official model-card note
(`https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507`): this is the updated
non-thinking Qwen3-4B mode and it does not generate `<think></think>` blocks;
`enable_thinking=False` is not required.

Local tokenizer check in `.venv-megatron`:

- Tokenizer class: `Qwen2TokenizerFast`.
- `enable_thinking` is not referenced in the chat template.
- The chat template contains no literal `<think>` or `</think>`.
- Rendering a user prompt with default kwargs, `enable_thinking=False`, or
  `enable_thinking=True` all produced the same generation prompt:
  `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`.

Conclusion: for this nonthinking model, do not leave empty think tags in the
target. Remove the whole `<think>...</think>` block and train directly on the
remaining Terminus JSON output.

Config prepared for this path:

```text
rlvr_demo/configs/qwen3_4b_instruct_terminal_sft_h200.yaml
```

Important dataset settings:

```yaml
sft_format: trajectory
strip_assistant_thinking: true
enable_thinking: false
truncate_long: true
```

This does not need one row per assistant turn because the target no longer
contains reasoning that must be hidden from future turns.

## Qwen3-4B-Instruct Strip-Think Estimate

Estimator:

```bash
python -m rlvr_demo.terminal_instruct_strip_report \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset nvidia/Nemotron-Terminal-Corpus \
  --configs dataset_adapters skill_based_easy skill_based_medium skill_based_mixed
```

Sampling: 4,096 evenly spaced rows from each skill split and 6,144 total adapter
rows, sampled evenly across the three adapter parquet files.

| config | rows | mean total strip tok/trajectory | mean trained strip tok/trajectory | estimated total strip tokens | estimated trained strip tokens | estimated removed total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `skill_based_easy` | 44,809 | 9,278.33 | 3,511.19 | 0.416B | 0.157B | 0.179B |
| `skill_based_medium` | 89,343 | 14,458.25 | 6,086.39 | 1.292B | 0.544B | 0.357B |
| `skill_based_mixed` | 5,689 | 9,985.58 | 3,946.84 | 0.057B | 0.022B | 0.017B |
| `dataset_adapters` | 226,313 | 11,457.13 | 4,735.56 | 2.593B | 1.072B | 2.127B |
| `skill_based_all` | 139,841 | 12,616.50 | 5,174.18 | 1.764B | 0.724B | 0.554B |
| `full_mix` | 366,154 | 11,899.92 | 4,903.08 | 4.357B | 1.795B | 2.681B |

The ETA below uses the measured 4B Thinking total-token throughput as a proxy
for Qwen3-4B-Instruct because the architectures are the same size. The Instruct
weights were not fully cached at the time of this estimate, so this is an
estimate rather than a direct Instruct training smoke result.

| dataset | estimated total strip tokens | ETA at 32768 proxy throughput | ETA at 40960 proxy throughput |
| --- | ---: | ---: | ---: |
| `skill_based_medium`, full split | 1.292B | 5.58h | 5.84h |
| `skill_based_medium`, train split after 512-row holdout | 1.284B | 5.54h | 5.81h |
| `skill_based_all` | 1.764B | 7.62h | 7.98h |
| `full_mix` | 4.357B | 18.81h | 19.71h |

Using the 32768 proxy throughput, trained-token throughput would be about
27.1k trained tok/s on `skill_based_medium` and about 26.5k trained tok/s on
`full_mix`, because stripped trajectory rows have about 41-42% supervised
tokens.

## Think Content Versus JSON Analysis/Plan

The content inside `<think>...</think>` is longer than the JSON `analysis` plus
`plan` fields in every sampled split.

| config | estimated think-inner tokens | estimated analysis+plan tokens | think / analysis+plan |
| --- | ---: | ---: | ---: |
| `skill_based_easy` | 0.178B | 0.049B | 3.61x |
| `skill_based_medium` | 0.355B | 0.118B | 3.00x |
| `skill_based_mixed` | 0.017B | 0.006B | 2.86x |
| `dataset_adapters` | 2.120B | 0.292B | 7.26x |
| `skill_based_all` | 0.550B | 0.174B | 3.17x |
| `full_mix` | 2.671B | 0.466B | 5.74x |

By characters, the same conclusion holds: full-mix think-inner content is about
10.42B characters versus about 2.03B characters for JSON `analysis` plus
`plan`, or 5.14x longer.

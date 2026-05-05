# Qwen3-4B-Instruct Terminal-Agent Medium Results

This note records the completed Qwen3-4B-Instruct terminal-agent runs from
2026-05-03 through 2026-05-05. Large artifacts are intentionally not checked in;
all artifact paths below are under the shared experiment root.

## Scope

- Base model: `Qwen/Qwen3-4B-Instruct-2507`.
- Dataset: `nvidia/Nemotron-Terminal-Corpus`.
- Dataset config: `skill_based_medium`.
- Split: `train`.
- Hardware for training: the local node only, 8 H200 GPUs.
- Evaluation: Terminal-Bench through Harbor on Slurm `l40s-1gpu` Docker nodes,
  with inference served from the local H200 node.

The instruct model is non-thinking. For SFT and teacher-answer-RL data, all
`<think>...</think>` blocks were stripped and `enable_thinking=false` was used.
The model was trained to emit the visible Terminus-2 JSON format, including
`analysis`, `plan`, `commands`, and `task_complete`, but not hidden think tags.

## Data Preparation

Retention manifest:

```bash
.venv/bin/python -m rlvr_demo.terminal_retention_manifest \
  --dataset nvidia/Nemotron-Terminal-Corpus \
  --config skill_based_medium \
  --split train \
  --output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/data/skill_based_medium.stripthink_valid_terminus_json_v1.jsonl \
  --summary-output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/data/skill_based_medium.stripthink_valid_terminus_json_v1.jsonl.summary.json
```

Retention summary:

| Field | Value |
| --- | ---: |
| Rows seen | 89,343 |
| Rows retained | 89,172 |
| Row retention | 99.81% |
| Assistant turns seen | 715,780 |
| Assistant turns retained | 674,205 |
| Assistant-turn retention | 94.19% |

## SFT Recipe

Config:

```text
rlvr_demo/configs/qwen3_4b_instruct_terminal_sft_h200.yaml
```

Run:

```bash
AREAL_VENV=.venv-megatron \
bash rlvr_demo/scripts/run_terminal_sft_h200.sh \
  rlvr_demo/configs/qwen3_4b_instruct_terminal_sft_h200.yaml
```

Key settings:

| Setting | Value |
| --- | --- |
| Actor backend | `megatron:d8p1t1` |
| GPUs | 8 H200 |
| SFT format | stripped full trajectory |
| Batch size | 512 |
| Max length | 32,768 |
| Packing | `ffd`, `max_tokens_per_mb=32768` |
| Epochs | 1 |
| Optimizer steps | 174 |
| Checkpoint frequency | every 25 steps |

Final checkpoint:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-sft-medium-stripthink-retained-trajectory32768-h200/trial0/default/epoch0epochstep173globalstep173
```

Logs:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/logs/ewer/qwen3-4b-instruct-terminal-sft-medium-stripthink-retained-trajectory32768-h200/trial0/metrics.jsonl
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-sft-medium-stripthink-retained-trajectory32768-h200/trial0/checkpoint_events.jsonl
```

Final checkpoint event:

| Field | Value |
| --- | ---: |
| Optimizer step | 174 |
| Fractional epoch | 1.0000 |
| Examples/tasks seen | 89,088 |
| Elapsed wall-clock | 22,875.30 sec |
| Timestamp saved | `2026-05-04T02:42:54.264590+00:00` |
| Final `sft/loss/avg` | 0.355384 |
| Final `sft/ppl/avg` | 1.445379 |

## Teacher-Answer-RL Recipe

Config:

```text
rlvr_demo/configs/qwen3_4b_instruct_terminal_teacher_answer_rl_from_sft99_h200_safe32768.yaml
```

Run:

```bash
AREAL_VENV=.venv-megatron \
AREAL_VLLM_PYTHON=.venv-rollout-vllm/bin/python \
bash rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh \
  rlvr_demo/configs/qwen3_4b_instruct_terminal_teacher_answer_rl_from_sft99_h200_safe32768.yaml
```

Initialization checkpoint:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-sft-medium-stripthink-retained-trajectory32768-h200/trial0/default/epoch0epochstep99globalstep99
```

Teacher-answer-RL semantics:

- The student starts from a new assistant turn and generates the visible JSON
  prefix through the point just before `"commands"`.
- PPO trains the student-generated prefix tokens.
- Reward is computed only from the teacher answer continuation, meaning
  `"commands"` and `"task_complete"`.
- Reasoning text is not directly compared to the teacher, and hidden think tags
  are disabled for this instruct model.

Key settings:

| Setting | Value |
| --- | --- |
| Actor backend | `megatron:d4p1t1` |
| Rollout backend | `vllm:d4p1t1` |
| Actor GPUs | 4 H200 |
| Rollout GPUs | 4 H200 |
| Prompt batch size | 128 |
| Samples per prompt | 2 |
| Max context | 32,768 |
| Max new tokens | 768 |
| Optimizer steps | 500 |
| Checkpoint frequency | every 50 steps |
| Learning rate | `1e-6` |

Logs:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/logs/ewer/qwen3-4b-instruct-terminal-teacher-answer-rl-from-sft99-medium-b128-s2-500step-safe32768-o768-s50-h200/trial0/metrics.jsonl
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-teacher-answer-rl-from-sft99-medium-b128-s2-500step-safe32768-o768-s50-h200/trial0/checkpoint_events.jsonl
```

Wall-clock matched checkpoint:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-teacher-answer-rl-from-sft99-medium-b128-s2-500step-safe32768-o768-s50-h200/trial0/default/epoch0epochstep249globalstep249
```

Final checkpoint:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/checkpoints/ewer/qwen3-4b-instruct-terminal-teacher-answer-rl-from-sft99-medium-b128-s2-500step-safe32768-o768-s50-h200/trial0/default/epoch0epochstep499globalstep499
```

Teacher-answer-RL checkpoint events:

| Point | Optimizer step | Fractional epoch | Elapsed wall-clock | Timestamp saved |
| --- | ---: | ---: | ---: | --- |
| Closest to SFT wall-clock | 250 | 0.0508 | 25,198.58 sec | `2026-05-04T19:03:25.578648+00:00` |
| Final | 500 | 0.1016 | 50,364.72 sec | `2026-05-05T02:02:51.717746+00:00` |

The SFT final checkpoint was saved at 22,875.30 sec, so the closest saved
teacher-answer-RL checkpoint is step 250, 2,323.28 sec later.

Final teacher-answer-RL metrics:

| Metric | Value |
| --- | ---: |
| `teacher_answer_reward/avg` | -0.676431 |
| `teacher_optimized_len/avg` | 307.15625 |
| `teacher_optimized_len/max` | 768 |
| `teacher_scoring_dropped_tokens/max` | 0 |
| `ppo_actor/no_eos_ratios/avg` | 0.015625 |
| `ppo_actor/update/actor_loss/avg` | 0.001361 |

## Terminal-Bench Evaluation

Evaluation task set, 5 trials per task:

- `modernize-scientific-stack`
- `log-summary-date-ranges`
- `multi-source-data-merger`
- `nginx-request-logging`
- `git-leak-recovery`
- `fix-git`
- `constraints-scheduling`
- `vulnerable-secret`
- `regex-log`
- `sqlite-db-truncate`

Run pattern:

```bash
# Serve on the local H200 node.
CUDA_VISIBLE_DEVICES=0 TENSOR_PARALLEL_SIZE=1 MAX_MODEL_LEN=40960 \
GPU_MEMORY_UTILIZATION=0.86 \
bash rlvr_demo/scripts/serve_terminal_model_vllm.sh \
  /path/to/checkpoint terminal-local 30080 --max-num-seqs 32

# Submit Docker-dependent Terminal-Bench jobs to Slurm.
OPENAI_API_KEY=EMPTY \
PARTITION=l40s-1gpu \
TIME=08:00:00 \
N_ATTEMPTS=5 \
N_CONCURRENT=5 \
MAX_INPUT_TOKENS=32000 \
MAX_OUTPUT_TOKENS=6144 \
PROACTIVE_SUMMARIZATION_THRESHOLD=12000 \
CPUS_PER_TASK=8 \
MEM=32G \
OVERRIDE_CPUS=3 \
OVERRIDE_MEMORY_MB=10000 \
bash rlvr_demo/scripts/run_terminal_bench_easy10_split_slurm_cpu.sh \
  EVAL_NAME \
  openai/SERVED_MODEL_NAME \
  http://10.0.159.57:30080/v1 \
  /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/harbor_jobs/EVAL_NAME

# Summarize.
.venv/bin/python -m rlvr_demo.terminal_experiment summarize-harbor \
  --jobs-dir /path/to/harbor_jobs/EVAL_NAME \
  --output /path/to/harbor_jobs/EVAL_NAME/summary.csv \
  --trials-per-task 5
```

The matched-time step-250 checkpoint was served with `MAX_MODEL_LEN=32768` and
`--max-num-seqs 16` because the 40,960-token server launch was unreliable for
that checkpoint on a single H200. The request cap stayed at 32,000 input tokens
and 6,144 output tokens.

Summary files:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/harbor_jobs/tb-base-easy10-5-sum32000-o6144/summary.summary.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/harbor_jobs/tb-sft-medium-easy10-5-sum32000-o6144/summary.summary.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/harbor_jobs/tb-ta-base-step50-easy10-5-sum32000-o6144/summary.summary.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/harbor_jobs/tb-ta-sft99-step250-easy10-5-sum32000-o6144-l40s/summary.summary.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/harbor_jobs/tb-ta-sft99-step500-easy10-5-sum32000-o6144-l40s/summary.summary.json
```

Results:

| Model/checkpoint | Passes | Selected subset | Rewarded trials | Full-suite lower bound |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3-4B-Instruct | 0/50 | 0.00% | 0/50 | 0.00% |
| SFT final, step 174 | 7/50 | 14.00% | 7/50 | 1.57% |
| Teacher-answer-RL from base, step 50 | 3/50 | 6.00% | 3/50 | 0.67% |
| Teacher-answer-RL from SFT99, step 250 matched-time | 3/50 | 6.00% | 3/41 | 0.67% |
| Teacher-answer-RL from SFT99, step 500 final | 5/50 | 10.00% | 5/48 | 1.12% |

Task pass rates for the three primary comparison points:

| Task | SFT final | TA-RL step 250 | TA-RL step 500 |
| --- | ---: | ---: | ---: |
| `constraints-scheduling` | 0.00 | 0.00 | 0.00 |
| `fix-git` | 0.00 | 0.00 | 0.00 |
| `git-leak-recovery` | 0.20 | 0.00 | 0.00 |
| `log-summary-date-ranges` | 0.00 | 0.00 | 0.20 |
| `modernize-scientific-stack` | 1.00 | 0.60 | 0.80 |
| `multi-source-data-merger` | 0.20 | 0.00 | 0.00 |
| `nginx-request-logging` | 0.00 | 0.00 | 0.00 |
| `regex-log` | 0.00 | 0.00 | 0.00 |
| `sqlite-db-truncate` | 0.00 | 0.00 | 0.00 |
| `vulnerable-secret` | 0.00 | 0.00 | 0.00 |

The paper baseline of roughly 10% is for all 89 Terminal-Bench tasks. These
numbers are from an easier 10-task subset. The comparable conservative number is
the full-suite lower bound, computed as `passes / (89 * 5)` while assuming every
unrun task fails.

## Offline Evaluation

Offline evaluation remains available through:

```text
rlvr_demo/terminal_offline_eval.py
```

It reports JSON parse validity, commands schema validity, `task_complete`
validity, normalized command sequence similarity, command exact match rate, and
`task_complete` prediction accuracy. Terminal-Bench was prioritized for these
final checkpoints because Docker access became available through Slurm.

## Limitations And Next Steps

- Teacher-answer-RL did not beat the SFT final checkpoint on this 10-task
  subset, either at matched wall-clock time or after 500 steps.
- The teacher-answer-RL run saw only about 10.16% of an epoch because rollout and
  PPO were much slower than SFT.
- Several Terminal-Bench trials ended in setup timeouts; the selected-subset
  score counts those as failures, while the rewarded-trials column excludes
  them.
- The 10-task subset is intentionally easier and not a replacement for the full
  89-task benchmark.
- A stronger next recipe should keep the SFT baseline, then test RL reward
  variants that reduce timeout-heavy behavior and improve command completion,
  with matched checkpoint evaluation on the same 10-task subset and a later
  full-suite run.

# Terminal Agent SFT and Teacher-Answer-RL Reproduction

This document records the final completed terminal-agent runs for
`nvidia/Nemotron-Terminal-Corpus` on the `skill_based_medium` configuration with
`Qwen/Qwen3-4B-Thinking-2507`.

The completed publication-trace run used a matched 512-turn subset because a
full pass over `skill_based_medium` was not feasible in this session. The full
split contains about 715,780 trainable assistant turns; at the measured H200 SFT
throughput, a full-medium pass would take multiple days. Both completed
algorithms used the same training subset and the same offline evaluation set.

## Scope

- Algorithms run: SFT and teacher-answer-RL.
- Algorithms not run: GRPO, per revised scope.
- Dataset: `nvidia/Nemotron-Terminal-Corpus`.
- Dataset config: `skill_based_medium`.
- Base model: `Qwen/Qwen3-4B-Thinking-2507`.
- Hardware used: 8x NVIDIA H200.
- Evaluation: offline, non-Docker format and command-sequence metrics.
- Max sequence length: 40960 tokens.

## Important Qwen3 Chat Template Handling

Qwen3's chat template removes prior assistant reasoning from multi-turn history.
For these terminal-agent trajectories, the data pipeline therefore trains one
assistant response per row:

- Previous assistant turns are included as history after stripping previous
  `<think>...</think>` blocks.
- The current assistant response remains the only supervised target.
- SFT trains the current full assistant response, including current reasoning,
  `analysis`, `plan`, `commands`, and `task_complete`.
- Teacher-answer-RL always splits each current assistant response at
  `"commands"`.
- For teacher-answer-RL, the student prefix is everything before `"commands"`;
  the teacher answer is only `"commands"` through `"task_complete"`.
- Teacher-answer reward must be computed only from the teacher answer. The run
  scripts set `TEACHER_ANSWER_FORMAT_BONUS=0.0` and
  `TEACHER_ANSWER_LENGTH_PENALTY=0.0`.

The chat-template validation command is:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL

.venv/bin/python -m rlvr_demo.terminal_chat_template_check \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --dataset nvidia/Nemotron-Terminal-Corpus \
  --dataset-config skill_based_medium \
  --split train \
  --max-length 40960 \
  --output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/chat_template_check.json
```

The saved validation artifact for the completed run is:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/chat_template_check.json`

## Reproduction Script

The easiest way to reproduce the completed subset-512 run is:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL

rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh all
```

The wrapper has explicit subcommands:

```bash
rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh chat-check
rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh sft
rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh teacher-rl
rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh eval
rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh compile
```

Useful overrides:

```bash
SFT_EXPERIMENT=my-sft-run \
TEACHER_EXPERIMENT=my-teacher-run \
FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh all
```

The wrapper encodes the exact final run parameters, including subset limits,
checkpoint cadence, eval settings, and result compilation.

## Environment

The run scripts choose `.venv-megatron` when present and fall back to `.venv`.
They also set the local H200 networking and memory flags used by the completed
runs:

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `GLOO_SOCKET_IFNAME=enp71s0`
- `NCCL_SOCKET_IFNAME=enp71s0`
- `NCCL_CUMEM_ENABLE=0`
- `NCCL_NVLS_ENABLE=0`
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `HF_HOME=/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache`
- `HF_HUB_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `TRITON_CACHE_DIR=/wbl-fast/usrs/ee/teacher-answer-rl/triton`

Teacher-answer-RL uses:

- `AREAL_VLLM_PYTHON=/wbl-fast/usrs/ee/teacher-answer-rl/AReaL/.venv-rollout-vllm/bin/python`
- `AREAL_SGLANG_PYTHON=/wbl-fast/usrs/ee/teacher-answer-rl/AReaL/.venv-rollout-sglang/bin/python`

The completed teacher-answer-RL run used vLLM rollout. SGLang support is kept in
the config and environment, but the final reported run used:

```yaml
rollout.backend: "vllm:d4p1t1"
```

## Configs and Scripts

Primary configs:

- `rlvr_demo/configs/qwen3_4b_terminal_sft_h200_1000.yaml`
- `rlvr_demo/configs/qwen3_4b_terminal_teacher_answer_rl_h200_1000.yaml`

Primary training scripts:

- `rlvr_demo/scripts/run_terminal_sft_h200.sh`
- `rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh`

Utility scripts:

- `rlvr_demo/terminal_chat_template_check.py`
- `rlvr_demo/terminal_data_report.py`
- `rlvr_demo/terminal_offline_eval.py`
- `rlvr_demo/terminal_compile_results.py`
- `rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh`

The checkpoint saver writes `checkpoint_events.jsonl` with wall-clock timestamps.
The stats logger writes `metrics.jsonl`. The compiler joins checkpoint events,
metrics, and eval results into `checkpoint_log.csv` and `checkpoint_log.jsonl`.

Required checkpoint-log fields are present:

- `algorithm`
- `base_model`
- `dataset_split`
- `examples_seen`
- `tasks_seen`
- `optimizer_step`
- `epoch`
- `fractional_epoch`
- `elapsed_wallclock_seconds`
- `timestamp_saved`
- `metrics`

## Exact Final SFT Command

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL

bash rlvr_demo/scripts/run_terminal_sft_h200.sh \
  rlvr_demo/configs/qwen3_4b_terminal_sft_h200_1000.yaml \
  experiment_name=qwen3-4b-terminal-sft-tree40960-h200-subset512 \
  total_train_epochs=1 total_train_steps=8 \
  train_dataset.batch_size=64 \
  train_dataset.dataset_kwargs.split_part=null \
  +train_dataset.dataset_kwargs.limit_rows=1024 \
  +train_dataset.dataset_kwargs.limit=512 \
  valid_dataset=null \
  saver.freq_steps=2 \
  evaluator.eval_before_train=false evaluator.freq_steps=999 \
  stats_logger.wandb.mode=disabled
```

SFT recipe details:

- Backend: `megatron:d1p1t8`.
- GPUs: all 8 H200s for Megatron tensor parallelism.
- Batch size: 64 turns.
- Optimizer steps: 8.
- Trainable turns seen: 512.
- Max packed sequence/microbatch tokens: 40960.
- AReaL tree training: enabled.
- Checkpoint frequency: every 2 optimizer steps.
- Final checkpoint elapsed wall clock: 517.850 seconds.

Final SFT checkpoint:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-terminal-sft-tree40960-h200-subset512/trial0/default/epoch0epochstep7globalstep7`

SFT logs:

- Metrics: `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/logs/ewer/qwen3-4b-terminal-sft-tree40960-h200-subset512/trial0/metrics.jsonl`
- Checkpoint events: `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-terminal-sft-tree40960-h200-subset512/trial0/checkpoint_events.jsonl`

## Exact Final Teacher-Answer-RL Command

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL

bash rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh \
  rlvr_demo/configs/qwen3_4b_terminal_teacher_answer_rl_h200_1000.yaml \
  experiment_name=qwen3-4b-terminal-teacher-answer-rl-tree40960-h200-subset512 \
  total_train_epochs=1 total_train_steps=16 \
  train_dataset.batch_size=32 rollout.consumer_batch_size=32 \
  train_dataset.dataset_kwargs.split_part=null \
  +train_dataset.dataset_kwargs.limit_rows=1024 \
  +train_dataset.dataset_kwargs.limit=512 \
  valid_dataset=null \
  gconfig.n_samples=2 gconfig.max_new_tokens=1024 \
  eval_gconfig.max_new_tokens=1024 actor.max_new_tokens=1024 \
  vllm.max_num_seqs=128 rollout.max_concurrent_rollouts=128 rollout.queue_size=2048 \
  saver.freq_steps=2 \
  evaluator.eval_before_train=false evaluator.freq_steps=999 \
  stats_logger.wandb.mode=disabled
```

Teacher-answer-RL recipe details:

- Actor backend: `megatron:d1p1t4`.
- Actor GPUs: 4 H200s.
- Rollout backend: `vllm:d4p1t1`.
- Rollout GPUs: 4 H200s.
- vLLM prefix caching: enabled.
- vLLM chunked prefill: enabled by vLLM for this long-context run.
- Prompt batch size: 32.
- Samples per prompt: 2.
- Optimizer steps: 16.
- Prompt tasks seen: 512.
- Max packed sequence/microbatch tokens: 40960.
- AReaL tree training: enabled.
- Checkpoint frequency: every 2 optimizer steps.
- Final checkpoint elapsed wall clock: 1502.347 seconds.
- Closest checkpoint to full SFT wall clock: global step 1, 587.032 seconds.

Teacher-answer-RL closest wall-clock checkpoint:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-terminal-teacher-answer-rl-tree40960-h200-subset512/trial0/default/epoch0epochstep1globalstep1`

Teacher-answer-RL final checkpoint:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-terminal-teacher-answer-rl-tree40960-h200-subset512/trial0/default/epoch0epochstep15globalstep15`

Teacher-answer-RL logs:

- Metrics: `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/logs/ewer/qwen3-4b-terminal-teacher-answer-rl-tree40960-h200-subset512/trial0/metrics.jsonl`
- Checkpoint events: `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-terminal-teacher-answer-rl-tree40960-h200-subset512/trial0/checkpoint_events.jsonl`

## Offline Evaluation

The completed evaluation used 64 held-out turns, greedy decoding, and
`max_new_tokens=1024`. It skipped the first 512 trainable turns to avoid overlap
with the subset used for training.

Example command for one checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m rlvr_demo.terminal_offline_eval \
  --checkpoint /path/to/checkpoint \
  --limit-rows 4096 \
  --skip-turns 512 \
  --limit 64 \
  --greedy \
  --max-new-tokens 1024 \
  --output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/name.json \
  --predictions-output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/name.jsonl
```

The reproduction wrapper runs all three evaluation points:

```bash
rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh eval
```

Evaluation metrics:

- JSON parse validity.
- Commands schema validity.
- `task_complete` validity.
- Normalized command sequence similarity.
- Command exact match rate.
- `task_complete` prediction accuracy.

Evaluation outputs:

- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/sft_final_subset512.json`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/sft_final_subset512.jsonl`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/teacher_closest_wallclock_subset512.json`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/teacher_closest_wallclock_subset512.jsonl`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/teacher_final_subset512.json`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval/teacher_final_subset512.jsonl`

## Result Compilation

Regenerate the result tables with:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL

.venv/bin/python -m rlvr_demo.terminal_compile_results \
  --experiment qwen3-4b-terminal-sft-tree40960-h200-subset512 \
  --experiment qwen3-4b-terminal-teacher-answer-rl-tree40960-h200-subset512 \
  --eval-dir /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/eval \
  --output-dir /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/subset512-sft-vs-teacher-rl
```

Compiled outputs:

- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/subset512-sft-vs-teacher-rl/checkpoint_log.csv`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/subset512-sft-vs-teacher-rl/checkpoint_log.jsonl`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/subset512-sft-vs-teacher-rl/comparison_table.json`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/results/subset512-sft-vs-teacher-rl/final_report.md`

## Final Comparison

| Point | Tasks seen | Wall clock | JSON valid | Schema valid | Cmd exact | Cmd similarity | task_complete acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final SFT | 512 | 517.85s | 0.1250 | 0.1250 | 0.0156 | 0.0545 | 0.1250 |
| Teacher-RL closest wall-clock | 64 | 587.03s | 0.2188 | 0.1250 | 0.0156 | 0.0189 | 0.1563 |
| Final Teacher-RL | 512 | 1502.35s | 0.2031 | 0.1250 | 0.0156 | 0.0156 | 0.0938 |

## Scaling Beyond the Completed Subset

The checked-in configs are full-medium recipes. The completed subset run was
created by overriding:

```bash
+train_dataset.dataset_kwargs.limit_rows=1024
+train_dataset.dataset_kwargs.limit=512
```

To run larger matched subsets:

1. Increase both `limit_rows` and `limit` by the same amount for SFT and
   teacher-answer-RL.
2. Increase `total_train_steps` so `steps * batch_size` matches the desired
   number of turns.
3. Keep `max_length=40960`, `actor.mb_spec.max_tokens_per_mb=40960`, and
   `actor.enable_tree_training=true`.
4. Keep one assistant response per data row unless the model chat template is
   changed and revalidated.
5. Re-run `terminal_chat_template_check.py` before training.
6. Re-run offline eval and `terminal_compile_results.py` after training.

For a full-medium pass, remove the subset overrides entirely and choose
`total_train_steps` from the actual post-filter trainable turn count and batch
size. Expect a multi-day run on this node unless throughput improves.

## Known Limitations

- The completed comparison is a real training comparison, but only on a
  512-turn subset.
- The offline eval is format-based and non-executing; it does not use
  terminal-bench or Docker.
- The eval sample has 64 held-out turns.
- Greedy eval used `max_new_tokens=1024`; some long valid answers may be
  truncated.
- Teacher-answer-RL did not improve command similarity or command exact match in
  this small run, although it produced higher JSON parse validity than SFT.

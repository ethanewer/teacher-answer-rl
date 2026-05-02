# Terminal-Agent Full Recipes: Qwen3-8B SFT and Teacher-Answer-RL

This document is the reproduction recipe for the Qwen3-8B terminal-agent runs in
this repo. It records the exact data construction, chat-template handling,
training commands, GPU layout, optimization settings, checkpointing, and eval
commands used for the SFT baseline and the teacher-answer-RL continuation.

## Common Setup

Run from:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL
```

Primary artifact root:

```bash
export FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b
```

Primary environments:

- Training env: `.venv-megatron`
- vLLM rollout/env serving env: `.venv-rollout-vllm`
- Optional SGLang env: `.venv-rollout-sglang`
- HF cache: `/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache`
- GPUs: 8x H200 on one node

The launch scripts set the important runtime variables:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=enp71s0
export NCCL_SOCKET_IFNAME=enp71s0
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/wbl-fast/usrs/ee/teacher-answer-rl/triton
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
```

Use the wrapper script unless debugging the trainer directly:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh <command>
```

## Data and Chat Formatting

Dataset:

- Hugging Face dataset: `nvidia/Nemotron-Terminal-Corpus`
- Split: `train`
- Small-run config/subset: `skill_based_medium`
- Full paper-style config: `full_mix`
- Local subset used for completed comparison: first 1000 released rows
- Validation holdout for offline eval: 512 turns from the loader partition

Qwen3 chat-template handling is critical. Qwen3 removes previous-turn thinking
from multi-turn conversations, so each assistant response is made into its own
training row.

For every row:

- The prompt is the conversation prefix up to the current assistant turn.
- Prior assistant messages have `<think>...</think>` stripped before applying
  the chat template.
- The current assistant target keeps its `<think>...</think>` content for SFT.
- `tokenizer.apply_chat_template(..., enable_thinking=True)` is used.
- Rows longer than the configured max length are filtered rather than truncated.
- Training uses one assistant response target per row.

For teacher-answer-RL:

- Each current assistant response is split at the top-level `"commands"` field.
- Student prefix is everything before `"commands"`.
- Teacher answer is `"commands"` through `"task_complete"` plus the closing JSON
  brace.
- Reward is computed only on teacher answer tokens.
- Reasoning text is used only as prefill/context and is never directly rewarded
  or compared.

The required Terminus-2 response shape is:

```text
<think>
...
</think>

{
  "analysis": "...",
  "plan": "...",
  "commands": [
    {
      "keystrokes": "ls -la \n",
      "duration": 0.1
    }
  ],
  "task_complete": true
}
```

Run the chat-template check before launching a new recipe:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh chat-check
```

Expected output path:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3_8b_chat_template_check.json
```

## SFT Recipe

Primary config:

```text
rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200.yaml
```

Paper-style full-mix command:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh sft-full
```

Single-node `skill_based_medium` comparison command actually used for the local
completed SFT run:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh sft-skill-final
```

Equivalent explicit command:

```bash
bash rlvr_demo/scripts/run_terminal_sft_h200.sh \
  rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200.yaml \
  experiment_name=qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200 \
  train_dataset.dataset_kwargs.name=skill_based_medium \
  +train_dataset.dataset_kwargs.limit_rows=1000 \
  train_dataset.batch_size=128 \
  total_train_epochs=2 \
  total_train_steps=24 \
  saver.freq_steps=4
```

SFT model and tokenizer:

- `actor.path: Qwen/Qwen3-8B`
- `tokenizer_path: ${actor.path}`
- `enable_thinking: true` in dataset kwargs
- `strip_prior_assistant_thinking: true`

SFT data settings:

- `train_dataset.path: nvidia/Nemotron-Terminal-Corpus`
- `train_dataset.split: train`
- `train_dataset.type: sft`
- Full recipe dataset config: `full_mix`, `split_part=train`
- Local completed recipe override: `skill_based_medium`, `limit_rows=1000`
- `shuffle_records: false`
- `shuffle_source_groups: true`
- `holdout_size: 512`
- `lazy_tokenize: true`

SFT sequence and packing settings:

- `train_dataset.max_length: 32768`
- `actor.mb_spec.max_tokens_per_mb: 32768`
- `actor.mb_spec.packing_algorithm: ffd`
- `actor.pad_to_maximum: true`
- `actor.enable_tree_training: true`
- Rows over max length are filtered by the dataset loader.

SFT GPU and precision settings:

- Single node, 8 H200 GPUs
- Actor backend: `megatron:d1p1t8`
- Tensor parallelism: 8-way
- `dtype: bfloat16`
- `gradient_checkpointing: true`
- `disable_dropout: true`
- `enable_offload: false`

SFT optimizer:

- Adam
- Learning rate: `2e-5`
- Weight decay: `1e-4`
- Betas: `(0.9, 0.95)`
- Epsilon: `1e-8`
- Scheduler: cosine
- Warmup proportion: `0.10`
- Gradient clipping: `1.0`

SFT checkpointing and logs:

- Full paper-style save cadence: `saver.freq_steps=500`
- Local completed run save cadence: `saver.freq_steps=4`
- Checkpoints:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/<experiment>/trial0`
- Logs:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/logs/ewer/<experiment>/trial0`
- Checkpoint wall-clock log:
  `checkpoint_events.jsonl` in the trial checkpoint directory

Completed local SFT run:

- Experiment:
  `qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200`
- Prepared data: 1000 usable rows, 5966 assistant turns
- Batch size: 128
- Steps: 24
- Examples/tasks seen: 3072
- Final checkpoint elapsed: `2068.8735690116882` seconds
- Final metric elapsed: `1938.5065271960339` seconds
- Final loss: `0.4692123234272003`
- Final checkpoint:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200/trial0/default/epoch1epochstep1globalstep23`

SFT smoke test:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh sft-smoke
```

## Teacher-Answer-RL Recipe

Primary optimized config:

```text
rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_from_nemotron_h200.yaml
```

Optimized full command:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-from-nemotron-full
```

Equivalent explicit command:

```bash
bash rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh \
  rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_from_nemotron_h200.yaml
```

Optimized smoke test:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-from-nemotron-smoke
```

Equivalent smoke overrides:

```bash
bash rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh \
  rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_from_nemotron_h200.yaml \
  experiment_name=qwen3-8b-terminal-teacher-answer-rl-from-nemotron-smoke \
  total_train_epochs=1 \
  total_train_steps=2 \
  +train_dataset.dataset_kwargs.limit_rows=64 \
  +train_dataset.dataset_kwargs.limit=32 \
  train_dataset.batch_size=8 \
  rollout.consumer_batch_size=8 \
  saver.freq_steps=1 \
  vllm.max_num_seqs=32 \
  rollout.max_concurrent_rollouts=32 \
  rollout.queue_size=256
```

Teacher-answer-RL model and tokenizer:

- `actor.path: nvidia/Nemotron-Terminal-8B`
- `tokenizer_path: ${actor.path}`
- This initializes from the released SFT checkpoint rather than Qwen3 base.
- This was the optimized recipe after the base-initialized teacher run
  underperformed on offline metrics.

Teacher-answer-RL data settings:

- `train_dataset.path: nvidia/Nemotron-Terminal-Corpus`
- `train_dataset.split: train`
- `train_dataset.type: rl`
- `dataset_kwargs.name: skill_based_medium`
- `dataset_kwargs.limit_rows: 1000`
- `dataset_kwargs.holdout_size: 512`
- `dataset_kwargs.strip_prior_assistant_thinking: true`
- `dataset_kwargs.enable_thinking: true`
- `dataset_kwargs.lazy_tokenize: true`
- `shuffle_records: false`
- `shuffle_source_groups: true`

Teacher-answer split/reward settings:

- Student prefix: response text before the top-level `"commands"` field.
- Teacher answer: top-level `"commands"` field, `"task_complete"` field, and
  closing JSON brace.
- Rewarded/computed tokens: teacher answer only.
- Reasoning tokens are not compared or rewarded.
- Format bonus: `TEACHER_ANSWER_FORMAT_BONUS=0.0`
- Length penalty: `TEACHER_ANSWER_LENGTH_PENALTY=0.0`
- `teacher_format_found` is logged as a metric to detect split failures.

Teacher-answer-RL sequence and packing settings:

- `train_dataset.max_length: 32768`
- `gconfig.max_tokens: 32768`
- `actor.mb_spec.max_tokens_per_mb: 32768`
- `actor.mb_spec.packing_algorithm: ffd`
- `actor.pad_to_maximum: true`
- `actor.enable_tree_training: true`
- Rows over max length are filtered by the dataset loader.

Teacher-answer-RL rollout/generation settings:

- Rollout backend: `vllm:d4p1t1`
- vLLM model: `${actor.path}`
- vLLM replicas: 4 data-parallel rollout workers
- `gconfig.n_samples: 2`
- `gconfig.max_new_tokens: 512`
- `gconfig.temperature: 0.6`
- `gconfig.top_p: 0.95`
- `gconfig.top_k: 20`
- `gconfig.greedy: false`
- `rollout.max_concurrent_rollouts: 256`
- `rollout.queue_size: 2048`
- `vllm.max_num_seqs: 128`
- `vllm.max_model_len: 32768`
- `vllm.gpu_memory_utilization: 0.80`
- vLLM prefix caching enabled by leaving `no_enable_prefix_caching: false`

Teacher-answer-RL actor/GPU settings:

- Single node, 8 H200 GPUs
- Actor backend: `megatron:d1p1t4`
- Actor GPUs: 4 H200s
- Rollout GPUs: 4 H200s
- `dtype: bfloat16`
- `gradient_checkpointing: true`
- `disable_dropout: true`
- `enable_offload: false`

Teacher-answer-RL optimizer/PPO-style settings:

- Adam
- Learning rate: `1e-6`
- Weight decay: `0.01`
- Betas: `(0.9, 0.999)`
- Epsilon: `1e-8`
- Scheduler: constant
- Warmup proportion: `0.001`
- Gradient clipping: `1.0`
- `eps_clip: 0.2`
- `kl_ctl: 0.0`
- `ppo_n_minibatches: 1`
- `recompute_logprob: true`
- `use_decoupled_loss: true`
- Rejection sampling: ratio upper bound `5.0`
- Reward normalization: group mean/std, group size `n_samples`
- Advantage normalization: batch mean/std
- Weight update mode: `xccl`

Teacher-answer-RL checkpointing and logs:

- Save every 8 optimizer steps.
- Checkpoints:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200/trial0`
- Logs:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/logs/ewer/qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200/trial0`
- Checkpoint wall-clock log:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200/trial0/checkpoint_events.jsonl`

Completed optimized teacher-answer-RL run:

- Experiment:
  `qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200`
- Prepared data: 1000 usable rows, 5333 trainable teacher-answer turns
- Train batch size: 32
- Rollout samples: 2
- Steps: 40
- Final examples/tasks seen: 1280 prompts
- Total logged training time: `2035.15` seconds
- Final checkpoint event elapsed: `2362.9116473197937` seconds
- Final checkpoint reward avg: `-0.8450348377227783`
- Best wall-clock-matched checkpoint: step 31, elapsed
  `1956.5410830974579` seconds
- Best offline command-similarity checkpoint: final step 39

Optimized teacher checkpoints:

| global step | optimizer step | elapsed sec | reward avg | checkpoint |
| ---: | ---: | ---: | ---: | --- |
| 7 | 8 | 784.6861 | -0.8217 | `epoch0epochstep7globalstep7` |
| 15 | 16 | 1171.3023 | -0.6462 | `epoch0epochstep15globalstep15` |
| 23 | 24 | 1562.3459 | -0.6605 | `epoch0epochstep23globalstep23` |
| 31 | 32 | 1956.5411 | -0.4826 | `epoch0epochstep31globalstep31` |
| 39 | 40 | 2362.9116 | -0.8450 | `epoch0epochstep39globalstep39` |

Earlier base-initialized teacher-answer-RL comparison recipe:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-skill-final
```

That recipe used `Qwen/Qwen3-8B`, batch 64, 48 steps, `max_new_tokens=2048`,
and 2 rollout samples on the same `skill_based_medium` 1000-row subset. It is
kept for comparison, but the optimized recipe above is the one to use going
forward.

## Offline Eval Recipe

Evaluate the released SFT model and all optimized teacher checkpoints:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh eval-teacher-from-nemotron
```

Eval settings:

- Dataset: `nvidia/Nemotron-Terminal-Corpus`
- Config: `skill_based_medium`
- Split part: validation holdout
- `limit_rows=1000`
- `limit=32`
- `max_length=40960`
- `max_new_tokens=2048`
- Sampled generation: temperature `0.6`, top-p `0.95`, top-k `20`
- Metrics: JSON parse validity, commands schema validity,
  `task_complete` validity, normalized command sequence similarity, command
  exact match, and `task_complete` prediction accuracy.

Output directory:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/eval_teacher_from_nemotron_40step
```

Compile SFT-vs-optimized-teacher checkpoint logs:

```bash
.venv/bin/python -m rlvr_demo.terminal_compile_results \
  --fileroot /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b \
  --experiment qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200 \
  --experiment qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200 \
  --eval-dir /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/eval_teacher_from_nemotron_40step \
  --output-dir /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3-8b-sft-vs-teacher-from-nemotron-40step
```

Compiled outputs:

- `checkpoint_log.jsonl`
- `checkpoint_log.csv`
- `comparison_table.json`

## Terminal-Bench Eval Recipe

Serve a checkpoint with vLLM using the Qwen3 reasoning parser and
`enable_thinking=true`. Example for optimized teacher step 39:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache \
.venv-rollout-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200/trial0/default/epoch0epochstep39globalstep39 \
  --served-model-name terminal-teacher-step39 \
  --host 0.0.0.0 \
  --port 30080 \
  --dtype bfloat16 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --trust-remote-code
```

Run Harbor/Terminal-Bench from a Docker-capable CPU Slurm node:

```bash
rlvr_demo/scripts/run_terminal_bench_eval_harbor.sh \
  tb-teacher-step39-modernize \
  openai/terminal-teacher-step39 \
  http://10.0.159.57:30080/v1 \
  /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/harbor_jobs/tb-teacher-step39-modernize \
  --task modernize-scientific-stack \
  --n-attempts 1 \
  --n-concurrent 1 \
  --max-turns 40 \
  --max-input-tokens 40960 \
  --max-output-tokens 8192 \
  --override-cpus 4 \
  --override-memory-mb 16384
```

Summarize Harbor outputs:

```bash
.venv/bin/python -m rlvr_demo.terminal_experiment summarize-harbor \
  --jobs-dir /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/harbor_jobs/tb-teacher-step39-modernize \
  --output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/results/tb_teacher_step39_modernize_summary.json
```

Completed one-task Terminal-Bench checks:

- Released SFT `nvidia/Nemotron-Terminal-8B`: `modernize-scientific-stack`,
  `1/1`, reward `1.0`
- Optimized teacher step 39: `modernize-scientific-stack`, `1/1`, reward `1.0`

These are skipped-hard-task smoke checks, not full Terminal-Bench scores.

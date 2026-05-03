# Terminal-Agent Full Recipes: Qwen3-8B SFT and Teacher-Answer-RL

This is the canonical runbook for the full-scale terminal-agent comparison.
Older Qwen3-4B and 1k-row notes in this repo are historical and are not the
current publication recipe.

## Current Scope

- Base model: `Qwen/Qwen3-8B`.
- Released SFT reference model for sanity checking:
  `nvidia/Nemotron-Terminal-8B`.
- Dataset: `nvidia/Nemotron-Terminal-Corpus`.
- Training dataset config: `full_mix`, which expands locally to
  `dataset_adapters`, `skill_based_easy`, `skill_based_medium`, and
  `skill_based_mixed`.
- Full released mix size observed at launch: 366,154 trajectories.
- Algorithms: SFT first, then teacher-answer-RL initialized from the completed
  SFT checkpoint.
- Algorithms not used in the current comparison: GRPO.
- Training context length: 32,768 tokens.
- Offline generation/eval context target: 40,960 tokens where supported.

Paper alignment notes from arXiv:2602.21193:

- Tables 5-8 show best SFT results with no data filtering and 264k total
  examples in the paper setup.
- The released corpus available here is larger than that count; the local
  paper-style run uses the complete released `full_mix` rather than regenerating
  or filtering data.
- The SFT recipe keeps the paper-style 32k training context and global batch
  size 128.

## Environment

Run from:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL
export FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b
```

Primary environments:

- Training: `.venv-megatron`
- vLLM rollout/serving: `.venv-rollout-vllm`
- Optional SGLang rollout: `.venv-rollout-sglang`
- HF cache: `/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache`
- Triton cache: `/wbl-fast/usrs/ee/teacher-answer-rl/triton`

Important runtime settings are encoded in the launch scripts and configs:

```bash
export HF_HOME=/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export NCCL_SOCKET_IFNAME=enp71s0
export GLOO_SOCKET_IFNAME=enp71s0
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Data and Chat Formatting

The data loader has two different SFT/RL formats because Qwen3 removes thinking
from prior assistant turns when using its normal multi-turn chat template.

SFT full-scale format:

- `sft_format: trajectory`
- One training row is one released terminal trajectory.
- The serializer writes Qwen ChatML tokens directly so all released assistant
  responses, including `<think>...</think>`, remain in the supervised target.
- The loss mask covers assistant content and assistant end-of-message tokens.
- User/system/tool tokens are context only.
- `truncate_long: true` keeps every trajectory row while capping at the 32k
  paper context.

Teacher-answer-RL format:

- One training row is one trainable assistant response.
- Prior assistant turns are context with previous `<think>...</think>` stripped.
- The student starts at a fresh assistant generation prompt produced by the
  Qwen3 tokenizer/chat template.
- The student samples until a top-level `"commands"` key is reached, or until
  the generation limit.
- All sampled student-prefix tokens up to `"commands"` are in the PPO loss mask.
- The scalar reward assigned to those sampled tokens is computed only from the
  teacher answer span: `"commands"` through `"task_complete"` plus the closing
  JSON brace.
- Reasoning text is generated and trained, but it is never string-compared or
  directly rewarded.

Terminus-2 target shape:

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

Run the tokenizer/template check before changing recipes:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh chat-check
```

## SFT Baseline

Config:

```text
rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200_slurm4.yaml
```

Launch command:

```bash
FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b \
bash rlvr_demo/scripts/run_terminal_sft_h200.sh \
  rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200_slurm4.yaml
```

Equivalent wrapper command:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh sft-full
```

Run identity:

- Experiment:
  `qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4`
- Slurm job for the clean current run: `7399`.
- Start timestamp: 2026-05-03 00:38 UTC.
- Worker nodes: 4 H200 nodes, 32 GPUs total.
- Backend: `megatron:d16p1t2`.
- Global batch size: 128 trajectories.
- Steps per epoch: 2,860.
- Total steps: 5,720.
- Total trajectory examples seen: 732,160.
- Sequence packing: FFD, `max_tokens_per_mb=32768`.
- `pad_to_maximum: true`.
- `enable_tree_training: true`.
- Precision: bf16.
- Optimizer: AdamW-compatible Adam, LR `2e-5`, betas `(0.9, 0.95)`,
  weight decay `1e-4`, cosine schedule, 10% warmup, grad clip `1.0`.

Checkpointing:

- Periodic checkpoints every 500 optimizer steps.
- A final checkpoint is forced on the last configured training step even when it
  is not a multiple of 500.
- Checkpoint events:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4/trial0/checkpoint_events.jsonl`
- Metrics:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/logs/ewer/qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4/trial0/metrics.jsonl`

An earlier pre-patch SFT attempt was cancelled before any checkpoint because
the original saver cadence would not have produced a true final checkpoint. Its
logs were preserved with an `aborted_pre_finalsave_20260503T001650Z` suffix.

## Teacher-Answer-RL

Config:

```text
rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_paper_h200.yaml
```

Launch after SFT completes:

```bash
SFT_EXPERIMENT=qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4 \
TEACHER_EXPERIMENT=qwen3-8b-terminal-teacher-answer-rl-released-fullmix-nofilter-b128-s2-5720step-h200 \
FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b \
DATASET_CONFIG=full_mix \
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-full
```

The wrapper resolves `TERMINAL_TEACHER_INIT_PATH` to the final SFT checkpoint.
Set it manually only when resuming from a specific checkpoint:

```bash
export TERMINAL_TEACHER_INIT_PATH=/path/to/sft/checkpoint
```

For unattended handoff after the SFT job finishes:

```bash
FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b \
SFT_EXPERIMENT=qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4 \
TEACHER_EXPERIMENT=qwen3-8b-terminal-teacher-answer-rl-released-fullmix-nofilter-b128-s2-5720step-h200 \
TARGET_SFT_STEP=5720 \
rlvr_demo/scripts/watch_sft_then_launch_teacher.sh \
  total_train_steps=5720
```

The script passes `TEACHER_EXPERIMENT` into the Hydra config as
`experiment_name` and forwards any trailing Hydra overrides to the teacher
launch.

Run identity:

- Experiment:
  `qwen3-8b-terminal-teacher-answer-rl-released-fullmix-nofilter-b128-s2-5720step-h200`
- Initialization: final local SFT checkpoint, not the released SFT model.
- Backend: Megatron actor plus vLLM rollout by default.
- Actor backend: `megatron:d2p1t2`.
- Rollout backend: `vllm:d4p1t1`.
- Actor GPUs: 4 H200s.
- Rollout GPUs: 4 H200s.
- Global prompt batch size: 128.
- Samples per prompt: 2.
- Total optimizer steps: 5,720.
- Prompt tasks seen: 732,160.
- Sequence packing: FFD, `max_tokens_per_mb=32768`.
- Generation: sampled, temperature `0.6`, top-p `0.95`, top-k `20`,
  `max_new_tokens=2048`.
- Optimizer: Adam, LR `1e-6`, betas `(0.9, 0.999)`, weight decay `0.01`,
  constant schedule, grad clip `1.0`.
- PPO settings: `eps_clip=0.25`, `ppo_n_minibatches=1`,
  `recompute_logprob=true`, `use_decoupled_loss=true`, ratio rejection upper
  bound `5.0`, group reward normalization with group size 2.

Checkpointing:

- Periodic checkpoints every 500 optimizer steps.
- A final checkpoint is forced on step 5,720.
- The final teacher checkpoint has the same number of prompt tasks seen as the
  full SFT trajectory examples seen.
- The comparison table will also select the teacher checkpoint closest to the
  completed SFT wall-clock time.

SGLang status:

- SGLang was tested with `.venv-rollout-sglang`.
- A PYTHONPATH isolation bug was fixed so rollout backends do not inherit
  incompatible site-packages from `.venv-megatron`.
- `sglang.skip_tokenizer_init` is false because string stop sequences are needed
  to stop the student at `"commands"`.
- A 64-row, 2-step SGLang teacher-answer-RL smoke completed successfully.
- Production teacher-answer-RL defaults to vLLM because it has been stable and
  has lower operational risk for the long run.

## Released SFT Terminal-Bench Sanity Check

The released `nvidia/Nemotron-Terminal-8B` model was evaluated through the
Terminus-2 agent on a 10-task, 5-trial subset before launching the local
full-scale SFT. This is a sanity check that the local Terminal-Bench/serving
path works on an easier subset; it is not directly comparable to the paper's
10%/13% Terminal-Bench scores, which are full 89-task scores.

Result file:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/results/tb_nemotron8b_easy10_5_final.summary.json
```

Summary:

- Trials: 50.
- Passes: 22.
- Easy-subset pass rate: 44%.
- Conservative full-suite lower bound: 22 / (89 tasks * 5 trials) = 4.94%,
  assuming every unrun task fails.

Task pass rates:

| task | pass rate |
| --- | ---: |
| constraints-scheduling | 0.20 |
| fix-git | 0.20 |
| git-leak-recovery | 0.60 |
| log-summary-date-ranges | 1.00 |
| modernize-scientific-stack | 1.00 |
| multi-source-data-merger | 0.80 |
| nginx-request-logging | 0.60 |
| regex-log | 0.00 |
| sqlite-db-truncate | 0.00 |
| vulnerable-secret | 0.00 |

## Offline Evaluation

Offline eval script:

```text
rlvr_demo/terminal_offline_eval.py
```

Metrics:

- JSON parse validity.
- Commands schema validity.
- `task_complete` validity.
- `task_complete` prediction accuracy.
- Normalized command sequence similarity.
- Command exact match rate.

Run after both training runs:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh eval
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh compile
```

The compiler emits:

```text
$FILEROOT/results/qwen3-8b-released-fullmix-nofilter-sft-vs-teacher-rl/checkpoint_log.jsonl
$FILEROOT/results/qwen3-8b-released-fullmix-nofilter-sft-vs-teacher-rl/checkpoint_log.csv
$FILEROOT/results/qwen3-8b-released-fullmix-nofilter-sft-vs-teacher-rl/comparison_table.json
```

The checkpoint log includes:

- `algorithm`
- `base_model`
- `dataset`
- `dataset_split`
- `dataset_config`
- `examples_seen`
- `tasks_seen`
- `optimizer_step`
- `epoch`
- `fractional_epoch`
- `elapsed_wall_clock_sec`
- `timestamp_saved`
- loss/reward metrics
- offline eval metrics when available

## Terminal-Bench Evaluation After Training

Serve a checkpoint with:

```bash
bash rlvr_demo/scripts/serve_terminal_model_vllm.sh \
  /path/to/checkpoint \
  terminal-local-checkpoint \
  30080
```

Submit CPU-node Harbor/Terminal-Bench with:

```bash
bash rlvr_demo/scripts/run_terminal_bench_eval_slurm_cpu.sh \
  tb-local-checkpoint-easy10-5 \
  openai/terminal-local-checkpoint \
  http://10.0.159.57:30080/v1 \
  $FILEROOT/terminal_bench_eval/harbor_jobs/tb-local-checkpoint-easy10-5
```

Post-training eval must use at least 10 tasks and 5 trials per task for each
reported checkpoint. The currently selected 10-task set is:

- `modernize-scientific-stack`
- `fix-git`
- `git-leak-recovery`
- `log-summary-date-ranges`
- `multi-source-data-merger`
- `nginx-request-logging`
- `vulnerable-secret`
- `constraints-scheduling`
- `regex-log`
- `sqlite-db-truncate`

The original requested set also included `prove-plus-comm` and `pypi-server`;
these can be used if Docker capacity is sufficient, but the faster 10-task set
above is the current efficient evaluation set.

Every subset Terminal-Bench result must report both:

- The pass rate on the selected subset.
- The conservative full-suite lower bound:
  `subset_passes / (89 * trials_per_task)`, treating all unrun Terminal-Bench
  tasks as failures. This is the only subset-derived number that can be compared
  against the paper's full-suite 10%/13% figures without overstating coverage.

`rlvr_demo.terminal_experiment summarize-harbor` writes these fields
automatically as `selected_subset_pass_rate` and
`full_suite_lower_bound_pass_rate`.

## Final Reporting Checklist

Fill these after the long runs finish:

- Final SFT checkpoint path, optimizer step, examples seen, elapsed wall-clock.
- Teacher checkpoint closest to full SFT wall-clock.
- Final teacher checkpoint path, optimizer step, examples seen, elapsed
  wall-clock.
- Offline eval table for the three comparison points.
- Terminal-Bench 10-task x 5-trial table for each reported checkpoint,
  including selected-subset score and conservative full-suite lower bound.
- Limitations and next steps.

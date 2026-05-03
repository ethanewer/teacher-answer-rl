# Qwen3-8B Terminal-Agent Paper Baseline

This file tracks the Qwen3-8B reproduction of the Nemotron-Terminal setup from
arXiv:2602.21193 and the follow-on teacher-answer-RL comparison.

The full runbook is:

```text
rlvr_demo/TERMINAL_AGENT_FULL_RECIPES.md
```

## Paper Alignment

Relevant setup points from the paper tables and recipe:

- Smallest paper model size: 8B.
- Best SFT data condition: no filtering, 264k total examples in the paper run.
- Best SFT context length: 32k.
- Local base model: `Qwen/Qwen3-8B`.
- Local released data: `nvidia/Nemotron-Terminal-Corpus`.
- Local training data: released `full_mix`, 366,154 trajectories observed at
  launch.
- Local SFT batch size: 128.
- Local SFT epochs: 2.
- Local SFT max context: 32,768 tokens.

The released paper SFT checkpoint `nvidia/Nemotron-Terminal-8B` was evaluated
first to verify that the local serving and Terminal-Bench path can reproduce a
score above 10%.

## Released SFT Terminal-Bench Check

Model served:

```text
nvidia/Nemotron-Terminal-8B
```

Serving:

- vLLM OpenAI-compatible server.
- Qwen3 reasoning parser.
- Model name: `terminal-sft-baseline`.
- Max model length: 40,960.
- Tensor parallelism: 8.

Terminal-Bench result:

- Tasks: 10.
- Trials per task: 5.
- Total trials: 50.
- Passes: 22.
- Pass rate: 44%.

Result file:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/results/tb_nemotron8b_easy10_5_final.summary.json
```

Per-task pass rates:

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

## Current SFT Run

Config:

```text
rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200_slurm4.yaml
```

Command:

```bash
FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b \
bash rlvr_demo/scripts/run_terminal_sft_h200.sh \
  rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200_slurm4.yaml
```

Experiment:

```text
qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4
```

Current clean Slurm job:

```text
7399
```

The run was restarted after adding forced final checkpoint saving. The cancelled
pre-patch attempt did not produce checkpoints; its logs are preserved with an
`aborted_pre_finalsave_20260503T001650Z` suffix.

## Planned Teacher-Answer-RL Run

Teacher-answer-RL starts from the completed local SFT checkpoint:

```bash
SFT_EXPERIMENT=qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4 \
TEACHER_EXPERIMENT=qwen3-8b-terminal-teacher-answer-rl-released-fullmix-nofilter-b128-s2-5720step-h200 \
FILEROOT=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b \
DATASET_CONFIG=full_mix \
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-full
```

Teacher-answer-RL semantics:

- The student samples the assistant reasoning/prefix from a fresh assistant turn.
- Generation stops at top-level `"commands"` when possible.
- The PPO loss mask covers all sampled student-prefix tokens.
- The scalar reward for those tokens is computed only from the likelihood of
  the teacher `"commands"` and `"task_complete"` span.
- No reward or string comparison is applied directly to reasoning text.

## Final Outputs Pending

After SFT and teacher-answer-RL complete, update this file with:

- Final SFT checkpoint and wall-clock time.
- Teacher checkpoint closest to SFT wall-clock.
- Final teacher checkpoint and wall-clock time.
- Offline eval metrics for the three comparison points.
- Terminal-Bench 10-task x 5-trial results for each reported checkpoint.
- Limitations and next steps.

# Comparable Default-Agent Terminal-Bench Eval, 2026-05-12

This rerun compares four conditions on the same 10-task Terminal-Bench subset, with 5 attempts per task.  All evals used `rlvr_demo/terminal_bench_default_agent_service_eval.py`, the default-agent tool harness, and the remote task service at `http://10.0.148.27:39080`.

Eval root:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_bench_eval/comparable_default_agent_nonthinking_20260512`

Dataset:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_bench_eval/adapted_easy10_tools_20260511`

Eval settings:

- `n_attempts: 5`
- `n_concurrent: 5`
- `max_turns: 40`
- `max_tokens_per_turn: 8192`
- `trajectory_timeout: 3600`
- `model_timeout: 900`
- `temperature: 0.7`
- `top_p: 0.8`
- `top_k: 20` for local Qwen/vLLM evals
- DeepSeek eval used `--omit-top-k-extra-body --thinking-type disabled`

## Harness Matrix

| Condition | Training harness | Eval harness | Training updates | Student rollouts | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3-4B baseline | N/A | Default-agent tool harness | 0 | 0 | `Qwen/Qwen3-4B-Thinking-2507` |
| Qwen3-4B GRPO-only | Default-agent tool harness | Default-agent tool harness | 15 | 3,840 | `batch_size=32`, `n_trajs=8` |
| Qwen3-4B GRPO + teacher-answer RL | Default-agent tool harness | Default-agent tool harness | 15 | 3,840 | 13 updates in the main run plus 2 updates from the step-12 checkpoint continuation; DeepSeek teacher had `thinking.type=disabled` |
| DeepSeek teacher | N/A | Default-agent tool harness | 0 | 0 | `deepseek-v4-pro`, non-reasoning via `thinking.type=disabled` |

The two trained Qwen runs used the same default-agent harness and the same number of student rollouts.  The teacher-answer run continued from the last complete step-12 checkpoint after a rollout-engine stall; the last two updates were run from that model checkpoint with the same harness and rollout count, but optimizer state was not recovered because the recipe had recovery disabled.

## Results

`mean_reward_all_trials` counts errors as zero and is the primary mean reward used below.  DeepSeek had one malformed tool-call JSON row on `sqlite-db-truncate` attempt 2; a single-attempt retry produced the same kind of malformed JSON, so the original row is retained as an error/failure.

| Condition | Eval output dir | Trials | Completed | Errors | Mean reward, all trials | Pass@1 | Pass@2 | Pass@4 | Pass@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B baseline | `qwen3-baseline-easy10-5` | 50 | 50 | 0 | 0.3958 | 0.2000 | 0.3200 | 0.4000 | 0.4000 |
| Qwen3-4B GRPO-only | `qwen3-grpo-matched-s15-easy10-5` | 50 | 50 | 0 | 0.3780 | 0.2000 | 0.2800 | 0.3800 | 0.4000 |
| Qwen3-4B GRPO + teacher-answer RL | `qwen3-tarl-nonthinking-matched-s15-easy10-5` | 50 | 50 | 0 | 0.3995 | 0.1800 | 0.2600 | 0.3000 | 0.3000 |
| DeepSeek non-reasoning teacher | `deepseek-v4-pro-nonthinking-easy10-5` | 50 | 49 | 1 | 0.7800 | 0.7800 | 0.8000 | 0.8000 | 0.8000 |

Among the Qwen variants, teacher-answer RL has the highest all-trial mean reward, but the baseline and GRPO-only runs have higher pass@k.  DeepSeek is much stronger on this easy subset, despite the single malformed sqlite row.

## Checkpoints

Baseline:

`Qwen/Qwen3-4B-Thinking-2507`

GRPO-only:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-thinking-default-agent-terminal-grpo-matched-easy-nodepmedium-h200/qwen3_4b_thinking_easymedium50_default_grpo_matched_a1tp2r6_b32x8_ctx48k_o8192_t20_obs1k_mb48k_r16_tt1500_req2400_ckpt1_s15/default/epoch0epochstep14globalstep14`

GRPO + teacher-answer RL:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-thinking-default-agent-terminal-online-teacher-answer-rl-nonthinking-easy-nodepmedium-h200/qwen3_4b_thinking_easymedium50_default_online_teacher_deepseekv4pro_nonthinking_matched_resume_gs12_a1tp2r6_b32x8_ctx48k_o8192_t20_obs1k_mb48k_r16_tt1500_req2400_ttok8192_ckpt1_s2/default/epoch0epochstep1globalstep1`

## Recipe Files

- `rlvr_demo/configs/qwen3_4b_thinking_terminal_grpo_easy_medium_default_h200_matched_s15.yaml`
- `rlvr_demo/configs/qwen3_4b_thinking_terminal_online_teacher_answer_rl_easy_medium_default_h200_nonthinking_matched_s15.yaml`
- `rlvr_demo/configs/qwen3_4b_thinking_terminal_online_teacher_answer_rl_easy_medium_default_h200_nonthinking_matched_resume_gs12_s2.yaml`
- `rlvr_demo/manifests/easy_medium_nodep_50_50_b32_s15_resume_from_row896.csv`

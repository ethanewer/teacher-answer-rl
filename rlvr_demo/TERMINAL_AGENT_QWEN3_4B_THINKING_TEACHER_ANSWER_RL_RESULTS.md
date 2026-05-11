# Qwen3 4B Thinking Default-Agent Teacher-Answer RL

## Training Run

Algorithm: online teacher-answer RL on top of default-agent terminal rollouts.

- Student rollouts use the default-agent tool-call harness with exactly one user message.
- For each student turn, DeepSeek `deepseek-v4-pro` generates the next tool call from the student prefix.
- The teacher-answer log-prob reward is applied only to the student reasoning span for that turn.
- The verifier reward is the same terminal task verifier reward used by the GRPO recipe.

Main recipe:

```text
rlvr_demo/configs/qwen3_4b_thinking_terminal_online_teacher_answer_rl_easy_medium_default_h200.yaml
```

Final checkpoint:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-thinking-default-agent-terminal-online-teacher-answer-rl-easy-nodepmedium-h200/qwen3_4b_thinking_easymedium50_default_online_teacher_deepseekv4pro_a1tp2r6_b32x8_ctx48k_o8192_t20_obs1k_mb48k_r16_tt1500_req2400_ttok8192_ckpt1_s15/default/epoch0epochstep14globalstep14
```

TensorBoard logdir:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/tensorboard/qwen3-4b-thinking-default-agent-terminal-online-teacher-answer-rl-easy-nodepmedium-h200/qwen3_4b_thinking_easymedium50_default_online_teacher_deepseekv4pro_a1tp2r6_b32x8_ctx48k_o8192_t20_obs1k_mb48k_r16_tt1500_req2400_ttok8192_ckpt1_s15
```

Console log:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/console_logs/qwen3_4b_thinking_online_teacher_deepseekv4pro_real_20260511_041316.log
```

Outcome: training completed all 15 configured steps and saved the final checkpoint above.

## Terminal-Bench Easy-10 Eval

The Slurm Harbor eval path could not be used on the newly started `m7i-cpu` nodes because those nodes did not expose a Docker daemon. The eval below used the existing Docker-capable terminal task service at:

```text
http://10.0.148.27:39080
```

The eval still ran real Terminal-Bench task containers and real verifiers. It used the default-agent harness rather than the Terminus JSON harness, matching the training harness for this checkpoint. The task adapters were prebuilt on shared storage with `tmux` and `asciinema` injected for Terminal-Bench session support:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_bench_eval/adapted_easy10_tools_20260511
```

Settings:

- Tasks: `modernize-scientific-stack`, `log-summary-date-ranges`, `multi-source-data-merger`, `nginx-request-logging`, `git-leak-recovery`, `fix-git`, `constraints-scheduling`, `vulnerable-secret`, `regex-log`, `sqlite-db-truncate`
- Attempts: 5 per task, 50 trials per model
- Concurrency: 5 trials per model
- Max turns: 40
- Max tokens per turn: 8192
- Model context: 49152
- Sampling: temperature 0.7, top-p 0.8, top-k 20

Summary paths:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_bench_eval/service_eval/qwen3-thinking-before-ta-rl-easy10-5-20260511/summary.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_bench_eval/service_eval/qwen3-thinking-after-ta-rl-easy10-5-20260511/summary.json
```

## Results

| Model | Completed | Errors | Full passes | Mean reward | Mean reward, error=0 | pass@1 | pass@2 | pass@4 | pass@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 before RL | 50/50 | 0 | 8/50 | 0.3975 | 0.3975 | 0.160 | 0.240 | 0.300 | 0.300 |
| Qwen3 after teacher-answer RL | 50/50 | 0 | 9/50 | 0.4075 | 0.4075 | 0.180 | 0.290 | 0.380 | 0.400 |

Per-task full passes and mean rewards:

| Task | Before mean | Before passes | After mean | After passes | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| modernize-scientific-stack | 0.9000 | 4/5 | 0.8000 | 3/5 |  |
| log-summary-date-ranges | 0.4000 | 0/5 | 0.5000 | 0/5 |  |
| multi-source-data-merger | 0.3333 | 0/5 | 0.3333 | 1/5 |  |
| nginx-request-logging | 0.2750 | 0/5 | 0.1750 | 0/5 | one after-RL context-limit row was repaired with a targeted rerun |
| git-leak-recovery | 0.8000 | 2/5 | 0.8000 | 2/5 |  |
| fix-git | 0.0000 | 0/5 | 0.0000 | 0/5 |  |
| constraints-scheduling | 0.6000 | 2/5 | 0.8000 | 3/5 |  |
| vulnerable-secret | 0.6667 | 0/5 | 0.6667 | 0/5 |  |
| regex-log | 0.0000 | 0/5 | 0.0000 | 0/5 |  |
| sqlite-db-truncate | 0.0000 | 0/5 | 0.0000 | 0/5 |  |

## Notes

- The trained checkpoint slightly improved pass metrics on this default-agent easy-10 eval: full passes went from 8/50 to 9/50, and pass@5 went from 0.30 to 0.40.
- The initial after-RL eval had one `nginx-request-logging` trial exceed the vLLM request boundary with 40961 input tokens plus 8192 requested output tokens. The eval script was patched afterward so future context-limit rows break to verifier scoring instead of dropping the trial. That failed row was replaced with a targeted rerun of the same task/attempt, which completed with reward 0.375.
- The original CPU task service was left active after eval: `{"ok":true,"sessions":32,"max_sessions":128,"max_starts":24}`.

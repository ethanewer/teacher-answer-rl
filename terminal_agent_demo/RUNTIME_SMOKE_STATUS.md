# Runtime Smoke Status

Date: 2026-05-12

Runtime smoke root:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/runtime_smoke/runtime_smoke_20260512_201005
```

## Confirmed Working

| Recipe | Slurm job | Result | Evidence |
| --- | ---: | --- | --- |
| SFT | 41182 | Completed one AReaL SFT train step | `sft/update_successful = 1`, `sft/n_seqs = 8`, `sft/n_valid_tokens = 28675`, `Training completes!` |
| Terminal-Bench eval | 41185 | Completed one Harbor eval task with the Terminus tool-calling agent | `terminal-bench - terminus-tool-calling - deepseek-v4-pro`, `Trials 1`, `Exceptions 0`, `Mean 1.000` |
| Teacher-answer RL | 41186 | Completed one rollout/scoring/update step at 32k context | `ppo_actor/n_seqs = 4`, `ppo_actor/n_valid_tokens = 1024`, `teacher_scoring_dropped_tokens = 0`, `rollout/accepted = 1`, `Training completes!` |

The first teacher-answer-RL smoke job, 41183, used an 8192-token smoke context and failed because the converted examples were already longer than that. The confirmed rerun, 41186, used the intended 32768-token context.

## GRPO Status

GRPO is not yet confirmed working end to end. The smoke runs did verify that the harness can start Docker-backed terminal rollouts, but the recipe has not completed a trainer update.

Observed GRPO failures:

| Slurm job | Status | Problem | Current state |
| ---: | --- | --- | --- |
| 41184 | Failed | AReaL treated the GRPO workflow as an agent workflow and required a `.run()` method: `Agent must have a callable 'run' method`. | Fixed in `terminus_tool_calling.py` by making `TerminusToolTerminalGRPOWorkflow` subclass `RolloutWorkflow` directly. |
| 41187 | Failed | The AReaL OpenAI wrapper used `gconfig.max_new_tokens` as the total engine context cap, causing `len of prompt tokens 1072 exceeds max_total_tokens 1024`. | Fixed in `terminus_tool_calling.py` by passing `engine_max_tokens=max_tokens_per_trajectory`. |
| 41188 | Failed | The rollout path reached terminal generation, but the actor trainer got one outer trajectory with actor data parallel size 4: `Number of items (1) must be >= K (4)`. | Still open. |
| 41189 | Failed | Retrying with `n_trajs=4` started four rollout workers and generated multiple terminal sessions, but AReaL still returned one outer workflow item to the trainer, so the same `Number of items (1) must be >= K (4)` failure occurred. | Still open. |

Root cause for the remaining GRPO issue:

- AReaL partitions the outer list returned by `prepare_batch`, one item per dataloader example, across actor data-parallel ranks.
- `n_trajs` creates multiple sampled terminal trajectories inside a single workflow result; it does not increase the outer trainer item count.
- The current smoke overrides used `train_dataset.batch_size=1` and a one-row manifest, while the default actor backend is `megatron:d4p1t1`.

Likely fixes to test next:

- Run GRPO smoke with at least four dataloader examples per update, for example `train_dataset.batch_size=4` and a manifest with at least four task rows.
- Or run the smoke with actor data parallel size 1, for example a `d1p1t1` actor backend, while leaving the full production recipe at `d4p1t1`.

Do not treat the current GRPO recipe as confirmed until it completes one rollout plus one actor update.

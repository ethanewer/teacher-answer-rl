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
| GRPO matched synthetic tasks | local H200 node | Completed one grouped GRPO rollout/update step with Docker-backed synthetic tasks | `ppo_actor/update/update_successful = 1`, `ppo_actor/n_seqs = 16`, `ppo_actor/update/n_tokens = 54600`, `rollout/num_trajectories_failed = 0`, `Training completes!` |

The first teacher-answer-RL smoke job, 41183, used an 8192-token smoke context and failed because the converted examples were already longer than that. The confirmed rerun, 41186, used the intended 32768-token context.

## GRPO Status

GRPO is now confirmed working end to end for the matched synthetic-task path.
The confirmed local smoke log is:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/smoke_logs/grpo_matched_smoke_20260512_230912.log
```

Metrics are in:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/logs/ewer/qwen3-4b-thinking-terminus-tool-grpo-medium-even-matched-smoke/smoke/metrics.jsonl
```

The smoke used the local H200 node, loaded `Qwen/Qwen3-4B-Thinking-2507`,
materialized `nvidia/Nemotron-Terminal-Synthetic-Tasks` rows into
Terminal-Bench-compatible Docker task directories, generated terminal rollouts
with the Terminus tool-calling harness, ran the verifier with zero failed
trajectories, completed actor log-prob recomputation and PPO update, saved a
checkpoint, and cleaned up the local vLLM and actor workers.

The smoke did not produce a passing verifier reward on the two medium matched
tasks used for the short run: `ppo_actor/task_reward/avg = 0.0`. That is a model
performance result, not a recipe/runtime failure.

The real GRPO recipe is:

```text
terminal_agent_demo/grpo/config_even_medium_real.yaml
terminal_agent_demo/grpo/run_even_medium_real.sh
```

It uses the same medium even-row source split as SFT and teacher-answer-RL, but
intersects those source task IDs with the locally cached synthetic-task dataset.
The current matched manifest contains 23,120 executable synthetic tasks from
41,854 unique SFT source task IDs. The recipe uses 16 prompts/update and 4
completions/prompt, so each actor update runs 64 Docker-backed terminal
rollouts. The smoke update produced 54,600 update tokens at 2 prompts and 2
completions, which scales to about 873,600 update tokens for the real recipe
when task lengths are similar.

Observed GRPO failures:

| Slurm job | Status | Problem | Current state |
| ---: | --- | --- | --- |
| 41184 | Failed | AReaL treated the GRPO workflow as an agent workflow and required a `.run()` method: `Agent must have a callable 'run' method`. | Fixed in `terminus_tool_calling.py` by making `TerminusToolTerminalGRPOWorkflow` subclass `RolloutWorkflow` directly. |
| 41187 | Failed | The AReaL OpenAI wrapper used `gconfig.max_new_tokens` as the total engine context cap, causing `len of prompt tokens 1072 exceeds max_total_tokens 1024`. | Fixed in `terminus_tool_calling.py` by passing `engine_max_tokens=max_tokens_per_trajectory`. |
| 41188 | Failed | The rollout path reached terminal generation, but the actor trainer got one outer trajectory with actor data parallel size 4: `Number of items (1) must be >= K (4)`. | Fixed by using AReaL's normal grouped rollout path with multiple dataloader examples per update instead of relying on `n_trajs` to increase trainer batch size. |
| 41189 | Failed | Retrying with `n_trajs=4` started four rollout workers and generated multiple terminal sessions, but AReaL still returned one outer workflow item to the trainer, so the same `Number of items (1) must be >= K (4)` failure occurred. | Fixed by preserving `gconfig.n_samples` for trainer grouping and using a private one-sample generation config inside each grouped terminal rollout. |

Root cause of the previous GRPO issue:

- AReaL partitions the outer list returned by `prepare_batch`, one item per dataloader example, across actor data-parallel ranks.
- `n_trajs` creates multiple sampled terminal trajectories inside a single workflow result; it does not increase the outer trainer item count.
- The old smoke overrides used `train_dataset.batch_size=1` and a one-row manifest, while the actor backend used multiple data-parallel ranks.
- `TerminusToolTerminalGRPOWorkflow` also temporarily mutated the shared `gconfig.n_samples` to 1, which broke grouped GRPO accounting outside the workflow.

The confirmed fix keeps the shared `gconfig.n_samples` intact, uses a private
one-sample generation config inside each terminal workflow, and sizes
`train_dataset.batch_size` so AReaL has enough outer examples for actor data
parallelism and grouped GRPO.

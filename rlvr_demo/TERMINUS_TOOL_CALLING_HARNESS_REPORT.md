# Terminus Tool-Calling Harness Report

Branch: `terminus-tool-calling-harness`  
Base commit: `88e76aeb1c39fe781eda2581f7a5773b939df345`

## Summary

This branch adds a Terminus-compatible tool-calling harness in `rlvr_demo/terminus_tool_calling.py`.
The harness keeps the Terminus-2 action payload shape, but moves it out of visible assistant JSON and into a single OpenAI tool call named `execute_commands`.

The tool arguments exactly match the Terminus-2 response object:

```json
{
  "analysis": "short state analysis",
  "plan": "short next-step plan",
  "commands": [
    {"keystrokes": "shell command or keystrokes", "duration": 0.1}
  ],
  "task_complete": false
}
```

This makes Terminus-2 trajectories easy to map:

- Old assistant JSON payload -> assistant `tool_calls[0].function.arguments`.
- Old user terminal observation -> `tool` message with `tool_call_id`.
- Initial task prompt remains the only real user message.

## System Prompt Changes

The new prompt identifies the agent as Terminus and instructs it to call `execute_commands` exactly once per turn. It preserves the Terminus-2 field meanings for `analysis`, `plan`, `commands`, and `task_complete`, but explicitly says not to put the Terminus JSON payload in visible assistant text. All action fields belong in the tool-call arguments.

The important template invariant is that terminal observations are appended as tool responses rather than fresh user prompts. This avoids the Qwen reasoning-template behavior that strips earlier assistant `<think>...</think>` blocks after a new user message.

## Qwen Template Check

Command:

```bash
.venv-megatron/bin/python -m rlvr_demo.terminus_tool_calling check-qwen-template \
  --local-files-only \
  --output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/qwen_template_append_only_render_default.txt
```

Result:

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Roles: `system`, `user`, `assistant`, `tool`, `assistant`
- Real user messages: 1
- First assistant thinking preserved: yes
- Second assistant thinking preserved: yes
- Tool response rendered: yes

This confirms the append-only property needed for reasoning-model SFT/RL.

## GRPO Rollout Smoke

The old GRPO rollout runner in this commit uses old Terminal-Bench `task.yaml` layout and Python Docker SDK. On this H200 node, the Docker CLI can run through its setgid binary but Python Docker SDK cannot access `/var/run/docker.sock`, so this branch adds a CLI-backed terminal wrapper for the GRPO smoke path.

Smoke task:

- Manifest: `rlvr_demo/manifests/terminus_tool_old_layout_smoke.csv`
- Task: `rlvr_demo/smoke_tasks/terminus_tool_old_layout_file_pass`
- Verifier: pytest checks `/app/answer.txt == "pass"`

Command:

```bash
.venv-megatron/bin/python -m rlvr_demo.terminus_tool_calling deepseek-synthetic-smoke \
  --manifest /wbl-fast/usrs/ee/teacher-answer-rl/AReaL/rlvr_demo/manifests/terminus_tool_old_layout_smoke.csv \
  --limit 1 \
  --stop-after-pass \
  --max-turns 4 \
  --max-tokens 4096 \
  --temperature 0.2 \
  --top-p 0.8 \
  --results-output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/deepseek_smoke/grpo_rollout_smoke_results.jsonl
```

Result:

- Model: `deepseek-v4-pro`
- Reward: 1.0
- Verifier passed: yes
- Log path: `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/deepseek_smoke`

DeepSeek returned a 400 when `tool_choice` was forced because this endpoint is routed as `deepseek-reasoner`; the harness retries without `tool_choice` while still providing the `execute_commands` tool schema. The model then emits valid tool calls.

## Terminal-Bench Harbor Smoke

Harbor initially failed before agent setup because importing the GRPO stack installed uvloop, and Harbor's Docker environment needs asyncio subprocess child watchers. The branch now keeps GRPO imports lazy so the Harbor agent import does not mutate the event-loop policy.

Run:

```bash
.venv/bin/harbor run \
  --config /wbl-fast/usrs/ee/teacher-answer-rl/AReaL/rlvr_demo/results/terminal_bench_eval_configs/terminus-tool-deepseek-fix-git-smoke.yaml \
  --job-name terminus-tool-deepseek-fix-git-smoke-lazyimports \
  --jobs-dir /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/harbor_smoke/jobs_lazyimports \
  --yes
```

Result:

- Task: `fix-git`
- Model: `deepseek-v4-pro`
- Agent: `terminus-tool-calling`
- Trials: 1
- Exceptions: 0
- Mean reward: 1.0
- Verifier reward: 1.0
- Result path: `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/harbor_smoke/jobs_lazyimports/terminus-tool-deepseek-fix-git-smoke-lazyimports/result.json`

CPU Slurm probe note: `m7i-cpu` and `m7i-cpu2` nodes tested in this run did not have a `docker` binary in PATH, so the Harbor smoke was run locally through Harbor's Docker CLI path. It used a unique Harbor compose project and the temporary task container was removed; existing GRPO containers remained running.

## Corpus Conversion

The converter reads local cached Arrow files for `nvidia/Nemotron-Terminal-Corpus`, config `skill_based_medium`, split `train`, and emits JSONL records with:

- `messages`: converted `system/user/assistant/tool/...` conversation.
- `tools`: the single `execute_commands` schema.
- Source metadata: task, trial name, model, agent, dataset config.

Commands used:

```bash
.venv-megatron/bin/python -m rlvr_demo.terminus_tool_calling convert-corpus \
  --limit 1000 \
  --output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/data/skill_based_medium.terminus_tool.sample1000.jsonl \
  --summary-output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/data/skill_based_medium.terminus_tool.sample1000.summary.json

.venv-megatron/bin/python -m rlvr_demo.terminus_tool_calling inspect-converted \
  --input /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/data/skill_based_medium.terminus_tool.sample5.jsonl \
  --output /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/data/skill_based_medium.terminus_tool.sample5.inspect.md
```

Sample-1000 result:

- Converted: 993
- Failed: 7
- Failed rows are malformed source assistant turns, usually unfinished prose or unrelated data JSON instead of a Terminus action payload. They are skipped rather than repaired, because inventing missing tool actions would change the source trajectory.

Visual inspection of the converted sample showed the expected alignment: the first assistant tool call contains the original Terminus `analysis`, `plan`, `commands`, and `task_complete`, and the following tool message contains the old terminal observation.

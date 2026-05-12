# Terminal Agent RL

This repository contains terminal-agent training and evaluation recipes built
around AReaL, Terminal-Bench, and a Terminus-compatible tool-calling harness.

The active code lives in the `AReaL` submodule. The top-level repository keeps
the experiment reports and points `AReaL` at the terminal-agent branch that is
ready to clone and run.

## Clone

```bash
git clone --recurse-submodules https://github.com/ethanewer/teacher-answer-rl
cd teacher-answer-rl/AReaL
```

If the repo was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Terminal Agent Demo

The main entry point is:

```text
AReaL/terminal_agent_demo/
```

That directory is self-contained. It includes:

- `terminus_tool_calling.py`: Terminus-style terminal harness using a single
  `execute_commands` tool.
- `sft/`: SFT recipe for converted Terminus trajectories.
- `teacher_answer_rl/`: teacher-answer-RL recipe using DeepSeek teacher
  continuations.
- `grpo/`: GRPO terminal rollout recipe.
- `eval/`: Harbor / Terminal-Bench evaluation launchers.
- `scripts/`: environment setup, corpus conversion, template checks, and smoke
  checks.

The harness preserves Qwen reasoning-model histories by keeping a single real
user task message and appending terminal observations as tool responses instead
of new user messages.

## Data

The conversion pipeline maps `nvidia/Nemotron-Terminal-Corpus` Terminus-2
trajectories into the tool-calling format:

```bash
cd AReaL
terminal_agent_demo/scripts/prepare_converted_data.sh
```

Default converted output:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_medium.terminus_tool.jsonl
```

## Recipes

From `AReaL/`:

```bash
terminal_agent_demo/sft/run.sh
terminal_agent_demo/teacher_answer_rl/run.sh
terminal_agent_demo/grpo/run.sh
```

Terminal-Bench evaluation:

```bash
terminal_agent_demo/eval/serve_terminal_model_vllm.sh /path/to/checkpoint terminal-local 30080
terminal_agent_demo/eval/run_terminal_bench_easy10_split_slurm_cpu.sh eval-name openai/terminal-local http://127.0.0.1:30080/v1
```

## Current Smoke Status

The confirmed runtime smoke tests are documented in:

```text
AReaL/terminal_agent_demo/RUNTIME_SMOKE_STATUS.md
```

As of the latest update:

- SFT completed one AReaL train step.
- Teacher-answer-RL completed one rollout/scoring/update step at 32k context.
- Terminal-Bench eval passed one easy task using the tool-calling harness.
- GRPO reached terminal rollouts but still needs a follow-up fix for actor
  data-parallel batch sizing before it is considered confirmed.

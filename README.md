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

For the real medium-split experiments, the reproducible data-prep script keeps
only original even-index source rows:

```bash
cd AReaL
terminal_agent_demo/scripts/prepare_even_medium_data.sh
```

Default even-row output:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_medium.even_original.terminus_tool.jsonl
```

## Recipes

From `AReaL/`:

```bash
terminal_agent_demo/sft/run.sh
terminal_agent_demo/teacher_answer_rl/run.sh
terminal_agent_demo/grpo/run.sh
```

## Real Medium Experiments

The real H200 recipes use `Qwen/Qwen3-4B-Thinking-2507`, preserve reasoning in
the converted trajectories, use the `terminus_tool_calling` harness, and train on
the full medium even-row split.

Launch both real runs from a submit node:

```bash
cd AReaL
terminal_agent_demo/scripts/launch_real_even_medium_runs.sh
```

The launcher records the exact sbatch/config paths in:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/launch_logs/
```

Individual real recipes:

```text
AReaL/terminal_agent_demo/sft/config_even_medium_real.yaml
AReaL/terminal_agent_demo/sft/run_even_medium_real.sbatch
AReaL/terminal_agent_demo/teacher_answer_rl/config_even_medium_real.yaml
AReaL/terminal_agent_demo/teacher_answer_rl/run_even_medium_real.sbatch
```

The SFT recipe trains full converted trajectories with sequence packing at 32k
context and about 0.5M-0.7M tokens per update. The teacher-answer-RL recipe is a
turn-level method: it uses the same even-row trajectories, expands them into
assistant-turn prompts, samples 32 prompts with 2 completions per prompt, and
uses Terminal-Bench-style GRPO settings with 32k context.

Runtime metrics are written under:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/logs/
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

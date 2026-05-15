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
- `grpo/`: GRPO terminal rollout recipes, including matched synthetic-task GRPO.
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
AReaL/terminal_agent_demo/grpo/config_even_medium_real.yaml
AReaL/terminal_agent_demo/grpo/run_even_medium_real.sh
```

The SFT recipe trains full converted trajectories with sequence packing at 32k
context and about 0.5M-0.7M tokens per update. The default teacher-answer-RL
recipe starts from the final SFT checkpoint, trains on odd medium rows, and uses
the robust full-turn teacher-answer method: reference-model teacher-answer
scoring from the tool-call span, group-normalized rewards, a small supervised
teacher-answer prefix loss on thinking tokens, tool-call syntax reward, and
length penalties. It samples 16 prompts with 2 completions per prompt at 32k
context and 2048 max new tokens.

The real GRPO recipe uses `nvidia/Nemotron-Terminal-Synthetic-Tasks`, intersected
with the same even-row source task IDs used by SFT and teacher-answer-RL. It
runs 16 prompts/update with 4 completions/prompt, for 64 Docker-backed terminal
rollouts/update, 32k context, 1024 max new tokens per turn, and a 2 actor GPU /
6 rollout GPU split on one 8-GPU H200 node. Synthetic task directories are
materialized lazily into Terminal-Bench-compatible task directories under the
run fileroot.

Runtime metrics are written under:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/logs/
```

Terminal-Bench evaluation:

```bash
terminal_agent_demo/eval/serve_terminal_model_vllm.sh /path/to/checkpoint terminal-local 30080
terminal_agent_demo/eval/run_terminal_bench_easy10_split_slurm_cpu.sh eval-name openai/terminal-local http://127.0.0.1:30080/v1
```

## Confirmed Results

Terminal-Bench scores below use the Terminus tool-calling Harbor harness and the
easy-10 split.

| Model / recipe | Checkpoint | Eval setting | Score |
| --- | --- | --- | --- |
| Qwen3-4B-Thinking base | `Qwen/Qwen3-4B-Thinking-2507` | 5 trials/task, `harbor_jobs_r6` | 3/50 = 6% |
| SFT medium even rows | `epoch0epochstep1384globalstep1384` | 5 trials/task, `harbor_jobs_r6` | 13/50 = 26% |
| Robust teacher-answer-RL | `ta-ref-lenpen-w25-p128-syn08-o2048-s50`, step 19 | 1 trial/task, first eval | 3/10 = 30% |
| Robust teacher-answer-RL | same step 19 checkpoint | 1 trial/task, eval rerun | 3/10 = 30% |
| Robust teacher-answer-RL full rerun | `ta-ref-lenpen-w25-p128-syn08-o2048-s50-repro1`, step 24 | 1 trial/task | 3/10 = 30% |
| Robust teacher-answer-RL full rerun | `ta-ref-lenpen-w25-p128-syn08-o2048-s50-repro1`, step 44 | 1 trial/task | 3/10 = 30% |

The default teacher-answer-RL recipe is:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config.yaml
```

It matches the successful robust recipe:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config_odd_medium_from_sft_refscore_lenpen_w25_p128_syn08_o2048_local_s50.yaml
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
- Matched synthetic-task GRPO completed one local rollout/update step with zero
  failed trajectories and `ppo_actor/update_successful = 1`.

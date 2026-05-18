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
recipe is a hand-crafted turn-based reward function for this terminal-agent
harness: it scores similarity between the student's generated `execute_commands`
action and the corpus teacher action with command overlap/presence/completion
rewards, tool-call syntax reward, repeated-command penalties, group reward
normalization, and output-length penalties. This is not the domain-general
teacher-answer likelihood algorithm; it does not apply supervised
teacher-answer loss.

The comparable GRPO recipe starts from the final SFT checkpoint and trains on
odd medium synthetic-task rows matched to the teacher-answer-RL setup. It runs
16 prompts/update with 2 completions/prompt, for 32 Docker-backed terminal
rollouts/update, 32k context, 2048 max new tokens per turn, and a 4 actor GPU /
4 rollout GPU split on one 8-GPU H200 node. Synthetic task directories are
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

Terminal-Bench scores below use the Terminus tool-calling Harbor harness on the
easy-10 split. Each row is a 10-task, 50-trial evaluation: 5 trials per task.

| Model / recipe | Checkpoint | Eval setting | Score |
| --- | --- | --- | --- |
| Base | `Qwen/Qwen3-4B-Thinking-2507` | easy-10, 5 trials/task, `harbor_jobs_r6` | 3/50 = 6% |
| SFT | `epoch0epochstep1384globalstep1384` | easy-10, 5 trials/task, `harbor_jobs_r6` | 13/50 = 26% |
| SFT + hand-crafted turn-reward TA-RL | `ta-cmdpresence-rlonly-gs39-strongcomplete-local-s40-repro-full-r1`, selected step 39 | easy-10, 5 trials/task, `ta-strongcomplete-visibletool-reminders-v3-lowtemp-easy10-a5-o4096-fullrerun-r1` | 20/50 = 40% |
| SFT + GRPO | `grpo-ta-comparable-from-sft-medium-odd-b16-s2-32k-o2048-a4r4-s50`, selected step 34 | easy-10, 5 trials/task, `grpo-medium-b16s2-s34-ecrbuild-easy10-t4096-a5-20260515` | 14/50 = 28% |

The default teacher-answer-RL recipe is:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config.yaml
```

It is the RL-only, hand-crafted turn-reward command-presence/completion recipe
used for the confirmed SFT + hand-crafted turn-reward TA-RL result above.

An experimental domain-general teacher-answer recipe is:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config_general_action_likelihood_prefix.yaml
```

This recipe follows the likelihood-reward direction studied in
`https://arxiv.org/abs/2602.03979`: for each tool-loop state it samples the
student's prefix before the next serialized tool call, then rewards that prefix
by the average log-probability of the corpus teacher's next tool-call block. It
uses only the message history, tool schema, sampled prefix, and teacher
continuation, so it is intended to apply to any tool-calling agent domain with
reference trajectories. It does not parse terminal commands or use
terminal-specific action-similarity rewards.

The SFT + GRPO row uses:

```text
AReaL/terminal_agent_demo/grpo/config_odd_medium_from_sft_ta_comparable_b16_s2_o2048_s50.yaml
AReaL/terminal_agent_demo/grpo/run_odd_medium_from_sft_ta_comparable_b16_s2_o2048_s50.sbatch
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

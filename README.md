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

The default GRPO recipe starts from the same final SFT checkpoint as TA-RL and
uses real Terminal-Bench-style environments and final verifier rewards rather
than teacher trajectories. The current default/best recipe is the
OpenThoughts-style easy-task run with 8 prompts/update, 8 rollouts/prompt, full
trajectory rewards, 1024 max new tokens, 8 turns, no KL, no uniform-reward
filter, and asymmetric PPO clipping. The validated 45-step run completed in
10620.94s / 2.95h. Its held-out easy-subset eval reward increased from 31.25 to
65.625 out of 100 over training. The full external eval row below uses the
held-out-best step-39 checkpoint reached at 9502.97s / 2.64h.

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

Terminal-Bench scores below use the Terminus tool-calling Harbor harness. The
combined score is the old easy-10 split plus the new additional-10 split:
20 tasks, 5 trials per task, 100 trials total. Rows without complete 100-trial
coverage are omitted here and kept in
`AReaL/terminal_agent_demo/additional-results.md`.

| Model / recipe | Training data | Train runtime | Full eval score |
| --- | --- | ---: | ---: |
| Base Qwen3-4B-Thinking | none | 0h | 3/100 |
| SFT medium-even | `skill_based_medium.even_original.terminus_tool.jsonl` | ~5.8h | 17/100 |
| SFT + hand-crafted turn-reward TA-RL, easy-selected | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 27/100 |
| SFT + domain-general likelihood TA-RL, easy-selected | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 27/100 |
| SFT + GRPO default/best, easy-selected | `terminal_synthetic_tasks/easy/manifest.csv` | 9502.97s / 2.64h | 17/100 |

The top-level table is intentionally limited to full 100-trial evals. Per-task
additional-10 results, medium-only rows, mixed GRPO, and eval job IDs are in:

```text
AReaL/terminal_agent_demo/additional-results.md
```

Training-time figures are committed under `figures/`:

```text
figures/tb_perf_vs_rl_training_time.{png,svg,pdf}
figures/grpo_task_reward_vs_training_step.{png,svg,pdf}
figures/default_grpo_train_eval_vs_time.{png,svg,pdf}
```

Regenerate them with:

```bash
AReaL/.venv/bin/python figures/plot_tb_perf_vs_rl_time.py
AReaL/.venv/bin/python figures/plot_grpo_reward_vs_step.py
AReaL/.venv/bin/python figures/plot_default_grpo_train_eval_vs_time.py
```

The hand-crafted turn-reward TA-RL easy-selected row uses:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config_easy_cmdpresence_rlonly_cont_gs39_strongcomplete_local_s40.yaml
```

This is an RL-only, hand-crafted turn-reward command-presence/completion recipe.
It is terminal-harness-specific: it scores similarity between the student's
generated `execute_commands` action and the corpus teacher action.

A confirmed domain-general teacher-answer recipe is:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config_general_action_likelihood_prefix_short_n4.yaml
```

This recipe follows the likelihood-reward direction studied in
`https://arxiv.org/abs/2602.03979`: for each tool-loop state it samples the
student's prefix before the next serialized tool call, then rewards that prefix
by the average log-probability of the corpus teacher's next tool-call block. The
matched recipe uses 4 samples per prompt, 512 max sampled prefix tokens, group
reward normalization across samples, and a YAML-configured output-length
penalty. It uses only the message history, tool schema, sampled prefix, and
teacher continuation, so it is intended to apply to any tool-calling agent
domain with reference trajectories. It does not parse terminal commands or use
terminal-specific action-similarity rewards.

The default GRPO best recipe is:

```text
AReaL/terminal_agent_demo/grpo/config.yaml
AReaL/terminal_agent_demo/grpo/config_easy_openthoughts_b8_s8_o1024_t8_trajectory_valid_nofilter_nokl_s45.yaml
```

No-argument `terminal_agent_demo/grpo/run.sh` launches this default recipe. The
default/best b8/s8 recipe is selected by its improving train reward and
held-out Terminal-Bench subset eval curve. The 100-trial external eval uses:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/checkpoints/ewer/grpo-openthoughts-easy-from-sft-b8-s8-o1024-t8-trajectory-valid-nofilter-nokl-s45/trial0/default/epoch0epochstep39globalstep39
```

The mixed TA-RL rows in `additional-results.md` use:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config_mixed_easy50_mediumodd50_cmdpresence_s1000.yaml
AReaL/terminal_agent_demo/teacher_answer_rl/config_mixed_easy50_mediumodd50_general_likelihood_prefix_short_n4_s1000.yaml
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

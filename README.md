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

The current top-level submodule pin is `AReaL@a48eda20`, on the
`terminus-tool-calling-harness` branch. A fresh `git clone --recurse-submodules`
checks out that commit, which includes the default TA-RL recipe described below.

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
context and about 0.5M-0.7M tokens per update. The default
teacher-answer-RL recipe is now the domain-general likelihood recipe in
`AReaL/terminal_agent_demo/teacher_answer_rl/config.yaml`, matching
`config_general_action_likelihood_prefix_short_n4.yaml`. It samples a partial
next action, appends the teacher continuation, and rewards the student prefix by
reference-model likelihood of that teacher continuation. It uses 4
samples/prompt, 512 max new tokens, `generic_likelihood_prefix`, score mode
`all`, logp reward weight 1.0, length penalty 0.16, and group reward
normalization. The older hand-crafted turn-reward TA-RL recipe is still kept as
a terminal-harness-specific baseline.

The comparable GRPO recipe starts from the final SFT checkpoint and trains on
odd medium synthetic-task rows matched to the teacher-answer-RL setup. It runs
16 prompts/update with 2 completions/prompt, for 32 Docker-backed terminal
rollouts/update, 32k context, 2048 max new tokens per turn, and a 4 actor GPU /
4 rollout GPU split on one 8-GPU H200 node. Synthetic task directories are
materialized lazily into Terminal-Bench-compatible task directories under the
run fileroot.

The default GRPO recipe starts from the same final SFT checkpoint as TA-RL and
uses real Terminal-Bench-style environments and final verifier rewards rather
than teacher trajectories. The current default/best recipe uses the easy
synthetic task manifest, 12 prompts/update, 4 completions/prompt, individual
turn exports, interleaved grouped rollouts, group mean-only reward
normalization, 25 turns, 1024 max new tokens, no KL, and asymmetric PPO
clipping. Its 10-step train reward windows increased from 0.183742 at steps
1-10 to 0.263940 at steps 30-39; the 36-45 window continued to 0.312020. The
external 20-task Terminal-Bench eval improved from the 17/100 SFT baseline to
18/100 at step 19 and 24/100 at step 39. The full external eval row below uses
the step-39 checkpoint reached at 7527.63s / 2.09h.

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
| SFT + domain-general likelihood TA-RL default, easy-selected | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 27/100 |
| SFT + GRPO default/best, easy-selected | `terminal_synthetic_tasks/easy/manifest.csv` | 7527.63s / 2.09h | 24/100 |

The default domain-general likelihood TA-RL score is the true comparable eval
result: 20/50 on easy-10 plus 7/50 on additional-10, with task-scoped eval
repairs and evaluator-side task solutions disabled. Its held-out
eval-over-training checks were not consistently improving under this true
evaluator. Per-task additional-10 results, medium-only rows, mixed GRPO, and
eval job IDs are in:

```text
AReaL/terminal_agent_demo/additional-results.md
```

Training-time figures are committed under `figures/`:

```text
figures/default_grpo_train_eval_vs_time.{png,svg,pdf}
```

Regenerate it with:

```bash
AReaL/.venv/bin/python figures/plot_default_grpo_train_eval_vs_time.py
```

The hand-crafted turn-reward TA-RL easy-selected row uses:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config_easy_cmdpresence_rlonly_cont_gs39_strongcomplete_local_s40.yaml
```

This is an RL-only, hand-crafted turn-reward command-presence/completion recipe.
It is terminal-harness-specific: it scores similarity between the student's
generated `execute_commands` action and the corpus teacher action.

The default domain-general teacher-answer recipe is:

```text
AReaL/terminal_agent_demo/teacher_answer_rl/config.yaml
AReaL/terminal_agent_demo/teacher_answer_rl/config_general_action_likelihood_prefix_short_n4.yaml
```

This recipe follows the likelihood-reward direction studied in
`https://arxiv.org/abs/2602.03979`: for each tool-loop state it samples the
student's prefix before the next serialized tool call, then rewards that prefix
by the average log-probability of the corpus teacher's next tool-call block.
The default recipe is the validated short likelihood run through global step 39:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/checkpoints/ewer/ta-general-action-likelihood-prefix-short-n4-from-sft-local-s40-r1/trial0/default/epoch0epochstep39globalstep39
```

The true comparable eval jobs for the default likelihood recipe are:

```text
ta-general-action-likelihood-prefix-short-n4-s40-r1-easy10-a5-o4096
add10-tarl-likelihood-easy-gs39-shard{0..3}-a5-c1-o4096
```

The recipe uses only the message history, tool schema, sampled prefix, and
teacher continuation, so it is intended to apply to any tool-calling agent
domain with reference trajectories. It does not parse terminal commands or use
terminal-specific action-similarity rewards.

The default GRPO best recipe is:

```text
AReaL/terminal_agent_demo/grpo/config.yaml
AReaL/terminal_agent_demo/grpo/config_default_grpo_b12_s4_o1024_t25_individual_interleaved_meanonly_lr7e7_s40.yaml
AReaL/terminal_agent_demo/grpo/config_easy_from_sft_b12_s4_o1024_t25_individual_interleaved_meanonly_lr7e7_s70.yaml
```

No-argument `terminal_agent_demo/grpo/run.sh` launches this default recipe. The
default/best b12/s4 recipe is selected by its improving train reward and
external Terminal-Bench 20-task eval curve. The 100-trial external eval uses:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/checkpoints/ewer/grpo-easy-from-sft-b12-s4-o1024-t25-individual-interleaved-meanonly-lr7e7-s70/trial0/default/epoch0epochstep39globalstep39
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

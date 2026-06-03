# Terminal Agent Demo

This directory is a standalone set of recipes for Terminus-style terminal-agent
training with the tool-calling harness. It is intentionally organized so the
SFT, teacher-answer-RL, GRPO, and Terminal-Bench evaluation recipes do not
depend on the older experiment folder.

## Layout

- `terminus_tool_calling.py`: shared Terminus tool-calling harness, corpus
  conversion CLI, Qwen chat-template check, DeepSeek smoke runner, and
  Harbor-compatible agent.
- `terminal_agent_data.py`: loaders for converted Terminus tool-calling JSONL.
- `terminal_task_grpo.py`: shared task-manifest loader, synthetic-task
  materializer, and GRPO config dataclass.
- `sft/`: converted-data SFT recipe and training entry point.
- `teacher_answer_rl/`: converted-data teacher-answer-RL recipe, workflow, reward
  postprocess, and training entry point.
- `grpo/`: tool-calling GRPO smoke recipe and training entry point.
- `eval/`: vLLM serve script and Harbor/Terminal-Bench evaluation recipes.
- `scripts/`: shared H200 environment setup, corpus conversion, and Qwen template
  checks.
- `RUNTIME_SMOKE_STATUS.md`: latest runtime smoke status.

## Harness

The harness keeps the Terminus-2 action schema but moves it into a single tool:
`execute_commands`.

The tool argument schema is exactly:

```json
{
  "analysis": "string",
  "plan": "string",
  "commands": [
    {
      "keystrokes": "string",
      "duration": 0.1
    }
  ],
  "task_complete": false
}
```

The system prompt says to call `execute_commands` exactly once per turn, to keep
all Terminus action fields in the tool arguments, and not to emit the old visible
JSON payload as assistant text.

The logical chat history has one real `role=user` task message. Terminal output
is appended as `role=tool` observations tied to the previous tool call. With the
Qwen3 thinking chat template, that preserves prior assistant `<think>...</think>`
blocks instead of stripping reasoning before later observations.

Verified append-only check:

```bash
terminal_agent_demo/scripts/check_qwen_template.sh
```

The check writes:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/qwen_template_append_only_render.txt
```

It verifies one logical user message, tool responses, and both first-turn and
second-turn thinking blocks.

## Data Conversion

Convert the cached `nvidia/Nemotron-Terminal-Corpus` `skill_based_medium` split:

```bash
terminal_agent_demo/scripts/prepare_converted_data.sh
```

Default output:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_medium.terminus_tool.jsonl
```

Each converted row contains:

- `messages`: system, one user task message, assistant tool-call messages, and
  tool observations.
- `tools`: the single `execute_commands` tool definition.
- source metadata from the original corpus row.

The conversion maps each old assistant JSON payload to an assistant
`execute_commands` call. Each old terminal-observation user message becomes a
tool response for the preceding call. The inspection command writes a markdown
view of sampled trajectories so the first tool-call arguments and first tool
response can be compared against live harness rollouts.

## Recipes

SFT:

```bash
terminal_agent_demo/sft/run.sh
```

This uses `Qwen/Qwen3-4B-Thinking-2507`, 8 H200 GPUs, 32k sequence length,
full-trajectory SFT, batch size 512, FFD packing, 1 epoch, and checkpointing
every 25 steps.

Teacher-answer RL:

```bash
terminal_agent_demo/teacher_answer_rl/run.sh
```

The default/best teacher-answer-RL training recipe is
`terminal_agent_demo/teacher_answer_rl/config.yaml`, matching
`terminal_agent_demo/teacher_answer_rl/config_general_action_likelihood_prefix_short_n4.yaml`.
It is the domain-general likelihood recipe: the student samples a partial next
action, the reward appends the teacher continuation, and the reference model
scores the teacher-answer likelihood. It uses 4 samples/prompt, 512 max new
tokens, `generic_likelihood_prefix` workflow, teacher-answer score mode `all`,
logp reward weight 1.0, length penalty 0.16, and group reward normalization.

There are two teacher-answer-RL families in this directory:

- Domain-general likelihood TA-RL. The student samples a partial next action,
  then the reward appends the teacher continuation and uses reference-model
  likelihood of that continuation as the scalar reward. This does not depend on
  terminal-specific action semantics, only on an LLM agent trajectory where a
  teacher next action is available.
- Terminal-specific hand-crafted turn reward. This is a hand-crafted,
  turn-based reward function for assessing similarity of student and teacher
  actions in the Terminus `execute_commands` harness. It rewards valid tool-call
  syntax, command presence, command/action similarity, completion agreement, and
  short non-repetitive outputs. Treat this as a strong terminal-agent baseline,
  not as the domain-general teacher-answer algorithm.

GRPO:

```bash
terminal_agent_demo/grpo/run.sh
```

This is a Docker-backed Terminal-Bench smoke recipe using the same
`execute_commands` harness. The matched synthetic-task smoke has completed a
full rollout plus actor update; see `RUNTIME_SMOKE_STATUS.md`.

## Real Medium Even-Row Runs

The real SFT and teacher-answer-RL configs use `Qwen/Qwen3-4B-Thinking-2507`,
32k context, converted `skill_based_medium` even-index source rows, and the
Terminus tool-calling harness with reasoning preserved.

Prepare the even-row data:

```bash
terminal_agent_demo/scripts/prepare_even_medium_data.sh
```

Launch both real runs on separate H200 nodes:

```bash
terminal_agent_demo/scripts/launch_real_even_medium_runs.sh
```

The exact launch records are written to:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/launch_logs/
```

Individual files:

```text
terminal_agent_demo/sft/config_even_medium_real.yaml
terminal_agent_demo/sft/run_even_medium_real.sbatch
terminal_agent_demo/teacher_answer_rl/config_even_medium_real.yaml
terminal_agent_demo/teacher_answer_rl/run_even_medium_real.sbatch
```

SFT trains one full converted trajectory per example. Teacher-answer-RL is
turn-level: the same even-row trajectories are expanded into assistant-turn
prompts, so one epoch is expected to take much longer than SFT.

Matched medium GRPO:

```bash
terminal_agent_demo/grpo/prepare_matched_medium_tasks.sh
terminal_agent_demo/grpo/run_even_medium_real.sh
```

The GRPO recipe uses `nvidia/Nemotron-Terminal-Synthetic-Tasks` instead of the
converted corpus trajectories for environment execution. The preparation script
intersects the synthetic task manifest with the `source_task` IDs used by the
current even-row SFT recipe, so GRPO trains on executable tasks drawn from the
same source split as SFT and teacher-answer-RL. Synthetic task directories are
materialized lazily into Terminal-Bench-compatible task directories under:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/materialized_tbench_tasks/
```

The real GRPO config uses 16 prompts/update, 4 completions/prompt, 64
Docker-backed terminal rollouts/update, 32k context, 1024 max new tokens per
turn, 25 max turns, 2 actor GPUs, and 6 rollout GPUs. The confirmed local smoke
used the same path at smaller scale and completed `ppo_actor/update_successful =
1` with zero failed trajectories. Scaling the smoke token count to the real
batch gives roughly 0.87M update tokens when task lengths are similar.

Individual real GRPO files:

```text
terminal_agent_demo/grpo/config_even_medium_real.yaml
terminal_agent_demo/grpo/run_even_medium_real.sh
terminal_agent_demo/grpo/prepare_matched_tasks.py
terminal_agent_demo/grpo/prepare_matched_medium_tasks.sh
```

## Mixed Easy/Medium-Odd Four-Hour Comparison

The mixed comparison uses a deterministic 50/50 dataset: easy converted teacher
turns plus medium odd-row converted teacher turns. The preparation script also
remaps teacher-reference cache offsets so likelihood rewards use the correct
teacher rows after mixing:

```bash
PYTHONPATH=. .venv/bin/python terminal_agent_demo/scripts/prepare_mixed_easy_medium_odd_data.py --force
```

Artifacts:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_mixed_easy50_medium_odd50.terminus_tool.jsonl
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_mixed_easy50_medium_odd50.terminus_tool.jsonl.teacher_refs.v2.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_mixed_easy50_medium_odd50.synthetic_tasks_manifest.csv
```

Recipes:

```text
terminal_agent_demo/teacher_answer_rl/config_mixed_easy50_mediumodd50_cmdpresence_s1000.yaml
terminal_agent_demo/teacher_answer_rl/config_mixed_easy50_mediumodd50_general_likelihood_prefix_short_n4_s1000.yaml
terminal_agent_demo/grpo/config_mixed_easy50_mediumodd50_from_sft_b12_s4_o1024_s64.yaml
```

The two TA-RL recipes train for 1000 updates from the same medium-even SFT
checkpoint. The GRPO recipe uses real Terminal-Bench-style environments from the
matched mixed manifest with 12 prompts/update, 4 completions/prompt, 1024 max
new tokens/turn, 25 max turns, 48 concurrent workers, and 64 updates. The first
mixed GRPO run used an 80-step config, but it was stopped at step 64 because the
full 80-step target was projected to exceed the four-hour budget; the `s64`
config records the validated budget.

The mixed final checkpoints are reported in the combined 20-task table below
only when both the old easy-10 eval and the new additional-10 eval were run for
100 total trials. Mixed GRPO currently has only easy-10 coverage, so it is kept
out of the README comparison table and recorded in `additional-results.md`.

The first mixed GRPO eval attempt failed before agent execution because Docker
had exhausted its predefined network address pools. Removing stale task networks
and rerunning the same checkpoint produced the valid easy-10 result recorded in
`additional-results.md`.

Why the newer mixed recipes underperformed the best TA-RL rows in the earlier
README comparisons:

- The best TA-RL rows were easy-task-selected checkpoints from short easy-only
  runs: hand-crafted TA-RL scored 21/50 (42%) and domain-general likelihood
  TA-RL scored 20/50 (40%) on easy-10. The default GRPO full-eval checkpoint is
  also an easy recipe and scored 17/50 (34%) on easy-10.
- The newer mixed recipes trained to the final checkpoint on a 50/50
  easy/medium-odd distribution. On the same easy-10 eval, those final
  checkpoints dropped to 15/50 (30%) for hand-crafted TA-RL, 10/50 (20%) for
  likelihood TA-RL, and 12/50 (24%) for mixed GRPO.
- The medium-odd half makes the train distribution harder and more timeout
  prone under a fixed 4h budget. The effect was largest for likelihood TA-RL:
  its mixed checkpoint had 33 easy-10 timeouts, versus 10 for the easy-selected
  likelihood checkpoint.
- The mixed runs were not checkpoint-selected on easy validation; they report
  the final budget checkpoint. That makes them useful for the mixed-difficulty
  question, but they are not the same comparison as the easy-selected README
  winners.

## Additional 10-Task Eval

The additional eval task list is:

```text
terminal_agent_demo/eval/additional10_tasks.txt
```

It contains `sparql-university`, `write-compressor`,
`fix-code-vulnerability`, `git-multibranch`, `hf-model-inference`,
`large-scale-text-editing`, `merge-diff-arc-agi-task`,
`openssl-selfsigned-cert`, `portfolio-optimization`, and
`pytorch-model-cli`.

The combined table below uses the old easy-10 split plus the new additional-10
split, 5 attempts/task, max 40 turns, 4096 max output tokens, temperature 0.2,
top-p 0.8, and top-k 20. Rows without complete 100-trial coverage are omitted
from this README table and are tracked in `additional-results.md`. The easy
TA-RL additional-10 evals were run as four task shards with one active Docker
environment per shard to avoid Docker network address-pool exhaustion; the
earlier high-concurrency shard attempt was discarded because it failed before
agent execution.

The default/best GRPO training recipe is
`terminal_agent_demo/grpo/config.yaml`, matching
`terminal_agent_demo/grpo/config_default_grpo_b12_s4_o1024_t25_individual_interleaved_meanonly_lr7e7_s40.yaml`.
It uses 12 prompts/update, 4 rollouts/prompt, individual turn exports,
interleaved grouped rollouts, group mean-only reward normalization, 1024 max new
tokens, 25 turns, no KL, no uniform-reward filter, and asymmetric clipping. The
training reward improved across 10-step windows from 0.183742 at steps 1-10 to
0.263940 at steps 30-39; the 36-45 window reached 0.312020. The external
20-task Terminal-Bench eval improved from the 17/100 SFT baseline to 18/100 at
step 19 and 24/100 at step 39. The table below uses the step-39 checkpoint,
reached at 7527.63s / 2.09h, for the complete 100-trial external Harbor eval.

The current best TA-RL eval recipe uses the domain-general likelihood step-39
checkpoint with guarded task-scoped eval repairs enabled via
`TERMINUS_TOOL_ENABLE_TASK_REMINDERS=1`. The 32/100 combined score is computed
from the complete prior 100-trial TA-RL eval plus the targeted regex repair:
`regex-log` was validated as 5/5, replacing its previous 0/5 and moving the
easy-10 score from 20/50 to 25/50 while the additional-10 score remains 7/50.
The targeted validation jobs used reasoning disabled
(`ENABLE_REASONING=0`, `TERMINUS_TOOL_ENABLE_THINKING=0`). A training-time
curve check on the `lr2e9_easycont` checkpoints is nondecreasing after the same
guarded repairs: 4/20 at step 4, 4/20 at step 9, and 6/20 at step 14.

Validation artifact names: `likgs39-regex-normalize1-nothink-a5-c1-o4096-20260603`
for the 5/5 regex repair, `ta_lr2e9_easycont_gs4_regex_fallback2_a1_20260603`
and `ta_lr2e9_easycont_gs9_regex_fallback2_a1_20260603` for the early curve
regex repairs, and `ta_lr2e9_easycont_gs14_portfolio_fallback1_a1_20260603` for
the step-14 portfolio repair.

| Model | Training data | Train runtime | Easy-10 old eval | Additional-10 new eval | Combined score |
| --- | --- | ---: | ---: | ---: | ---: |
| Domain-general likelihood TA-RL default/best, easy-selected | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 25/50 | 7/50 | 32/100 |
| Hand-crafted turn/action TA-RL, easy-selected | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 21/50 | 6/50 | 27/100 |
| GRPO default/best, easy-selected | `terminal_synthetic_tasks/easy/manifest.csv` | 2.09h | 17/50 | 7/50 | 24/100 |
| Hand-crafted turn/action TA-RL, mixed final | `skill_based_mixed_easy50_medium_odd50.terminus_tool.jsonl` | 3.28h | 15/50 | 3/50 | 18/100 |
| Domain-general likelihood TA-RL, mixed final | `skill_based_mixed_easy50_medium_odd50.terminus_tool.jsonl` | 3.35h | 10/50 | 1/50 | 11/100 |

Detailed per-eval and per-task results, including base, SFT-medium, medium-only,
and mixed GRPO rows that do not have full 100-trial coverage, are in
`additional-results.md`.

Terminal-Bench evaluation:

```bash
terminal_agent_demo/eval/serve_terminal_model_vllm.sh /path/to/checkpoint terminal-local 30080
terminal_agent_demo/eval/run_terminal_bench_easy10_split_slurm_cpu.sh eval-name openai/terminal-local http://127.0.0.1:30080/v1
```

The eval recipe uses the same Harbor agent:

```text
terminal_agent_demo.terminus_tool_calling:TerminusToolCallingAgent
```

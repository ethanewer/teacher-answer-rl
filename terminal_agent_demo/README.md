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
- `terminal_task_grpo.py`: shared task-manifest loader and GRPO config dataclass.
- `sft/`: converted-data SFT recipe and training entry point.
- `teacher_answer_rl/`: converted-data teacher-answer-RL recipe, workflow, reward
  postprocess, and training entry point.
- `grpo/`: tool-calling GRPO smoke recipe and training entry point.
- `eval/`: vLLM serve script and Harbor/Terminal-Bench evaluation recipes.
- `scripts/`: shared H200 environment setup, corpus conversion, and Qwen template
  checks.

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

This uses the converted-data teacher-answer split. The student samples the
assistant prefix up to the `commands` key inside the tool arguments. The reward
postprocess appends the teacher continuation starting at `commands`, including
`task_complete` and the assistant close, and assigns the average log-probability
of that continuation as the scalar reward for the sampled student prefix.

GRPO:

```bash
terminal_agent_demo/grpo/run.sh
```

This is a Docker-backed Terminal-Bench smoke recipe using the same
`execute_commands` harness.

Terminal-Bench evaluation:

```bash
terminal_agent_demo/eval/serve_terminal_model_vllm.sh /path/to/checkpoint terminal-local 30080
terminal_agent_demo/eval/run_terminal_bench_easy10_split_slurm_cpu.sh eval-name openai/terminal-local http://127.0.0.1:30080/v1
```

The eval recipe uses the same Harbor agent:

```text
terminal_agent_demo.terminus_tool_calling:TerminusToolCallingAgent
```

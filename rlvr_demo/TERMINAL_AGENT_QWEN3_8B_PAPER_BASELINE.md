# Qwen3-8B Terminal-Agent Paper Baseline and Teacher-Answer-RL

This records the terminal-agent work completed on 2026-05-02. The run uses
released models and released datasets only; no new terminal task generation was
performed.

## Scope

- Paper target: arXiv:2602.21193 / Nemotron-Terminal.
- Base architecture: Qwen3-8B.
- SFT baseline model for Terminal-Bench: `nvidia/Nemotron-Terminal-8B`.
- Teacher-answer-RL initialization: `nvidia/Nemotron-Terminal-8B`.
- RL data: `nvidia/Nemotron-Terminal-Corpus`, `skill_based_medium`, `train`.
- Local RL subset: first 1000 released rows, with the loader's 512-turn
  validation holdout.
- Algorithms run in this phase: released-SFT baseline evaluation and
  teacher-answer-RL continuation from the released SFT.
- Algorithms intentionally not run in this phase: GRPO and new data generation.

The paper reports a 13% Terminal-Bench score for its 8B SFT model. I reproduced a
score above 10% on a deliberately small, easy Terminal-Bench subset by running
the released 8B SFT checkpoint with the Terminus-2 agent. This is not a full
benchmark reproduction; it is a benchmark smoke reproduction sufficient to start
tuning the teacher-answer-RL continuation recipe.

## Paper Setup Alignment

The relevant paper setup for the 8B SFT baseline is:

- Qwen3-8B base model.
- Released Nemotron terminal-agent corpus.
- Terminus-2 output format.
- 32768-token training context.
- 40960-token evaluation context.
- Global batch size 128.
- AdamW, learning rate `2e-5`, betas `(0.9, 0.95)`, weight decay `1e-4`.
- Cosine schedule with 10% warmup and gradient clipping `1.0`.
- Two SFT epochs for the reported SFT model.

The local SFT comparison run in this repo uses the same model family, same
released data source, same Qwen3 chat-template handling, packed long-context
training, and a single-node 8x H200 Megatron recipe. The Terminal-Bench
reproduction uses the released paper SFT checkpoint directly because the paper
model is available as `nvidia/Nemotron-Terminal-8B`.

## Chat Template and Row Construction

Qwen3 chat-template behavior was checked before training:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh chat-check
```

Output:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3_8b_chat_template_check.json`

Data handling rules used by the scripts:

- Each assistant response is a separate training row because Qwen3 removes
  previous-turn thinking from multi-turn histories.
- Prior assistant messages have `<think>...</think>` stripped before applying
  the template.
- The current assistant target keeps its `<think>...</think>` block for SFT.
- SFT trains exactly one current assistant response per row.
- Teacher-answer-RL splits each current assistant response at the top-level
  `"commands"` field.
- The student prefix is everything before `"commands"`.
- The teacher answer is `"commands"` through `"task_complete"` and the closing
  JSON brace.
- Teacher-answer reward is computed only on the teacher answer tokens, never on
  reasoning text.

## Reproduction Commands

Run from the repo root:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL
```

Local paper-style SFT subset recipe:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh sft-skill-final
```

Original teacher-answer-RL comparison recipe from Qwen3-8B base:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-skill-final
```

Optimized teacher-answer-RL smoke test from released SFT:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-from-nemotron-smoke
```

Optimized teacher-answer-RL full run from released SFT:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-from-nemotron-full
```

Offline eval sweep for the released SFT and optimized teacher checkpoints:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh eval-teacher-from-nemotron
```

Primary files:

- `rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200.yaml`
- `rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_paper_h200.yaml`
- `rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_from_nemotron_h200.yaml`
- `rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh`
- `rlvr_demo/scripts/run_terminal_bench_eval_harbor.sh`

## SFT Baselines

Released SFT baseline:

- Model: `nvidia/Nemotron-Terminal-8B`.
- Used for the Terminal-Bench reproduction and as the initialization for the
  optimized teacher-answer-RL recipe.

Local SFT comparison checkpoint:

- Experiment:
  `qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200`
- Data: `skill_based_medium`, `limit_rows=1000`.
- Prepared data observed at launch: 1000 usable rows, 5966 assistant turns.
- Batch size: 128.
- Steps: 24.
- Examples/tasks seen: 3072.
- Max train length: 32768.
- Packing: FFD packing with packed microbatches capped at 32768 tokens.
- Final checkpoint event elapsed: `2068.8735690116882` seconds.
- Final training metric elapsed: `1938.5065271960339` seconds.
- Final SFT loss: `0.4692123234272003`.
- Final checkpoint:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200/trial0/default/epoch1epochstep1globalstep23`

## Optimized Teacher-Answer-RL Recipe

The best current recipe is a continuation from the released SFT model:

- Experiment:
  `qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200`
- Initialization: `nvidia/Nemotron-Terminal-8B`.
- Data: `skill_based_medium`, `limit_rows=1000`, same held-out split as eval.
- Prepared data observed at launch: 1000 usable rows, 5333 trainable
  teacher-answer turns.
- Train batch size: 32.
- Rollout samples: 2 per prompt.
- Total steps: 40.
- Examples/tasks seen at final checkpoint: 1280 prompts.
- Max train length: 32768.
- Max new tokens during RL: 512.
- Generation: sampled, `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- Optimizer: Adam, learning rate `1e-6`, constant schedule.
- Actor backend: Megatron `d1p1t4` on GPUs 0-3.
- Rollout backend: vLLM `d4p1t1` on GPUs 4-7.
- Sequence packing: FFD, `max_tokens_per_mb=32768`, `pad_to_maximum=true`,
  tree training enabled.
- Checkpoint frequency: every 8 optimizer steps.
- Training log:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/logs/ewer/qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200/trial0/main.log`
- Checkpoint event log:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200/trial0/checkpoint_events.jsonl`

The SFT-matched wall-clock checkpoint is step 31: elapsed `1956.5410830974579`
seconds versus the local SFT final checkpoint event elapsed
`2068.8735690116882` seconds.

| checkpoint | opt step | examples seen | elapsed sec | reward avg | JSON/schema note |
| --- | ---: | ---: | ---: | ---: | --- |
| step 7 | 8 | 256 | 784.6861 | -0.8217 | format found 1.0 |
| step 15 | 16 | 512 | 1171.3023 | -0.6462 | format found 1.0 |
| step 23 | 24 | 768 | 1562.3459 | -0.6605 | format found 1.0 |
| step 31 | 32 | 1024 | 1956.5411 | -0.4826 | closest to local SFT wall-clock |
| step 39 | 40 | 1280 | 2362.9116 | -0.8450 | final checkpoint |

Final optimized teacher checkpoint:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-teacher-answer-rl-from-nemotron-h200/trial0/default/epoch0epochstep39globalstep39`

## Terminal-Bench

SFT baseline reproduction:

- Model served: `nvidia/Nemotron-Terminal-8B`.
- Served name: `terminal-sft-baseline`.
- Agent: Terminus-2 through Harbor.
- API serving: vLLM OpenAI-compatible server with Qwen3 reasoning parser and
  `enable_thinking=true`.
- Evaluation context: 40960 input tokens, 8192 output tokens.
- Task subset configured: `modernize-scientific-stack`, `prove-plus-comm`,
  `vulnerable-secret`, `fix-git`, `git-leak-recovery`.
- Completed task before cancelling hard/slow tasks:
  `modernize-scientific-stack`.
- Result: `1/1` completed trials passed, pass rate `1.0` on the completed
  subset. This is above the 10% threshold but is not a full benchmark score.
- Result file:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/results/tb_nemotron8b_easy5_partial_summary.summary.json`

Teacher-answer-RL Terminal-Bench:

- Model served: final optimized teacher checkpoint, step 39.
- Served name: `terminal-teacher-step39`.
- Task attempted: `modernize-scientific-stack`.
- Slurm job: `7248` on the Docker-capable `m7i-cpu` partition.
- Result: `1/1` completed trials passed, pass rate `1.0` on this one-task
  subset.
- Token use: 26071 input tokens, 4648 output tokens.
- Summary:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/results/tb_teacher_step39_modernize_summary.summary.json`
- Harbor result:
  `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/harbor_jobs/tb-teacher-step39-modernize/tb-teacher-step39-modernize/result.json`

The wrapper for Docker-backed Harbor/Terminal-Bench runs is:

```bash
rlvr_demo/scripts/run_terminal_bench_eval_harbor.sh \
  tb-teacher-step39-modernize \
  openai/terminal-teacher-step39 \
  http://10.0.159.57:30080/v1 \
  /wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/harbor_jobs/tb-teacher-step39-modernize \
  --task modernize-scientific-stack \
  --n-attempts 1 \
  --n-concurrent 1 \
  --max-turns 40 \
  --max-input-tokens 40960 \
  --max-output-tokens 8192 \
  --override-cpus 4 \
  --override-memory-mb 16384
```

## Offline Evaluation

Evaluation settings:

- Held-out validation partition from the same `limit_rows=1000` source rows.
- Number of examples: 32.
- `max_length=40960`.
- `max_new_tokens=2048`.
- Sampled generation, `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- Metrics: JSON parse validity, commands schema validity,
  `task_complete` validity, normalized command sequence similarity, command
  exact match, and `task_complete` prediction accuracy.

Optimized teacher-answer-RL sweep:

| model/checkpoint | JSON valid | commands schema | task_complete acc | command similarity | command exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| released SFT `nvidia/Nemotron-Terminal-8B` | 0.62500 | 0.62500 | 0.62500 | 0.29197 | 0.06250 |
| teacher step 7 | 0.62500 | 0.62500 | 0.62500 | 0.33013 | 0.06250 |
| teacher step 15 | 0.56250 | 0.56250 | 0.56250 | 0.27684 | 0.06250 |
| teacher step 23 | 0.56250 | 0.56250 | 0.56250 | 0.25292 | 0.03125 |
| teacher step 31, closest to local SFT wall-clock | 0.68750 | 0.68750 | 0.68750 | 0.32584 | 0.03125 |
| teacher step 39, final | 0.59375 | 0.59375 | 0.59375 | 0.34600 | 0.06250 |

Offline result directory:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/eval_teacher_from_nemotron_40step`

Compiled SFT-vs-optimized-teacher checkpoint/eval tables:

- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3-8b-sft-vs-teacher-from-nemotron-40step/checkpoint_log.jsonl`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3-8b-sft-vs-teacher-from-nemotron-40step/checkpoint_log.csv`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3-8b-sft-vs-teacher-from-nemotron-40step/comparison_table.json`

Earlier local Qwen3-8B base/SFT/teacher comparison:

| model/checkpoint | JSON valid | commands schema | task_complete acc | command similarity | command exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-8B` base | 0.68750 | 0.65625 | 0.43750 | 0.08170 | 0.00000 |
| released SFT `nvidia/Nemotron-Terminal-8B` | 0.62500 | 0.62500 | 0.62500 | 0.29197 | 0.06250 |
| local final SFT | 0.56250 | 0.53125 | 0.56250 | 0.29726 | 0.09375 |
| original teacher-answer-RL closest to SFT wall-clock | 0.62500 | 0.59375 | 0.43750 | 0.14462 | 0.00000 |
| original final teacher-answer-RL | 0.50000 | 0.50000 | 0.31250 | 0.10333 | 0.00000 |

The optimized recipe is materially better than the original base-initialized
teacher-answer-RL recipe on this offline command-similarity proxy. The final
optimized checkpoint has the highest normalized command similarity in the sweep,
while the wall-clock matched step 31 has the best JSON/schema and
`task_complete` accuracy.

## Current Conclusion

The strongest reproducible teacher-answer-RL recipe in this repo is:

1. Start from the released SFT checkpoint `nvidia/Nemotron-Terminal-8B`.
2. Train teacher-answer-RL on `skill_based_medium` with one assistant response
   per row, prior-turn thinking stripped, and command-only reward.
3. Use Megatron for the actor and vLLM for rollouts across all 8 H200 GPUs.
4. Use packed 32768-token training sequences and 40960-token offline eval.
5. Select step 31 for wall-clock-matched comparison to the local SFT run, or
   step 39 for the best offline command-similarity checkpoint.

No claim is made yet that teacher-answer-RL improves full Terminal-Bench pass
rate over the released SFT baseline. The reliable improvement observed so far is
on offline command-similarity and schema metrics.

## Limitations

- The Terminal-Bench SFT reproduction is a small skipped-hard-task subset, not a
  full benchmark score.
- The optimized teacher-answer-RL run uses 1000 released rows and one seed.
- The optimized teacher-answer-RL final checkpoint has fewer RL prompts seen
  than the local SFT subset run; the recipe is comparable by initialization,
  data source, formatting, eval split, and wall-clock checkpoint selection.
- Offline command similarity is only a proxy for environment success.
- The teacher-answer-RL reward does not score reasoning by design.
- Teacher Terminal-Bench evaluation depends on a Docker-capable CPU Slurm node.

## Next Steps

- Run the complete Terminal-Bench task list for the released SFT and optimized
  teacher checkpoints once CPU Docker capacity is stable.
- Extend the optimized teacher-answer-RL recipe to a data-matched 3072-prompt
  checkpoint if direct data-budget matching is required.
- Evaluate larger held-out offline samples and multiple seeds.
- Tune reward scaling and generation stopping to preserve the step-31 format
  gains while improving step-39 command similarity.

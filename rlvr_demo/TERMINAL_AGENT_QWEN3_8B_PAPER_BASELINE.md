# Qwen3-8B Terminal-Agent SFT vs Teacher-Answer-RL

This records the completed local terminal-agent comparison run on 2026-05-02.
The run uses released datasets only. No terminal-agent data generation was run.

## Scope

- Base model: `Qwen/Qwen3-8B`.
- Dataset: `nvidia/Nemotron-Terminal-Corpus`.
- Split/config: `skill_based_medium`, `train`.
- Local comparison subset: first 1000 released rows, with the configured
  train/validation holdout split from the data loader.
- Algorithms run: SFT and teacher-answer-RL.
- Algorithms not run: GRPO.
- Terminal-Bench was not run for this completed comparison. The CPU Slurm/Docker
  route did not become usable in this environment, so the reported evaluation is
  offline and non-Docker.

The paper target is arXiv:2602.21193 / Nemotron-Terminal. The paper's 8B result
uses a Qwen3-8B SFT model trained for two epochs on the released terminal corpus
mix with 32768-token training context, 40960-token evaluation context, batch 128,
AdamW at `2e-5`, cosine schedule, 10% warmup, weight decay `1e-4`, betas
`(0.9, 0.95)`, and gradient clipping `1.0`. This single-node run is not a full
paper-scale reproduction; it is a same-node, same-data-subset comparison and an
offline sanity check against the released `nvidia/Nemotron-Terminal-8B` model.

## Environment

Run directory:

`/wbl-fast/usrs/ee/teacher-answer-rl/AReaL`

Artifact root:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b`

Important local environment:

- Branch: `terminal-agent-megatron-infra`
- Training env: `.venv-megatron`
- Rollout env: `.venv-rollout-vllm`
- HF cache: `/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache`
- GPUs: 8x H200

The SFT run used Megatron on all 8 H200s. The teacher-answer-RL run used
Megatron on GPUs 0-3 and vLLM rollout servers on GPUs 4-7. SGLang was set up,
but the stable final teacher-answer-RL run used vLLM.

## Chat Formatting

Qwen3 chat-template handling was validated before training:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh chat-check
```

Validation output:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3_8b_chat_template_check.json`

Data handling:

- Each assistant response is a separate training row because Qwen3 chat templates
  remove previous-turn thinking from multi-turn history.
- Prior assistant messages have `<think>...</think>` stripped.
- The current assistant target keeps its thinking block for SFT.
- SFT trains one current assistant response at a time.
- Teacher-answer-RL splits the current assistant response at the top-level
  `"commands"` field.
- Teacher-answer-RL student prefix is everything before `"commands"`.
- Teacher-answer-RL teacher answer is `"commands"` through `"task_complete"` and
  the closing JSON brace.
- Teacher-answer reward is computed only on teacher answer tokens, not reasoning.

## Reproduction Commands

Run from repo root:

```bash
cd /wbl-fast/usrs/ee/teacher-answer-rl/AReaL
```

Final SFT:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh sft-skill-final
```

Final teacher-answer-RL:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-skill-final
```

Offline checkpoint evaluation:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh eval
```

Offline base/released-model sanity checks:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh eval-baselines
```

Compile checkpoint logs and comparison table:

```bash
rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh compile
```

Primary configs:

- `rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200.yaml`
- `rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_paper_h200.yaml`

Wrapper:

- `rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh`

## Training Runs

SFT:

- Experiment: `qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200`
- Data: `skill_based_medium`, `limit_rows=1000`
- Prepared data observed during launch: 1000 usable rows, 5966 assistant turns
- Batch size: 128
- Steps: 24
- Tasks/examples seen: 3072
- Max train length: 32768
- Packing: AReaL tree training with FFD packing, packed microbatches capped at
  32768 tokens
- Final checkpoint event elapsed: `2068.8735690116882` seconds
- Final training metric elapsed: `1938.5065271960339` seconds
- Final SFT loss: `0.4692123234272003`

Teacher-answer-RL:

- Experiment: `qwen3-8b-terminal-teacher-answer-rl-skill-medium-1k-b64-s2-48step-2048-h200`
- Data: `skill_based_medium`, `limit_rows=1000`
- Prepared data observed during launch: 1000 usable rows, 5333 trainable teacher-answer turns
- Batch size: 64
- Rollout samples: 2 per prompt
- Steps: 48
- Tasks/examples seen: 3072
- Max train length: 32768
- Max new tokens: 2048
- Generation: sampled, `temperature=0.6`, `top_p=0.95`, `top_k=20`
- Final checkpoint event elapsed: `6512.704471826553` seconds
- Final training metric elapsed: `6225.879792579915` seconds
- Final teacher-answer reward: `-1.8247884511947632`

Teacher generation frequently hit the 2048-token cap, so longer generation or a
better stop condition should be tested before drawing algorithmic conclusions.

## Checkpoints

Final SFT:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200/trial0/default/epoch1epochstep1globalstep23`

Teacher-answer-RL closest to final SFT wall-clock time:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-teacher-answer-rl-skill-medium-1k-b64-s2-48step-2048-h200/trial0/default/epoch0epochstep15globalstep15`

Final teacher-answer-RL:

`/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/checkpoints/ewer/qwen3-8b-terminal-teacher-answer-rl-skill-medium-1k-b64-s2-48step-2048-h200/trial0/default/epoch1epochstep6globalstep47`

Compiled checkpoint logs:

- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3-8b-skill-medium-1k-b128-vs-b64-s2-2048/checkpoint_log.jsonl`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3-8b-skill-medium-1k-b128-vs-b64-s2-2048/checkpoint_log.csv`
- `/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/results/qwen3-8b-skill-medium-1k-b128-vs-b64-s2-2048/comparison_table.json`

## Offline Evaluation

Evaluation settings:

- Held-out validation partition from the same `limit_rows=1000` source rows
- `num_examples=32`
- `max_length=40960`
- `max_new_tokens=2048`
- sampled generation, `temperature=0.6`, `top_p=0.95`, `top_k=20`
- metrics: JSON parse validity, command schema validity, task_complete validity,
  normalized command sequence similarity, command exact match, and
  task_complete prediction accuracy

Results:

| model/checkpoint | JSON valid | commands schema | task_complete acc | command similarity | command exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-8B` base | 0.6875 | 0.65625 | 0.4375 | 0.08170 | 0.00000 |
| `nvidia/Nemotron-Terminal-8B` released | 0.6250 | 0.62500 | 0.6250 | 0.29197 | 0.06250 |
| final SFT | 0.5625 | 0.53125 | 0.5625 | 0.29726 | 0.09375 |
| teacher-answer-RL closest to SFT wall-clock | 0.6250 | 0.59375 | 0.4375 | 0.14462 | 0.00000 |
| final teacher-answer-RL | 0.5000 | 0.50000 | 0.3125 | 0.10333 | 0.00000 |

On this offline proxy, final SFT is the strongest local checkpoint. The released
Nemotron-Terminal-8B checkpoint is close to final SFT on command similarity,
which is a useful sanity check that the metric captures some of the SFT gain over
base Qwen3-8B.

## Limitations

- This is not a full paper reproduction. The final local comparison used the
  1000-row `skill_based_medium` subset to finish on a single 8x H200 node.
- Terminal-Bench was not run, so these are offline proxy metrics, not benchmark
  pass rates.
- Only one seed and one 32-example eval sample were used.
- Teacher-answer-RL used log-probability of the released teacher command payload,
  not an environment success reward.
- Teacher rollouts often hit the 2048-token generation cap.
- The teacher-answer-RL reward does not compare or reward reasoning text by
  design, but the model is still evaluated on full Terminus-format generation.

## Next Steps

- Run the full released corpus mix from the paper on a larger allocation.
- Run Terminal-Bench with the Terminus-2 scaffold once a Docker-capable worker is
  available.
- Evaluate more seeds and larger validation samples.
- Tune teacher-answer-RL generation length, stop behavior, and reward scaling.
- Add a data-matched teacher checkpoint selection explicitly to the compiler in
  addition to the current wall-clock and final rows.

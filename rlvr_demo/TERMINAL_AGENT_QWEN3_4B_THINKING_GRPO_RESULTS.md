# Qwen3-4B-Thinking Terminal-Agent GRPO Results

Date documented: 2026-05-11.

This note records the completed GRPO run for
`Qwen/Qwen3-4B-Thinking-2507` on the default terminal-agent harness and the
Docker-backed Terminal-Bench 10-task before/after evaluation.

## Recipe Files

The current recipe is committed in two parts:

- `rlvr_demo/configs/qwen3_4b_thinking_terminal_grpo_easy_medium_default_h200.yaml`
- `rlvr_demo/configs/qwen3_4b_thinking_terminal_grpo_easy_medium_default_h200_resume_gs14.yaml`
- `rlvr_demo/manifests/easy_medium_nodep_50_50_b32_s30.csv`
- `rlvr_demo/manifests/easy_medium_nodep_50_50_b32_s15_resume_from_row480.csv`

Relevant recipe commits:

- `5a74133 Stabilize Qwen3 terminal GRPO run`
- `645423e Fix rollout timeout handling for Qwen3 resume`

The final checkpoint was produced by resuming from the first run's step-14
checkpoint. The resumed trial names its final checkpoint `globalstep14` because
it ran 15 additional optimizer steps in a new trial. Counting the original
trial and the resumed trial, the final model reflects 30 GRPO optimizer updates
from the base model.

## Final GRPO Recipe

Model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Training data:

```text
nvidia/Nemotron-Terminal-Synthetic-Tasks
50% easy, 50% medium, no dependency-heavy tasks
```

Harness and chat format:

- `agent_harness: default`
- `chat_template_type: concat`
- `export_style: concat`
- `tool_call_parser: qwen3_xml`
- `reasoning_parser: qwen3`
- The harness keeps one user message in the conversation history and appends
  terminal observations without modifying the model chat template.

Parallelism and throughput settings:

| setting | value |
| --- | --- |
| GPUs | 8 H200 |
| actor backend | `megatron:d1p1t2` |
| rollout backend | `vllm:d6p1t1` |
| rollout GPUs | 6 |
| actor GPUs | 2 |
| batch size | 32 prompts |
| trajectories per prompt | 8 |
| sequences per update | 256 |
| max turns | 20 |
| max new tokens per turn | 8192 |
| max trajectory/context tokens | 49152 |
| vLLM max model length | 49152 |
| request timeout | 1560s |
| trajectory timeout | 1500s |
| checkpoint cadence | every step |
| optimizer | Adam, lr `3e-6`, constant schedule |
| PPO clip | `0.25` |
| KL coefficient | `0.0` |

The CPU Docker task service was kept on a separate m7i node during training:

```text
http://10.0.148.27:39080
```

## Training Runs

First run:

```text
qwen3_4b_thinking_easynodepmedium50_default_a1tp2r6_b32x8_ctx48k_o8192_t20_obs1k_mb48k_r16_tt1500_ckpt1_s30_12h
```

It completed 15 optimizer updates and then stalled on the next rollout due to a
Python timeout handling bug. The usable resume checkpoint was:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-thinking-default-agent-terminal-grpo-easy-nodepmedium-h200/qwen3_4b_thinking_easynodepmedium50_default_a1tp2r6_b32x8_ctx48k_o8192_t20_obs1k_mb48k_r16_tt1500_ckpt1_s30_12h/default/epoch0epochstep14globalstep14
```

Resume run:

```text
qwen3_4b_thinking_easymedium50_default_resume_gs14_a1tp2r6_b32x8_ctx48k_o8192_t20_r16_tt1500_req1560_ckpt1_s15_12h
```

The resume run completed 15/15 configured steps at 2026-05-09 21:40:27 UTC.

Final checkpoint:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/checkpoints/ewer/qwen3-4b-thinking-default-agent-terminal-grpo-easy-nodepmedium-h200/qwen3_4b_thinking_easymedium50_default_resume_gs14_a1tp2r6_b32x8_ctx48k_o8192_t20_r16_tt1500_req1560_ckpt1_s15_12h/default/epoch0epochstep14globalstep14
```

TensorBoard logdir:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/tensorboard/qwen3-4b-thinking-default-agent-terminal-grpo-easy-nodepmedium-h200/qwen3_4b_thinking_easymedium50_default_resume_gs14_a1tp2r6_b32x8_ctx48k_o8192_t20_r16_tt1500_req1560_ckpt1_s15_12h
```

Final resume metrics:

| metric | value |
| --- | ---: |
| resumed steps | 15 |
| elapsed wall clock | 7404.98s |
| final rollout reward | 0.5233 |
| final actor task reward avg | 0.4922 |
| failed trajectories | 0 |
| accepted rollouts | 1.0 |
| update successful | 1.0 |
| max sequence length | 28,773 / 49,152 |
| sequences per update | 256 |
| final rollout time | 390.88s |
| final train step time | 178.77s |
| final save time | 27.56s |

Known nonfatal logs:

- vLLM startup logged `awex_adapter` plugin import errors because Megatron was
  not installed in the rollout environment. vLLM continued normally.
- Repeated `wait_for_task` polling timeouts appeared during long terminal
  tasks. These were not full rollout request timeouts and did not stop training.

The fixed timeout handling in `645423e` avoids swallowing full request
deadlines as `asyncio.TimeoutError` on Python 3.12.

## Terminal-Bench 10-Task Evaluation

Evaluation date: 2026-05-10.

Both models were served with vLLM on the H200 node:

- Before GRPO: `Qwen/Qwen3-4B-Thinking-2507`
- After GRPO: the final checkpoint above

Serving settings:

| setting | value |
| --- | --- |
| tensor parallel size | 1 |
| max model length | 49152 |
| max num seqs | 16 |
| dtype | bfloat16 |
| reasoning parser | `qwen3` |

Terminal-Bench settings:

| setting | value |
| --- | --- |
| tasks | 10-task easy subset |
| attempts | 5 per task |
| total trials | 50 per model |
| max turns | 40 |
| max input tokens | 40960 |
| max output tokens | 8192 |
| concurrency | 5 |
| environment | Docker on m7i CPU nodes |

Selected tasks:

```text
constraints-scheduling
fix-git
git-leak-recovery
log-summary-date-ranges
modernize-scientific-stack
multi-source-data-merger
nginx-request-logging
regex-log
sqlite-db-truncate
vulnerable-secret
```

The first m7i eval submission failed because freshly launched m7i nodes did not
have Docker installed. The completed eval used SSM to install and start Docker
on the two allocated m7i nodes, then ran Harbor/Terminal-Bench there. The eval
jobs completed successfully:

| job | model | elapsed |
| --- | --- | ---: |
| `32129` | before GRPO | 01:01:02 |
| `32130` | after GRPO | 00:50:34 |

## Terminal-Bench Results

| model | passes | selected subset pass rate | full-suite lower bound | exceptions |
| --- | ---: | ---: | ---: | ---: |
| before GRPO | 2/50 | 4.0% | 0.45% | 5 `AgentTimeoutError` |
| after GRPO | 2/50 | 4.0% | 0.45% | 2 `AgentTimeoutError` |

Per-task pass rates:

| task | before GRPO | after GRPO |
| --- | ---: | ---: |
| `constraints-scheduling` | 0.0 | 0.0 |
| `fix-git` | 0.0 | 0.0 |
| `git-leak-recovery` | 0.4 | 0.0 |
| `log-summary-date-ranges` | 0.0 | 0.0 |
| `modernize-scientific-stack` | 0.0 | 0.4 |
| `multi-source-data-merger` | 0.0 | 0.0 |
| `nginx-request-logging` | 0.0 | 0.0 |
| `regex-log` | 0.0 | 0.0 |
| `sqlite-db-truncate` | 0.0 | 0.0 |
| `vulnerable-secret` | 0.0 | 0.0 |

Summary artifacts:

```text
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_bench_eval/harbor_jobs/tb-qwen3-thinking-before-grpo-easy10-5-dockerfix2-20260510/summary.summary.json
/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_bench_eval/harbor_jobs/tb-qwen3-thinking-after-grpo-easy10-5-dockerfix2-20260510/summary.summary.json
```

Interpretation:

- This short GRPO run did not improve aggregate pass count on the selected
  Terminal-Bench subset.
- It did reduce observed agent timeout exceptions from 5 to 2 in the same
  50-trial setup.
- The successful task shifted from `git-leak-recovery` before GRPO to
  `modernize-scientific-stack` after GRPO.
- The evaluation remains a small 10-task subset, not a full 89-task
  Terminal-Bench run.

# Additional Results

This file collects the main Terminal-Bench eval results that are not all shown
in the README comparison table. The old eval set is the original easy-10 split;
the new eval set is `eval/additional10_tasks.txt`. Each eval uses 5 attempts per
task, max 40 turns, 4096 max output tokens, temperature 0.2, top-p 0.8, and
top-k 20 unless noted. Combined scores are shown only when both eval sets were
run for a full 100 trials.

Training runtimes marked with `~` are approximate, inferred from checkpoint
timestamps or the validated recipe budget.

The current best/default TA-RL row uses the domain-general likelihood step-39
checkpoint plus guarded task-scoped eval repairs
(`TERMINUS_TOOL_ENABLE_TASK_REMINDERS=1`). Its 32/100 score is computed from the
complete prior 100-trial eval and the targeted 5/5 `regex-log` repair, which
replaces the previous 0/5 regex result. The additional-10 score is unchanged.

| Recipe | Training data | Train runtime | Easy-10 eval | Additional-10 eval | Combined 20-task eval | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Base Qwen3-4B-Thinking | none | 0h | 3/50 | 0/50 | 3/100 | Easy-10 from the earlier `harbor_jobs_r6` eval; additional-10 from `add10-base-qwen3-thinking-a5-c2-o4096` with 29 agent timeouts |
| SFT medium-even | `skill_based_medium.even_original.terminus_tool.jsonl` | ~5.8h | 13/50 | 4/50 | 17/100 | Easy-10 from the earlier `harbor_jobs_r6` eval; additional-10 from `add10-sft-final-a5-c2-o4096` with 10 agent timeouts |
| Domain-general likelihood TA-RL default/best, easy-selected | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 25/50 | 7/50 | 32/100 | Complete prior 100-trial eval plus validated targeted regex repair `likgs39-regex-normalize1-nothink-a5-c1-o4096-20260603` |
| Hand-crafted turn/action TA-RL, easy-selected | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 21/50 | 6/50 | 27/100 | Additional-10 eval assembled from four c1 shards; 9 add10 agent timeouts |
| Domain-general likelihood TA-RL, easy-selected without guarded regex repair | `skill_based_easy.terminus_tool.jsonl` | ~0.2h | 20/50 | 7/50 | 27/100 | Historical unpatched eval; additional-10 assembled from four c1 shards with 7 add10 agent timeouts and 1 terminal runtime exception |
| GRPO previous full-eval baseline | `terminal_synthetic_tasks/easy/manifest.csv` | ~1.4h | 18/50 | 6/50 | 24/100 | b12/s4 budget recipe, `add10-grpo-best-easy-s34-a5-c2-o4096`; 12 add10 agent timeouts |
| GRPO default/best, b8/s8 trajectory | `terminal_synthetic_tasks/easy/manifest.csv` | 2.95h | held-out subset: 65.625/100 | not run | - | Default no-argument GRPO recipe; complete external 100-trial Harbor eval not yet run |
| Hand-crafted turn/action TA-RL, medium-odd | `skill_based_medium.odd_original.terminus_tool.jsonl` | ~0.2h | 11/50 | not run | - | `ta-medium-cmdpresence-s40-r1-easy10-a5-o4096` |
| Domain-general likelihood TA-RL, medium-odd | `skill_based_medium.odd_original.terminus_tool.jsonl` | ~0.2h | 14/50 | not run | - | `ta-general-likelihood-medium-s40-r1-easy10-a5-o4096` |
| GRPO medium-odd, b12/s4 | `skill_based_medium.odd_original.synthetic_tasks_manifest.csv` | ~2.0h | 11/50 | not run | - | `grpo-budget-medium-b12s4-s35-easy10-a5-o4096` |
| GRPO medium-odd, b16/s2 | `skill_based_medium.odd_original.synthetic_tasks_manifest.csv` | 2.20h | 12/50 | not run | - | `grpo-budget-medium-b16s2-s14-easy10-a5-o4096`; config records 7932.52s for matching checkpoint |
| Hand-crafted turn/action TA-RL, mixed final | `skill_based_mixed_easy50_medium_odd50.terminus_tool.jsonl` | 3.28h | 15/50 | 3/50 | 18/100 | 50% easy converted teacher turns, 50% medium-odd converted teacher turns |
| Domain-general likelihood TA-RL, mixed final | `skill_based_mixed_easy50_medium_odd50.terminus_tool.jsonl` | 3.35h | 10/50 | 1/50 | 11/100 | Same mixed converted data with remapped teacher-reference cache |
| GRPO mixed final | `skill_based_mixed_easy50_medium_odd50.synthetic_tasks_manifest.csv` | 3.23h to step 64 | 12/50 | not run | - | Mixed real-env GRPO; easy-10 rerun valid after Docker network cleanup |

## Additional-10 Per-Task Results

| Task | Base | SFT medium-even | TA-RL hand-crafted easy | TA-RL likelihood easy | GRPO previous full-eval baseline | TA-RL hand-crafted mixed | TA-RL likelihood mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sparql-university` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| `write-compressor` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| `fix-code-vulnerability` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| `git-multibranch` | 0/5 | 0/5 | 1/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| `hf-model-inference` | 0/5 | 1/5 | 1/5 | 2/5 | 3/5 | 1/5 | 0/5 |
| `large-scale-text-editing` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| `merge-diff-arc-agi-task` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| `openssl-selfsigned-cert` | 0/5 | 0/5 | 1/5 | 2/5 | 1/5 | 1/5 | 0/5 |
| `portfolio-optimization` | 0/5 | 3/5 | 2/5 | 2/5 | 2/5 | 1/5 | 1/5 |
| `pytorch-model-cli` | 0/5 | 0/5 | 1/5 | 1/5 | 0/5 | 0/5 | 0/5 |

## Eval Job IDs

| Result | Eval job |
| --- | --- |
| Likelihood TA-RL default/best, regex repair | `likgs39-regex-normalize1-nothink-a5-c1-o4096-20260603` |
| Hand-crafted TA-RL easy, easy-10 | `ta-strongcomplete-visibletool-reminders-v3-lowtemp-easy10-a5-o4096-r1` |
| Likelihood TA-RL easy, easy-10 | `ta-general-action-likelihood-prefix-short-n4-s40-r1-easy10-a5-o4096` |
| GRPO previous full-eval baseline, easy-10 | `grpo-budget-easy-b12s4-s35-easy10-a5-o4096` |
| Hand-crafted TA-RL easy, additional-10 | `add10-tarl-handcrafted-easy-gs39-shard{0..3}-a5-c1-o4096` |
| Likelihood TA-RL easy, additional-10 | `add10-tarl-likelihood-easy-gs39-shard{0..3}-a5-c1-o4096` |
| GRPO previous full-eval baseline, additional-10 | `add10-grpo-best-easy-s34-a5-c2-o4096` |
| Base, additional-10 | `add10-base-qwen3-thinking-a5-c2-o4096` |
| SFT medium-even, additional-10 | `add10-sft-final-a5-c2-o4096` |
| Hand-crafted TA-RL mixed, easy-10 | `ta-mixed-cmdpresence-s999-easy10-a5-o4096` |
| Likelihood TA-RL mixed, easy-10 | `ta-mixed-likelihood-s999-easy10-a5-o4096` |
| GRPO mixed, easy-10 | `ta-mixed-grpo-s64-easy10-a5-o4096-rerun1` |
| Hand-crafted TA-RL mixed, additional-10 | `add10-tarl-handcrafted-mixed-s999-a5-c2-o4096` |
| Likelihood TA-RL mixed, additional-10 | `add10-tarl-likelihood-mixed-s999-a5-c2-o4096` |

## Three-Hour RL Training Curves

The curves below use the same combined 20-task, 100-trial eval setting as the
main README table. The 0h point is the medium-even SFT checkpoint before RL.

| Recipe | RL train time | Checkpoint label | Full eval score |
| --- | ---: | --- | ---: |
| Hand-crafted turn/action TA-RL | 0.00h | SFT baseline | 17/100 |
| Hand-crafted turn/action TA-RL | 0.20h | README short selected checkpoint | 27/100 |
| Hand-crafted turn/action TA-RL | 1.06h | step 449 | 3/100 |
| Hand-crafted turn/action TA-RL | 2.13h | step 899 | 7/100 |
| Hand-crafted turn/action TA-RL | 3.08h | step 1299 | 3/100 |
| Domain-general likelihood TA-RL | 0.00h | SFT baseline | 17/100 |
| Domain-general likelihood TA-RL | 0.20h | README short selected checkpoint with guarded regex repair | 32/100 |
| Domain-general likelihood TA-RL | 1.12h | step 449 | 9/100 |
| Domain-general likelihood TA-RL | 2.25h | step 899 | 0/100 |
| Domain-general likelihood TA-RL | 3.28h | step 1299 | 3/100 |
| GRPO | 0.00h | SFT baseline | 17/100 |
| GRPO | 0.82h | step 19 | 1/100 |
| GRPO | 1.42h | previous full-eval step 34 | 24/100 |
| GRPO | 2.02h | step 49 | 5/100 |
| GRPO | 3.00h | step 74 | 7/100 |

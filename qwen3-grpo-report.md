# Report: Qwen/Qwen3-0.6B GRPO on GSM8K

## Recommendation

Use **Qwen/Qwen3-0.6B → GRPO on GSM8K** as the first working RL baseline.

This recipe best matches your constraints:

| Requirement                     | Status                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------ |
| Already a reasoning model       | Yes. Qwen3 thinking mode is enabled by default and emits `<think>...</think>`. |
| Not Qwen-3.5                    | Yes. It is Qwen3.                                                              |
| Open source model               | Yes. `Qwen/Qwen3-0.6B`, Apache-2.0.                                            |
| Open source data                | Yes. GSM8K is MIT-licensed.                                                    |
| Public before/after improvement | Yes. Public card reports **56.33% → 60.65%** GSM8K test.                       |
| Small enough to iterate on      | Yes. 0.6B model, short GSM8K task.                                             |

The strongest public anchor is **x32/Qwen3-0.6B-GRPO-GSM8K-Think**, which says it is a fine-tuned version of **Qwen/Qwen3-0.6B**, trained with **TRL**, using **GRPO**, and evaluated on GSM8K. Its reported GSM8K test improvement is from **56.33%** for the local Qwen3-0.6B reproduction to **60.65%** after GRPO checkpoint-180. ([Hugging Face][1])

## Why Qwen/Qwen3-0.6B qualifies

Qwen’s official model card says Qwen3 thinking mode is enabled by default when using `enable_thinking=True`, and in that mode the model generates thinking content wrapped in a `<think>...</think>` block followed by the final response. The same card recommends thinking-mode sampling settings of `temperature=0.6`, `top_p=0.95`, `top_k=20`, and warns against greedy decoding. ([Hugging Face][2])

The model card lists **Qwen/Qwen3-0.6B** under the Apache-2.0 license. ([Hugging Face][2]) The Qwen3 technical report also describes Qwen3 as integrating thinking mode for complex, multi-step reasoning and non-thinking mode in a unified framework. 

## Public result to reproduce

Use the x32 model card as the benchmark target:

| Model / checkpoint                 |  GSM8K val | GSM8K test |         Improvement |
| ---------------------------------- | ---------: | ---------: | ------------------: |
| Qwen/Qwen3-0.6B official           |          — | **59.59%** |                   — |
| Qwen/Qwen3-0.6B local reproduction | **53.25%** | **56.33%** |                   — |
| GRPO checkpoint-160                |          — | **59.67%** | **+5.92% relative** |
| GRPO checkpoint-180                |  **62.8%** | **60.65%** | **+7.66% relative** |

The absolute test-set gain versus the local reproduction baseline is:

```text
60.65% - 56.33% = +4.32 percentage points
```

The card states that the “official” number refers to results published by ModelScope or Hugging Face, while “local reproduction” refers to the project’s local pipeline; the improvement is relative to the local reproduction baseline. ([Hugging Face][1])

## Dataset

Use **GSM8K main**. The Hugging Face dataset card describes GSM8K as 8.5K grade-school math word problems requiring multi-step reasoning, with natural-language solutions and final numeric answers. For the `main` configuration, each example has `question` and `answer`; the `answer` field contains reasoning plus the final numeric solution. The split sizes are **7,473 train** and **1,319 validation/test** examples, and the dataset is MIT-licensed. ([Hugging Face][3])

Use:

```text
Dataset: openai/gsm8k
Config:  main
Train:   train split, 7473 examples
Eval:    test/validation split, 1319 examples
```

The x32 card lists `modelscope/gsm8k`; I would use `openai/gsm8k` for HF-native replication unless exact ModelScope compatibility is important. The task content and answer format should be equivalent enough for a baseline, but this is one source of possible score drift. ([Hugging Face][1])

## Prompt format

Use the prompt style from the x32 card:

```text
Please solve the math problem step by step. Use <think> tags to show your reasoning process, then provide the final numerical answer.

IMPORTANT: Your final answer must be a pure number only.

Format:
<think>
Your step-by-step reasoning here...
</think>
Final answer: [pure number only]
```

This matches the public checkpoint’s displayed inference format, where the model emits a reasoning block and then `Final answer: 18`. ([Hugging Face][1])

## Training framework

Use **TRL GRPOTrainer**, because the public x32 card says the model was trained using TRL and GRPO, and reports the framework versions:

```text
TRL:          0.19.0
Transformers: 4.52.4
PyTorch:      2.7.1
Datasets:     3.6.0
Tokenizers:   0.21.2
```

([Hugging Face][1])

TRL’s GRPO documentation supports custom reward functions, including Python reward functions that receive prompts, completions, tokenized completions, trainer state, and dataset columns, and return one float reward per completion. ([GitHub][4])

## Reward

Use a simple verifiable reward:

```text
R = R_correct + 0.1 * R_format
```

Where:

```text
R_correct = 1.0 if the extracted final answer numerically equals the GSM8K gold answer, else 0.0

R_format = 1.0 if the output contains:
           <think>...</think>
           Final answer: <pure number>
           else 0.0
```

The important part is that **accuracy reward dominates**. The format reward should be small, because the target metric is GSM8K exact-match accuracy, not merely producing valid-looking `<think>` tags.

## Generation settings

Use the Qwen3 thinking-mode defaults for both training rollouts and evaluation:

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
min_p       = 0
do_sample   = true
```

Qwen explicitly recommends these settings for thinking mode and warns not to use greedy decoding. ([Hugging Face][2])

## Evaluation protocol

Evaluate these three points:

```text
1. Base:           Qwen/Qwen3-0.6B
2. Checkpoint:     GRPO checkpoint-160
3. Final:          GRPO checkpoint-180
```

Use:

```text
Split:        openai/gsm8k main test/validation
Metric:       exact numeric match
n:            1319
Prompt:       same <think> + Final answer prompt
Decoding:     sample@1, temperature 0.6, top_p 0.95, top_k 20
Max tokens:   512 as first replication target
```

Report:

```text
GSM8K exact-match accuracy
format compliance rate
mean completion length
reward mean
reward std
fraction of prompts with zero reward std across generations
wall-clock time
GPU type
seed
```

For the 1,319-example GSM8K evaluation split, the finite-test-set standard error is about **1.36 percentage points** at 56–61% accuracy. A 95% binomial interval around a single score is about **±2.6–2.7 pp**. For the before/after delta, the independent-test approximation gives about **±3.8 pp**, though paired evaluation on the same questions should usually be tighter.

## Expected outcome

Use this as the target:

```text
Initial test accuracy:  ~56.3%
Final test accuracy:    ~60.6%
Expected gain:          +4.3 percentage points
```

This is a modest but real-looking improvement signal. It is much better for algorithm development than Qwen3-1.7B on GSM8K, where the model appears too strong and the improvement can vanish into noise.

## Caveats

The public x32 model card is enough to justify this as a starting baseline, but it is **not** a full exact-reproduction package. It does not publish the complete training script, optimizer hyperparameters, seed, reward implementation, answer parser, or exact eval harness. The “checkpoint-160/180” names strongly suggest trainer global steps, but the card does not explicitly define them as steps, so treat `max_steps=180` as a reasonable replication choice rather than a confirmed fact. ([Hugging Face][1])

My recommended wording for your baseline would be:

> We reproduce a Qwen3-0.6B GSM8K GRPO baseline inspired by x32/Qwen3-0.6B-GRPO-GSM8K-Think, using the same base model, task, thinking-style prompt, package versions, and target checkpoint schedule, with an explicitly specified correctness reward and evaluation harness.

That avoids overclaiming bit-for-bit replication while preserving the key point: this is the best small, strict reasoning-model recipe I found with public evidence of GRPO improving math accuracy.

[1]: https://huggingface.co/x32/Qwen3-0.6B-GRPO-GSM8K-Think "x32/Qwen3-0.6B-GRPO-GSM8K-Think · Hugging Face"
[2]: https://huggingface.co/Qwen/Qwen3-0.6B "Qwen/Qwen3-0.6B · Hugging Face"
[3]: https://huggingface.co/datasets/openai/gsm8k "openai/gsm8k · Datasets at Hugging Face"
[4]: https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md "trl/docs/source/grpo_trainer.md at main · huggingface/trl · GitHub"

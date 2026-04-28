# Qwen3-0.6B GSM8K RLVR Demo

This demo trains `Qwen/Qwen3-0.6B` on GSM8K with AReaL GRPO, a Megatron actor,
and SGLang rollout workers. It follows the prompt, reward shape, and decoding
settings from `../qwen3-grpo-report.md`, while using AReaL instead of TRL.

The verified node is a single 4x NVIDIA B200 host.

## Environment

From `AReaL/`:

```bash
$HOME/.local/bin/uv sync --extra cuda
$HOME/.local/bin/uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com transformer-engine-cu12==2.13.0
.venv/bin/python -m pip install --no-deps transformer-engine==2.13.0
MAX_JOBS=8 $HOME/.local/bin/uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com --no-build-isolation transformer-engine-torch==2.13.0
$HOME/.local/bin/uv pip install --python .venv/bin/python cuda-python==12.9.0 cuda-bindings==12.9.4
```

Make sure `.venv/bin/ninja` is available. The local setup used `uv sync --extra
cuda`, which installed it.

One local AReaL patch is required on this server: `areal/api/cli_args.py`
uses `sys.executable -m ...` in `get_py_cmd()`. Without that, SGLang child
processes can pick up the system Python CUDA packages instead of the venv CUDA
bindings.

## Node Settings

`rlvr_demo/scripts/run_b200_fast.sh` exports the important settings:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
GLOO_SOCKET_IFNAME=eth0
NCCL_SOCKET_IFNAME=eth0
NCCL_CUMEM_ENABLE=0
NCCL_NVLS_ENABLE=0
CUDA_DEVICE_MAX_CONNECTIONS=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HOME=/NHNHOME/areal_cache/huggingface
TRITON_CACHE_DIR=/NHNHOME/areal_cache/triton
```

The working topology is a 2+2 GPU split:

- Actor backend: `megatron:d2p1t1`
- Rollout backend: `sglang:d2p1t1`
- Actor workers use GPUs 0-1.
- SGLang rollout workers use GPUs 2-3.

This avoids duplicate-GPU XCCL communicator failures that appeared when actor
and rollout workers were both placed across all 4 GPUs. B200 also needs
`sglang.attention_backend: flashinfer`; SGLang FA3 asserted on SM100 during
bring-up.

## Smoke Test

```bash
bash rlvr_demo/scripts/run_b200_fast.sh rlvr_demo/configs/qwen3_06b_gsm8k_areal_megatron_smoke.yaml
```

This is a one-step plumbing test with 8 train examples and 8 eval examples. It
has been verified to initialize the Megatron actor, SGLang rollout servers, and
XCCL weight update path successfully.

## Report-Style Reproduction

```bash
bash rlvr_demo/scripts/run_b200_fast.sh rlvr_demo/configs/qwen3_06b_gsm8k_areal_megatron_report.yaml
```

This keeps the report anchor:

- `Qwen/Qwen3-0.6B`
- `openai/gsm8k`, `main`
- report prompt with `<think>` and `Final answer:`
- reward `1.0 * exact_numeric_match + 0.1 * strict_report_format`
- sample@1 eval with `temperature=0.6`, `top_p=0.95`, `top_k=20`
- GRPO with `n_samples=4`, `max_new_tokens=512`, and `total_train_steps=180`
- full 1,319-example GSM8K test evaluation before training and at step 180

The public report target is the x32 card's local-repro baseline to checkpoint
180 improvement, approximately `56.33% -> 60.65%` exact match. This AReaL config
is an implementation-level reproduction of the recipe, not a bit-for-bit TRL
reproduction.

Verified report-style result on the full 1,319-example GSM8K test split:

```text
step=  1 n=1319 mean_reward=0.419864 correct=550/1319 acc=0.4170 strict=0.0288 avg_gen_len=471.7 no_eos=0.6376
step=180 n=1319 mean_reward=0.811979 correct=945/1319 acc=0.7165 strict=0.9553 avg_gen_len=282.6 no_eos=0.0455
```

This is a +29.95 percentage point exact-match gain in this AReaL reproduction.
The initial score is lower than the public card's local reproduction because this
evaluation uses the strict report prompt plus sampled thinking-mode decoding; the
important reproduction signal is that the same RLVR recipe produces a large
before/after math gain with AReaL, Megatron, and SGLang.

## B200 Fast Recipe

```bash
bash rlvr_demo/scripts/run_b200_fast.sh
```

The default config is `rlvr_demo/configs/qwen3_06b_gsm8k_areal_megatron_b200_fast.yaml`.
It is the short, optimized recipe for this node:

- `total_train_steps: 10`
- `train_dataset.batch_size: 32`
- `gconfig.n_samples: 8`
- up to 256 sampled completions per train step
- `gconfig.max_new_tokens: 384`
- `eval_gconfig.max_new_tokens: 512`
- `actor.optimizer.lr: 8e-6`
- `actor.mb_spec.max_tokens_per_mb: 32768`
- validation limited to the first 256 GSM8K test examples

On this node, a 40-step exploratory run peaked early and then degraded as EOS
behavior collapsed. The 10-step stop is intentional.

Verified 10-step result on the fixed 256-example eval subset:

```text
step=  1 n=256 mean_reward=0.441797 correct=113/256 acc=0.4414 strict=0.0039 avg_gen_len=465.9 no_eos=0.6250
step= 10 n=256 mean_reward=0.785156 correct=176/256 acc=0.6875 strict=0.9766 avg_gen_len=191.2 no_eos=0.0234
```

That is a +24.61 percentage point exact-match gain on the short eval subset in
about 64 seconds of post-initialization training time.

Repeated short runs with only the seed and experiment name changed:

```text
seed=7  step 1: 113/256 acc=0.4414  step 10: 176/256 acc=0.6875  delta=+24.61 pp  time=64.23s
seed=8  step 1: 108/256 acc=0.4219  step 10: 169/256 acc=0.6602  delta=+23.83 pp  time=60.32s
seed=9  step 1: 111/256 acc=0.4336  step 10: 158/256 acc=0.6172  delta=+18.36 pp  time=63.76s
```

Mean over these three seeds: `43.23% -> 65.50%`, a +22.27 percentage point
exact-match gain in about 63 seconds after initialization.

To repeat with another seed:

```bash
bash rlvr_demo/scripts/run_b200_fast.sh \
  rlvr_demo/configs/qwen3_06b_gsm8k_areal_megatron_b200_fast.yaml \
  seed=10 experiment_name=qwen3-06b-gsm8k-grpo-b200-fast10-seed10
```

The failed 40-step exploratory run is still useful context:

```text
step=  1 acc=0.4648 mean_reward=0.467188 no_eos=0.5977
step= 10 acc=0.5742 mean_reward=0.672266 no_eos=0.0156
step= 20 acc=0.3945 mean_reward=0.487109 no_eos=0.1445
step= 30 acc=0.3906 mean_reward=0.429688 no_eos=1.0000
step= 40 acc=0.4492 mean_reward=0.485547 no_eos=1.0000
```

## Summarizing Eval Dumps

Training eval rollouts are written under:

```text
/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/logs/$USER/<experiment>/trial0/eval-rollout/
```

Summarize them with:

```bash
.venv/bin/python -m rlvr_demo.summarize_eval_rollouts \
  /NHNHOME/areal_runs/qwen3-gsm8k-rlvr/logs/$USER/qwen3-06b-gsm8k-grpo-b200-fast10/trial0/eval-rollout
```

Standalone SGLang evaluation is also available:

```bash
.venv/bin/python -m rlvr_demo.eval_qwen3_gsm8k \
  --config rlvr_demo/configs/qwen3_06b_gsm8k_areal_megatron_b200_fast.yaml
```

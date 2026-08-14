"""Project-wide constants. Import these; never inline the values."""

TIME_BINS = 128
MIN_MS = 1
MAX_MS = 120_000

# Qwen3.5-4B, the same family as the 0.8B the local shakedown trained
# (docs/results-08b-shakedown.md), scaled to what an H100 makes cheap. A
# hybrid: 24 of 32 layers are linear attention, which needs
# flash-linear-attention AND causal-conv1d on CUDA -- without BOTH,
# transformers' fast path disarms entirely and decode drops to ~15 tok/s
# (measured; training is barely affected at ~5k tok/s). Embeddings are tied,
# which halves what modules_to_save must hold.
#
# Qwen2.5-7B-Instruct was the previous default and remains fully supported.
# meta-llama/Meta-Llama-3.1-8B-Instruct stays gated: metadata reads work for
# anyone, weight downloads 403 without an approved access request.
BASE_MODEL = "Qwen/Qwen3.5-4B"
QWEN25_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

DEFAULT_SEED = 0

# Replayed text within this similarity of the target counts as having typed
# it. Lives here rather than in run_eval because the portal's per-sample
# validity badge must mean exactly what the Tier-1 validity gate means; two
# copies of the number would drift and the badge would quietly stop
# predicting the gate.
REPLAY_SIM_MIN = 0.80

"""Project-wide constants. Import these; never inline the values."""

TIME_BINS = 128
MIN_MS = 1
MAX_MS = 120_000

# The plan allows an 8B-class Llama or Qwen. Qwen is the default because
# meta-llama/Meta-Llama-3.1-8B-Instruct is gated: reading its metadata works
# for anyone, but downloading the weights returns 403 without an approved
# access request. Switch back once that approval is granted.
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

DEFAULT_SEED = 0

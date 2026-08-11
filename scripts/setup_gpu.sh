#!/usr/bin/env bash
# Provisions a fresh Linux GPU box for the Phase-1 motor fine-tune.
#
# Assumes: Ubuntu with NVIDIA drivers already present (true on Lambda), and
# this repository already checked out at the current directory.
#
#   bash scripts/setup_gpu.sh
#
# Takes roughly 40 minutes, nearly all of it the corpus download and the
# dataset rebuild. Safe to re-run: every step is idempotent.

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
AALTO_URL="https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"
RAW_DIR="data/raw/aalto"

echo "==> base tools (driver-only images may lack curl/unzip)"
for tool in curl unzip; do
  command -v "$tool" >/dev/null 2>&1 || sudo apt-get install -y "$tool"
done

echo "==> GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "no NVIDIA GPU visible; this script is for a CUDA box" >&2; exit 1; }
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# Measured for the default Qwen3.5-4B (tied embeddings): 24.1 GiB peak at
# --batch 32. The old >=48 GB gate was sized for the Qwen2.5-7B alternative,
# whose untied embedding tables roughly double the trainable footprint.
if [ "${GPU_MB}" -lt 22000 ]; then
  echo "FATAL: ${GPU_MB} MiB GPU memory; the default 4B run peaks at ~24.1 GiB (>=32 GB comfortable; Qwen2.5-7B needs >=48 GB)" >&2
  exit 1
elif [ "${GPU_MB}" -lt 26000 ]; then
  echo "WARNING: ${GPU_MB} MiB is borderline for the measured 24.1 GiB peak -- expect --freeze-embeddings or a smaller --batch"
fi

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# Python 3.14 has no stable torch wheels; pin something torch actually ships for.
echo "==> virtualenv on Python ${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> dependencies from the committed lockfile (torch is the CUDA build, ~2.5 GB)"
# --locked installs the exact versions this codebase was tested against,
# instead of re-resolving the loose ranges in pyproject.
uv sync --locked --extra train --extra dev

echo "==> fast kernels for the hybrid linear-attention default model"
# transformers disarms Qwen3.5's ENTIRE fast path unless BOTH import; without
# them decode drops to ~15 tok/s (measured) and every eval crawls. These are
# not in the lockfile because causal-conv1d compiles against whatever CUDA
# toolkit the box has -- it is a per-box build, not a lockable wheel.
uv pip install flash-linear-attention
if ! python -c "import causal_conv1d" 2>/dev/null; then
  # needs an nvcc matching torch's CUDA major (13); Lambda images ship 12.8
  if command -v nvcc >/dev/null && nvcc --version | grep -q "release 13"; then
    CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install causal-conv1d --no-build-isolation ||
      echo "WARNING: causal-conv1d build failed; eval decode will be ~15 tok/s"
  else
    echo "WARNING: no CUDA 13 nvcc; install cuda-toolkit-13-0 (NVIDIA apt repo), then:"
    echo "  CUDA_HOME=/usr/local/cuda-13.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install causal-conv1d --no-build-isolation"
  fi
fi

python - <<'PY'
import sys
import torch
cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
print(f"torch {torch.__version__} cuda={cuda} bf16={bf16}")
if cuda:
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}, {p.total_memory/1e9:.0f} GB")
if not (cuda and bf16):
    # Without this gate a CPU-only torch wheel passes setup and the run
    # silently trains on CPU fp32 for days instead of hours.
    sys.exit("FATAL: CUDA with bf16 is required on this box; refusing to continue")
PY

echo "==> Aalto corpus (1.6 GB download, expands to 18 GB)"
mkdir -p "${RAW_DIR}"
if [ ! -f "${RAW_DIR}/Keystrokes/files/metadata_participants.txt" ]; then
  if [ ! -f "${RAW_DIR}/Keystrokes.zip" ]; then
    curl -L --fail --retry 3 -o "${RAW_DIR}/Keystrokes.zip" "${AALTO_URL}"
  fi
  unzip -q -o "${RAW_DIR}/Keystrokes.zip" -d "${RAW_DIR}/"
  echo "    extracted $(ls "${RAW_DIR}/Keystrokes/files" | wc -l) files"
else
  echo "    already extracted, skipping"
fi

echo "==> test suite (offline; should be all green before training)"
python -m pytest -q

mkdir -p logs checkpoints  # referenced by the train/eval commands below

echo "==> dataset rebuild (parallel; measured ~3 min for the full corpus on 24 cores)"
if [ ! -f "data/processed/split.json" ]; then
  python scripts/build_dataset.py --klicke /nonexistent
else
  echo "    data/processed/split.json exists, skipping"
fi

cat <<'EOF'

==> ready

Smoke test first (about 10 minutes, proves the loop end to end):

  python -m typeshi.train_motor --mode transcription \
    --out checkpoints/smoke --epochs 0.002

Then the real run:

  python -m typeshi.train_motor --mode transcription --out checkpoints/motor

Then score it:

  python scripts/run_eval.py --checkpoint checkpoints/motor --n 200

EOF

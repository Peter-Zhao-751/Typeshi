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

echo "==> GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "no NVIDIA GPU visible; this script is for a CUDA box" >&2; exit 1; }

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

echo "==> dataset rebuild (~25 min single-process)"
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

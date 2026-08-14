#!/usr/bin/env bash
# Install CUDA 13 toolkit (nvcc) and build causal-conv1d from source against
# the venv's torch 2.13+cu130. Background job; log tells the story.
set -euo pipefail
cd /home/ubuntu/Typeshi

echo "==> [$(date +%T)] NVIDIA CUDA apt repo (ubuntu2204)"
if ! ls /usr/local/cuda-13* >/dev/null 2>&1; then
  cd /tmp/claude-1000/-home-ubuntu/72f558a9-1197-4088-85e6-9e4ba657ab8f/scratchpad
  curl -sL --fail -o cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring.deb
  sudo apt-get update -qq
  echo "==> [$(date +%T)] installing cuda-toolkit-13-0 (several GB)"
  sudo apt-get install -y -qq cuda-toolkit-13-0 > /dev/null
fi
ls -d /usr/local/cuda-13*

echo "==> [$(date +%T)] building causal-conv1d from source"
cd /home/ubuntu/Typeshi
source .venv/bin/activate
export CUDA_HOME=$(ls -d /usr/local/cuda-13* | head -1)
export PATH="$CUDA_HOME/bin:$PATH"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE   # skip the 404-ing prebuilt-wheel fetch
export MAX_JOBS=20
nvcc --version | tail -2
uv pip install causal-conv1d --no-build-isolation

echo "==> [$(date +%T)] numerical validation vs pure-torch reference"
python - <<'PY'
import torch, torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
torch.manual_seed(0)
B, D, L, W = 2, 4096, 64, 4
x = torch.randn(B, D, L, device="cuda", dtype=torch.bfloat16)
w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
b = torch.randn(D, device="cuda", dtype=torch.bfloat16)
out = causal_conv1d_fn(x, w, b, activation="silu")
ref = F.silu(F.conv1d(x.float(), w.unsqueeze(1).float(), b.float(), padding=W-1, groups=D)[..., :L])
err = (out.float() - ref).abs().max().item()
print(f"fn err: {err:.5f}"); assert err < 0.05
state = torch.randn(B, D, W, device="cuda", dtype=torch.bfloat16)
xt = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
o1 = causal_conv1d_update(xt, state.clone(), w, b, activation="silu")
rolled = torch.roll(state, shifts=-1, dims=-1); rolled[..., -1] = xt
ref_o = F.silu((rolled.float() * w.float().unsqueeze(0)).sum(-1) + b.float())
err2 = (o1.float() - ref_o).abs().max().item()
print(f"update err: {err2:.5f}"); assert err2 < 0.05
from transformers.utils.import_utils import is_causal_conv1d_available
print("transformers sees causal_conv1d:", is_causal_conv1d_available())
print("BUILD VALIDATED")
PY
echo "==> [$(date +%T)] DONE"

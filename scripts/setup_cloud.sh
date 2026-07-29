#!/usr/bin/env bash
# One-shot setup for a rented cloud GPU box (Ubuntu + CUDA, root/sudo available).
# For the Docker path use docker/Dockerfile instead. Run from the repo root.
#
#   bash scripts/setup_cloud.sh
#
# Assumes a CUDA toolkit (nvcc) is already present — true for vast.ai/runpod/lambda
# "cuda-devel" / pytorch images. Verify with `nvcc --version` first.
set -euo pipefail

SUDO="$(command -v sudo || true)"

echo "==> system deps (libsparsehash for torchsparse, ninja, unzip)"
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends libsparsehash-dev git ninja-build libgl1 wget unzip

echo "==> python deps"
pip install --no-cache-dir -r requirements.txt

echo "==> torchsparse v2.1 (from source; builds for the detected GPU arch)"
# If building on a box whose GPU differs from the run box, set e.g.
#   export TORCH_CUDA_ARCH_LIST="8.9"   # single-arch, faster build
FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation \
  git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0

echo "==> verify"
python -c "import torch, torchsparse; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchsparse', torchsparse.__version__)"
echo "==> smoke test (synthetic, no dataset)"
PYTHONPATH=. python -m scripts.smoke_test

cat <<'EOF'

Setup OK. Next:
  1) bash scripts/download_semantickitti.sh /data/semantickitti   # ~80GB, needs disk
  2) edit configs/data/semantickitti.yaml -> root: /data/semantickitti/dataset
  3) python -m src.train task=semantic model=minkunet             # reproduce mIoU on seq 08
EOF

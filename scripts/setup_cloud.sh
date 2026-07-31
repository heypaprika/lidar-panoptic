#!/usr/bin/env bash
# One-shot setup for a rented cloud GPU box (Ubuntu + CUDA, root/sudo available).
# For the Docker path use docker/Dockerfile instead. Run from the repo root.
#
#   bash scripts/setup_cloud.sh
#
# spconv ships prebuilt CUDA wheels, so there is NO source build (no libsparsehash, no nvcc needed
# at install time). Just make sure the box has an NVIDIA GPU + driver.
set -euo pipefail

SUDO="$(command -v sudo || true)"

echo "==> system deps (git, unzip, libgl for open3d)"
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends git libgl1 wget unzip

echo "==> python deps"
# system python often ships blinker via distutils, which pip can't cleanly uninstall to upgrade.
pip install --no-cache-dir --ignore-installed blinker
# requirements.txt pins spconv-cu120 (CUDA 12.x). For a different CUDA, install the matching wheel:
#   pip install spconv-cu118   # CUDA 11.8
pip install --no-cache-dir -r requirements.txt

echo "==> verify"
python -c "import torch, spconv; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('spconv', spconv.__version__)"
echo "==> smoke test (synthetic, no dataset)"
PYTHONPATH=. python -m scripts.smoke_test

cat <<'EOF'

Setup OK. Next:
  1) bash scripts/download_semantickitti.sh /data/semantickitti   # ~80GB, needs disk
  2) python -m scripts.debug_backbone /data/semantickitti/dataset val   # sanity: nnz shrinks, BACKBONE OK
  3) python -m src.train task=semantic model=minkunet data.root=/data/semantickitti/dataset
EOF

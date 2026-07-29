# Reproducible env for train/eval on a rented cloud GPU. torchsparse needs CUDA + libsparsehash.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsparsehash-dev git ninja-build libgl1 wget unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# Build torchsparse for all common cloud GPUs at once (no GPU visible during `docker build`):
#   7.0 V100 · 7.5 T4/2080 · 8.0 A100 · 8.6 3090/A6000 · 8.9 4090/4070Ti · 9.0 H100
ENV FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" MAX_JOBS=4
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-build-isolation \
      git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0
COPY . .

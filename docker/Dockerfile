# Reproducible env for train/eval. torchsparse needs CUDA + libsparsehash.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsparsehash-dev git ninja-build libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-build-isolation \
      git+https://github.com/mit-han-lab/torchsparse.git@v2.1.0
COPY . .

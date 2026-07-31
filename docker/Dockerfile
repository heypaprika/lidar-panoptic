# Reproducible env for train/eval on a rented cloud GPU. spconv ships prebuilt CUDA wheels (no build).
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 wget unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# requirements pins spconv-cu120 (CUDA 12.x). Match the wheel to the base image's CUDA if you change it.
RUN pip install --no-cache-dir --ignore-installed blinker \
 && pip install --no-cache-dir -r requirements.txt
COPY . .

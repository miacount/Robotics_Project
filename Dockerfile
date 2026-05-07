FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Workdir
WORKDIR /workspace

# PyTorch (CUDA 12.1 binary, CUDA 12.2 호환)
RUN pip install --no-cache-dir \
    torch==2.1.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Project requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /workspace

CMD ["/bin/bash"]

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.8.5
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget curl git build-essential \
    libffi-dev libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libgdbm-dev libnss3-dev \
    liblzma-dev tk-dev xz-utils && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils python3-pip && \
    ln -sf python3.8 /usr/bin/python && \
    wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python get-pip.py "pip==20.3" && rm get-pip.py && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch 1.12.1 + CUDA 11.3 via pip (official wheels)
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Copy requirements and install remaining packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set default working directory
WORKDIR /workspace

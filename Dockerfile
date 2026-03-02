FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY pyproject.toml README*.md ./

RUN pip install --no-cache-dir \
    torch==2.1.1 \
    torchvision==0.16.0 \
    "numpy<2.0.0" \
    matplotlib>=3.7.0 \
    wandb>=0.15.0 \
    tqdm>=4.65.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.2.0 \
    Pillow>=9.5.0 \
    lpips>=0.1.4 \
    ml-collections>=0.1.1 \
    ipython>=8.0.0

COPY src/ ./src/
# Original pixel-space training scripts
COPY train_cifar10.py ./
COPY train_cm.py ./
COPY compute_fid_cm.py ./
COPY generate_images_cm.py ./
# Latent space training scripts
COPY train_latent_cifar10.py ./
COPY train_latent_cm.py ./
COPY generate_latent_cifar10.py ./
COPY generate_latent_cm.py ./

RUN pip install --no-cache-dir . --no-deps

RUN mkdir -p /app/data /app/checkpoints /app/config

ENV PYTHONPATH=/app
ENV WANDB_DIR=/app/wandb

# ENTRYPOINT задается в docker-compose или при запуске
# ENTRYPOINT ["python", "train_cifar10.py"]

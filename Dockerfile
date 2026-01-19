FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt* ./

RUN pip install --no-cache-dir \
    torch==2.1.1 \
    torchvision \
    "numpy<2.0.0" \
    matplotlib \
    wandb \
    tqdm \
    scipy \
    Pillow

RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY src/ ./src/
COPY train_cifar10.py ./
COPY train_cm.py ./
COPY compute_fid_cm.py ./

RUN mkdir -p /app/data /app/checkpoints /app/config

ENV PYTHONPATH=/app
ENV WANDB_DIR=/app/wandb

RUN pip install ml_collections IPython

# ENTRYPOINT задается в docker-compose или при запуске
# ENTRYPOINT ["python", "train_cifar10.py"]

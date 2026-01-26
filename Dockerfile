# Base Image
FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1

ENV PYTORCH_ROCM_ARCH="gfx1100"
ENV ROCM_HOME=/opt/rocm
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512

WORKDIR /app

# System deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget libgl1 libglib2.0-0 build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y apex || true
RUN pip install "numpy<2.0"

# --- CACHE BUSTER ---
# Cambiando questa data forziamo Docker a reinstallare tutto da qui in giù
ENV BUILD_DATE="2024-06-15"

# Installazione pacchetti (incluso TIKTOKEN che mancava)
RUN pip install --no-cache-dir \
    accelerate \
    sentencepiece \
    tiktoken \
    protobuf \
    timm \
    einops \
    gradio \
    opencv-python-headless \
    pillow

# Core AI - Aggiorniamo anche transformers all'ultima versione
RUN pip install --no-cache-dir --no-deps "git+https://github.com/huggingface/transformers.git"
RUN pip install --no-cache-dir --no-deps "git+https://github.com/huggingface/diffusers.git"
RUN pip install --no-cache-dir huggingface_hub regex requests tokenizers filelock safetensors pyyaml

# Fix finale numpy
RUN pip install --force-reinstall "numpy<2.0"

CMD ["/bin/bash"]

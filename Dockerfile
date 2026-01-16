FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1

ENV ROCM_HOME=/opt/rocm
ENV PYTORCH_ROCM_ARCH="gfx1100"
ENV TORCH_BLAS_PREFER_HIPBLASLT=1

WORKDIR /app

# Dipendenze di sistema
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget libgl1 libglib2.0-0 build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# 1. RIMUOVE APEX (Causa crash con BFloat16 su ROCm)
RUN pip uninstall -y apex

# 2. Downgrade Numpy (Per compatibilità)
RUN pip install "numpy<2.0" --force-reinstall

# 3. Installa le librerie per GLM-Image
RUN pip install --no-cache-dir \
    accelerate \
    sentencepiece \
    protobuf \
    timm \
    einops \
    gradio \
    "git+https://github.com/huggingface/transformers.git" \
    "git+https://github.com/huggingface/diffusers.git"

CMD ["/bin/bash"]

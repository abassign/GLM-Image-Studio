#!/bin/bash
set -euo pipefail

# --- CONFIGURAZIONE ---
IMAGE_NAME="glm-image-rocm"
# SCRIPT_TO_RUN="run_inference.py"
SCRIPT_TO_RUN="web_ui.py"  # GUI
HF_CACHE_DIR="$HOME/AI/hf_cache"
OUTPUT_DIR="$(pwd)/outputs"
# ----------------------

mkdir -p "$HF_CACHE_DIR" "$OUTPUT_DIR"

# Configurazione AMD
RENDER_ARG=""
if getent group render >/dev/null; then
    RENDER_GID=$(getent group render | cut -d: -f3)
    RENDER_ARG="--group-add=$RENDER_GID"
fi

# Costruzione degli argomenti da passare a Python
# "$@" mantiene le virgolette e gli spazi corretti
PY_ARGS="$@"

echo "=================================================="
echo "   GLM-Image Generator (AMD 7900 XTX)"
echo "=================================================="

docker run --rm -it \
  --name glm_runner \
  --network=host \
  --ipc=host \
  --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video $RENDER_ARG \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -v "$HF_CACHE_DIR":/root/.cache/huggingface \
  -v "$OUTPUT_DIR":/app/outputs \
  -v "$(pwd)/$SCRIPT_TO_RUN":/app/$SCRIPT_TO_RUN \
  "$IMAGE_NAME" \
  bash -c "pip uninstall -y apex >/dev/null 2>&1 && python $SCRIPT_TO_RUN"

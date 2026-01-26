#!/bin/bash
set -euo pipefail

IMAGE_NAME="glm-image-rocm"
SCRIPT_TO_RUN="server.py"

HF_CACHE_DIR="$HOME/AI/hf_cache"
OUTPUT_DIR="$(pwd)/outputs"
LORA_DIR="$(pwd)/loras"

mkdir -p "$HF_CACHE_DIR" "$OUTPUT_DIR" "$LORA_DIR"

RENDER_ARG=""
if getent group render >/dev/null; then
    RENDER_GID=$(getent group render | cut -d: -f3)
    RENDER_ARG="--group-add=$RENDER_GID"
fi

echo "=================================================="
echo "   GLM-Image STUDIO PRO (FastAPI + JS)"
echo "   Server running on: http://localhost:7860"
echo "   LoRA Directory: $LORA_DIR"
echo "=================================================="

# Added PYTORCH_HIP_ALLOC_CONF for ROCm stability
docker run --rm -it \
  --name glm_runner \
  --network=host \
  --ipc=host \
  --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video $RENDER_ARG \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -v "$HF_CACHE_DIR":/root/.cache/huggingface \
  -v "$OUTPUT_DIR":/app/outputs \
  -v "$(pwd)":/app \
  -v "$LORA_DIR":/app/loras \
  "$IMAGE_NAME" \
  bash -c "python $SCRIPT_TO_RUN"

# 🧩 GLM-Image Studio (ROCm)

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![ROCm](https://img.shields.io/badge/ROCm-6.4+-red?style=for-the-badge&logo=amd&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**GLM-Image Studio** is a professional, high-performance AI creative suite designed specifically for **AMD GPUs** (ROCm). It features a modern, responsive Web UI that orchestrates Text-to-Image generation, Image-to-Image transformation, and advanced Visual Reasoning (Image-to-Text) using the latest GLM models.

> **Optimized for**: AMD Radeon RX 7900 series / Instinct accelerators running ROCm 6.x via Docker.

---

## ✨ Key Features

### 🎨 Creative Suites
*   **Text-to-Image (T2I)**: Generate high-fidelity images using Flux-based diffusion pipelines (`zai-org/GLM-Image`).
*   **Image-to-Image (I2I)**: Transform existing images with natural language prompts.
*   **Image-to-Text (I2T) with Thinking**: Analyze images using **`zai-org/GLM-4.1V-9B-Thinking`**.
    *   **Visual Thinking Process**: View the model's internal step-by-step reasoning (collapsible view).
    *   **Native Resolution**: Supports up to 4K inputs for analyzing fine details.
    *   **Structured Output**: Separates the "Thinking Process" from the "Final Answer" for clarity.
*   **Upscale & Refine**: Tiled upscaling using `stable-diffusion-x4-upscaler` with variable scale (1x-4x), `bfloat16` precision for artifact-free results, and post-process alignment.

### 🚀 Advanced-Grade UI
*   **Smart History Gallery**:
    *   **Auto-Sorting**: Newest generations always appear at the top.
    *   **Universal Loaders**: Load any history image into any input slot (`[➜ 1]`, `[➜ 2]`) regardless of origin.
    *   **Bulk Restore**: One-click **`[All]`** button instantly reloads dual-source inputs.
    *   **Compact Layout**: Optimized 128px view with high-contrast timestamps (~20% more space efficient).
    *   **Persistence**: Automatically saves all generations to disk.
*   **Model Management**: Dynamic model loading from `models/` directory with UI-based Trash/Delete operations.
*   **Advanced Control**: Selectable Samplers (Euler a, DPM++ 2M Karras, etc.) and native **GGUF** model support for low-VRAM environments.
*   **LoRA Management**: Hot-swappable LoRA adapters with strength control.
*   **Real-time Monitoring**: Integrated system status, timer, and console logs directly in the dashboard.
*   **State Isolation**: Independent prompt and result buffers for T2I, I2I, and I2T modes prevent accidental data loss.
*   **Cross-Flow**: Send generated images instantly from T2I -> I2I or analysis text from I2T -> T2I prompt.
*   **Robust Lifecycle**: Enhanced process management ensures clean shutdowns for all background workers.

### ⚙️ Backend Engineering
*   **Modular Architecture**: Isolated subprocesses for T2I, I2I, and I2T ensure stability and clean VRAM management.
*   **Unified Storage**: All uploads and generations are centrally managed in `outputs/` with automatic collision handling (auto-renaming).
*   **Zero-Config Deploy**: Docker-based setup handles all ROCm dependencies and library conflicts.

---

## 🛠️ Prerequisites

1.  **Linux OS**: Ubuntu 22.04 or compatible.
2.  **AMD Hardware**: GPU with ROCm support (e.g., RX 7900 XTX, MI300).
3.  **Docker & ROCm**:
    *   [Install Docker](https://docs.docker.com/engine/install/)
    *   Ensure your user is in the `video` and `render` groups.
4.  **HuggingFace Token**: Required to download the models.

---

## 🚀 Quick Start

### 1. Clone & Prepare
```bash
git clone https://github.com/your-username/glm-image-studio-rocm.git
cd glm-image-studio-rocm

# Create directories
mkdir -p outputs loras
```

### 2. Configure Environment
Create a `.env` file (optional, or pass via command line) if you modify the script, but the default script uses your host's HF cache.
*Ensure you have logged in to HuggingFace or have your token ready.*

### 3. Build Container
```bash
docker build -t glm-image-rocm .
```

### 4. Run Studio
Use the provided script to mount volumes and map GPUs correctly:
```bash
chmod +x run_glm.sh
./run_glm.sh
```
*The Web UI will be available at:* `http://localhost:7860`

---

## 📁 Project Structure

```text
/app
├── server.py           # FastAPI Backend & Orchestrator
├── process_t2i.py      # Independent T2I Worker
├── process_i2i.py      # Independent I2I Worker
├── process_i2t.py      # Independent I2T Worker
├── process_upscale.py  # Independent Upscale Worker
├── process_zimage.py   # Z-Image Turbo / GGUF Worker
├── shared_utils.py     # Shared logging & config logic
├── lora_manager.py     # LoRA scanning & config generation
├── run_glm.sh          # Docker launch script
├── Dockerfile          # ROCm Environment Definition
├── static/             # Frontend Assets (HTML/CSS/JS)
├── outputs/            # Stores ALL Generations & Uploads
└── loras/              # Place .safetensors adapters here
```

---

## 🧠 Supported Models

*   **Generation**: `zai-org/GLM-Image` (Flux.1 / SDXL styled pipelines)
*   **Vision/Reasoning**: `zai-org/GLM-4.1V-9B-Thinking`
*   **Turbo**: `zai-org/Z-Image-Turbo` (GGUF Quantized, fast inference)

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **VRAM OOM** | The system auto-clears VRAM when switching tabs. Wait 2-3s between mode switches. |
| **Model Load Fail** | Verify your HuggingFace token and internet connectivity. |
| **Permission Denied** | Ensure `run_glm.sh` is executable (`chmod +x`). |
| **Upload Error** | Check if the `outputs/` directory is writable by the container user. |

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).

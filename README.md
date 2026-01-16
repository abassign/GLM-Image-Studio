# GLM-Image Studio (AMD ROCm Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Platform](https://img.shields.io/badge/Platform-AMD%20ROCm-red)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Status](https://img.shields.io/badge/Status-Stable-green)

A fully containerized, optimized Web UI to run the **GLM-Image** model (16B parameters) on **AMD GPUs** using ROCm. 
Specifically tested and optimized for the **Radeon RX 7900 XTX (24GB)**.

## ✨ Features

- **AMD ROCm Support:** Built on `rocm/pytorch` with critical fixes for BFloat16 and Apex conflicts.
- **Hybrid Memory Management:** Custom sequential offloading pipeline enabling the massive 16B model to run on 24GB VRAM without OOM crashes.
- **Robust Web UI (Gradio):**
  - **Text-to-Image & Image-to-Image:** Seamless mode switching.
  - **Reactive Controls:** Sliders automatically adjust to maintain aspect ratios.
  - **Safety First:** Stop Generation and Emergency Exit buttons included.
  - **Clean Interface:** redundant progress bars hidden for a cleaner look.
- **Queue System:** Prevents race conditions and state corruption during multiple requests.
- **One-Click Run:** `run_glm.sh` handles Docker mounting, permissions, and GPU access.

## 🛠️ Prerequisites

- **OS:** Linux (Arch/Manjaro recommended).
- **GPU:** AMD Radeon GPU (RDNA3 / 7900 XTX recommended) with ROCm drivers.
- **Software:** Docker installed and configured with appropriate permissions.
- **RAM:** At least 32GB System RAM recommended for model offloading.

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/abassign/GLM-Image-Studio.git
cd GLM-Image-Studio
```

### 2. Build the Docker Image
```bash
docker build -t glm-image-rocm .
```

### 3. Run the Studio
```bash
./run_glm.sh
```
The application will launch at `http://localhost:7860`.

## 🖥️ Interface Controls

| Control | Description |
| :--- | :--- |
| **Prompt** | Enter your text description here. |
| **Input Image** | Drag & drop an image for Image-to-Image mode. |
| **Aspect Ratio** | "Keep Aspect Ratio" locks dimensions to prevent distortion. |
| **Steps / Guidance** | Adjust generation quality and adherence to prompt. |
| **🚀 GENERATE** | Starts the diffusion process. |
| **⏹️ STOP** | Immediately interrupts the generation. |
| **❌ EXIT** | Terminates the application and container. |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created by [abassign](https://github.com/abassign)*

# Project Log - GLM-Image Studio

## 2026-01-27
- **Status**: Resumed work. User confirmed Z-Image Turbo tests are positive.
- **Focus**: Shifting to "Upscale" feature refinement.
- **Action**: 
    - Analyzed `process_upscale.py`:
         - Uses `stabilityai/stable-diffusion-x4-upscaler`.
         - Implements **Tiled Upscaling** (256x256 tiles with overlap) to support high-res inputs on limited VRAM.
         - Uses `sequential_cpu_offload` and `attention_slicing` for memory safety.
    - Created this log structure in `logs/project_log.md`.
    - Planned verification of upscaling pipeline.
    - Implemented **Variable Scale** for Upscale:
      - Added Scale Slider (1x - 4x) and Output Resolution Preview in Frontend.
      - Updated `server.py` and `process_upscale.py` to handle post-process resizing.
      - User can now see `Input -> Output` resolution before generating.
    - **UI Refinement**: Moved Upscale Parameters block to be below the Prompt area for better workflow availability.
    - **User Feedback**: User verified Upscale is good ("buon upscale") but asks about improving **Depth of Field** (reducing Z-Image's tendency for shallow focus).
    - **Analysis**:
        - Z-Image/Flux often default to shallow DoF (Bokeh).
        - Generative Upscalers (SD x4) can "fix" this by hallucinating details in blurry areas if prompted: "highly detailed background, deep depth of field".
        - Plan: Advise using the Prompt field with high noise to recover background details.
    - **Fix Implementation**:
        - User reported "Noise Level" was ineffective.
        - **Cause**: UI sent 0-100 logic, but Model `noise_level` parameter acts on ~0-1000 scale. Also Guidance was too low (1.5) to enforce prompts.
        - **Action**:
            - Backend: Remapped UI `noise_level` * 3.5 (Range 0-100 -> 0-350 effective).
            - Frontend: Set default Guidance to **7.0** when switching to Upscale mode.
    - **UI Refinement**: Updated **Steps Slider**:
        - Min: 5, Step: 5, Default: 15 (Requested by user).
    - **Design Decision**: User asked if a **2nd Image** is needed for Upscale.
        - **Answer**: No. The `SD x4 Upscaler` pipeline is strictly Single-Input (Source + Prompt). Adding a second image would require complex ControlNet workflows not currently implemented.
        - **Status**: The second upload slot remains **hidden** in Upscale mode to prevent confusion.
        - **Action**: Modified History Gallery to **hide "-> 2" button** when in Upscale mode, matching T2I behavior.
    - **Refinement**: **Noise Level 0-1000 (Log/Power Scale)**.
        - User requested full range 0-1000 with logarithmic control.
        - Implemented Quadratic Curve (`1000 * (slider/100)^2`) in Frontend.
        - Slider 20% -> Noise 40. Slider 50% -> Noise 250. Slider 100% -> Noise 1000.
        - Removed Backend remapping.
    - **Debugging**: User reports "No effect" at Noise 1000 / Guidance 7.
        - **Action**: Added detailed debug prints in `process_upscale.py` (`Noise`, `Steps`, `Guidance`, `Prompt`) to verify input values in the System Log.
        - **Findings**: Script received `Noise Level: 20` despite user setting 1000.
        - **Hypothesis**: `script.js` changes might not be loaded (browser cache) OR `server.py` logic is flawed.
        - **Action**: Added debug print in `server.py`.
        - **CRITICAL FIX**: `server.py` logic `if req.noise_level else 20` was incorrect (python evaluates 0 as false, though 1000 should work). Replaced with strict `is not None` check.
        - **JS Update**: Added `console.log` in `script.js` directly on the slider movement `oninput`.
        - **Cache Buster**: Forced v8 on `script.js` to ensure browsers drop the old file.
        - **ROOT CAUSE FOUND**: `script.js` contained **TWO** definitions of `handleGeneration()`. The second one (without my fixes) was overwriting the first one.
        - **Fix**: Deleted duplicate function and merged Upscale logic into the correct one.
        - **Model Limit**: User found model crashes at Noise > 350.
        - **Recalibration**: Updated Frontend scale to map 0-100% -> **0-350** (Quadratic). Max power is now safe.
        - **Cleanup**: Removed real-time debug console logs.
    - **Feature Addition**: Downloaded **Deep DoF LoRA** (`deep_dof_sdxl.safetensors`) into `/loras` directory to help user combat "Burnt-in Bokeh".
        - Model: Ostris/polyhedron-skinny-all-in-focus-sdxl.
    - **UI Refinement**: **Upscale Mode Cleanup**
        - Hidden: **Top K**, **Temperature**, **LoRA Loader** (irrelevant for Upscale).
        - Visible: **Steps**, **Guidance**, **Seed**, **Noise Level**.
    - **Feature Addition**: **Image Comparison Slider** (Before/After).
        - Auto-activates in **I2I** and **Upscale** modes when generation completes.
    - **Feature Addition**: **Image Comparison Slider** (Before/After).
        - **ABANDONED**: Feature removed at user request due to UI instability.
        - Reverted to standard Single Image View.
    - **Bug Fix**: **Upscale White Artifacts / Exposure Spots**.
    - **Bug Fix**: **Upscale White Artifacts / Exposure Spots**.
        - Cause: VAE decoding overflow in `float16` mode (common SD x4 issue).
        - Fix 1 (Slow): VAE float32 cast (Abandoned due to performance hit).
        - Fix 2 (Fast & Stable): Switched entire pipeline to **`bfloat16`**.
            - `bfloat16` prevents overflow (fixing white spots) while maintaining fast inference on ROCm.
    - **Feature Addition**: **Image Comparison Slider** (Before/After).
        - Auto-activates in **I2I** and **Upscale** modes when generation completes.
        - **REBUILT (v2)**: completely rewrote the UI using `clip-path` and a simplified Stacking Context.
        - **Precision Alignment**: Implemented Backend Reference Generation (`DTOOL_REF`).
            - Upscaler now saves a resized version of the input to match the output dimensions exactly.
            - Slider uses this reference image to guarantee pixel-perfect overlap.
        - **Race Condition Fix**: Added "Hot Swap" logic. If the reference image arrives after the main image, the slider dynamically updates to the high-precision version immediately.
            - Upscaler now saves a resized version of the input to match the output dimensions exactly.
            - Slider uses this reference image to guarantee pixel-perfect overlap, eliminating "cut off" or alignment issues caused by resolution differences.
        - **Race Condition Fix**: Added "Hot Swap" logic. If the reference image arrives after the main image, the slider dynamically updates to the high-precision version immediately.
        - **CSS Alignment Fix**: Corrected `.comp-overlay` to be full-width (was 50%), ensuring the "Before" image is rendered with the exact same aspect-ratio constraints as the "After" image.

## 2026-01-28
- **UI Restoration (Legacy Layout)**:
    - **Goal**: Restore the specific Upload Panel layout requested by the user ("History Gallery" top button, separate Browse/Trash row, side-by-side Drop Zones, full-width Swap button).
    - **Implementation**:
        - Reconstructed `index.html` structure to match screenshots exactly.
        - Restored the "Browse 1" / "Browse 2" buttons with hidden trash icons that appear on upload.
        - Used `flex` row for side-by-side Drop Zones.
- **UI Refinement (Typography & Spacing)**:
    - **Typography**:
        - Enforced **Title Case** for all buttons ("History Gallery", "Browse 1", "Swap Images").
        - Identified and removed a global `text-transform: uppercase` rule in `style.css` (`.btn`) that was overriding HTML casing.
        - Set text color to **White** (`#ffffff`) for readability against dark backgrounds.
    - **Spacing**:
        - Equalized spacing around the "Swap Images" button: 10px above (to drop zones) and 10px below (to Denoise controls).
        - Reduced `upload-panel` bottom margin to 10px.
        - Removed `margin-top` from Swap button.
    - **Iconography**:
        - Added spacing between Folder icons and "Browse" text.
        - Allowed native Emoji colors for the Trash icon (removed white filter).
        - Added spacing between Folder icons and "Browse" text.
- **Layout Adjustment**:
    - **Goal**: History Gallery top button, separate Browse/Trash row, side-by-side Drop Zones, full-width Swap button.

## 2026-02-01
- **Feature Implementation**: **Sampler Selection**
    - **Goal**: Allow users to choose different sampling algorithms (e.g., Euler a, DPM++ 2M Karras).
    - **Backend**: Implemented `apply_scheduler` in `shared_utils.py` to dynamically switch schedulers on the pipeline.
    - **Frontend**: Added "Scheduler" dropdown in the UI.
    - **Fix**: Solved `AssertionError` with **Z-Image/SD3** models by preserving `time_shift_type` and forcing `use_dynamic_shifting=True` when switching samplers.
    - **Refinement**: Automatically mapped `Euler a` -> `FlowMatchEulerDiscreteScheduler` for SD3 models to prevent compatibility errors.

- **Feature Implementation**: **Model Management**
    - **Goal**: Full CRUD (Create/Read/Delete) control over the `models/` directory.
    - **Backend**: Added `DELETE /api/models` endpoint checking for path safety.
    - **Frontend**: Added "Trash" icon (🗑️) to the model list to delete selected models/folders with confirmation.
    - **Configuration**:
        - Updated `.gitignore` to exclude `models/*` (preventing git bloat).
        - Updated `server.py` to auto-create and initialize `models/` with a `README.md` if missing.

- **Critical Fix**: **OOM & Memory Optimization**
    - **Issue**: **Z-Image (SD3)** caused Out of Memory (OOM) errors during VAE decoding on ROCm (24GB VRAM).
    - **Cause**: A conflict between `enable_model_cpu_offload()` and a manual `pipe.to("cuda")` call was forcing the entire model into VRAM, disabling offloading.
    - **Fix**: Removed the manual move. Now the pipeline correctly swaps text encoders/transformers to CPU when not in use.
    - **Optimization**: Enabled `enable_vae_tiling()` and `enable_vae_slicing()` in T2I/I2I to process high-res images in small chunks.

- **Pipeline Refinement**: **I2I Single-File Loading**
    - **Issue**: `process_i2i.py` was hardcoded to `zai-org/GLM-Image`.
    - **Fix**: Updated I2I to accept dynamic `--model_path` and use `AutoPipelineForImage2Image` (or `.from_single_file`) to support custom checkpoints (e.g., Pony, CyberRealistic).
    - Moved the **Prompt** text area to the bottom of the parameter list (below Denoise Strength/Mix Ratio) as requested.

## 2026-01-29
- **Status**: Maintenance & Bug Fixing.
- **Focus**: UI Stability.
- **Action**:
    - **Fix**: Resolved "null reference" errors in browser console causing UI freezes.
    - **UI Restoration**: Restored missing elements:
        - Prompt Input field.
        - Image-to-Image (I2I) sliders.
        - History action buttons.
    - **Code Integrity**: Verified and corrected HTML element types and references in `script.js` to match `index.html`.

## 2026-01-30
- **Status**: Planning Phase (Model Structure).
- **Focus**: Scalable Model Management.
- **Context**: Growing number of models requires a structured organization system with metadata.
- **Plan**:
    - **Directory Structure**: Create a `models/` root directory to classify models (e.g., `models/stable-diffusion/`, `models/flux/`).
    - **Metadata**: Introduce `dataModels.json` in model directories to define:
        - Model display name.
        - VRAM requirements and resolution constraints.
        - Trigger words / Prompt prefixes.
    - **Backend**: Update `server.py` to scan this structure dynamically instead of hardcoding models.
    - **Workers**: Modify `process_*.py` scripts to accept a specific `model_path`.
- **Status Update**: Comprehensive Implementation Plan designated. Execution pending.

## 2026-01-31
- **Status**: Infrastructure & Configuration.
- **Focus**: Automatic Directory Management (User Request).
- **Objective**: Ensure critical directories (`outputs`, `models`, `loras`) are automatically created at runtime with external visibility.
- **Action**:
    - **Script Update**: Modified `run_glm.sh` to:
        - Define `MODELS_DIR`.
        - Auto-create `outputs`, `models`, `loras` on host before container start.
        - Mount `models` directory explicitly.
    - **Backend Update**: Updated `server.py` startup logic to:
        - Check/Create `/app/models` alongside `outputs` and `loras`.
        - Enforce `chmod 777` on these directories to guarantee Write/Read access for external users (fixing potential Docker root permission issues).
- **Outcome**: The system now self-heals by generating missing folder structures on launch, preventing "missing folder" errors and ensuring data persistence.


## 2026-02-01 (Late)
- **Feature Implementation**: **Z-Image Turbo GGUF Support (Modular Loading)**
    - **Goal**: Load quantized Z-Image Turbo GGUF models (`.gguf`) which failed with standard `from_single_file` due to dimension mismatches (160 vs 256) in `adaLN_modulation`.
    - **Solution**: Adopted "UnetLoader" strategy (inspired by ComfyUI):
        1. **Modular Loading**: Do NOT load the whole pipeline from GGUF.
        2. **Transformer**: Load only the `ZImageTransformer2DModel` from the GGUF file using `GGUFQuantizationConfig`.
           ```python
           from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
           quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
           transformer = ZImageTransformer2DModel.from_single_file(gguf_path, quantization_config=quantization_config, ...)
           ```
        3. **Base Components**: Load Tokenizer, Text Encoder, VAE, and Scheduler from the standard un-quantized base model directory.
        4. **Assembly**: Manually construct `ZImagePipeline` with these mixed components.
    - **Fixes**:
        - Resolves `SingleFileComponentError` and dimension mismatches.
        - Fixed `ImportError` for `GGUFQuantizationConfig` by importing from `diffusers.quantizers.quantization_config`.
        - Fixed `AttributeError` caused by variable shadowing (`scheduler` argument vs local variable).

## 2026-02-05
- **Status**: Verification & Audit.
- **Action**: Verified codebase integrity upon user request.
- **Findings**:
    - **Codebase State**: Corresponds to Feb 1st snapshot (Modular GGUF Loading).
    - **Missing Features**: The "Cache-DiT" integration (discussed in Feb 4th session) is **NOT** present in the file system.
    - **Conclusion**: The environment is stable but rolled back or pre-dated relative to the Cache-DiT experiments.

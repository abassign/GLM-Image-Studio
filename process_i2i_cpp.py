import sys
import os
import argparse
import subprocess
import time
import json
import shared_utils
import select
import re

# This script wraps the 'sd' C++ binary from stable-diffusion.cpp for I2I
# ensuring fully compatible output logging and JSON metadata generation
# for GLM-Image Studio.

import shutil
import time

# Helper for VRAM
def get_gpu_memory():
    """
    Returns the current VRAM usage in MB.
    Attempts to use rocm-smi or nvidia-smi.
    Returns 0 if neither is available.
    """
    try:
        # 1. Try rocm-smi (AMD)
        # We need to find the binary if not in path
        rocm_smi_cmd = ["rocm-smi", "--showmeminfo", "vram", "--json"]
        
        # Check specific paths if not in PATH
        if shutil.which("rocm-smi") is None:
             if os.path.exists("/opt/rocm/bin/rocm-smi"):
                 rocm_smi_cmd[0] = "/opt/rocm/bin/rocm-smi"
             elif os.path.exists("/usr/bin/rocm-smi"):
                 rocm_smi_cmd[0] = "/usr/bin/rocm-smi"
             else:
                 rocm_smi_cmd = None

        if rocm_smi_cmd:
            result = subprocess.run(rocm_smi_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # rocm-smi json structure: {"card0": {"VRAM Total Memory (B)": "...", "VRAM Total Used Memory (B)": "..."}}
                total_used = 0
                for card in data:
                    if "VRAM Total Used Memory (B)" in data[card]:
                        total_used += int(data[card]["VRAM Total Used Memory (B)"])
                return total_used / (1024 * 1024) # Convert to MB

        # 2. Try nvidia-smi (NVIDIA)
        if shutil.which("nvidia-smi"):
             result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             if result.returncode == 0:
                 lines = result.stdout.strip().split('\n')
                 total_mb = sum([int(x) for x in lines])
                 return float(total_mb)

    except Exception:
        pass
    return 0.0

SD_BINARY_PATH = "/usr/local/bin/sd"

def run_sd_cpp_i2i(prompt, width, height, steps, guidance, seed, init_img, strength, lora_config=None, top_k=1, temperature=0.6, model_path=None, clip_skip=1, scheduler=None, llm_override=None):
    global SD_BINARY_PATH
    start_time = time.time()
    llm_path = None # Initialize to avoid UnboundLocalError
    print(f"--> [CPP I2I Worker] Starting process PID: {os.getpid()}", flush=True)

    if not os.path.exists(SD_BINARY_PATH):
        # Check local path as fallback if not in /usr/local/bin
        local_sd = os.path.join(os.getcwd(), "sd")
        if os.path.exists(local_sd):
            SD_BINARY_PATH = local_sd
        else:
            print(f"--> [CPP I2I Worker] ERROR: C++ Binary not found at {SD_BINARY_PATH}. Please rebuild container.", flush=True)

    # 1. Determine Output Filename
    out_filename = f"i2i_cpp_{int(time.time())}.png"
    save_path = os.path.join("/app/outputs", out_filename)
    os.makedirs("/app/outputs", exist_ok=True)

    # 2. Determine Model Path
    final_model = model_path
    if model_path and os.path.isdir(model_path):
         # Try to find a specific weight file if a directory was passed
         # We MUST ignore "qwen" files as they are LLMs, not diffusion models
         cands = [f for f in sorted(os.listdir(model_path)) if f.endswith(".gguf") and "qwen" not in f.lower()]
         if not cands: 
             cands = [f for f in sorted(os.listdir(model_path)) if f.endswith(".safetensors") and "qwen" not in f.lower()]
         
         if cands:
             final_model = os.path.join(model_path, cands[0])
         else:
             # Deep search for Z-Image
             if "z_image" in model_path.lower() or "z-image" in model_path.lower():
                 # Look for a GGUF variant folder
                 parent = os.path.dirname(model_path)
                 for d in os.listdir(parent):
                     if "z" in d.lower() and "image" in d.lower() and "gguf" in d.lower():
                         gguf_path = os.path.join(parent, d)
                         if os.path.isdir(gguf_path):
                             cands_gguf = [f for f in sorted(os.listdir(gguf_path)) if f.endswith(".gguf") and "qwen" not in f.lower()]
                             if cands_gguf:
                                 final_model = os.path.join(gguf_path, cands_gguf[0])
                                 break

    if not final_model or (not os.path.exists(final_model)):
        print(f"data: LOG|❌ ERROR: Model file or directory not found: {final_model}", flush=True)
        sys.exit(1)

    # 3. Construct Command
    # Check if it's a Z-Image model (Lumina2 architecture)
    is_zimage = "z_image" in final_model.lower() or "z-image" in final_model.lower()

    cmd = [
        SD_BINARY_PATH,
        "--mode", "img_gen", # I2I is handled by img_gen + init_img
        "-t", str(os.cpu_count() or 4), # Threads
        "--prompt", prompt,
        "-W", str(width),
        "-H", str(height),
        "--steps", str(steps),
        "--cfg-scale", str(guidance),
        "--seed", str(seed),
        "-i", init_img, # Init Image
        "--strength", str(strength), # Denoise Strength
        "-o", save_path,
        "-v",
        "--fa", # Force Flash Attention (Correct Flag)
        "--vae-tiling" # Enable VAE tiling to prevent OOM during decode
    ]

    if is_zimage:
        print(f"--> [CPP I2I Worker] Detected Z-Image Architecture. Using specialized flags.", flush=True)
        cmd.extend(["-m", final_model])
        cmd.extend(["--diffusion-model", final_model])
        
        # Try to find VAE and LLM (Same logic as T2I)
        models_root = "/app/models"
        zimage_base = os.path.join(models_root, "Z-Image-Turbo")
        
        # 1. Look for VAE (ae.sft or from Z-Image folder)
        vae_path = None
        vae_search_paths = [
            os.path.join(zimage_base, "vae/diffusion_pytorch_model.safetensors"),
            os.path.join(models_root, "ae.sft"),
            os.path.join(os.path.dirname(final_model), "ae.sft")
        ]
        for p in vae_search_paths:
            if os.path.exists(p):
                vae_path = p
                break
        
        if vae_path:
            cmd.extend(["--vae", vae_path])
        else:
            print(f"data: LOG|⚠️ WARNING: Z-Image VAE (ae.sft) not found. Generation might fail.", flush=True)

        # 2. Look for LLM (Qwen GGUF)
        llm_path = None
        
        if llm_override and os.path.exists(llm_override):
             print(f"--> [CPP I2I Worker] Using Manual LLM Override: {llm_override}", flush=True)
             llm_path = llm_override
        else:
            # Auto-detect logic
            llm_search_dirs = [
                os.path.join(models_root, "Z-Image-Turbo-GGUF"),
                models_root,
                os.path.join(zimage_base, "text_encoder"),
                os.path.join(os.path.dirname(final_model), "text_encoders")
            ]
            
            found_llms = []
            for d in llm_search_dirs:
                if not os.path.exists(d): continue
                for f in os.listdir(d):
                    if f.endswith(".gguf") and ("qwen" in f.lower()):
                        found_llms.append(os.path.join(d, f))
            
            if found_llms:
                # Prioritize 4B
                for p in found_llms:
                    if "4b" in os.path.basename(p).lower():
                        llm_path = p
                        break
                if not llm_path:
                    llm_path = found_llms[0]

        if llm_path:
            print(f"--> [CPP I2I Worker] Using LLM: {llm_path}", flush=True)
            cmd.extend(["--llm", llm_path])
        else:
            print(f"data: LOG|⚠️ WARNING: Qwen LLM GGUF not found for Z-Image. Attempting with standard flags.", flush=True)
    else:
        cmd.extend(["-m", final_model])

    # Scheduler Mapping
    sched_map = {
        "euler_a": "euler_a",
        "euler": "euler",
        "dpm++_2m_karras": "dpm++2m", 
        "dpm++_sde_karras": "dpm++2s_a",
        "ddim": "euler"
    }
    if scheduler and scheduler in sched_map:
        cmd.extend(["--sampling-method", sched_map[scheduler]])

    # 4. LoRA Configuration
    if lora_config and os.path.exists(lora_config):
        try:
            with open(lora_config, 'r') as f:
                loras = json.load(f)
                for lora in loras:
                    path = lora[0]
                    strength = float(lora[1])
                    active = lora[2]
                    

                    if active and os.path.exists(path):
                        print(f"--> [CPP I2I Worker] Applying LoRA: {os.path.basename(path)} (Strength: {strength})", flush=True)
                        lora_dir = os.path.dirname(path)
                        
                        # Add lora-model-dir only once
                        if "--lora-model-dir" not in cmd:
                             cmd.extend(["--lora-model-dir", lora_dir])

                        # Append to prompt: <lora:filename_stem:strength>
                        stem = os.path.splitext(os.path.basename(path))[0]
                        # We must update the prompt in the command list, which is at index 4 (after binary, mode, -t threads)
                        # wait, cmd index is not safe.
                        # We will append to a local prompt variable and finding where to update cmd.

                        # Actually 'prompt' arg is passed to function.
                        
                        stem = os.path.splitext(os.path.basename(path))[0]
                        esc_stem = re.escape(stem)
                        pattern = rf"<lora:{esc_stem}:([0-9.]+)>"
                        
                        if re.search(pattern, prompt):
                            print(f"--> [CPP I2I Worker] Updating existing LoRA tag in prompt for {stem}", flush=True)
                            prompt = re.sub(pattern, f"<lora:{stem}:{strength}>", prompt)
                        else:
                            prompt += f" <lora:{stem}:{strength}>"

        except Exception as e:
            print(f"--> [CPP I2I Worker] Error parsing LoRA config: {e}", flush=True)

    # 4b. Handle Prompt-Only LoRAs (if no UI LoRAs selected)
    if "--lora-model-dir" not in cmd and "<lora:" in prompt:
        print(f"--> [CPP I2I Worker] Detected LoRA tags in prompt but no LoRA directory set.", flush=True)
        # Try to find a default LoRA directory
        # 1. Z-Image Turbo Default
        default_lora_dir = "/app/models/Z-Image-Turbo-GGUF/loras"
        # 2. General /app/loras
        if not os.path.exists(default_lora_dir):
            default_lora_dir = "/app/loras"
        
        # 3. If model path is deep, check relative 'loras' folder
        if os.path.exists(final_model):
            model_dir_loras = os.path.join(os.path.dirname(final_model), "loras")
            if os.path.exists(model_dir_loras):
                default_lora_dir = model_dir_loras

        if os.path.exists(default_lora_dir):
            print(f"--> [CPP I2I Worker] Auto-setting LoRA directory to: {default_lora_dir}", flush=True)
            cmd.extend(["--lora-model-dir", default_lora_dir])
        else:
            print(f"--> [CPP I2I Worker] Warning: Could not find a default LoRA directory.", flush=True)

    # UPDATE PROMPT IN CMD
    # cmd[4] is prompt because:
    # 0: binary
    # 1: --mode
    # 2: img_gen
    # 3: -t
    # 4: threads
    # 5: --prompt
    # 6: PROMPT <--- Index 6
        
        # safely find --prompt index
        try:
            p_idx = cmd.index("--prompt")
            if p_idx + 1 < len(cmd):
                cmd[p_idx + 1] = prompt
                print(f"--> [CPP I2I Worker] Updated Prompt with LoRA: {prompt}", flush=True)
        except:
             print("--> [CPP I2I Worker] Warning: Could not update prompt in command list", flush=True)


    # 5. Execution Loop (Adaptive Tiling)
    # i2i had tiling enabled by default, but we will make it adaptive now.
    if "--vae-tiling" in cmd:
        cmd.remove("--vae-tiling")

    current_cmd = cmd.copy()
    
    # Check Resolution Threshold for Auto-Tiling
    pixel_count = width * height
    threshold = 1024 * 1024 # 1 Megapixel
    
    using_tiling = False
    if pixel_count > threshold:
        print(f"--> [CPP I2I Worker] Resolution {width}x{height} > 1MP. Auto-enabling VAE Tiling.", flush=True)
        current_cmd.append("--vae-tiling")
        using_tiling = True

    # Retry Loop
    max_retries = 2
    final_returncode = -1
    max_vram = 0.0
    compute_buffer_mb = 0.0
    
    for attempt in range(max_retries):
        print(f"--> [CPP I2I Worker] Execution Attempt {attempt+1}/{max_retries} (Tiling: {using_tiling})", flush=True)
        print(f"--> [CPP I2I Worker] Executing: {' '.join(current_cmd)}", flush=True)
        
        # Reset metrics for new attempt
        max_vram = 0.0
        compute_buffer_mb = 0.0
        last_vram_check = 0

        # Streaming Output for Progress
        try:
            process = subprocess.Popen(current_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                # Check VRAM every 1 second
                if time.time() - last_vram_check > 1.0:
                     current_vram = get_gpu_memory()
                     if current_vram > max_vram: max_vram = current_vram
                     last_vram_check = time.time()
                
                if line:
                    raw = line.strip()
                    
                    # Parse Compute Buffer Size
                    if "z_image compute buffer size:" in raw:
                        try:
                            parts = raw.split("size:")
                            if len(parts) > 1:
                                val_part = parts[1].strip().split(" ")[0]
                                compute_buffer_mb = float(val_part)
                        except: pass

                    if "%" in raw or "step" in raw.lower():
                         print(f"data: LOG|{raw}", flush=True)
                    else:
                         print(raw, flush=True)

            final_returncode = process.poll()
            
            if final_returncode == 0:
                print("--> [CPP I2I Worker] Process finished successfully.", flush=True)
                break # Success!
            else:
                print(f"--> [CPP I2I Worker] Process failed with code {final_returncode}", flush=True)
                
                # If failed and NOT using tiling, try enabling it for next attempt
                if not using_tiling and attempt < max_retries - 1:
                     print(f"--> [CPP I2I Worker] ⚠️ Failure detected. Retrying with VAE Tiling enabled...", flush=True)
                     if "--vae-tiling" not in current_cmd:
                         current_cmd.append("--vae-tiling")
                     using_tiling = True
                     continue # Retry
                else:
                     break # Failed even with tiling or other error

        except Exception as e:
            print(f"--> [CPP I2I Worker] Execution Error: {e}", flush=True)
            break

    if final_returncode == 0:
        print(f"SUCCESS_OUTPUT:{save_path}", flush=True)
        
        # SAVE JSON METADATA
        inputs_data = {"prompt": prompt, "source_image_1": init_img}
        params_data = {
            "width": width, "height": height, "steps": steps, 
            "guidance": guidance, "seed": seed, 
            "strength": strength,
            "backend": "stable-diffusion.cpp",
            "mode": "i2i",
            "model": os.path.basename(final_model) + " (Diffusion)",
            "llm": os.path.basename(llm_path) + " (Text Encoder)" if llm_path else "auto"
        }
        outputs_data = {
            "type": "image", 
            "files": [out_filename],
            "exec_time": round(time.time() - start_time, 2),
            "gpu_vram_usage_mb": round(max_vram, 2),
            "compute_buffer_mb": round(compute_buffer_mb, 2)
        }
        shared_utils.save_generation_log("i2i", inputs_data, params_data, outputs_data, image_path_for_filename=save_path, model_name="sd-cpp")
        
    else:
         print(f"--> [CPP I2I Worker] Process failed with code {final_returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_config", type=str, default="")
    parser.add_argument("--top_k", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--clip_skip", type=int, default=1)
    parser.add_argument("--scheduler", type=str, default="euler")
    parser.add_argument("--llm_path", type=str, default=None)
    # I2I Specific
    parser.add_argument("--init_img", type=str, required=True)
    parser.add_argument("--strength", type=float, default=0.75)
    
    args = parser.parse_args()
    
    run_sd_cpp_i2i(
        args.prompt, args.width, args.height, args.steps, args.guidance, 
        args.seed, args.init_img, args.strength,
        args.lora_config, args.top_k, args.temperature, 
        args.model_path, args.clip_skip, args.scheduler, args.llm_path
    )

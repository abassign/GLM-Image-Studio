import argparse
import torch
import sys
import os
import time
import traceback
import gc
import shared_utils
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

# Model ID for SD x4 Upscaler
MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"

# Setup logging
shared_utils.setup_logging()

def run_upscale(image_path, prompt, seed=42, noise_level=20, steps=20, guidance=1.0, scale=4.0):
    print(f"--> [Upscale Worker] Starting process PID: {os.getpid()}", flush=True)
    
    # Noise Level is now passed correctly mapped (0-1000) from Frontend.
    effective_noise = noise_level
    print(f"--> [Upscale Worker] Noise Level: {effective_noise} | Steps: {steps} | Guidance: {guidance} | Scale: {scale}", flush=True)
    print(f"--> [Upscale Worker] Prompt: '{prompt}'", flush=True)

    if not os.path.exists(image_path):
        print(f"ERROR: Input image not found: {image_path}", flush=True)
        sys.exit(1)

    try:
        gc.collect()
        torch.cuda.empty_cache()

        print("--> [Upscale Worker] Loading Pipeline...", flush=True)
        # Using bfloat16 for stability on ROCm (avoids NaNs/White Spots) vs float16
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16
        )
        
        # VAE float32 cast (Critical for ROCm to avoid white/black artifacts)
        pipe.vae = pipe.vae.to(dtype=torch.float32)
        # PERFORMANCE: Move straight to GPU (User has 24GB VRAM, no need for offload)
        pipe.to("cuda")

        # 2. Enable Attention Slicing
        pipe.enable_attention_slicing()
        
        # 3. Enable VAE Tiling
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        except AttributeError:
             print("Warning: VAE Tiling/Slicing not available on this VAE object.", flush=True)
        
        print(f"--> [Upscale Worker] Loading Image: {os.path.basename(image_path)}", flush=True)
        low_res_img = Image.open(image_path).convert("RGB")
        w_orig, h_orig = low_res_img.size

        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        final_prompt = prompt if prompt and prompt.strip() else "high resolution, authentic details"

        # TILING LOGIC (Fixed at x4 native model resolution)
        # We always produce x4 first, then resize if needed.
        
        TILE_SIZE = 256 
        OVERLAP = 32
        w, h = low_res_img.size
        
        upscaled_image = None

        if w > TILE_SIZE or h > TILE_SIZE:
            print(f"--> [Upscale Worker] Image too large ({w}x{h}), using Tiled Upscaling...", flush=True)
            
            full_upscaled = Image.new("RGB", (w * 4, h * 4))
            
            grid_cols = (w + TILE_SIZE - 1) // TILE_SIZE
            grid_rows = (h + TILE_SIZE - 1) // TILE_SIZE
            
            for row in range(grid_rows):
                for col in range(grid_cols):
                    x1 = max(0, col * TILE_SIZE - OVERLAP)
                    y1 = max(0, row * TILE_SIZE - OVERLAP)
                    x2 = min(w, (col + 1) * TILE_SIZE + OVERLAP)
                    y2 = min(h, (row + 1) * TILE_SIZE + OVERLAP)
                    
                    tile = low_res_img.crop((x1, y1, x2, y2))
                    
                    print(f"--> [Upscale Worker] Processing Tile {col},{row} ({tile.size})...", flush=True)
                    
                    upscaled_tile = pipe(
                        prompt=final_prompt,
                        image=tile,
                        noise_level=effective_noise,
                        num_inference_steps=steps, 
                        guidance_scale=guidance, 
                        generator=generator
                    ).images[0]
                    
                    in_start_x = OVERLAP if col > 0 else 0
                    in_start_y = OVERLAP if row > 0 else 0
                    in_end_x = tile.width - (OVERLAP if col < grid_cols - 1 else 0)
                    in_end_y = tile.height - (OVERLAP if row < grid_rows - 1 else 0)
                    
                    valid_upscaled = upscaled_tile.crop((
                        in_start_x * 4, 
                        in_start_y * 4, 
                        in_end_x * 4, 
                        in_end_y * 4
                    ))
                    
                    paste_x = (x1 + in_start_x) * 4
                    paste_y = (y1 + in_start_y) * 4
                    
                    full_upscaled.paste(valid_upscaled, (paste_x, paste_y))
                    
                    torch.cuda.empty_cache()
                    gc.collect()

            upscaled_image = full_upscaled
            
        else:
            print(f"--> [Upscale Worker] Upscaling Direct...", flush=True)
            upscaled_image = pipe(
                prompt=final_prompt,
                image=low_res_img,
                noise_level=effective_noise,
                num_inference_steps=steps, 
                guidance_scale=guidance, 
                generator=generator
            ).images[0]
        
        # --- RESIZE IF SCALE != 4.0 ---
        if scale != 4.0:
            target_w = int(w_orig * scale)
            target_h = int(h_orig * scale)
            print(f"--> [Upscale Worker] Resizing from {upscaled_image.width}x{upscaled_image.height} to {target_w}x{target_h} (Scale: {scale}x)", flush=True)
            upscaled_image = upscaled_image.resize((target_w, target_h), Image.LANCZOS)
        
        out_filename = f"upscale_{int(time.time())}.png"
        save_path = os.path.join("/app/outputs", out_filename)
        os.makedirs("/app/outputs", exist_ok=True)
        upscaled_image.save(save_path)
        print(f"SUCCESS_OUTPUT:{save_path}", flush=True)



        inputs_data = {
            "prompt": final_prompt,
            "source_images": [os.path.basename(image_path)]
        }
        params_data = {
            "noise_level": noise_level,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "model": "stable-diffusion-x4-upscaler",
            "scale": scale
        }
        outputs_data = {
            "type": "image", 
            "files": [os.path.basename(save_path)]
        }

        shared_utils.save_generation_log("upscale", inputs_data, params_data, outputs_data, image_path_for_filename=save_path, model_name="sd-x4-upscaler")
        print("--> [Upscale Worker] Task Completed.", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR IN UPSCALE WORKER: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_level", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=4.0)
    args = parser.parse_args()

    run_upscale(
        image_path=args.image_path, 
        prompt=args.prompt, 
        seed=args.seed,
        noise_level=args.noise_level,
        steps=args.steps,
        guidance=args.guidance,
        scale=args.scale
    )

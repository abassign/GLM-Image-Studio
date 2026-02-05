import argparse
import torch
import sys
import os
import json
import time
import traceback
import gc
import shared_utils
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image
from PIL import Image

# Model IDs
MODEL_ID_TURBO = "Tongyi-MAI/Z-Image-Turbo"
MODEL_ID_STANDARD = "Tongyi-MAI/Z-Image" # Standard version

# Setup logging
shared_utils.setup_logging()


def run_zimage(prompt, width, height, steps, guidance, seed, lora_config=None, top_k=1, temperature=0.6, image_path=None, strength=0.75, model_type="turbo", model_path=None):
    print(f"--> [Z-Image Worker] Starting process PID: {os.getpid()}", flush=True)

    try:
        gc.collect()
        torch.cuda.empty_cache()

        # Select Model Source
        if model_path and os.path.exists(model_path):
            print(f"--> [Z-Image Worker] Loading LOCAL Model from: {model_path}", flush=True)
            target_model_id = model_path
        else:
            target_model_id = MODEL_ID_STANDARD if model_type == "standard" else MODEL_ID_TURBO
            print(f"--> [Z-Image Worker] Selected HuggingFace Model: {target_model_id} ({model_type})", flush=True)

        print("--> [Z-Image Worker] Loading Pipeline...", flush=True)
        
        # Determine loader
        is_single_file = os.path.isfile(target_model_id)
        
        if image_path and os.path.exists(image_path):
             print(f"--> [Z-Image Worker] Mode: I2I (Image: {os.path.basename(image_path)})", flush=True)
             if is_single_file:
                 pipe = AutoPipelineForImage2Image.from_single_file(
                    target_model_id, 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                 )
             else:
                 pipe = AutoPipelineForImage2Image.from_pretrained(
                    target_model_id, 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                 )
        else:
             print("--> [Z-Image Worker] Mode: T2I", flush=True)
             if is_single_file:
                 pipe = DiffusionPipeline.from_single_file(
                    target_model_id, 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                 )
             else:
                 pipe = DiffusionPipeline.from_pretrained(
                    target_model_id, 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                 )

        # Z-Image likely supports CPU offload
        print("--> [Z-Image Worker] Enabling CPU Offload...", flush=True)
        try: pipe.enable_model_cpu_offload()
        except: pass

        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"--> [Z-Image Worker] Generating...", flush=True)
        
        # Z-Image Turbo args check (steps often low, e.g., 4-8 for turbo)
        # We pass standard params, assuming pipe handles them or ignores extras.
        # Z-Image Turbo usually needs fewer steps.
        
        extra_args = {}
        if image_path and os.path.exists(image_path):
            init_image = Image.open(image_path).convert("RGB")
            # Resize logic similar to GLM or keeping original? usually better to resize to generation dim
            init_image = init_image.resize((width, height), Image.LANCZOS)
            extra_args["image"] = init_image
            extra_args["strength"] = strength
        
        image = pipe(
            prompt=prompt, 
            width=width, 
            height=height,
            num_inference_steps=steps, 
            guidance_scale=guidance,
            generator=generator,
            **extra_args
        ).images[0]

        out_filename = f"z_t2i_{int(time.time())}.png"
        save_path = os.path.join("/app/outputs", out_filename)
        os.makedirs("/app/outputs", exist_ok=True)
        image.save(save_path)
        print(f"SUCCESS_OUTPUT:{save_path}", flush=True)

        # SAVE JSON for History
        inputs_data = {"prompt": prompt}
        params_data = {
            "width": width, "height": height, "steps": steps, 
            "guidance": guidance, "seed": seed, "model": f"z-image-{model_type}",
            "strength": strength if image_path else None
        }
        if image_path:
             inputs_data["source_images"] = [image_path]
        outputs_data = {"type": "image", "files": [os.path.basename(save_path)]}

        # Use standard "t2i"/"i2i" modes for history compatibility
        mode_str = "i2i" if image_path else "t2i" 
        shared_utils.save_generation_log(mode_str, inputs_data, params_data, outputs_data, image_path_for_filename=save_path, model_name=f"z-image-{model_type}")

        print("--> [Z-Image Worker] Task Completed.", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR IN Z-IMAGE WORKER:", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10) # Turbo default lower
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_config", type=str, default="")
    parser.add_argument("--top_k", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--model_type", type=str, default="turbo", choices=["turbo", "standard"])
    parser.add_argument("--model_path", type=str, default=None, help="Local path to model folder or safetensors file")
    args = parser.parse_args()

    run_zimage(args.prompt, args.width, args.height, args.steps, args.guidance, args.seed, args.lora_config, args.top_k, args.temperature, args.image_path, args.strength, args.model_type, args.model_path)

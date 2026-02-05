import argparse
import torch
import sys
import os
import json
import time
import inspect
import traceback
import gc
import datetime
import shared_utils
from PIL import Image
from diffusers import AutoPipelineForImage2Image, DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline

MODEL_ID = "zai-org/GLM-Image"

# Setup logging
shared_utils.setup_logging()



def run_i2i(prompt, image_path, width, height, steps, guidance, seed, lora_config=None, top_k=1, temperature=0.6, image_path_2=None, strength=0.75, mix_ratio=0.5, clip_skip=1, model_path=None, scheduler=None):
    print(f"--> [I2I Worker] Starting process PID: {os.getpid()}", flush=True)

    target_model = model_path if model_path else MODEL_ID

    if not os.path.exists(image_path):
        print(f"ERROR: Input image not found at: {image_path}", flush=True)
        sys.exit(1)

    try:
        gc.collect()
        torch.cuda.empty_cache()

        print(f"--> [I2I Worker] Loading Pipeline from {target_model}...", flush=True)
        print(f"--> [I2I Worker] Loading Pipeline from {target_model}...", flush=True)
        
        # Handle single file vs directory
        if os.path.isfile(target_model):
             print("--> [I2I Worker] Detected Single File. Attempting to load as SDXL I2I...", flush=True)
             try:
                pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(target_model, torch_dtype=torch.bfloat16)
             except Exception as sdxl_err:
                print(f"--> [I2I Worker] SDXL Load failed ({sdxl_err}). Trying SD1.5 I2I...", flush=True)
                try:
                    pipe = StableDiffusionImg2ImgPipeline.from_single_file(target_model, torch_dtype=torch.bfloat16)
                except Exception as sd15_err:
                     print(f"--> [I2I Worker] Failed to load single file model: {sd15_err}")
                     raise sd15_err
        elif os.path.isdir(target_model):
             # It's a directory. Check if it's a Diffusers repo (has model_index.json or config.json)
             if os.path.exists(os.path.join(target_model, "model_index.json")) or os.path.exists(os.path.join(target_model, "config.json")):
                  try:
                       pipe = AutoPipelineForImage2Image.from_pretrained(target_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
                  except Exception as auto_err:
                       print(f"--> [I2I Worker] AutoPipeline failed ({auto_err}). Fallback to standard DiffusionPipeline...", flush=True)
                       pipe = DiffusionPipeline.from_pretrained(target_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
             else:
                 # Directory without index, maybe contains the single file?
                 # Find first .safetensors in the dir
                 files = [f for f in os.listdir(target_model) if f.endswith(".safetensors")]
                 if files:
                     single_file_path = os.path.join(target_model, files[0])
                     print(f"--> [I2I Worker] Found single file in directory: {single_file_path}. Loading...", flush=True)
                     try:
                        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(single_file_path, torch_dtype=torch.bfloat16)
                     except Exception as sdxl_err:
                        print(f"--> [I2I Worker] SDXL Load failed ({sdxl_err}). Trying SD1.5 I2I...", flush=True)
                        try:
                            pipe = StableDiffusionImg2ImgPipeline.from_single_file(single_file_path, torch_dtype=torch.bfloat16)
                        except Exception as sd15_err:
                             print(f"--> [I2I Worker] Failed to load single file model: {sd15_err}")
                             raise sd15_err
                 else:
                     # Fallback to default load attempt (remote repo id?)
                     try:
                        pipe = AutoPipelineForImage2Image.from_pretrained(target_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
                     except:
                        raise FileNotFoundError(f"No model_index.json or .safetensors found in {target_model}")
        else:
             # Assume repo ID
             try:
                  pipe = AutoPipelineForImage2Image.from_pretrained(target_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
             except Exception as auto_err:
                  print(f"--> [I2I Worker] AutoPipeline failed ({auto_err}). Fallback to standard DiffusionPipeline...", flush=True)
                  pipe = DiffusionPipeline.from_pretrained(target_model, torch_dtype=torch.bfloat16, trust_remote_code=True)

        # Apply Clip Skip if requested (Logging only, relies on params)
        if clip_skip and int(clip_skip) > 1:
            print(f"--> [I2I Worker] Requested Clip Skip: {clip_skip} (Passed to pipeline if supported)", flush=True)

        shared_utils.load_loras(pipe, lora_config)

        # Apply Scheduler
        if scheduler:
            shared_utils.apply_scheduler(pipe, scheduler)


        print("--> [I2I Worker] Enabling CPU Offload...", flush=True)
        pipe.enable_model_cpu_offload()

        # Fix Vision Encoder Pinning (Required for model_cpu_offload)
        vision_components = ["vision_language_encoder", "vision_model", "image_encoder"]
        for name in vision_components:
            if hasattr(pipe, name) and getattr(pipe, name) is not None:
                getattr(pipe, name).to("cuda")

        try: pipe.enable_vae_tiling()
        except: pass
        try: pipe.enable_attention_slicing("max")
        except: pass

        # Load and resize first image
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((width, height), Image.LANCZOS)
        
        final_init_image = init_image
        
        # Load and resize second image if present and blend
        if image_path_2 and os.path.exists(image_path_2):
            print(f"--> [I2I Worker] Loading secondary image for blending: {os.path.basename(image_path_2)}", flush=True)
            init_image_2 = Image.open(image_path_2).convert("RGB")
            init_image_2 = init_image_2.resize((width, height), Image.LANCZOS)
            
            print(f"--> [I2I Worker] Blending images with ratio: {mix_ratio}", flush=True)
            # Use Image.blend: out = image1 * (1.0 - alpha) + image2 * alpha
            final_init_image = Image.blend(init_image, init_image_2, alpha=float(mix_ratio))

        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"--> [I2I Worker] Generating (Strength: {strength}, TopK: {top_k}, Temp: {temperature})...", flush=True)
        kwargs = {
            "prompt": prompt, 
            "image": [final_init_image], 
            "width": width, "height": height,
            "num_inference_steps": steps, "generator": generator,
            "guidance_scale": guidance,
        }
        
        # Filter unsupported kwargs
        sig_params = inspect.signature(pipe.__call__).parameters
        
        if "top_k" in sig_params and top_k is not None: 
            kwargs["top_k"] = int(top_k)
        if "temperature" in sig_params and temperature is not None: 
            kwargs["temperature"] = float(temperature)

        if "strength" in sig_params: 
            kwargs["strength"] = float(strength)
        else:
            print("--> [I2I Warning] 'strength' parameter not supported by this pipeline version.", flush=True)

        image = pipe(**kwargs).images[0]

        out_filename = f"i2i_{int(time.time())}.png"
        save_path = os.path.join("/app/outputs", out_filename)
        os.makedirs("/app/outputs", exist_ok=True)
        image.save(save_path)
        print(f"SUCCESS_OUTPUT:{save_path}", flush=True)

        # SAVE JSON (V2)
        
        inputs_data = {
            "prompt": prompt,
            "source_images": [os.path.basename(image_path)],
            "loras": []
        }
        if image_path_2:
            inputs_data["source_images"].append(os.path.basename(image_path_2))
            
        if lora_config and os.path.exists(lora_config):
            try:
                with open(lora_config, 'r') as f: inputs_data["loras"] = json.load(f)
            except: pass

        params_data = {
            "width": width, "height": height, "steps": steps, "guidance": guidance, 
            "seed": seed, "strength": strength, "top_k": top_k, "temperature": temperature, 
            "mix_ratio": mix_ratio
        }

        outputs_data = {
            "type": "image",
            "files": [os.path.basename(save_path)]
        }

        shared_utils.save_generation_log("i2i", inputs_data, params_data, outputs_data, image_path_for_filename=save_path, model_name="glm-4")

        print("--> [I2I Worker] Task Completed.", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR IN I2I WORKER:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--image_path_2", type=str, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_config", type=str, default="")
    parser.add_argument("--top_k", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--mix_ratio", type=float, default=0.5)
    parser.add_argument("--clip_skip", type=int, default=1)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    args = parser.parse_args()

    run_i2i(args.prompt, args.image_path, args.width, args.height, args.steps, args.guidance, args.seed, args.lora_config, args.top_k, args.temperature, args.image_path_2, args.strength, args.mix_ratio, args.clip_skip, args.model_path, args.scheduler)

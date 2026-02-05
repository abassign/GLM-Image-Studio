import sys
import os

# Ensure we use our persistent libs
libs_path = "/app/libs"
if libs_path not in sys.path:
    sys.path.insert(0, libs_path) # High priority
    print(f"--> [T2I Worker] Added {libs_path} to sys.path", flush=True)

try:
    import gguf
    print(f"--> [T2I Worker] GGUF Module Verified (Import successful)", flush=True)
except ImportError as e:
    print(f"--> [T2I Worker] WARNING: Could not import gguf from {libs_path}: {e}", flush=True)

import argparse
import torch
import json
import time
import traceback
import gc
import datetime
import shared_utils
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline

# ... rest of script

def run_t2i(prompt, width, height, steps, guidance, seed, lora_config=None, top_k=1, temperature=0.6, model_path=None, clip_skip=1, scheduler=None):
    print(f"--> [T2I Worker] Starting process PID: {os.getpid()}", flush=True)

    try:
        gc.collect()
        torch.cuda.empty_cache()

        if not model_path:
            print("--> [T2I Worker] No model path provided. Exiting.")
            sys.exit(1)

        print(f"--> [T2I Worker] Loading Pipeline from {model_path}...", flush=True)
        # Handle single file vs directory
        if os.path.isfile(model_path):
             pipe = None
             found_z_image = False
             if "z_image" in model_path.lower() or "z-image" in model_path.lower():
                 print("--> [T2I Worker] Detected Z-Image GGUF. Using Modular Loading Strategy...", flush=True)
                 try:
                     from diffusers import ZImagePipeline, ZImageTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
                     from transformers import AutoTokenizer, AutoModel
                     try:
                        from diffusers.quantizers.quantization_config import GGUFQuantizationConfig
                     except ImportError:
                        from diffusers.utils import GGUFQuantizationConfig # Fallback for older/different versions

                     # Base components path
                     base_model_path = "/app/models/Z-Image-Turbo"
                     print(f"--> [T2I Worker] Loading helper components from {base_model_path}...", flush=True)
                     
                     tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_model_path, "tokenizer"))
                     text_encoder = AutoModel.from_pretrained(os.path.join(base_model_path, "text_encoder"))
                     vae = AutoencoderKL.from_pretrained(os.path.join(base_model_path, "vae"))
                     z_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(base_model_path, "scheduler"))

                     print(f"--> [T2I Worker] Loading Z-Image Transformer from GGUF: {model_path}", flush=True)
                     # Configure GGUF loading
                     quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
                     
                     transformer = ZImageTransformer2DModel.from_single_file(
                         model_path,
                         quantization_config=quantization_config,
                         torch_dtype=torch.bfloat16
                     )

                     print("--> [T2I Worker] Assembling Z-Image Pipeline...", flush=True)
                     pipe = ZImagePipeline(
                         vae=vae,
                         text_encoder=text_encoder,
                         tokenizer=tokenizer,
                         transformer=transformer,
                         scheduler=z_scheduler
                     )
                     
                     print("--> [T2I Worker] Z-Image GGUF loaded successfully (Modular)!", flush=True)
                     found_z_image = True
                 except Exception as z_err:
                     print(f"--> [T2I Worker] Z-Image Load failed: {z_err}. Falling back to SDXL/SD1.5...", flush=True)
                     # raise z_err # Optional: raise to see full trace if needed, but fallback is safer
                     found_z_image = False
             
             if not found_z_image: # If not Z-Image or Z-Image loading failed, try SDXL/SD1.5
                 print("--> [T2I Worker] Detected Single File. Attempting to load as SDXL...", flush=True)
                 try:
                    pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.bfloat16)
                 except Exception as sdxl_err:
                    print(f"--> [T2I Worker] SDXL Load failed ({sdxl_err}). Trying SD1.5...", flush=True)
                    try:
                        pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.bfloat16)
                    except Exception as sd15_err:
                         print(f"--> [T2I Worker] Failed to load single file model: {sd15_err}")
                         raise sd15_err
        else:
             # It's a directory. Check if it's a Diffusers repo (has model_index.json or config.json)
             if os.path.exists(os.path.join(model_path, "model_index.json")) or os.path.exists(os.path.join(model_path, "config.json")):
                 pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
             else:
                 # Directory without index, maybe contains the single file?
                 # Find first .safetensors in the dir
                 files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
                 if files:
                     single_file_path = os.path.join(model_path, files[0])
                     print(f"--> [T2I Worker] Found single file in directory: {single_file_path}. Loading...", flush=True)
                     try:
                        pipe = StableDiffusionXLPipeline.from_single_file(single_file_path, torch_dtype=torch.bfloat16)
                     except Exception as sdxl_err:
                        print(f"--> [T2I Worker] SDXL Load failed ({sdxl_err}). Trying SD1.5...", flush=True)
                        try:
                            pipe = StableDiffusionPipeline.from_single_file(single_file_path, torch_dtype=torch.bfloat16)
                        except Exception as sd15_err:
                             print(f"--> [T2I Worker] Failed to load single file model: {sd15_err}")
                             raise sd15_err
                 else:
                     raise FileNotFoundError(f"No model_index.json or .safetensors found in {model_path}")

        # Apply Clip Skip if requested
        # Note: Modifying config.num_hidden_layers directly is unsafe and can cause hangs/errors.
        # We rely on the pipeline accepting 'clip_skip' or 'num_hidden_layers' in the __call__ method,
        # or we accept that strict clip skip requires loading a specific CLIP model variant.
        # For now, we mainly rely on passing it via extra_kwargs if supported.
        if clip_skip and int(clip_skip) > 1:
            print(f"--> [T2I Worker] Requested Clip Skip: {clip_skip} (Passed to pipeline if supported)", flush=True)

        shared_utils.load_loras(pipe, lora_config)
        
        # Apply Scheduler
        if scheduler:
            shared_utils.apply_scheduler(pipe, scheduler)

        # Memory Optimizations
        print("--> [T2I Worker] Enabling Memory Optimizations (Offload, Tiling, Slicing)...", flush=True)
        
        # 1. CPU Offload (Models moved to CPU when not used)
        offload_enabled = False
        try:
            pipe.enable_model_cpu_offload()
            offload_enabled = True
        except Exception as e:
            print(f"--> [T2I Worker] enable_model_cpu_offload failed: {e}")

        # 2. VAE Tiling (Decode in chunks)
        try:
            pipe.enable_vae_tiling()
        except AttributeError:
             print("--> [T2I Worker] enable_vae_tiling not supported.")
        
        # 3. VAE Slicing (Decode batch of tiles one by one)
        try:
            pipe.enable_vae_slicing()
        except AttributeError:
             print("--> [T2I Worker] enable_vae_slicing not supported.")

        try:
            pipe.enable_attention_slicing("max")
        except AttributeError:
             print("--> [T2I Worker] enable_attention_slicing not supported.")
        
        # Disabled manual VAE cast to avoid conflict with enable_model_cpu_offload
        # if hasattr(pipe, "vae") and pipe.vae is not None:
        #      try:
        #         pipe.vae.to(dtype=torch.bfloat16)  
        #      except: pass
        # if hasattr(pipe, "vae"):
        #    pipe.vae.to(dtype=torch.float32)

        # Only move to CUDA manually if offload is NOT enabled
        if not offload_enabled:
             print("--> [T2I Worker] Moving pipeline to CUDA manually...", flush=True)
             pipe.to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"--> [T2I Worker] Generating (TopK: {top_k}, Temp: {temperature})...", flush=True)
        # Pass extra params ONLY IF supported by the pipeline's __call__ method
        # Standard Diffusion Pipelines usually do not support top_k/temperature, but we check to be safe/future-proof.
        import inspect
        sig_params = inspect.signature(pipe.__call__).parameters
        
        extra_kwargs = {}
        if "top_k" in sig_params and top_k is not None: 
            extra_kwargs["top_k"] = int(top_k)
        if "temperature" in sig_params and temperature is not None: 
            extra_kwargs["temperature"] = float(temperature)
        
        # Some new diffusers versions accept clip_skip in __call__
        if "clip_skip" in sig_params and clip_skip and int(clip_skip) > 1:
             extra_kwargs["clip_skip"] = int(clip_skip)

        image = pipe(
            prompt=prompt, width=width, height=height,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=generator,
            **extra_kwargs
        ).images[0]

        out_filename = f"t2i_{int(time.time())}.png"
        save_path = os.path.join("/app/outputs", out_filename)
        os.makedirs("/app/outputs", exist_ok=True)
        image.save(save_path)
        print(f"SUCCESS_OUTPUT:{save_path}", flush=True)

    # SAVE JSON (V2)
        
        # Inputs
        inputs_data = {
            "prompt": prompt,
            "loras": []
        }
        if lora_config and os.path.exists(lora_config):
            try:
                with open(lora_config, 'r') as f: inputs_data["loras"] = json.load(f)
            except: pass

        # Params
        params_data = {
            "width": width, "height": height, "steps": steps, 
            "guidance": guidance, "seed": seed, 
            "top_k": top_k, "temperature": temperature,
            "clip_skip": clip_skip
        }

        # Outputs
        outputs_data = {
            "type": "image",
            "files": [os.path.basename(save_path)]
        }

        shared_utils.save_generation_log("t2i", inputs_data, params_data, outputs_data, image_path_for_filename=save_path, model_name="glm-4")

        print("--> [T2I Worker] Task Completed.", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR IN T2I WORKER:", flush=True)
        traceback.print_exc()
        sys.exit(1)

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
    parser.add_argument("--scheduler", type=str, default="euler_a")
    args = parser.parse_args()

    run_t2i(args.prompt, args.width, args.height, args.steps, args.guidance, args.seed, args.lora_config, args.top_k, args.temperature, args.model_path, args.clip_skip, args.scheduler)

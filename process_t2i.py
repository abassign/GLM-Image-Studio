import argparse
import torch
import sys
import os
import json
import time
import traceback
import gc
import datetime
import shared_utils
from diffusers import DiffusionPipeline

MODEL_ID = "zai-org/GLM-Image"

# Setup logging
shared_utils.setup_logging()



def run_t2i(prompt, width, height, steps, guidance, seed, lora_config=None, top_k=1, temperature=0.6):
    print(f"--> [T2I Worker] Starting process PID: {os.getpid()}", flush=True)

    try:
        gc.collect()
        torch.cuda.empty_cache()

        print("--> [T2I Worker] Loading Pipeline...", flush=True)
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)

        shared_utils.load_loras(pipe, lora_config)

        print("--> [T2I Worker] Enabling CPU Offload & Tiling...", flush=True)
        pipe.enable_model_cpu_offload()
        try: pipe.enable_vae_tiling()
        except: pass
        try: pipe.enable_attention_slicing("max")
        except: pass

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
            "top_k": top_k, "temperature": temperature
        }

        # Outputs
        outputs_data = {
            "type": "image",
            "files": [os.path.basename(save_path)]
        }

        shared_utils.save_generation_log("t2i", inputs_data, params_data, outputs_data, image_path_for_filename=save_path)

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
    args = parser.parse_args()

    run_t2i(args.prompt, args.width, args.height, args.steps, args.guidance, args.seed, args.lora_config, args.top_k, args.temperature)

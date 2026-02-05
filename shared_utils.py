import os
import json
import logging
import diffusers
from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler
)


# Common Paths
OUTPUT_DIR = "/app/outputs"
LORA_DIR = "/app/loras"

def setup_logging():
    """Configures logging to suppress verbose library warnings."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    diffusers.logging.set_verbosity_error()

def save_json(full_path, data, prefix="System"):
    """
    Generic function to save data to a JSON file.
    """
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"--> [{prefix}] JSON Log saved: {full_path}", flush=True)
    except Exception as e:
        print(f"--> [{prefix} Warning] Failed to save JSON: {e}", flush=True)

def save_generation_log(mode, inputs, params, outputs, image_path_for_filename=None, model_name="unknown"):
    """
    Unified V2 JSON Saver.
    mode: 't2i' | 'i2i' | 'i2t'
    inputs: dict { 'prompt', 'source_images': [], 'loras': [] }
    params: dict { 'width', 'height', ... }
    outputs: dict { 'type', 'files': [], 'text_content': {} }
    image_path_for_filename: usage for determinig filename (optional)
    model_name: str (e.g. "glm-4", "z-image-turbo")
    """
    import datetime
    import uuid
    
    # Determine Filename
    if image_path_for_filename:
        base_name = os.path.basename(image_path_for_filename)
        # If it's an image, swap ext. If it's already a target json name, keep it.
        if base_name.lower().endswith(('.png', '.jpg', '.jpeg')):
             json_filename = os.path.splitext(base_name)[0] + ".json"
        else:
             json_filename = base_name if base_name.endswith(".json") else base_name + ".json"
    else:
        # Fallback timestamp name
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{mode}_{timestamp_str}.json"

    full_path = os.path.join(OUTPUT_DIR, json_filename)

    # V2 Structure
    data = {
        "meta": {
            "version": "2.0",
            "mode": mode,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "job_id": str(uuid.uuid4()),
            "model": model_name
        },
        "inputs": inputs,
        "parameters": params,
        "outputs": outputs
    }

    save_json(full_path, data, prefix=f"{mode.upper()} Worker")

def load_loras(pipe, config_path):
    """
    Loads and fuses LoRA adapters into the pipeline based on the config file.
    """
    if not config_path or not os.path.exists(config_path): 
        return
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        adapters, weights = [], []
        for i, item in enumerate(data):
            # Format expected: [path, strength, active]
            path, strength, active = item[0], float(item[1]), bool(item[2])
            
            if active and os.path.exists(path):
                name = f"lora_{i}"
                print(f"--> [LoRA] 🧩 Loading: {os.path.basename(path)} ({strength})", flush=True)
                pipe.load_lora_weights(path, adapter_name=name)
                adapters.append(name)
                weights.append(strength)
        
        if adapters:
            pipe.set_adapters(adapters, adapter_weights=weights)
            # Force fusion of weights
            try:
                pipe.fuse_lora(adapter_names=adapters, lora_scale=1.0)
                print(f"--> [LoRA] 🔗 Fused {len(adapters)} LoRAs into model weights", flush=True)
            except Exception as e:
                 print(f"--> [LoRA] ⚠️ Fusion Warning: {e}", flush=True)

            print(f"--> [LoRA] ✅ Activated {len(adapters)} LoRAs", flush=True)

    except Exception as e:
        print(f"--> [LoRA] ❌ Error: {e}", flush=True)

def apply_scheduler(pipe, scheduler_name="euler_a"):
    """
    Switches the scheduler of the pipeline.
    Common Schedulers for SDXL/Pony:
    - Euler a (Euler Ancestral) - Great for portraits
    - Euler - Default
    - DPM++ 2M Karras - Generic good quality
    - DPM++ SDE Karras - High quality but slower
    - DDIM - Fast
    """
    if not scheduler_name: return

    s_name = scheduler_name.lower().strip()
    config = pipe.scheduler.config

    try:
        # Check for SD3/Z-Image specific config keys that must be preserved
        # config is a FrozenDict, access via .get is safer than hasattr for keys
        extra_args = {}
        if config.get("use_dynamic_shifting", False):
            extra_args["use_dynamic_shifting"] = True
        
        # SD3/Z-Image Logic: If time_shift_type is present (exponential), 
        # use_dynamic_shifting MUST be True for the assertion to pass.
        is_sd3 = False
        if config.get("time_shift_type", None):
             is_sd3 = True
             extra_args["time_shift_type"] = config.get("time_shift_type")
             extra_args["use_dynamic_shifting"] = True # FORCE TRUE if shifting is used

        # Debug
        if extra_args:
             print(f"--> [Scheduler] Preserving Z-Image/SD3 args: {extra_args}", flush=True)

        # Z-Image / SD3 Compatibility Layer
        # SD3 pipelines pass flow-matching args (like 'mu') that standard schedulers REJECT.
        # We must force a Flow-Matching compatible scheduler.
        if is_sd3:
             if "FlowMatch" not in str(type(pipe.scheduler)):
                 print(f"--> [Scheduler] ⚠️ Z-Image/SD3 Detected. '{s_name}' is incompatible. Forcing FlowMatchEulerDiscreteScheduler.", flush=True)
                 pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(config, **extra_args)
             return # Stop here, do not attempt to apply other schedulers.

        # Standard SDXL/Pony Scheduler Logic
        if s_name == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config, **extra_args)
        
        elif s_name == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(config, **extra_args)
        
        elif s_name == "dpm++_2m_karras" or s_name == "dpm_2m_karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="dpmsolver++", **extra_args)
        
        elif s_name == "dpm++_2m" or s_name == "dpm_2m":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", **extra_args)
        
        elif s_name == "dpm++_sde_karras" or s_name == "dpm_sde_karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", **extra_args)
        
        elif s_name == "ddim":
             pipe.scheduler = DDIMScheduler.from_config(config)
        
        elif s_name == "lms_karras":
             pipe.scheduler = LMSDiscreteScheduler.from_config(config, use_karras_sigmas=True)
        
        print(f"--> [Scheduler] Applied: {s_name} ({type(pipe.scheduler).__name__})", flush=True)
    except Exception as e:
        print(f"--> [Scheduler] Failed to switch to {s_name}: {e}. Keeping default.", flush=True)

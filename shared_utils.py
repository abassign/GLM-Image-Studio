import os
import json
import logging
import diffusers


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

def save_generation_log(mode, inputs, params, outputs, image_path_for_filename=None):
    """
    Unified V2 JSON Saver.
    mode: 't2i' | 'i2i' | 'i2t'
    inputs: dict { 'prompt', 'source_images': [], 'loras': [] }
    params: dict { 'width', 'height', ... }
    outputs: dict { 'type', 'files': [], 'text_content': {} }
    image_path_for_filename: usage for determinig filename (optional)
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
            "job_id": str(uuid.uuid4())
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

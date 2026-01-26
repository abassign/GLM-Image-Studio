import argparse
import torch
import sys
import os
import gc
import re
import json
import datetime
import shared_utils
from PIL import Image
from transformers import AutoProcessor, AutoModel

# Thinking Version (GLM-4.1V)
MODEL_ID = "zai-org/GLM-4.1V-9B-Thinking"

def clean_output_text(text):
    text = text.replace("Ġ", " ").replace("Ċ", "\n")
    text = re.sub(r'\s+([?.!,:;])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def run_i2t(image_path, image_path_2, prompt, top_k, temperature, strength=0.75, mix_ratio=0.5):
    print(f"--> [I2T Worker] Starting Analysis (GLM-4.1V Thinking). PID: {os.getpid()}", flush=True)

    if not os.path.exists(image_path):
        print("ERROR: Image 1 not found", flush=True)
        sys.exit(1)
        
    if image_path_2 and not os.path.exists(image_path_2):
        print("WARNING: Image 2 provided but not found. Ignoring.", flush=True)
        image_path_2 = None

    try:
        gc.collect()
        torch.cuda.empty_cache()

        print(f"--> [I2T Worker] Loading Processor...", flush=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        print(f"--> [I2T Worker] Loading Model...", flush=True)
        # Class Inspection (Robust logic maintained)
        base_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cpu", torch_dtype=torch.bfloat16)
        module_name = base_model.__module__
        remote_module = sys.modules[module_name]
        TargetClass = None
        candidate_names = ["ChatGLMForConditionalGeneration", "Glm4vForCausalLM", "GLM4VForCausalLM", "ModelForCausalLM"]
        for name in candidate_names:
            if hasattr(remote_module, name): TargetClass = getattr(remote_module, name); break
        if TargetClass is None:
            for name, obj in vars(remote_module).items():
                if isinstance(obj, type) and (name.endswith("ForCausalLM") or name.endswith("ForConditionalGeneration")):
                    if obj != type(base_model): TargetClass = obj; break
        if TargetClass is None: raise ValueError(f"Could not find a CausalLM class in {module_name}")
        del base_model
        torch.cuda.empty_cache()

        model = TargetClass.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map="auto"
        ).eval()

        print("--> [I2T Worker] Processing Input...", flush=True)
        image1 = Image.open(image_path).convert("RGB")
        original_dims = image1.size
        
        # BLENDING LOGIC
        if image_path_2:
            print(f"--> [I2T Worker] Loading Second Image for Blending (Mix: {mix_ratio})...", flush=True)
            image2 = Image.open(image_path_2).convert("RGB")
            # Resize image2 to match image1 for blending
            image2 = image2.resize(image1.size, Image.LANCZOS)
            
            # Blend: out = image1 * (1.0 - alpha) + image2 * alpha
            # If mix_ratio is 0.0 -> Image 1 only
            # If mix_ratio is 1.0 -> Image 2 only
            final_image = Image.blend(image1, image2, alpha=float(mix_ratio))
        else:
            final_image = image1
        
        content_list = [{"type": "image", "image": final_image}]
            
        content_list.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content_list}]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to("cuda")

        print(f"--> [I2T Worker] Generating (TopK: {top_k}, Temp: {temperature})...", flush=True)

        # Cast top_k to int because generation expects integer
        k_val = int(top_k)
        
        gen_kwargs = {
            "max_new_tokens": 4096,
            "do_sample": True if temperature > 0 else False, 
            "top_k": k_val, 
            "temperature": temperature
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        outputs = outputs[:, inputs.input_ids.shape[1]:]
        raw_response = processor.decode(outputs[0], skip_special_tokens=True)
        final_response = clean_output_text(raw_response)

        # Custom logic to reconstruct what save_json_log did
        
        # 1. Parsing Thinking vs Answer
        thought = ""
        answer = final_response
        think_match = re.search(r'<think>(.*?)</think>', final_response, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            answer = final_response.replace(think_match.group(0), "").strip()

        answer = answer.replace("<answer>", "").replace("</answer>", "").strip()
        
        # 2. Ancestor Params Merging (Maintain compatibility with old/new schemas if possible)
        ancestor_params = {}
        try:
            base_path = os.path.splitext(image_path)[0]
            potential_json = base_path + ".json"
            if os.path.exists(potential_json):
                with open(potential_json, 'r') as f:
                    ancestor_data = json.load(f)
                    # Support v1 and v2
                    ancestor_params = ancestor_data.get("parameters", {})
        except Exception as e:
            print(f"[I2T Worker] Could not load ancestor params: {e}")

        # Params
        final_params = ancestor_params.copy()
        final_params.update(gen_kwargs)
        final_params["strength"] = strength
        final_params["mix_ratio"] = mix_ratio
        final_params["original_width"] = original_dims[0]
        final_params["original_height"] = original_dims[1]

        # Inputs
        inputs_data = {
            "prompt": prompt,
            "source_images": [os.path.basename(image_path)]
        }
        if image_path_2:
            inputs_data["source_images"].append(os.path.basename(image_path_2))
            
        # Outputs
        outputs_data = {
            "type": "text",
            "text_content": {
                "thinking_process": thought,
                "final_answer": answer,
                "raw_full_response": final_response
            }
        }
        
        # Filename construction handled by shared_utils if we pass a target name
        # We want i2t_...json
        base_name = os.path.basename(image_path)
        safe_name = base_name.replace(" ", "_")
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_json_name = f"i2t_{safe_name}_{timestamp_str}.json"
        
        shared_utils.save_generation_log("i2t", inputs_data, final_params, outputs_data, image_path_for_filename=target_json_name)
 
        print(f"TEXT_OUTPUT_START")
        print(final_response)
        print(f"TEXT_OUTPUT_END")

        print("--> [I2T Worker] Analysis Completed.", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {str(e)}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--image_path_2", type=str, default=None) # New Argument
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    # NEW ARGUMENTS
    parser.add_argument("--top_k", type=float, default=1.0) 
    parser.add_argument("--temperature", type=float, default=0.6) # RESTORED
    parser.add_argument("--strength", type=float, default=0.75) 
    parser.add_argument("--mix_ratio", type=float, default=0.5)

    args = parser.parse_args()

    run_i2t(args.image_path, args.image_path_2, args.prompt, args.top_k, args.temperature, args.strength, args.mix_ratio)

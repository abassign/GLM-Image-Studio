import gradio as gr
import torch
import os
import sys
import random
from datetime import datetime
from PIL import Image

# --- INITIAL SETUP ---
MODEL_ID = "zai-org/GLM-Image"
OUTPUT_DIR = "/app/outputs"

print(f"--> [Init] Loading GLM-Image model...")

try:
    from diffusers.pipelines.glm_image import GlmImagePipeline
    PipelineClass = GlmImagePipeline
except ImportError:
    from diffusers import DiffusionPipeline
    PipelineClass = DiffusionPipeline

try:
    pipe = PipelineClass.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Use Model CPU Offload for standard stability.
    pipe.enable_model_cpu_offload()
    print("--> [Init] Model loaded. Callback Fixed.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---

def round_32(x):
    """Rounds to nearest multiple of 32 (GLM architecture requirement)."""
    return int(round(x / 32) * 32)

def is_valid_image(image):
    """
    Returns True only if image exists AND is larger than 32x32.
    """
    if image is None:
        return False
    w, h = image.size
    if w < 32 or h < 32:
        return False
    return True

# --- REACTIVE UI CALLBACKS ---

def update_ui_state(image):
    """Updates Status Header and sliders based on image."""
    if not is_valid_image(image):
        status_md = "## ⚪ Text to Image Mode (Active)"
        return status_md, 1024, 1024

    status_md = "## 🟢 Image to Image Mode (Active)"
    w, h = image.size
    aspect_ratio = w / h

    if w > h:
        new_w = 1024
        new_h = round_32(new_w / aspect_ratio)
    else:
        new_h = 1024
        new_w = round_32(new_h * aspect_ratio)

    return status_md, new_w, new_h

def sync_height(width, image, keep_ratio):
    if not is_valid_image(image) or not keep_ratio: return gr.update()
    aspect_ratio = image.size[0] / image.size[1]
    return round_32(width / aspect_ratio)

def sync_width(height, image, keep_ratio):
    if not is_valid_image(image) or not keep_ratio: return gr.update()
    aspect_ratio = image.size[0] / image.size[1]
    return round_32(height * aspect_ratio)

    return round_32(height * aspect_ratio)

def shutdown():
    print("--> Shutting down GLM-Image Studio...")
    os._exit(0)

# --- MAIN GENERATION LOGIC ---

def generate_image(prompt, input_image, width, height, steps, guidance, seed, randomize_seed, progress=gr.Progress()):

    # 1. Mode Detection
    if is_valid_image(input_image):
        mode = "Image-to-Image"
        valid_image = input_image
    else:
        mode = "Text-to-Image"
        valid_image = None

    # 2. Seed
    if randomize_seed or seed < 0:
        seed = int(random.randint(0, 2**32 - 1))
    else:
        seed = int(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    target_w = int(width)
    target_h = int(height)
    steps = int(steps)

    print(f"--> [Request] Mode: {mode} | Size: {target_w}x{target_h} | Steps: {steps}")

    # 3. FIX: Updated Callback Signature
    # The pipeline now passes 4 arguments: (pipe, step_index, timestep, callback_kwargs)
    def callback_dynamic(pipe, step_index, timestep, callback_kwargs):
        progress(step_index / steps, desc=f"Generating step {step_index}/{steps}...")
        return callback_kwargs

    # 4. Pipeline Arguments
    pipe_args = {
        "prompt": prompt,
        "height": target_h,
        "width": target_w,
        "num_inference_steps": steps,
        "guidance_scale": float(guidance),
        "generator": generator,
        "callback_on_step_end": callback_dynamic,
        # Note: We keep this just in case, though new callbacks use callback_kwargs
        "callback_on_step_end_tensor_inputs": ["latents"]
    }

    # 5. Image-to-Image Logic (Manual Memory Fix)
    if valid_image is not None:
        try:
            print("--> [Memory] Manually moving Vision Encoder to GPU...")
            pipe.vision_language_encoder.to("cuda")

            img_prep = valid_image.convert("RGB")
            img_prep = img_prep.resize((target_w, target_h), Image.LANCZOS)
            pipe_args["image"] = [img_prep]

        except Exception as e:
            return None, f"Error preparing image: {e}", seed

    # 6. Inference
    try:
        with torch.no_grad():
            image = pipe(**pipe_args).images[0]

        # Cleanup
        if valid_image is not None:
            print("--> [Memory] Releasing Vision Encoder to CPU...")
            pipe.vision_language_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 7. Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "i2i" if valid_image is not None else "t2i"
        filename = f"glm_{prefix}_{timestamp}_s{seed}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        image.save(save_path)

        return save_path, f"Mode: {mode} | Saved: {filename}", seed

    except Exception as e:
        # Emergency Cleanup
        if valid_image is not None:
             try:
                 pipe.vision_language_encoder.to("cpu")
                 torch.cuda.empty_cache()
             except: pass

        import traceback
        traceback.print_exc()
        return None, f"GENERATION ERROR: {str(e)}", seed

# --- GRADIO INTERFACE ---
css = """
.no-progress .progress-level { display: none !important; }
.no-progress .meta-text { display: none !important; }
"""
with gr.Blocks(title="GLM-Image Studio", theme=gr.themes.Soft(), css=css) as demo:

    gr.Markdown("# 🎨 GLM-Image Studio")
    gr.Markdown("Hybrid Txt2Img & Img2Img Engine on AMD Radeon 7900 XTX")

    with gr.Row():
        with gr.Column(scale=4):
            prompt_input = gr.Textbox(
                label="Prompt",
                lines=3,
                value="A futuristic cyberpunk city, neon lights, rain, 8k, masterpiece"
            )

            with gr.Group():
                status_header = gr.Markdown("## ⚪ Text to Image Mode (Active)")
                input_image_box = gr.Image(label="Input Image (Drag & Drop)", type="pil", height=300)
                keep_ratio_chk = gr.Checkbox(label="Keep Aspect Ratio", value=True, info="Locks dimensions (I2I only)")

            with gr.Row():
                width_slider = gr.Slider(512, 1536, value=1024, step=32, label="Target Width")
                height_slider = gr.Slider(512, 1536, value=1024, step=32, label="Target Height")

            with gr.Group():
                gr.Markdown("### ⚙️ Quality Settings")
                with gr.Row():
                    steps_slider = gr.Slider(10, 60, value=50, step=1, label="Steps")
                    guidance_slider = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="Guidance")
                with gr.Row():
                    seed_input = gr.Number(value=-1, label="Seed", precision=0, scale=3)
                    random_btn = gr.Checkbox(label="🎲 Randomize", value=True, scale=1)

            with gr.Row():
                btn_generate = gr.Button("🚀 GENERATE", variant="primary", size="lg", scale=2)
                btn_stop = gr.Button("⏹️ STOP", variant="stop", size="lg", scale=1)
                btn_exit = gr.Button("❌ EXIT", variant="secondary", size="lg", scale=1)

        with gr.Column(scale=5):
            image_output = gr.Image(label="Result", type="filepath", interactive=False)
            with gr.Row():
                info_output = gr.Textbox(label="System Log", interactive=False, scale=3, elem_classes=["no-progress"])
                seed_output = gr.Number(label="Seed Used", interactive=False, scale=1, elem_classes=["no-progress"])

    # Events
    input_image_box.change(fn=update_ui_state, inputs=[input_image_box], outputs=[status_header, width_slider, height_slider])

    width_slider.input(fn=sync_height, inputs=[width_slider, input_image_box, keep_ratio_chk], outputs=[height_slider])
    height_slider.input(fn=sync_width, inputs=[height_slider, input_image_box, keep_ratio_chk], outputs=[width_slider])
    keep_ratio_chk.change(fn=sync_height, inputs=[width_slider, input_image_box, keep_ratio_chk], outputs=[height_slider])

    keep_ratio_chk.change(fn=sync_height, inputs=[width_slider, input_image_box, keep_ratio_chk], outputs=[height_slider])

    gen_event = btn_generate.click(
        fn=generate_image,
        inputs=[prompt_input, input_image_box, width_slider, height_slider, steps_slider, guidance_slider, seed_input, random_btn],
        outputs=[image_output, info_output, seed_output]
    )
    
    btn_stop.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])
    btn_exit.click(fn=shutdown, inputs=None, outputs=None)

print("--> Starting Web Server on port 7860...")
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)

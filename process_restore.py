import sys
import os
import argparse
import cv2
import glob
import torch
from pathlib import Path

# Add local libs to path
libs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs')
sys.path.append(libs_path)

# Print for debugging
print(f"--> [Restore] Added {libs_path} to sys.path", flush=True)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import restoration libraries: {e}")
    sys.exit(1)

def run_restore(image_path, scale=4, face_enhance=False, model_name='RealESRGAN_x4plus'):
    print(f"--> [Restore] Starting restoration on {image_path}", flush=True)
    print(f"--> [Restore] Scale: {scale}, Face Enhance: {face_enhance}", flush=True)

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}", flush=True)
        return

    # 1. Setup RealESRGAN
    # model_name options: RealESRGAN_x4plus, RealESRNet_x4plus, RealESRGAN_x4plus_anime_6B, RealESRGAN_x2plus
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    else:
        # Default to x4plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

    # Use half precision if supported, else float32
    # On ROCm, sometimes float32 is safer for these models too.
    # Let's try half=True first (default), or user can toggle?
    # Given previous issues, maybe half=False (float32) is safer for consistence.
    half_precision = False 

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_url, # Pass URL so it downloads automatically checks cache
        model=model,
        tile=512, # Use tiling to prevent OOM (79GB allocation issue)
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
        gpu_id=0 # 0 for first GPU
    )

    # 2. Setup GFPGAN if needed
    face_enhancer = None
    if face_enhance:
        print("--> [Restore] Loading GFPGAN for face enhancement...", flush=True)
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    # 3. Read Image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("ERROR: Failed to read image using cv2", flush=True)
        return

    # 4. Inference
    try:
        if face_enhancer:
            # GFPGAN inference (handles background upscale too via bg_upsampler)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            # RealESRGAN only
            output, _ = upsampler.enhance(img, outscale=scale)
    except Exception as e:
        print(f"ERROR during inference: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # 5. Save Output
    base_name = os.path.basename(image_path)
    filename, ext = os.path.splitext(base_name)
    out_filename = f"{filename}_restore{ext}"
    output_path = os.path.join("/app/outputs", out_filename)
    
    cv2.imwrite(output_path, output)
    print(f"SUCCESS_OUTPUT:{output_path}", flush=True)
    print("--> [Restore] Finished.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--scale", type=float, default=4) # User might ask for 2x
    parser.add_argument("--face_enhance", action='store_true')
    # Use boolean string parsing
    
    args = parser.parse_args()
    
    # Handle scale mapping if needed (RealESRGAN usually x2 or x4)
    # If user requests 3x, we might upscale x4 and resize? Or just x4.
    # For now, stick to x4plus which is x4.
    
    run_restore(args.image_path, scale=args.scale, face_enhance=args.face_enhance)

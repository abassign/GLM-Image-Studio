import os
import subprocess
import signal
import sys
import time
import select
import shutil
import random
import json
from huggingface_hub import HfApi
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import lora_manager
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI()

# Note: 'uploads' removed as per cleanup
# Note: 'uploads' removed as per cleanup
DIRS = ["/app/static", "/app/outputs", "/app/models"]
for d in DIRS: 
    os.makedirs(d, exist_ok=True)
    try:
        os.chmod(d, 0o777)
    except Exception as e:
        print(f"Warning: Could not set permissions for {d}: {e}")

# Initialize models dir if empty
models_readme = os.path.join("/app/models", "README.md")
if not os.path.exists(models_readme):
    try:
        with open(models_readme, "w") as f:
            f.write("# Models Directory\n\nPlace your .safetensors models here, or use the UI to download them.\nsubdirectories are also supported.")
    except: pass

app.mount("/static", StaticFiles(directory="/app/static"), name="static")
app.mount("/outputs", StaticFiles(directory="/app/outputs"), name="outputs")

current_process = None

# --- Helper Functions ---
def kill_server():
    print("--> [System] Shutdown initiated...", flush=True)
    time.sleep(1)
    print("--> [System] Killing process tree...", flush=True)
    try:
        # Initial attempt: Stop the Uvicorn reloader (Parent)
        os.kill(os.getppid(), signal.SIGTERM)
    except Exception:
        pass
    
    # Force kill self
    os._exit(0)

    # Force kill self
    os._exit(0)

# --- Model Management ---
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Warning: huggingface_hub not installed. Model auto-download may fail.")

def run_process(cmd):
    global current_process
    print(f"--> [Server] run_process called with: {' '.join(cmd)}", flush=True)
    if current_process:
        try:
            current_process.terminate()
            current_process.wait(timeout=1)
        except: pass
    
    # We use independent process via subprocess
    # BUT we need to forward output to our stdout so uvicorn captures it for the UI log.
    current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Forward logs in background
    import threading
    def log_forwarder(proc):
        for line in proc.stdout:
            print(line, end='', flush=True)
    
    t = threading.Thread(target=log_forwarder, args=(current_process,))
    t.daemon = True
    t.start()


def scan_models(model_type_filter=None):
    """
    Scans /app/models recursively for dataModels.json.
    If matching 'z-image' model found but folder is empty (no model_index.json or .safetensors),
    attempt to download the Base model from Hugging Face.
    Returns the resolved local path.
    """
    models_root = "/app/models"
    if not os.path.exists(models_root):
        return None
    
    for root, dirs, files in os.walk(models_root):
        if "dataModels.json" in files:
            try:
                json_path = os.path.join(root, "dataModels.json")
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                if model_type_filter and data.get("type") == model_type_filter:
                    # Found config. Check model content.
                    raw_path = data.get("path", ".")
                    if raw_path.startswith("./"):
                        abs_path = os.path.join(root, raw_path[2:])
                    elif raw_path.startswith("/"):
                        abs_path = raw_path
                        abs_path = os.path.join(root, raw_path)
                        
                    # Auto-create 'loras' subdirectory for this model (Model-Specific LoRAs)
                    model_dir = os.path.dirname(abs_path) if os.path.isfile(abs_path) else abs_path
                    lora_dir = os.path.join(model_dir, "loras")
                    if not os.path.exists(lora_dir):
                        try:
                            os.makedirs(lora_dir, exist_ok=True)
                            os.chmod(lora_dir, 0o777)
                            print(f"--> [Server] Auto-created LoRA directory for model: {lora_dir}")
                        except Exception as e:
                            print(f"--> [Server] Warning: Could not create LoRA dir {lora_dir}: {e}")

                    # --- Z-IMAGE AUTOLOAD LOGIC ---
                    # Check if model files exist. Diffusers usually needs model_index.json
                    # Or specific files user mentioned: qwen_3_4b, z_image_bf16, ae
                    # We check for general presence of large files or index.
                    
                    has_content = False
                    if os.path.exists(abs_path):
                         # check simple content
                         content = os.listdir(abs_path)
                         # Filter out json/hidden files to see if weights exist
                         weights = [x for x in content if x.endswith(".safetensors") or x.endswith(".bin") or x == "model_index.json"]
                         if len(weights) > 0: has_content = True

                    if not has_content and data.get("type") == "z-image":
                        print(f"--> [Server] Z-Image folder found at {abs_path} but appears empty.")
                        print(f"--> [Server] Initiating Auto-Download of Z-Image Base (Tongyi-MAI/Z-Image)...")
                        try:
                            # We download to the root of that model folder
                            snapshot_download(
                                repo_id="Tongyi-MAI/Z-Image",
                                local_dir=abs_path,
                                local_dir_use_symlinks=False,
                                resume_download=True
                            )
                            print("--> [Server] Download Completed.")
                            # Update JSON if needed? No, path './' is still valid.
                        except Exception as dl_err:
                            print(f"--> [Server] Download Failed: {dl_err}")
                    
                    # Return path if exists (or now exists)
                    if os.path.exists(abs_path):
                        return abs_path

            except Exception as e:
                print(f"Error reading/processing model config at {root}: {e}")
                
    return None

def scan_for_text_encoders(model_path):
    """
    Scans for text encoders (LLMs) associated with a model.
    Looks in:
    1. model_path/text_encoders/*.gguf
    2. model_path/../text_encoders/*.gguf (sibling folder)
    3. Global search in models types
    """
    if not model_path or not os.path.exists(model_path):
        return []

    encoders = []
    
    # helper to add unique
    seen = set()
    def add_enc(path, source):
        if path not in seen:
            encoders.append({"path": path, "name": os.path.basename(path), "source": source})
            seen.add(path)

    # 1. Direct subfolder
    sub_te = os.path.join(model_path, "text_encoders")
    if os.path.isdir(sub_te):
        for f in os.listdir(sub_te):
            if f.endswith(".gguf") and "qwen" in f.lower():
                 add_enc(os.path.join(sub_te, f), "Embedded")

    # 2. Sibling folder (common if model_path is a specific file or subfolder)
    # If model_path is .../Z-Image-Turbo-GGUF/z_image.gguf
    parent = os.path.dirname(model_path)
    sibling_te = os.path.join(parent, "text_encoders")
    if os.path.isdir(sibling_te):
        for f in os.listdir(sibling_te):
            if f.endswith(".gguf") and "qwen" in f.lower():
                 add_enc(os.path.join(sibling_te, f), "Folder")

    # 3. Z-Image-Turbo-GGUF common location
    common_te = "/app/models/Z-Image-Turbo-GGUF/text_encoders"
    if os.path.isdir(common_te):
        for f in os.listdir(common_te):
             if f.endswith(".gguf") and "qwen" in f.lower():
                 add_enc(os.path.join(common_te, f), "Shared")

    return encoders

# --- Pydantic Models ---

class LoraItem(BaseModel):
    folder: str
    filename: str
    strength: float

class GenRequest(BaseModel):
    mode: str = "t2i"
    prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    randomize: bool
    init_image: Optional[str] = None
    init_image_2: Optional[str] = None # New secondary image
    loras: List[LoraItem]
    top_k: Optional[float] = 1.0       
    temperature: Optional[float] = 0.6 
    strength: Optional[float] = 0.75
    mix_ratio: Optional[float] = 0.5
    noise_level: Optional[int] = 20 # For Upscale
    scale: Optional[float] = 4.0    # For Upscale
    face_enhance: Optional[bool] = False # For Restoration
    model_path: Optional[str] = None # Added for dynamic model selection
    clip_skip: Optional[int] = 1 # Added for Pony/Anime models
    scheduler: Optional[str] = "euler" # Added for Sampler Selection
    llm_path: Optional[str] = None # Added for Manual Text Encoder Selection
    save_log: Optional[bool] = False # Added for Debug Logging

class AnalyzeRequest(BaseModel):
    image_path: str
    image_path_2: Optional[str] = None 
    prompt: str
    top_k: int = 1
    temperature: float = 0.6
    strength: Optional[float] = 0.75
    mix_ratio: Optional[float] = 0.5

class DeleteRequest(BaseModel):
    filename: str

# --- Endpoints ---

@app.post("/api/text_encoders")
async def get_text_encoders(payload: dict):
    model_path = payload.get("model_path")
    encoders = scan_for_text_encoders(model_path)
    return {"encoders": encoders}

@app.get("/")
async def read_index():
    return RedirectResponse(url="/static/index.html")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save to outputs instead of uploads
        output_dir = "/app/outputs"
        base_name, ext = os.path.splitext(file.filename)
        
        counter = 1
        new_filename = file.filename
        
        # Collision detection loop
        while os.path.exists(os.path.join(output_dir, new_filename)):
            new_filename = f"{base_name}({counter}){ext}"
            counter += 1
            
        file_path = os.path.join(output_dir, new_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"path": file_path, "url": f"/outputs/{new_filename}"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/scan_loras")
async def scan_loras(payload: dict):
    folder = payload.get("folder", "/app/loras")
    files = lora_manager.list_lora_files(folder)
    return {"files": files}

@app.get("/api/history")
async def get_history():
    """Returns a list of generated images and their metadata, scanning JSONs first."""
    output_dir = "/app/outputs"
    history = []
    
    if not os.path.exists(output_dir):
        return {"history": []}

    try:
        # Scan for JSON files first (Source of Truth)
        json_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.json')]
        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)

        for json_filename in json_files:
            json_path = os.path.join(output_dir, json_filename)
            base_name = os.path.splitext(json_filename)[0]
            
            # --- STRICT MODE DETECTION ---
            inferred_mode = "unk"
            lower_name = json_filename.lower()
            if lower_name.startswith("i2t_"): inferred_mode = "i2t"
            elif lower_name.startswith("t2i_"): inferred_mode = "t2i"
            elif lower_name.startswith("i2i_"): inferred_mode = "i2i"

            # Load JSON
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error parsing {json_filename}: {e}")
                continue

            meta = data.get("meta", {})
            raw_params = data.get("parameters", {})
            raw_inputs = data.get("inputs", data.get("input", {}))
            raw_outputs = data.get("outputs", data.get("output", {}))

            params = raw_params.copy()
            
            # Mode detection
            if "mode" in meta: params["mode"] = meta["mode"]
            elif "mode" not in params:
                 if inferred_mode != "unk": params["mode"] = inferred_mode
                 else: params["mode"] = "unk"

            # Inject Inputs
            if "prompt" in raw_inputs: params["prompt"] = raw_inputs["prompt"]
            
            # Handle Source Images
            if "source_images" in raw_inputs and isinstance(raw_inputs["source_images"], list):
                for i, img in enumerate(raw_inputs["source_images"]):
                    key = f"source_image_{i+1}"
                    # Ensure full path mapping if needed, mostly backend handling
                    full_img_path = os.path.join(output_dir, img) if not img.startswith("/") else img
                    params[key] = full_img_path
                    
            elif "source_image" in raw_inputs: # V1 Legacy
                 if "source_image_1" not in params:
                     params["source_image_1"] = raw_inputs["source_image"]
            
            # Independent check for direct keys (Fix for I2I CPP)
            for k in ["source_image_1", "source_image_2", "source_image_3", "source_image_4"]:
                if k in raw_inputs: params[k] = raw_inputs[k]

            # Determine Image Filename for Display
            image_filename = None
            if "files" in raw_outputs and isinstance(raw_outputs["files"], list) and len(raw_outputs["files"]) > 0:
                image_filename = os.path.basename(raw_outputs["files"][0])
            
            if not image_filename:
                if "filename" in meta:
                    candidate = meta["filename"] 
                    if os.path.exists(os.path.join(output_dir, candidate)):
                        image_filename = candidate
                if not image_filename:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        if os.path.exists(os.path.join(output_dir, base_name + ext)):
                            image_filename = base_name + ext
                            break

            image_url = None
            display_filename = None

            if image_filename:
                image_url = f"/outputs/{image_filename}"
                display_filename = image_filename
            else:
                # Fallback for I2T (Source Image as Thumbnail)
                if params.get("mode") == "i2t":
                    src = params.get("source_image_1")
                    if src:
                        if src.startswith("/app"): image_url = src.replace("/app", "")
                        elif not src.startswith("/"): image_url = f"/outputs/{src}"
                        else: image_url = src 
                        display_filename = json_filename 
                    else:
                        image_url = "https://placehold.co/100/000000/00FF00?text=I2T"
                        display_filename = json_filename
                else:
                    continue
            
            if "prompt" not in params and "prompt" in raw_inputs:
                params["prompt"] = raw_inputs["prompt"]

            item = {
                "image": image_url,
                "filename": display_filename,
                "timestamp": os.path.getmtime(json_path),
                "params": params,
                "output": raw_outputs
            }
            history.append(item)
            
        return {"history": history}
    except Exception as e:
        print(f"History scan error: {e}")
        return {"history": []}

@app.post("/api/stop")
async def stop_process():
    global current_process
    if current_process:
        print("--> [System] Stopping process...")
        current_process.terminate()
        try:
            current_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            current_process.kill()
        current_process = None
        return {"status": "stopped"}
    return {"status": "no_process"}

@app.post("/api/exit")
async def exit_app(background_tasks: BackgroundTasks):
    global current_process
    if current_process: current_process.terminate()
    background_tasks.add_task(kill_server)
    return {"status": "exiting"}

@app.post("/api/delete_history")
async def delete_history(req: DeleteRequest):
    if not req.filename:
        return JSONResponse(content={"error": "No filename provided"}, status_code=400)
    
    safe_name = os.path.basename(req.filename) 
    file_path = os.path.join("/app/outputs", safe_name)
    base_name = os.path.splitext(safe_name)[0]
    json_path = os.path.join("/app/outputs", base_name + ".json")
    
    if safe_name.endswith(".json"):
        json_path = file_path
        for ext in [".png", ".jpg", ".jpeg"]:
            img = os.path.join("/app/outputs", base_name + ext)
            if os.path.exists(img):
                os.remove(img)
                break
    else:
        if os.path.exists(file_path): os.remove(file_path)

    try:
        if os.path.exists(json_path): os.remove(json_path)
        return {"status": "deleted"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/models")
async def list_models():
    """Lists all models found in /app/models, filtering internal files and finding GGUF."""
    models = []
    root_dir = "/app/models"
    
    if os.path.exists(root_dir):
        # Pass 1: Identitfy Diffusers Roots to prevent listing their internal files
        diffusers_roots = set()
        for r, d, f in os.walk(root_dir):
            if "model_index.json" in f:
                 diffusers_roots.add(r)
        
        for r, d, f in os.walk(root_dir):
            rel_path = os.path.relpath(r, root_dir)
            if rel_path == ".": rel_path = ""
            
            # Check if we are inside a diffusers internal folder (subdirectory of a root)
            is_internal = False
            for root in diffusers_roots:
                if r != root and r.startswith(root):
                    is_internal = True
                    break

            # 1. Standard Folder Models (Diffusers)
            if "model_index.json" in f:
                # This is a model root
                try:
                    # Try reading metadata if exists
                    display_name = rel_path if rel_path else os.path.basename(r)
                    if "dataModels.json" in f:
                        with open(os.path.join(r, "dataModels.json"), 'r') as jf:
                            data = json.load(jf)
                            if "name" in data: display_name = data["name"]
                    
                    models.append({
                        "id": rel_path if rel_path else os.path.basename(r),
                        "name": display_name,
                        "path": r,
                        "type": "diffusers",
                        "repo_id": rel_path
                    })
                except: pass

            # 2. File Scanning (Safetensors & GGUF)
            for file in f:
                # GGUF: Always include, even if internal
                if file.endswith(".gguf"):
                     # EXCLUDE LLMs (Text Encoders) from main list
                     if "text_encoder" in rel_path or "text_encoder" in file:
                         print(f"--> [System] Skipping Text Encoder in Model List: {rel_path}/{file}", flush=True)
                         continue

                     m_id = os.path.join(rel_path, file)
                     models.append({
                        "id": m_id,
                        "name": f"{rel_path}/{file}" if rel_path else file,
                        "path": os.path.join(r, file),
                        "type": "gguf",
                        "repo_id": rel_path
                    })
                
                # Safetensors: Only include if NOT internal diffusers file
                # If we are in a diffusers root, it's fine (UNLESS it's the internal diffusion_pytorch_model.safetensors which is usually handled by the folder)
                # Actually, standard Diffusers VAE/UNet .safetensors inside subfolders should be hidden.
                elif file.endswith(".safetensors"):
                    if is_internal: continue # Skip internal weights
                    if "model_index.json" in f: continue # Skip main weights if folder is listed
                    
                    m_id = os.path.join(rel_path, file)
                    models.append({
                        "id": m_id,
                        "name": f"{rel_path}/{file}" if rel_path else file,
                        "path": os.path.join(r, file),
                        "type": "custom_file",
                        "repo_id": rel_path
                    })

    return {"models": models}

class DeleteModelRequest(BaseModel):
    model_id: str

@app.post("/api/delete_model")
async def delete_model(req: DeleteModelRequest):
    """
    Deletes a model (file or directory) based on its relative ID.
    Securely ensures targeting only files within /app/models.
    """
    root_dir = "/app/models"
    
    # basic security check against traversal
    if ".." in req.model_id or req.model_id.startswith("/"):
        return JSONResponse(content={"error": "Invalid model ID"}, status_code=400)

    target_path = os.path.join(root_dir, req.model_id)
    target_path = os.path.abspath(target_path)
    
    if not target_path.startswith(root_dir):
        return JSONResponse(content={"error": "Access denied: Path outside models directory"}, status_code=403)
    
    if not os.path.exists(target_path):
        return JSONResponse(content={"error": "Model not found"}, status_code=404)
    
    try:
        if os.path.isdir(target_path):
            print(f"--> [System] Deleting model directory: {target_path}", flush=True)
            shutil.rmtree(target_path)
        else:
            print(f"--> [System] Deleting single model file: {target_path}", flush=True)
            os.remove(target_path)
            
        return {"status": "success", "message": f"Deleted {req.model_id}"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



class InspectModelRequest(BaseModel):
    repo_id: str
    token: Optional[str] = None

@app.post("/api/inspect_model")
async def inspect_model(req: InspectModelRequest):
    """
    Lists files in a Hugging Face repository with sizes.
    """
    try:
        api = HfApi(token=req.token)
        # model_info returns siblings with rfilename and size (blob LFS size)
        info = api.model_info(req.repo_id, files_metadata=True)
        files = []
        for s in info.siblings:
            files.append({
                "name": s.rfilename,
                "size": s.size if s.size is not None else 0
            })
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}

class DownloadModelRequest(BaseModel):
    repo_id: str
    token: str = None
    filenames: list[str] = None
    subdirectory: str = None

@app.post("/api/download_model")
async def download_model(req: DownloadModelRequest):
    """
    Downloads a model using a subprocess worker to capture logs in UI.
    """
    repo_name = req.repo_id.split("/")[-1]
    
    # Determine Target Directory
    if req.subdirectory:
        # User specified target (e.g. "Z-Image-Turbo/transformer/gguf")
        # Validate path
        if ".." in req.subdirectory or req.subdirectory.startswith("/"):
             return JSONResponse(content={"error": "Invalid subdirectory path"}, status_code=400)
        
        target_dir = os.path.join("/app/models", req.subdirectory)
    else:
        # Default: /app/models/{repo_name}
        # If it is a URL, we need a broad folder name
        if req.repo_id.startswith("http"):
            # Use Civitai ID or hash
            import hashlib
            slug = hashlib.md5(req.repo_id.encode()).hexdigest()[:8]
            # Try to be more descriptive if possible
            if "civitai" in req.repo_id and "/models/" in req.repo_id:
                 try:
                     # Extract ID
                     cid = req.repo_id.split("/models/")[1].split("?")[0].split("/")[0]
                     slug = f"Civitai-{cid}"
                 except: pass
            
            repo_name = slug
            target_dir = os.path.join("/app/models", repo_name)
        else:
            target_dir = os.path.join("/app/models", repo_name)
    
    if os.path.exists(target_dir):
        # Only skip if NOT doing selective download (which might add files)
        # AND if we are not downloading into a specific subdirectory (merging)
        is_merging = req.subdirectory is not None
        if not req.filenames and not is_merging and len(os.listdir(target_dir)) > 2:
            return {"status": "exists", "path": target_dir}

    cmd = [sys.executable, "-u", "download_worker.py", "--repo_id", req.repo_id]
    if req.token:
        cmd.extend(["--token", req.token])
    if req.filenames:
        cmd.append("--allow_patterns")
        cmd.extend(req.filenames)
    
    # Pass explicit target directory
    cmd.extend(["--target_dir", target_dir])

    # Run as active process so logs stream to UI
    run_process(cmd)
    
    return {"status": "loading", "message": "Download started (check logs)"}

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    global current_process
    if not req.image_path or not os.path.exists(req.image_path):
        return JSONResponse(content={"error": f"Image not found: {req.image_path}"}, status_code=400)

    cmd = [
        sys.executable, "-u", "process_i2t.py",
        "--image_path", req.image_path,
        "--prompt", req.prompt,
        "--top_k", str(req.top_k),
        "--temperature", str(req.temperature),
        "--strength", str(req.strength if req.strength is not None else 0.75),
        "--mix_ratio", str(req.mix_ratio if req.mix_ratio is not None else 0.5)
    ]
    
    if req.image_path_2 and os.path.exists(req.image_path_2):
        cmd.extend(["--image_path_2", req.image_path_2])

    def event_generator():
        global current_process
        try:
            current_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=os.environ.copy()
            )
            start_t = time.time()
            buffer_text = False
            while True:
                if current_process is None: break 
                reads = [current_process.stdout.fileno()]
                ret = select.select(reads, [], [], 0.1)
                if ret[0]:
                    line = current_process.stdout.readline()
                    if line:
                        clean = line.strip()
                        if "TEXT_OUTPUT_START" in clean:
                            buffer_text = True
                            continue
                        elif "TEXT_OUTPUT_END" in clean:
                            buffer_text = False
                            continue
                        if buffer_text: yield f"data: TXT|{line}\n\n"
                        elif clean: yield f"data: LOG|{clean}\n\n"
                if current_process.poll() is not None: break
            
            if current_process.returncode == 0: yield f"data: DONE|Finished in {time.time()-start_t:.1f}s\n\n"
            else: yield f"data: ERR|Process exited with code {current_process.returncode}\n\n"
        except Exception as e:
            yield f"data: ERR|Server Error: {str(e)}\n\n"
        finally:
            current_process = None

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/stop")
async def stop_generation():
    global current_process
    print("--> [Server] STOP REQUEST RECEIVED", flush=True)
    if current_process:
        try:
            current_process.terminate()
            # aggressive kill if needed?
            # current_process.kill()
            return {"message": "Process terminated"}
        except Exception as e:
            return {"error": str(e)}
    return {"message": "No process running"}

@app.post("/api/generate")
async def generate(req: GenRequest):
    global current_process
    if req.randomize or req.seed == -1: final_seed = random.randint(0, 2**32-1)
    else: final_seed = req.seed

    print(f"--> [Server] Request Mode: {req.mode} | Noise: {req.noise_level} | Scale: {req.scale}", flush=True)
    print(f"--> [Server] Model Path: {req.model_path} | Scheduler: {req.scheduler}", flush=True)

    lora_data = [{"path": os.path.join(l.folder, l.filename), "strength": l.strength} for l in req.loras]
    config_path = lora_manager.create_config_json(lora_data)

    cmd = []
    if req.mode == "i2i":
        if not req.init_image or not os.path.exists(req.init_image):
             return JSONResponse(content={"error": "Init image required"}, status_code=400)
        cmd = ["python", "-u", "process_i2i.py", "--prompt", req.prompt, "--image_path", req.init_image, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature), "--strength", str(req.strength), "--mix_ratio", str(req.mix_ratio), "--clip_skip", str(req.clip_skip or 1)]
        if req.init_image_2 and os.path.exists(req.init_image_2):
            cmd.extend(["--image_path_2", req.init_image_2])
        if req.model_path:
            cmd.extend(["--model_path", req.model_path])
        if req.scheduler:
             cmd.extend(["--scheduler", req.scheduler])
    elif req.mode == "z_t2i":
        # Check for local Z-Image model
        local_z_path = req.model_path if req.model_path else scan_models("z-image")
        
        # Default Z-Image steps to 10 if not explicitly high, but user controls it
        cmd = ["python", "-u", "process_zimage.py", "--prompt", req.prompt, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature)]
        # Z-Image might not support clip_skip natively yet, skipping for now unless Z-Image pipeline supports it.
        
        if local_z_path:
             print(f"--> [Server] Using Local Z-Image Model: {local_z_path}")
             cmd.extend(["--model_path", local_z_path, "--model_type", "standard"]) # Assume standard if local base provided

    elif req.mode == "z_i2i":
        if not req.init_image or not os.path.exists(req.init_image):
             return JSONResponse(content={"error": "Init image required for Z-Image I2I"}, status_code=400)
             
        # Check for local Z-Image model
        local_z_path = req.model_path if req.model_path else scan_models("z-image")
        
        cmd = ["python", "-u", "process_zimage.py", "--prompt", req.prompt, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature), "--image_path", req.init_image, "--strength", str(req.strength)]
        
        if local_z_path:
             print(f"--> [Server] Using Local Z-Image Model: {local_z_path}")
             cmd.extend(["--model_path", local_z_path, "--model_type", "standard"])
    elif req.mode == "upscale":
        if not req.init_image or not os.path.exists(req.init_image):
             return JSONResponse(content={"error": "Input image required for Upscale"}, status_code=400)
        # Pass optional prompt if provided
        p_arg = req.prompt if req.prompt and req.prompt.strip() else ""
        
        # Robust Logic: Only default to 20 if None. If 0, keep 0.
        final_noise = req.noise_level if req.noise_level is not None else 20
        final_scale = req.scale if req.scale is not None else 4.0
        
        cmd = ["python", "-u", "process_upscale.py", "--image_path", req.init_image, "--seed", str(final_seed), "--noise_level", str(final_noise), "--prompt", p_arg, "--steps", str(req.steps), "--guidance", str(req.guidance), "--scale", str(final_scale)]
    elif req.mode == "t2i_cpp":
        cmd = [sys.executable, "-u", "process_t2i_cpp.py", "--prompt", req.prompt, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature), "--clip_skip", str(req.clip_skip or 1)]
        if req.model_path:
             cmd.extend(["--model_path", req.model_path])
        if req.scheduler:
             cmd.extend(["--scheduler", req.scheduler])
    elif req.mode == "i2i_cpp":
        if not req.init_image or not os.path.exists(req.init_image):
             return JSONResponse(content={"error": "Init image required for CPP I2I"}, status_code=400)

        cmd = [sys.executable, "-u", "process_i2i_cpp.py", 
               "--prompt", req.prompt, 
               "--width", str(req.width), 
               "--height", str(req.height), 
               "--steps", str(req.steps), 
               "--guidance", str(req.guidance), 
               "--seed", str(final_seed), 
               "--lora_config", config_path, 
               "--top_k", str(req.top_k), 
               "--temperature", str(req.temperature), 
               "--clip_skip", str(req.clip_skip or 1),
               "--init_img", req.init_image,
               "--strength", str(req.strength)
        ]
        if req.model_path:
             cmd.extend(["--model_path", req.model_path])
        if req.scheduler:
             cmd.extend(["--scheduler", req.scheduler])
        # Pass LLM Path if present in request (though request object might need update if we pass it from frontend)
        # Actually standard GenerateRequest might not have llm_path field yet? 
        # Let's check GenerateRequest definition. 
        # Assuming we receive it or it's part of extra args. 
        # For now, let's verify GenerateRequest first or add it via **kwargs if flexible. 
        # Wait, I added it to t2i_cpp logic previously? 
        # Let's check lines 787... in view.
        if hasattr(req, "llm_path") and req.llm_path:
            cmd.extend(["--llm_path", req.llm_path])

    elif req.mode == "restore":
        if not req.init_image or not os.path.exists(req.init_image):
             return JSONResponse(content={"error": "Input image required for Restoration"}, status_code=400)
        
        cmd = [sys.executable, "-u", "process_restore.py", "--image_path", req.init_image, "--scale", str(req.scale if req.scale else 4.0)]
        if req.face_enhance:
            cmd.append("--face_enhance")
    elif req.mode == "t2i":
        # Check if the selected model is Z-Image (Diffusers version)
        # If so, redirect to Z-Image worker logic
        is_zimage_diffusers = False
        if req.model_path and "z-image" in req.model_path.lower() and not req.model_path.endswith(".gguf"):
             # It's likely a Z-Image Diffusers folder or file
             is_zimage_diffusers = True
        
        if is_zimage_diffusers:
             print(f"--> [Server] Z-Image Diffusers Model Detected: {req.model_path}. Redirecting to Z-Image Worker.")
             cmd = ["python", "-u", "process_zimage.py", "--prompt", req.prompt, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature)]
             cmd.extend(["--model_path", req.model_path, "--model_type", "standard"])
        else:
            cmd = [sys.executable, "-u", "process_t2i.py", "--prompt", req.prompt, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature), "--clip_skip", str(req.clip_skip or 1)]
            if req.model_path:
                 cmd.extend(["--model_path", req.model_path])
            if req.scheduler:
                 cmd.extend(["--scheduler", req.scheduler])

    def event_generator():
        global current_process
        try:
            # Use binary buffer (text=False) to allow unbuffered reads and handle \r
            current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False, bufsize=0, env=os.environ.copy())
            start_t = time.time()
            
            # Robust read buffer
            line_buffer = b""
            full_log_buffer = []
            target_log_path = None
            
            while True:
                if current_process is None: break 
                
                # Check for data
                reads = [current_process.stdout.fileno()]
                r, _, _ = select.select(reads, [], [], 0.1)
                
                if r:
                    # Read chunk
                    chunk = current_process.stdout.read(1024)
                    if not chunk: 
                        # EOF probably
                        pass 
                    else:
                        line_buffer += chunk
                        
                        while True:
                            # Process line by line
                            # We want to split by \n to get clean lines for log
                            # But also handle \r for progress bars if needed (though less critical for log file)
                            
                            idx = line_buffer.find(b'\n')
                            if idx == -1: break
                            
                            line_bytes = line_buffer[:idx+1]
                            line_buffer = line_buffer[idx+1:]
                            
                            try:
                                line = line_bytes.decode('utf-8', errors='replace')
                                clean = line.strip()
                                
                                # Buffer for log file
                                if req.save_log:
                                    full_log_buffer.append(line)

                                if "SUCCESS_OUTPUT:" in clean:
                                    parts = clean.split("SUCCESS_OUTPUT:")
                                    if len(parts) > 1:
                                        final_path = parts[1].strip()
                                        
                                        # Determine Log Path (Using Absolute Path)
                                        if req.save_log and not target_log_path:
                                            # Ensure we have an absolute path for writing
                                            log_base = final_path
                                            if not log_base.startswith("/"):
                                                # If relative, prepend outputs dir
                                                log_base = os.path.join("/app/outputs", log_base)
                                            # If it starts with /outputs/ (web path), correct to /app/outputs/
                                            elif log_base.startswith("/outputs/"):
                                                log_base = "/app" + log_base
                                            
                                            base, _ = os.path.splitext(log_base)
                                            target_log_path = base + ".log"

                                        # Fix for frontend: convert absolute path to URL path
                                        if final_path.startswith("/app/outputs/"):
                                            final_path = final_path.replace("/app/outputs/", "/outputs/")
                                        elif not final_path.startswith("/"):
                                            # relative path
                                            final_path = "/outputs/" + final_path
                                            
                                        yield f"data: IMG|{final_path}\n\n"

                                elif "REF_OUTPUT:" in clean:
                                     parts = clean.split("REF_OUTPUT:")
                                     if len(parts) > 1:
                                         ref_path = parts[1].strip()
                                         if ref_path.startswith("/app/outputs/"):
                                             ref_path = ref_path.replace("/app/outputs/", "/outputs/")
                                         yield f"data: REF|{ref_path}\n\n"
                                elif clean:
                                    yield f"data: LOG|{clean}\n\n"
                            except:
                                pass

                if current_process.poll() is not None:
                     # Check for remaining buffer
                     if line_buffer:
                         try:
                             line = line_buffer.decode('utf-8', errors='replace')
                             clean = line.strip()
                             if req.save_log: full_log_buffer.append(line + "\n")
                             if clean: yield f"data: LOG|{clean}\n\n"
                         except: pass
                     break
            
            # End of Process - Save Log
            if req.save_log and target_log_path and full_log_buffer:
                try:
                    with open(target_log_path, "w") as lf:
                        lf.writelines(full_log_buffer)
                    print(f"--> [System] Saved Log to {target_log_path}", flush=True)
                except Exception as e:
                    print(f"--> [System] Failed to save log: {e}", flush=True)

            if current_process.returncode == 0: yield f"data: DONE|Finished in {time.time()-start_t:.1f}s\n\n"
            else: yield f"data: ERR|Process exited with code {current_process.returncode}\n\n"
                                

        except Exception as e:
            yield f"data: ERR|{str(e)}\n\n"
        finally:
            current_process = None

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)

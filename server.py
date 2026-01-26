import os
import subprocess
import signal
import sys
import time
import select
import shutil
import random
import json
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import lora_manager
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI()

# Note: 'uploads' removed as per cleanup
DIRS = ["/app/static", "/app/outputs", "/app/loras"]
for d in DIRS: os.makedirs(d, exist_ok=True)

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

@app.get("/")
async def read_index():
    return RedirectResponse(url="/static/index.html")

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={}, status_code=204)

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

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    global current_process
    if not req.image_path or not os.path.exists(req.image_path):
        return JSONResponse(content={"error": f"Image not found: {req.image_path}"}, status_code=400)

    cmd = [
        "python", "-u", "process_i2t.py",
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

@app.post("/api/generate")
async def generate(req: GenRequest):
    global current_process
    if req.randomize or req.seed == -1: final_seed = random.randint(0, 2**32-1)
    else: final_seed = req.seed

    lora_data = [{"path": os.path.join(l.folder, l.filename), "strength": l.strength} for l in req.loras]
    config_path = lora_manager.create_config_json(lora_data)

    cmd = []
    if req.mode == "i2i":
        if not req.init_image or not os.path.exists(req.init_image):
             return JSONResponse(content={"error": "Init image required"}, status_code=400)
        cmd = ["python", "-u", "process_i2i.py", "--prompt", req.prompt, "--image_path", req.init_image, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature), "--strength", str(req.strength), "--mix_ratio", str(req.mix_ratio)]
        if req.init_image_2 and os.path.exists(req.init_image_2):
            cmd.extend(["--image_path_2", req.init_image_2])
    else:
        cmd = ["python", "-u", "process_t2i.py", "--prompt", req.prompt, "--width", str(req.width), "--height", str(req.height), "--steps", str(req.steps), "--guidance", str(req.guidance), "--seed", str(final_seed), "--lora_config", config_path, "--top_k", str(req.top_k), "--temperature", str(req.temperature)]

    def event_generator():
        global current_process
        try:
            current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=os.environ.copy())
            start_t = time.time()
            while True:
                if current_process is None: break 
                reads = [current_process.stdout.fileno()]
                ret = select.select(reads, [], [], 0.1)
                if ret[0]:
                    line = current_process.stdout.readline()
                    if line:
                        clean = line.strip()
                        if clean:
                            yield f"data: LOG|{clean}\n\n"
                            if "SUCCESS_OUTPUT:" in clean:
                                img_path = clean.replace("SUCCESS_OUTPUT:", "").strip()
                                yield f"data: IMG|/outputs/{os.path.basename(img_path)}\n\n"
                if current_process.poll() is not None: break

            if current_process.returncode == 0: yield f"data: DONE|Finished in {time.time()-start_t:.1f}s\n\n"
            else: yield f"data: ERR|Process exited with code {current_process.returncode}\n\n"
        except Exception as e: yield f"data: ERR|{str(e)}\n\n"
        finally: current_process = None

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)

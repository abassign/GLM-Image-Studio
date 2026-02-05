import os
import sys
import argparse
import json
from huggingface_hub import snapshot_download

def download(repo_id, token=None, allow_patterns=None, custom_target_dir=None):
    print(f"--> [Downloader] Starting download for {repo_id}...", flush=True)
    if allow_patterns:
        print(f"--> [Downloader] Selective download: {allow_patterns}", flush=True)

    repo_name = repo_id.split("/")[-1]
    
    if custom_target_dir:
        # Use provided directory (absolute or relative to /app/models handled by server)
        # Server should pass absolute path or we trust it.
        # Let's assume server passes the full path we want.
        target_dir = custom_target_dir
    else:
        target_dir = os.path.join("/app/models", repo_name)
    
    # Ensure dir exists
    print(f"--> [Downloader] Target Directory: {target_dir}", flush=True)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        kwargs = {
            "repo_id": repo_id,
            "local_dir": target_dir,
            "local_dir_use_symlinks": False,
            "token": token,
            "resume_download": True,
            "ignore_patterns": ["*.ckpt", "*.onnx", "*.pb", "*.h5", "*.msgpack", "*.xml", "*.tflite", "*.safetensors.sha256"]
        }
        
        # If user explicitly selects files, we override the ignore list to ensure selected files are downloaded
        # (e.g. if they really want a .ckpt)
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
            if "ignore_patterns" in kwargs:
                del kwargs["ignore_patterns"]

        snapshot_download(**kwargs)
        
        # Create dataModels.json if missing AND if we are creating a new root model folder
        # If we are downloading into a subdirectory (files), we probably don't want to overwrite the root JSON
        # logic: checks if target_dir is a direct child of /app/models OR if we are forcing it.
        # Simplest: check if json exists.
        json_path = os.path.join(target_dir, "dataModels.json")
        if not os.path.exists(json_path):
            path_val = "./"
            
            # If we downloaded specific files, check if there is exactly one safetensors
            if allow_patterns:
                # patterns might be ["*.safetensors"] or ["foo.safetensors"]
                # Let's scan the dir to see what we actually have
                files = [f for f in os.listdir(target_dir) if f.endswith(".safetensors")]
                if len(files) == 1:
                    path_val = f"./{files[0]}"
            
            conf = {
                "name": repo_name,
                "type": "custom",
                "path": path_val,
                "source": "huggingface",
                "repo_id": repo_id
            }
            # Only create dataModels.json if we are likely in a valid model root
            # If this is a subdirectory download (e.g. transformer/gguf), maybe we skip?
            # For now, let's keep it harmless.
            with open(json_path, 'w') as f:
                json.dump(conf, f, indent=4)
                
        # Auto-create loras dir if it's a root model dir
        # (heuristic: if custom_target_dir is NOT set or matches default)
        if not custom_target_dir or custom_target_dir.endswith(repo_name):
             os.makedirs(os.path.join(target_dir, "loras"), exist_ok=True)
        
        print(f"--> [Downloader] SUCCESS: Model downloaded to {target_dir}", flush=True)
        
    except Exception as e:
        print(f"--> [Downloader] ERROR: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--allow_patterns", type=str, nargs="*", default=None)
    parser.add_argument("--target_dir", type=str, default=None)
    
    args = parser.parse_args()
    download(args.repo_id, args.token, args.allow_patterns, args.target_dir)

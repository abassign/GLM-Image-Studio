import os
import sys
import argparse
import json
import requests
import re
from huggingface_hub import snapshot_download

def download_civitai(url, target_dir, token=None):
    print(f"--> [Downloader] Starting Civitai Download: {url}", flush=True)
    
    # 1. Parsing
    # We expect https://civitai.com/models/1125067?modelVersionId=2575945
    # or just https://civitai.com/models/1125067
    
    version_id = None
    model_id = None
    
    # Try getting version ID from query param
    if "modelVersionId=" in url:
        try:
            version_id = url.split("modelVersionId=")[1].split("&")[0]
        except: pass
        
    # Get model ID from path
    # Match /models/(\d+)
    match = re.search(r'/models/(\d+)', url)
    if match:
        model_id = match.group(1)
        
    if not version_id:
        if not model_id:
             print("--> [Downloader] Error: Could not parse Civitai URL.", flush=True)
             sys.exit(1)
             
        # Fetch model metadata to get default version
        print(f"--> [Downloader] Fetching metadata for model {model_id}...", flush=True)
        try:
            meta_url = f"https://civitai.com/api/v1/models/{model_id}"
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(meta_url, headers=headers)
            if resp.status_code != 200:
                 print(f"--> [Downloader] Error fetching metadata: {resp.status_code}", flush=True)
                 sys.exit(1)
            
            data = resp.json()
            # Get first version (usually latest) or preferred
            # Civitai returns modelVersions list. 0 is usually latest.
            if "modelVersions" in data and len(data["modelVersions"]) > 0:
                 version_id = data["modelVersions"][0]["id"]
                 print(f"--> [Downloader] Auto-selected latest version: {version_id}", flush=True)
            else:
                 print("--> [Downloader] Error: No versions found for this model.", flush=True)
                 sys.exit(1)
                 
        except Exception as e:
            print(f"--> [Downloader] Metadata Error: {e}", flush=True)
            sys.exit(1)
            
    # 2. Download
    download_url = f"https://civitai.com/api/download/models/{version_id}"
    print(f"--> [Downloader] Download URL: {download_url}", flush=True)
    
    # Ensure target dir
    os.makedirs(target_dir, exist_ok=True)
    
    # Stream download
    try:
        # We need to follow redirects to get the final filename
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        with requests.get(download_url, stream=True, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            
            # Extract filename from header
            filename = None
            if "Content-Disposition" in r.headers:
                cd = r.headers["Content-Disposition"]
                # filename="foo.safetensors"
                if "filename=" in cd:
                    filename = cd.split("filename=")[1].strip('"')
            
            if not filename:
                # Fallback
                filename = f"civitai_model_{version_id}.safetensors"
                
            print(f"--> [Downloader] Saving as: {filename}", flush=True)
            file_path = os.path.join(target_dir, filename)
            
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1MB
            wrote = 0
            
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(block_size):
                    if chunk:
                        f.write(chunk)
                        wrote += len(chunk)
                        if total_size > 0:
                            percent = (wrote / total_size) * 100
                            # Log every 5%
                            if int(percent) % 5 == 0:
                                # We print raw progress for UI to catch? or just standard logs
                                # The current UI catches lines.
                                sys.stdout.write(f"\r--> [Downloader] Progress: {percent:.1f}%")
                                sys.stdout.flush()
            
            print("\n--> [Downloader] Download Complete.", flush=True)
            
            # Create dataModels.json
            json_path = os.path.join(target_dir, "dataModels.json")
            if not os.path.exists(json_path):
                conf = {
                    "name": filename,
                    "type": "custom",
                    "path": f"./{filename}",
                    "source": "civitai",
                    "repo_id": url
                }
                with open(json_path, 'w') as jf:
                    json.dump(conf, jf, indent=4)
                    
            # Create loras dir just in case
            os.makedirs(os.path.join(target_dir, "loras"), exist_ok=True)
            
    except Exception as e:
         print(f"--> [Downloader] Download Failed: {e}", flush=True)
         sys.exit(1)

def download(repo_id, token, allow_patterns, custom_target_dir):
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
    
    if args.repo_id.startswith("http"):
        # Civitai or direct URL mode
        download_civitai(args.repo_id, args.target_dir, args.token)
    else:
        # Hugging Face mode
        download(args.repo_id, args.token, args.allow_patterns, args.target_dir)

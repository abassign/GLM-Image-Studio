import sys
import torch
import diffusers
import gguf
from diffusers.utils import is_gguf_available

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Diffusers: {diffusers.__version__}")
print(f"GGUF: {gguf.__version__}")
print(f"Diffusers thinks GGUF is available: {is_gguf_available()}")

try:
    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    print("load_gguf_checkpoint imported successfully")
except ImportError as e:
    print(f"Failed to import load_gguf_checkpoint: {e}")

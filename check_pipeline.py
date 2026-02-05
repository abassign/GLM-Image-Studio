import diffusers
print(dir(diffusers))
try:
    from diffusers import ZImagePipeline
    print("ZImagePipeline: FOUND")
except ImportError:
    print("ZImagePipeline: NOT FOUND")

# Check if we can load the base model to get components
try:
    from transformers import Qwen2Tokenizer, Qwen2Model
    # Note: model_index said Qwen3Model, but transformers might call it Qwen2 if compatible?
    # Or maybe we need a newer transformers version?
    print("Transformers clean import")
except ImportError as e:
    print(f"Transformers import error: {e}")

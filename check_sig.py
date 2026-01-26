import torch
from diffusers import DiffusionPipeline
import inspect

MODEL_ID = "zai-org/GLM-Image"
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
sig = inspect.signature(pipe.__call__)
print("PIPELINE CALL SIGNATURE:")
for param in sig.parameters.values():
    print(f"  {param}")

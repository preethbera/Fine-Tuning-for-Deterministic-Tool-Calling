import os
import torch
import gc

def enforce_gc():
    """Forces Python garbage collection and clears CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_dir(path: str):
    """Ensures a directory exists, platform-agnostic."""
    os.makedirs(path, exist_ok=True)

import os
import torch

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(model, 'save'): # YOLO
        model.save(path)
    else: # PyTorch
        torch.save(model.state_dict(), path)

def get_model_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0
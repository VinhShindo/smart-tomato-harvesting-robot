import time
import torch

def measure_latency(model, dummy_input, device='cpu', n_iter=20):
    model.to(device)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start) / n_iter * 1000  # ms

def compute_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            _, pred = torch.max(out, 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()
    return correct / total
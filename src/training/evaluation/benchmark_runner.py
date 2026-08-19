import os
import pandas as pd
from model.detection_model import DetectionModel
from model.classification_model import ClassificationModel
from evaluation.metrics import measure_latency, compute_accuracy
from model.utils import get_model_size_mb
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BenchmarkRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def run_benchmark_detection(self, model_names, data_yaml, epochs=10):
        for model_name in model_names:
            print(f"⏳ Benchmarking Detection: {model_name}")
            model = DetectionModel(model_name)
            # Train nhanh
            model.train(data_yaml, epochs=epochs, batch=16, imgsz=640, device=0, project_dir=os.path.join(self.output_dir, "temp"), name=model_name.replace(".pt",""))
            
            # Đo metric
            metrics = model.model.val()
            map_score = metrics.box.map
            size = get_model_size_mb(os.path.join(self.output_dir, "temp", model_name.replace(".pt",""), "weights", "best.pt"))
            
            # Đo latency
            dummy_img = torch.randn(1, 3, 640, 640)
            latency = measure_latency(model.model.model, dummy_img, device='cpu')
            
            self.results.append({"model": model_name, "mAP50-95": round(map_score, 4), "latency_ms": round(latency, 2), "size_mb": round(size, 2)})
        self._save_csv("detection_results.csv")

    def run_benchmark_classification(self, model_names, data_dir, epochs=10):
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        val_ds = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
        dummy_img = torch.randn(1, 3, 224, 224)
        
        for model_name in model_names:
            print(f"⏳ Benchmarking Classification: {model_name}")
            model = ClassificationModel(model_name)
            # Train nhanh (giả lập, sử dụng val chưa train)
            acc = compute_accuracy(model.model, val_loader, model.device)
            
            # Đo Latency & Size
            latency = measure_latency(model.model, dummy_img, device='cpu')
            model.save("temp_cls.pth")
            size = get_model_size_mb("temp_cls.pth")
            os.remove("temp_cls.pth")
            
            self.results.append({"model": model_name, "accuracy": round(acc, 4), "latency_ms": round(latency, 2), "size_mb": round(size, 2)})
        self._save_csv("classification_results.csv")

    def _save_csv(self, filename):
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        self.results = []
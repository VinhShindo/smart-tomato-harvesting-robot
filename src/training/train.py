import os
import yaml
from model.detection_model import DetectionModel
from model.classification_model import ClassificationModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CONFIG_PATH = "config/train.yaml"
BEST_MODEL_PATH = "output/benchmark_results/best_model.txt"

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Đọc best model
    if not os.path.exists(BEST_MODEL_PATH):
        print("⚠️ Chưa có best_model.txt. Vui lòng chạy benchmark.py trước.")
        return
    
    with open(BEST_MODEL_PATH, "r") as f:
        lines = f.readlines()
        BEST_DETECT = lines[0].strip().split("=")[1]
        BEST_CLASSIFY = lines[1].strip().split("=")[1]
    
    print(f"🚀 Bắt đầu Train chính thức Detection: {BEST_DETECT}")
    detector = DetectionModel(BEST_DETECT)
    detector.train(
        data_yaml=cfg["dataset_detection_yaml"],
        epochs=cfg["detection"]["epochs"],
        batch=cfg["detection"]["batch"],
        imgsz=cfg["detection"]["imgsz"],
        device=cfg["detection"]["device"],
        project_dir=cfg["save_weights_dir"],
        name="best_detection"
    )
    
    print(f"🚀 Bắt đầu Train chính thức Classification: {BEST_CLASSIFY}")
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((cfg["classification"]["img_size"], cfg["classification"]["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(cfg["dataset_classification_dir"], "train"), transform)
    train_loader = DataLoader(train_ds, batch_size=cfg["classification"]["batch"], shuffle=True, num_workers=4)
    val_ds = datasets.ImageFolder(os.path.join(cfg["dataset_classification_dir"], "valid"), transform)
    val_loader = DataLoader(val_ds, batch_size=cfg["classification"]["batch"], shuffle=False, num_workers=4)
    
    classifier = ClassificationModel(BEST_CLASSIFY)
    classifier.train(train_loader, val_loader, epochs=cfg["classification"]["epochs"], lr=cfg["classification"]["lr"])
    classifier.save(os.path.join(cfg["save_weights_dir"], "best_classification.pth"))
    
    print("🎉 Training hoàn tất! Model nằm tại:", cfg["save_weights_dir"])

if __name__ == "__main__":
    main()
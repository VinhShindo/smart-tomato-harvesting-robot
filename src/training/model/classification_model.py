import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from .base_model import BaseModel

class ClassificationModel(BaseModel):
    def __init__(self, model_name="resnet18", num_classes=3):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        
    def _build_model(self):
        if "resnet18" in self.model_name:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif "resnet34" in self.model_name:
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif "mobilenet" in self.model_name:
            model = models.mobilenet_v3_small(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif "efficientnet" in self.model_name:
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        return model

    def train(self, train_loader, val_loader, epochs, lr):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                optimizer.zero_grad()
                out = self.model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
            
            # Validation metric đơn giản
            if (epoch+1) % 5 == 0:
                self.model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, lbls in val_loader:
                        imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                        out = self.model(imgs)
                        _, pred = torch.max(out, 1)
                        total += lbls.size(0)
                        correct += (pred == lbls).sum().item()
                print(f"Epoch {epoch+1}/{epochs} | Val Acc: {correct/total:.4f}")
    
    def predict(self, image):
        if isinstance(image, str): # Nếu là path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_tensor)
            _, pred = torch.max(out, 1)
        return pred.item()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
from ultralytics import YOLO
from .base_model import BaseModel

class DetectionModel(BaseModel):
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.model_name = model_name

    def train(self, data_yaml, epochs, batch, imgsz, device, project_dir, name):
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project_dir,
            name=name,
            exist_ok=True
        )
    
    def predict(self, image):
        return self.model(image)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = YOLO(path)
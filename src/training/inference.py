import os
import cv2
import json
from model.detection_model import DetectionModel
from model.classification_model import ClassificationModel

# Đường dẫn mặc định đến model đã train (điều chỉnh nếu cần)
DETECT_WEIGHT = "output/weights/best_detection/weights/best.pt"
CLASSIFY_WEIGHT = "output/weights/best_classification.pth"

def load_models():
    if not os.path.exists(DETECT_WEIGHT) or not os.path.exists(CLASSIFY_WEIGHT):
        raise FileNotFoundError("⚠️ Model chưa được train. Hãy chạy train.py trước.")
    detector = DetectionModel(DETECT_WEIGHT)
    classifier = ClassificationModel()
    classifier.load(CLASSIFY_WEIGHT)
    return detector, classifier

def run_pipeline(image_path, detector, classifier):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Không đọc được ảnh"}
    
    results = detector.predict(img)
    outputs = []
    CLASS_MAP = {0: "ripe", 1: "green", 2: "half_ripe"}
    
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: continue
        
        # Classification
        cls_id = classifier.predict(roi)
        
        outputs.append({
            "object_id": f"tomato_{i}",
            "bbox": [x1, y1, x2, y2],
            "width_px": x2 - x1,
            "height_px": y2 - y1,
            "area_px": (x2 - x1) * (y2 - y1),
            "ripeness": CLASS_MAP.get(cls_id, "unknown"),
            "detection_confidence": conf
        })
    return json.dumps(outputs, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    try:
        detector, classifier = load_models()
        # Test thử với 1 ảnh bất kỳ
        test_path = "dataset_final_detection/train/images/IMG_0985.jpg"
        if os.path.exists(test_path):
            print(run_pipeline(test_path, detector, classifier))
        else:
            print("ℹ️ Hãy thay đường dẫn ảnh thật vào biến test_path trong inference.py")
    except Exception as e:
        print(e)
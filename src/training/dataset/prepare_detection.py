import os
import cv2
import numpy as np

BASE_DIR = "tomato-laboro"
OUTPUT_DIR = "dataset_final_classification"
SUBSETS = ["train", "valid", "test"]

def normalize_class_id(old_id):
    if old_id in [0, 3]: return 0      # Ripe
    elif old_id in [1, 4]: return 1   # Green
    elif old_id in [2, 5]: return 2   # Half_ripe
    return old_id

CLASS_MAP = {0: "ripe", 1: "green", 2: "half_ripe"}

def crop_exact_polygon(img, parts):
    try:
        old_id = int(parts[0])
        class_id = normalize_class_id(old_id)
        if class_id not in CLASS_MAP: return None, None
        
        coords = list(map(float, parts[1:]))
        xs = coords[0::2]; ys = coords[1::2]
        h, w, _ = img.shape
        
        x_min_norm, x_max_norm = min(xs), max(xs)
        y_min_norm, y_max_norm = min(ys), max(ys)
        x1 = int(x_min_norm * w); y1 = int(y_min_norm * h)
        x2 = int(x_max_norm * w); y2 = int(y_max_norm * h)
        
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1: return None, None
        
        roi = img[y1:y2, x1:x2].copy()
        # Chuyển đổi tọa độ sang ROI
        poly_points = []
        for x_norm, y_norm in zip(xs, ys):
            px = int(x_norm * w) - x1
            py = int(y_norm * h) - y1
            poly_points.append([px, py])
        poly_points = np.array(poly_points, np.int32)
        
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly_points], 255)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
        
        coords_mask = cv2.findNonZero(mask)
        if coords_mask is not None:
            x, y, w_mask, h_mask = cv2.boundingRect(coords_mask)
            roi_masked = roi_masked[y:y+h_mask, x:x+w_mask]
        
        return roi_masked, CLASS_MAP[class_id]
    except:
        return None, None

def prepare_classification():
    for subset in SUBSETS:
        src_img_dir = os.path.join(BASE_DIR, subset, "images")
        src_lbl_dir = os.path.join(BASE_DIR, subset, "labels")
        
        print(f"📂 Đang crop & chuẩn hóa Classification cho tập: {subset}")
        obj_count = 0
        
        for lbl_file in os.listdir(src_lbl_dir):
            if not lbl_file.endswith(".txt") or lbl_file == "classes.txt": continue
            
            img_name = lbl_file.replace(".txt", ".jpg")
            img_path = os.path.join(src_img_dir, img_name)
            if not os.path.exists(img_path):
                img_name = lbl_file.replace(".txt", ".JPG")
                img_path = os.path.join(src_img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            with open(os.path.join(src_lbl_dir, lbl_file), "r") as f:
                for line in f:
                    crop_img, class_folder = crop_exact_polygon(img, line.strip().split())
                    if crop_img is not None and crop_img.size > 0:
                        save_dir = os.path.join(OUTPUT_DIR, subset, class_folder)
                        os.makedirs(save_dir, exist_ok=True)
                        save_name = f"{os.path.splitext(img_name)[0]}_{obj_count}.jpg"
                        cv2.imwrite(os.path.join(save_dir, save_name), crop_img)
                        obj_count += 1
        print(f"✅ Xong {subset}: {obj_count} ảnh crop.")
    print(f"✅ Dataset Classification đã sẵn sàng tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_classification()
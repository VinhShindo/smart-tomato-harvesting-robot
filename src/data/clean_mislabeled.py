import os
import cv2
import numpy as np
import shutil
from collections import defaultdict

# ================= CẤU HÌNH =================
CLASSIFY_DIR = "dataset_final_classification"
OUTPUT_DIR = "dataset_final_classification_cleaned"
SUBSETS = ["train", "valid", "test"]

# Ngưỡng màu để phân biệt (Có thể điều chỉnh sau khi chạy test)
THRESHOLDS = {
    "ripe": {
        "min_red_ratio": 0.40,   # Nếu red_ratio < 0.4 mà nằm trong ripe -> bị gán sai
        "max_green_ratio": 0.30  # Nếu green_ratio > 0.3 mà nằm trong ripe -> bị gán sai
    },
    "green": {
        "min_red_ratio": 0.25,   # Nếu red_ratio > 0.25 mà nằm trong green -> bị gán sai
        "max_green_ratio": 0.35  # (Dự phòng, thường green có green_ratio cao)
    },
    "half_ripe": {
        "min_red_ratio": 0.30,
        "max_red_ratio": 0.55,
        "min_green_ratio": 0.25,
        "max_green_ratio": 0.45
    }
}
# =============================================

def get_color_ratios(image_path):
    """Tính tỉ lệ màu Đỏ và Xanh trên ảnh (đã bỏ nền đen)"""
    img = cv2.imread(image_path)
    if img is None: return None, None
    
    # Chuyển sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Tạo mask loại bỏ nền đen
    mask = np.all(img_rgb == [0, 0, 0], axis=-1)
    non_black = img_rgb[~mask]
    
    if len(non_black) == 0:
        return None, None
    
    r_mean = np.mean(non_black[:, 0])
    g_mean = np.mean(non_black[:, 1])
    b_mean = np.mean(non_black[:, 2])
    total = r_mean + g_mean + b_mean
    
    if total == 0:
        return None, None
    
    return r_mean / total, g_mean / total

def clean_dataset():
    print("🚀 Bắt đầu làm sạch nhãn (Label Noise Cleaning)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    stats = {
        "moved": defaultdict(int),
        "fixed": defaultdict(lambda: defaultdict(int))
    }
    
    for subset in SUBSETS:
        print(f"\n📂 Đang xử lý tập: {subset}")
        
        for class_name in ["ripe", "green", "half_ripe"]:
            src_dir = os.path.join(CLASSIFY_DIR, subset, class_name)
            if not os.path.exists(src_dir): continue
            
            dst_dir = os.path.join(OUTPUT_DIR, subset, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            
            files = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
            print(f"   🔍 Đang quét thư mục {class_name} ({len(files)} ảnh)...")
            
            for file_name in files:
                file_path = os.path.join(src_dir, file_name)
                red_ratio, green_ratio = get_color_ratios(file_path)
                
                if red_ratio is None:
                    # Ảnh lỗi (toàn màu đen) -> bỏ qua, vẫn giữ nguyên
                    shutil.copy(file_path, os.path.join(dst_dir, file_name))
                    continue
                
                # Logic xác định lớp đúng dựa trên màu sắc
                predicted_class = class_name
                
                if class_name == "ripe":
                    # Nếu quả chín mà quá xanh hoặc quá nhạt
                    if red_ratio < THRESHOLDS["ripe"]["min_red_ratio"]:
                        predicted_class = "green"
                    elif green_ratio > THRESHOLDS["ripe"]["max_green_ratio"]:
                        predicted_class = "green"
                
                elif class_name == "green":
                    # Nếu quả xanh mà quá đỏ
                    if red_ratio > THRESHOLDS["green"]["min_red_ratio"]:
                        predicted_class = "ripe"
                
                elif class_name == "half_ripe":
                    # Half_ripe nằm giữa: quá đỏ hoặc quá xanh đều sai
                    if red_ratio > THRESHOLDS["half_ripe"]["max_red_ratio"]:
                        predicted_class = "ripe"
                    elif red_ratio < THRESHOLDS["half_ripe"]["min_red_ratio"] and green_ratio > THRESHOLDS["half_ripe"]["max_green_ratio"]:
                        predicted_class = "green"
                    # Nếu red_ratio và green_ratio cân bằng hoặc lẫn lộn, giữ nguyên half_ripe
                
                # Nếu dự đoán khác với nhãn gốc -> chuyển thư mục
                if predicted_class != class_name:
                    dst_class_dir = os.path.join(OUTPUT_DIR, subset, predicted_class)
                    os.makedirs(dst_class_dir, exist_ok=True)
                    shutil.move(file_path, os.path.join(dst_class_dir, file_name))
                    stats["moved"][subset] += 1
                    stats["fixed"][subset][f"{class_name}->{predicted_class}"] += 1
                else:
                    # Nếu đúng nhãn, copy sang thư mục sạch
                    shutil.copy(file_path, os.path.join(dst_dir, file_name))
    
    # --- BÁO CÁO KẾT QUẢ ---
    print("\n" + "="*60)
    print("📊 BÁO CÁO LÀM SẠCH NHÃN")
    print("="*60)
    total_moved = 0
    for subset in SUBSETS:
        moved = stats["moved"].get(subset, 0)
        total_moved += moved
        print(f"\n📂 Tập {subset}:")
        if moved == 0:
            print(f"   ✅ Không có ảnh nào bị chuyển nhãn.")
        else:
            print(f"   ⚠️ Đã chuyển {moved} ảnh sang thư mục đúng:")
            for change, count in stats["fixed"][subset].items():
                print(f"      - {change}: {count} ảnh")
    print("\n" + "="*60)
    print(f"🎉 TỔNG CỘNG: Đã sửa {total_moved} ảnh bị gán nhãn sai!")
    print(f"📂 Dữ liệu sạch đã lưu tại: {OUTPUT_DIR}/")
    print("📌 Tiếp theo, hãy dùng thư mục này để chạy cân bằng dữ liệu (balance).")

if __name__ == "__main__":
    clean_dataset()
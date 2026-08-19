import os
import shutil

BASE_DIR = "tomato-laboro"             # Thư mục dataset gốc
OUTPUT_DIR = "dataset_final_detection" # Thư mục đích
SUBSETS = ["train", "valid", "test"]

def normalize_detection_id(old_id):
    # Gộp tất cả ID (0-5) về 0
    if old_id in [0, 1, 2, 3, 4, 5]:
        return 0
    return old_id

def prepare_detection():
    for subset in SUBSETS:
        src_img_dir = os.path.join(BASE_DIR, subset, "images")
        src_lbl_dir = os.path.join(BASE_DIR, subset, "labels")
        
        dst_img_dir = os.path.join(OUTPUT_DIR, subset, "images")
        dst_lbl_dir = os.path.join(OUTPUT_DIR, subset, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        print(f"📂 Đang chuẩn hóa Detection cho tập: {subset}")
        
        for label_file in os.listdir(src_lbl_dir):
            if not label_file.endswith(".txt") or label_file == "classes.txt": continue
            
            # Copy ảnh
            img_name = label_file.replace(".txt", ".jpg")
            if not os.path.exists(os.path.join(src_img_dir, img_name)):
                img_name = label_file.replace(".txt", ".JPG")
            shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
            
            # Chuẩn hóa ID trong label file
            new_lines = []
            with open(os.path.join(src_lbl_dir, label_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    old_id = int(parts[0])
                    new_id = normalize_detection_id(old_id)
                    parts[0] = str(new_id)
                    new_lines.append(" ".join(parts))
            
            with open(os.path.join(dst_lbl_dir, label_file), "w") as f:
                f.write("\n".join(new_lines))
    
    # Tạo data.yaml
    yaml_content = f"""
path: ../{OUTPUT_DIR}
train: train/images
val: valid/images
test: test/images
names:
  0: tomato
"""
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content.strip())
    print(f"✅ Dataset Detection đã sẵn sàng tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_detection()
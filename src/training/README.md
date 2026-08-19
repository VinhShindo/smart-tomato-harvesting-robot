# Smart Tomato Harvesting Robot - Model Training & Benchmark Pipeline

## Yêu cầu hệ thống
- Python 3.8+
- Cài đặt thủ công các thư viện chính:
  `pip install ultralytics torch torchvision pandas matplotlib seaborn opencv-python pyyaml`

## 1. Chuẩn bị dữ liệu
- Đặt dataset gốc (`tomato-laboro` và `tomato-laboro-big`) vào thư mục root.
- Chạy các script chuẩn bị trong `dataset/`:
  ```bash
  python dataset/prepare_detection.py
  python dataset/prepare_classification.py
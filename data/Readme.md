# BÁO CÁO TỔNG HỢP - MERGE DATASETS

**Ngày tạo:** 2026-08-19 20:39:45

## 1. Mô tả quy trình

Script này thực hiện gộp 2 dataset đã chuẩn hóa (V1 Clean và V2) thành một dataset cuối cùng duy nhất.

- **Dataset V1 Clean:** Bao gồm cả quả `Big` và `Little`, đã chuẩn hóa về 3 class 0,1,2.
- **Dataset V2:** Chỉ bao gồm quả `Big`, đã có sẵn 3 class 0,1,2.

## 2. Thống kê trước khi gộp

### Detection (Ảnh gốc)
| Subset | V1 Files | V2 Files | Tổng |
|:---|:---:|:---:|:---:|
| **train** | 563 | 309 | 872 |
| **valid** | 161 | 89 | 250 |
| **test** | 80 | 44 | 124 |

### Classification (Ảnh crop)
| Subset | V1 Files | V2 Files | Tổng |
|:---|:---:|:---:|:---:|
| **train** | 6812 | 2005 | 8817 |
| **valid** | 1878 | 582 | 2460 |
| **test** | 1087 | 323 | 1410 |

## 3. Kết quả sau khi gộp

### Detection (Ảnh gốc)
| Subset | V1 Files | V2 Files (_v2) | Tổng |
|:---|:---:|:---:|:---:|
| **train** | 872 | 872 | 1744 |
| **valid** | 250 | 250 | 500 |
| **test** | 124 | 124 | 248 |

### Classification (Ảnh crop)
| Subset | V1 Files | V2 Files (_v2) | Tổng |
|:---|:---:|:---:|:---:|
| **train** | 8817 | 10648 | 19465 |
| **valid** | 2460 | 2929 | 5389 |
| **test** | 1410 | 1600 | 3010 |

## 4. Kết quả đầu ra

- **Thư mục Detection cuối cùng:** `dataset_final_detection/`
- **Thư mục Classification cuối cùng:** `dataset_final_classification/`
- **Lưu ý:** Các file từ V2 đã được thêm hậu tố `_v2` để tránh trùng tên với V1.

---
*Báo cáo được tạo tự động bởi `merge_datasets.py`*
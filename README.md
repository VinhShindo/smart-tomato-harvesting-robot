> **Ảnh đầu vào → xác định quả cà chua → trích xuất đặc trưng → xác định giai đoạn chín → Server đánh giá điều kiện → kết luận `HARVEST` hoặc `WAIT`.**

Quan trọng nhất: **AI đưa ra nhận định, Server là nơi đưa ra quyết định cuối cùng.**

# 1. Sơ đồ xử lý đầy đủ

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE / CAMERA                                │
│                                                                             │
│  Ảnh cây cà chua: lá + thân + nhiều quả + nền                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1 — IMAGE PREPROCESSING                             │
│                                                                             │
│  - Đọc ảnh                                                                  │
│  - Resize                                                                    │
│  - Normalize                                                                 │
│  - Kiểm tra ảnh hợp lệ                                                       │
│  - Chuẩn bị input cho AI                                                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2 — TOMATO DETECTION                               │
│                                                                             │
│                         AI / YOLO                                           │
│                                                                             │
│  Input: Image                                                               │
│  Output:                                                                    │
│    - Class = tomato                                                          │
│    - Confidence                                                              │
│    - Bounding Box                                                            │
│                                                                             │
│  Ví dụ:                                                                     │
│    Tomato #01 → confidence = 0.96                                           │
│    bbox = [x1, y1, x2, y2]                                                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                         Confidence đạt ngưỡng?
                         ┌─────────┴─────────┐
                         │                   │
                        NO                  YES
                         │                   │
                         ▼                   ▼
                      IGNORE         ┌──────────────────────┐
                                     │ Xác định vùng quả    │
                                     │ Tomato ROI / Crop    │
                                     └──────────┬───────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3 — OBJECT VALIDATION                              │
│                                                                             │
│  Kiểm tra:                                                                  │
│  - Có phải tomato không?                                                    │
│  - Detection confidence                                                     │
│  - Bounding box hợp lệ                                                      │
│  - Kích thước ROI có đủ để phân tích không?                                 │
│                                                                             │
│  Output: Valid Tomato Object                                                │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  STEP 4 — FEATURE EXTRACTION                               │
│                                                                             │
│                         Tomato ROI                                          │
│                              │                                              │
│             ┌────────────────┼────────────────┐                             │
│             │                │                │                             │
│             ▼                ▼                ▼                             │
│        SIZE FEATURES    COLOR FEATURES   SHAPE FEATURES                    │
│                                                                             │
│        - width          - RGB/HSV/Lab    - width/height ratio              │
│        - height         - Hue            - contour                         │
│        - area           - Saturation     - circularity                     │
│        - relative size  - Value          - shape                           │
│                         - red ratio                                         │
│                         - green ratio                                       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  STEP 5 — RIPENESS CLASSIFICATION                          │
│                                                                             │
│                Xác định giai đoạn hiện tại của quả                          │
│                                                                             │
│        Feature +/hoặc Image Classification Model                            │
│                                                                             │
│        ┌───────────┬────────────┬─────────────┬───────────┐                │
│        │           │            │             │           │                │
│        ▼           ▼            ▼             ▼           │                │
│      GREEN      TURNING       PINK          RIPE          │                │
│        │           │            │             │           │                │
│        └───────────┴────────────┴─────────────┴───────────┘                │
│                                                                             │
│  Output:                                                                    │
│    stage = RIPE                                                              │
│    stage_confidence = 0.91                                                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 6 — RESULT AGGREGATION                             │
│                                                                             │
│  Tổng hợp toàn bộ thông tin của quả                                         │
│                                                                             │
│  Tomato #01                                                                  │
│  ├── Detection confidence = 0.96                                            │
│  ├── Bounding box = [x1,y1,x2,y2]                                          │
│  ├── Width / Height = ...                                                   │
│  ├── Color features = ...                                                   │
│  ├── Shape features = ...                                                   │
│  ├── Stage = RIPE                                                            │
│  └── Stage confidence = 0.91                                                │
│                                                                             │
│                    ↓                                                        │
│              JSON / API Payload                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║                     STEP 7 — SERVER                                        ║
║                       DECISION ENGINE                                       ║
║                                                                             ║
║  Server KHÔNG nhận ảnh để tự đoán lại.                                      ║
║  Server nhận kết quả từ AI và áp dụng các RULE.                             ║
║                                                                             ║
║  Kiểm tra:                                                                  ║
║                                                                             ║
║  1. detected == tomato?                                                     ║
║  2. detection_confidence >= threshold?                                      ║
║  3. stage_confidence >= threshold?                                          ║
║  4. stage có thuộc nhóm được thu hoạch?                                     ║
║  5. kích thước có đạt yêu cầu?                                              ║
║  6. Các điều kiện khác có đạt không?                                        ║
║                                                                             ║
╚══════════════════════════════════┬══════════════════════════════════════════╝
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ HARVEST DECISION   │
                         └─────────┬──────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
                     ▼                           ▼
              ┌──────────────┐            ┌──────────────┐
              │   HARVEST    │            │     WAIT     │
              │              │            │              │
              │ Đủ điều kiện │            │ Chưa đủ      │
              │ thu hoạch    │            │ điều kiện    │
              └──────┬───────┘            └──────┬───────┘
                     │                           │
                     ▼                           ▼
              Chờ robot thật              Theo dõi tiếp
              thực hiện sau này           ở lần tiếp theo
```

---

# 2. Cần phân biệt 3 khái niệm

Đây là phần **rất quan trọng khi triển khai**.

### `Detection`

Trả lời:

> **“Đây có phải quả cà chua không?”**

Ví dụ:

```json
{
  "class": "tomato",
  "confidence": 0.96
}
```

---

### `Classification`

Trả lời:

> **“Quả cà chua này đang ở giai đoạn nào?”**

Ví dụ:

```json
{
  "stage": "RIPE",
  "confidence": 0.91
}
```

---

### `Decision`

Trả lời:

> **“Với trạng thái hiện tại, có nên thu hoạch không?”**

Ví dụ:

```json
{
  "decision": "HARVEST",
  "reason": "RIPE_AND_SIZE_OK"
}
```

Vậy:

```text
Detection ≠ Classification ≠ Decision
```

Đây chính là kiến trúc mình khuyên nhóm giữ xuyên suốt dự án.

---

# 3. Output chuẩn mà AI nên trả cho Server

Ta nên thống nhất format ngay từ đầu.

Ví dụ:

```json
{
  "request_id": "REQ_000001",
  "timestamp": "2026-08-17T14:00:00",
  "objects": [
    {
      "object_id": "T001",
      "class": "tomato",
      "detection_confidence": 0.96,

      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 202,
        "y2": 159
      },

      "size": {
        "width_px": 82,
        "height_px": 79,
        "area_px": 6478
      },

      "color": {
        "red_ratio": 0.82,
        "green_ratio": 0.03,
        "orange_ratio": 0.11
      },

      "shape": {
        "aspect_ratio": 1.04,
        "circularity": 0.87
      },

      "ripeness": {
        "stage": "RIPE",
        "confidence": 0.91
      }
    }
  ]
}
```

Server sau đó bổ sung:

```json
{
  "object_id": "T001",
  "decision": {
    "harvest": true,
    "status": "READY_TO_HARVEST",
    "reason": "RIPE_AND_SIZE_OK"
  }
}
```

---

# 4. Server Decision Engine

Không nên hard-code logic kiểu:

```python
if red_ratio > 0.7:
    harvest = True
```

ngay từ đầu.

Nên thiết kế:

```text
AI RESULT
    ↓
Decision Engine
    ↓
Rules / Configuration
    ↓
Decision
```

Ví dụ configuration:

```yaml
decision:
  detection_confidence: 0.70
  ripeness_confidence: 0.80

  harvest_stages:
    - RIPE

  minimum_size:
    width_px: 40
    height_px: 40
```

Sau này nếu khảo sát thực tế thấy:

```text
RIPE → thu hoạch
PINK → chưa thu hoạch
```

thì chỉ cần thay rule.

Nếu sau này phát hiện:

```text
PINK + size lớn → cũng có thể thu hoạch
```

thì thay rule mà **không cần huấn luyện lại detection model**.

---

# 5. Một điểm quan trọng về "kích thước"

Ở phiên bản đầu tiên, ta sẽ lưu:

```text
width_px
height_px
area_px
```

Chứ **chưa gọi đó là kích thước thực tế của quả**.

Ví dụ:

```text
80 pixel
```

không đồng nghĩa:

```text
80 mm
```

Sau này nếu cần kích thước thật:

```text
Pixel
  ↓
Camera Calibration
  ↓
Real-world Measurement
  ↓
mm / cm
```

Ta có thể bổ sung sau.

---

# 6. README.md — phiên bản chốt cho module hiện tại

Bạn có thể dùng nguyên README dưới đây.

````markdown
# Smart Tomato Harvesting Robot
## AI Tomato Detection, Ripeness Classification & Harvest Decision

---

## 1. Overview

This module is the AI and server decision pipeline of the
**Smart Tomato Harvesting Robot** project.

The current development phase focuses only on the software pipeline
for analyzing tomato images and determining whether a tomato is ready
for harvesting.

Hardware components such as the mobile robot, robot arm, ESP32,
motors and gripper/cutter are temporarily excluded.

The system follows the pipeline:

```text
Image
  ↓
Tomato Detection
  ↓
Feature Extraction
  ↓
Ripeness Classification
  ↓
Server Decision Engine
  ↓
HARVEST / WAIT
````

---

# 2. Objective

The objective of this module is to develop a system capable of:

1. Receiving a tomato plant image.
2. Detecting tomato fruits.
3. Confirming whether the detected object is a tomato.
4. Extracting tomato features.
5. Estimating tomato size.
6. Analyzing tomato color.
7. Analyzing tomato shape.
8. Classifying the current ripeness stage.
9. Sending the AI result to the server.
10. Applying predefined decision rules.
11. Determining whether the tomato is ready for harvesting.

The final output is:

```text
HARVEST
```

or

```text
WAIT
```

---

# 3. System Architecture

```text
                         INPUT IMAGE
                              │
                              ▼
                    Image Preprocessing
                              │
                              ▼
                     Tomato Detection
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  Invalid              Valid
                    │                   │
                  Ignore                ▼
                              Object Validation
                                      │
                                      ▼
                              Feature Extraction
                                      │
                       ┌──────────────┼──────────────┐
                       │              │              │
                       ▼              ▼              ▼
                     Size           Color          Shape
                       │              │              │
                       └──────────────┼──────────────┘
                                      ▼
                           Ripeness Classification
                                      │
                                      ▼
                              Result Aggregation
                                      │
                                      ▼
                                  API / JSON
                                      │
                                      ▼
                              Server Decision
                                  Engine
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                       HARVEST                  WAIT
```

---

# 4. Processing Pipeline

## Step 1 — Image Input

The system receives an image containing a tomato plant.

The image may contain:

* Tomato fruits
* Leaves
* Stems
* Soil
* Background
* Multiple tomato fruits

Example:

```text
Camera / Image File
        ↓
     Image
```

---

# 5. Step 2 — Image Preprocessing

Before AI inference, the image is prepared.

Operations may include:

* Image loading
* Resize
* Normalization
* Format conversion
* Input validation

Output:

```text
Preprocessed Image
```

---

# 6. Step 3 — Tomato Detection

The detection model identifies tomato fruits in the image.

The model returns:

* Object class
* Confidence score
* Bounding box

Example:

```json
{
  "class": "tomato",
  "confidence": 0.96,
  "bbox": [120, 80, 202, 159]
}
```

The detection stage answers:

> Is this object a tomato?

It does NOT determine whether the tomato is ready for harvesting.

---

# 7. Step 4 — Object Validation

Detected objects are validated before further processing.

Validation criteria may include:

* Object class must be `tomato`.
* Detection confidence must exceed a threshold.
* Bounding box must be valid.
* ROI must have sufficient resolution.

Example:

```text
confidence >= 0.70
        ↓
Valid Tomato
```

Objects that do not satisfy the criteria are ignored.

---

# 8. Step 5 — Feature Extraction

The detected tomato is cropped from the original image.

```text
Original Image
      ↓
Bounding Box
      ↓
Tomato ROI
      ↓
Feature Extraction
```

The system extracts three main groups of features.

## 8.1 Size Features

Initial prototype features:

* Width in pixels
* Height in pixels
* Pixel area
* Width/height ratio

Example:

```json
{
  "width_px": 82,
  "height_px": 79,
  "area_px": 6478
}
```

The initial system uses image-space measurements.

Real-world size in mm/cm requires camera calibration
and can be added later.

---

## 8.2 Color Features

The tomato ROI is analyzed using an appropriate color space,
such as HSV or Lab.

Possible features:

* Hue
* Saturation
* Value
* Red pixel ratio
* Green pixel ratio
* Orange pixel ratio

Example:

```json
{
  "red_ratio": 0.82,
  "green_ratio": 0.03,
  "orange_ratio": 0.11
}
```

---

## 8.3 Shape Features

Possible features:

* Aspect ratio
* Contour
* Area
* Circularity
* Shape characteristics

Example:

```json
{
  "aspect_ratio": 1.04,
  "circularity": 0.87
}
```

---

# 9. Step 6 — Ripeness Classification

The system estimates the current ripeness stage.

The initial classification scheme is:

```text
GREEN
TURNING
PINK
RIPE
```

The exact number of classes can be adjusted after dataset analysis.

Example output:

```json
{
  "stage": "RIPE",
  "confidence": 0.91
}
```

The classification stage answers:

> What is the current ripeness stage of this tomato?

It does NOT directly determine whether the robot should harvest.

---

# 10. Step 7 — AI Result Aggregation

All information related to the tomato is combined.

Example:

```json
{
  "object_id": "T001",
  "class": "tomato",
  "detection_confidence": 0.96,

  "bbox": {
    "x1": 120,
    "y1": 80,
    "x2": 202,
    "y2": 159
  },

  "size": {
    "width_px": 82,
    "height_px": 79,
    "area_px": 6478
  },

  "color": {
    "red_ratio": 0.82,
    "green_ratio": 0.03,
    "orange_ratio": 0.11
  },

  "shape": {
    "aspect_ratio": 1.04,
    "circularity": 0.87
  },

  "ripeness": {
    "stage": "RIPE",
    "confidence": 0.91
  }
}
```

This information is sent to the server.

---

# 11. Step 8 — Server Decision Engine

The server receives the AI result.

The server does not replace the AI model.

Instead, it applies predefined rules to determine
whether the tomato is ready for harvesting.

Conceptually:

```text
AI Result
   ↓
Decision Engine
   ↓
Decision Rules
   ↓
Harvest Decision
```

Example rules:

```yaml
decision:
  detection_confidence: 0.70
  ripeness_confidence: 0.80

  harvest_stages:
    - RIPE

  minimum_size:
    width_px: 40
    height_px: 40
```

---

# 12. Harvest Decision

Example:

```text
Detection confidence = 0.96
Ripeness confidence  = 0.91
Stage                = RIPE
Size                 = PASS
```

Decision:

```text
HARVEST
```

Another example:

```text
Detection confidence = 0.95
Ripeness confidence  = 0.88
Stage                = GREEN
Size                 = PASS
```

Decision:

```text
WAIT
```

---

# 13. Decision Output

The server returns a final decision.

Example:

```json
{
  "object_id": "T001",
  "decision": "HARVEST",
  "status": "READY_TO_HARVEST",
  "reason": "RIPE_AND_SIZE_OK"
}
```

If the tomato is not ready:

```json
{
  "object_id": "T002",
  "decision": "WAIT",
  "status": "NOT_READY",
  "reason": "RIPENESS_STAGE_NOT_ELIGIBLE"
}
```

---

# 14. Detection vs Classification vs Decision

These three stages must remain separate.

## Detection

Question:

> Is this a tomato?

Output:

```text
tomato + confidence + bounding box
```

---

## Classification

Question:

> What is the current ripeness stage?

Output:

```text
GREEN / TURNING / PINK / RIPE
```

---

## Decision

Question:

> Should this tomato be harvested?

Output:

```text
HARVEST / WAIT
```

Therefore:

```text
Detection
    ≠
Classification
    ≠
Decision
```

---

# 15. API Data Flow

The AI module provides the analysis result to the server.

Example endpoint:

```text
POST /api/v1/tomatoes/analyze
```

Example request:

```json
{
  "image_id": "image_001.jpg"
}
```

Example AI response:

```json
{
  "object_id": "T001",
  "class": "tomato",
  "detection_confidence": 0.96,
  "bbox": [120, 80, 202, 159],

  "size": {
    "width_px": 82,
    "height_px": 79
  },

  "color": {
    "red_ratio": 0.82,
    "green_ratio": 0.03
  },

  "ripeness": {
    "stage": "RIPE",
    "confidence": 0.91
  }
}
```

Server response:

```json
{
  "object_id": "T001",
  "decision": "HARVEST",
  "status": "READY_TO_HARVEST",
  "reason": "RIPE_AND_SIZE_OK"
}
```

---

# 16. Current Repository Structure

```text
smart-tomato-harvesting-robot/
│
├── README.md
├── .gitignore
│
├── server/
│   ├── api/
│   ├── decision/
│   ├── schemas/
│   └── main.py
│
├── ai/
│   ├── detection/
│   ├── classification/
│   ├── features/
│   ├── inference/
│   └── models/
│
├── data/
│   ├── raw/
│   ├── labeled/
│   └── processed/
│
├── simulation/
│   └── mock_robot/
│
├── tests/
│   ├── ai/
│   ├── server/
│   └── integration/
│
└── docs/
    └── architecture/
```

---

# 17. Development Phases

## Phase 1 — Tomato Detection

```text
Image
 ↓
YOLO
 ↓
Tomato Detection
 ↓
Bounding Box
```

---

## Phase 2 — Feature Extraction

```text
Tomato ROI
 ↓
Size
Color
Shape
```

---

## Phase 3 — Ripeness Classification

```text
Tomato ROI + Features
 ↓
Ripeness Model
 ↓
GREEN / TURNING / PINK / RIPE
```

---

## Phase 4 — Decision Engine

```text
AI Result
 ↓
Server
 ↓
Rules
 ↓
HARVEST / WAIT
```

---

## Phase 5 — Mock Robot

The real robot is not required at this stage.

The system can simulate:

```text
HARVEST
   ↓
Mock Robot
   ↓
"Harvesting tomato T001..."
```

This allows the complete software pipeline to be tested
before integrating hardware.

---

# 18. Evaluation Metrics

## Detection

* Precision
* Recall
* mAP
* Confidence
* Inference time

## Ripeness Classification

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## Decision Engine

* Correct harvest decisions
* False harvest decisions
* Missed harvest decisions
* Decision consistency

---

# 19. Important Design Principles

### Principle 1

AI performs perception.

```text
AI = What do we see?
```

### Principle 2

Server performs decision making.

```text
Server = What should we do?
```

### Principle 3

Decision rules must be configurable.

The harvesting criteria should not be hard-coded
inside the AI model.

### Principle 4

Hardware is separated from the AI pipeline.

The current system must be fully testable without
the physical robot.

### Principle 5

The system must support multiple tomatoes in one image.

Each detected tomato should have its own:

```text
object_id
bbox
features
ripeness
decision
```

---

# 20. Future Integration

After the AI and server pipeline is stable:

```text
Server
   ↓
HARVEST
   ↓
Robot Command
   ↓
Mobile Robot
   ↓
Robot Arm
   ↓
Gripper / Cutter
   ↓
Physical Harvest
```

The hardware integration will be implemented as a separate phase
without changing the core AI classification pipeline.

---

# 21. Project Goal

The final software pipeline aims to achieve:

```text
IMAGE
  ↓
TOMATO DETECTED
  ↓
FEATURES EXTRACTED
  ↓
RIPENESS CLASSIFIED
  ↓
SERVER DECISION
  ↓
┌───────────────┐
│               │
▼               ▼
HARVEST        WAIT
```

The current development target is to make this pipeline
work reliably using images and simulated inputs before
connecting it to the physical robot.

````

---

## 7. Và đây là thứ tự chúng ta sẽ triển khai

Mình đề xuất **không nhảy sang server ngay**. Ta làm từng tầng, mỗi tầng chạy được mới sang tầng tiếp theo:

```text
                    ┌──────────────────┐
                    │  01. IMAGE INPUT │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 02. DETECTION    │
                    │    Tomato        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 03. ROI +        │
                    │    FEATURES      │
                    │ Size/Color/Shape │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 04. RIPENESS     │
                    │    CLASSIFICATION│
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 05. JSON RESULT  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 06. SERVER API   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 07. DECISION     │
                    │    ENGINE        │
                    └────────┬─────────┘
                             ↓
                       ┌─────┴─────┐
                       ↓           ↓
                   HARVEST        WAIT
````

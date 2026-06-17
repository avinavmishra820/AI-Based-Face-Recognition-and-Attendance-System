# FaceX — AI Face Recognition & Attendance System

> Real-time face recognition · Anti-spoofing (MiniFASNet) · Mask detection · Automated attendance logging · Excel export · Analytics dashboard

**Authors:** Aashutosh Mishra & Ujjwal Sharma  
**Department:** Computer Science, Nitte Meenakshi Institute of Technology, Bangalore  
**Degree:** B.Tech (Computer Science), 8th Semester, 2024–2025

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Models Used](#models-used)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Module Reference](#module-reference)
8. [Configuration](#configuration)
9. [GUI Dashboard](#gui-dashboard)
10. [Analytics & Metrics](#analytics--metrics)
11. [Folder Outputs](#folder-outputs)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

---

## Overview

FaceX is a fully offline, real-time face recognition and attendance management system. It runs entirely on a standard webcam and CPU — no cloud services, no internet connection, and no infrared camera required.

The system integrates three deep learning models:

| Model | Role | Architecture |
|---|---|---|
| dlib ResNet-34 | Face encoding & identity matching | ResNet-34 + triplet loss, 128-d embedding |
| MiniFASNetV2 + V1SE | Liveness / anti-spoofing | Lightweight depthwise CNN, 80×80 input |
| MobileNetV2 *(optional)* | Mask detection classifier | Inverted residual blocks, transfer learning |

Each video frame goes through a sequential pipeline: **capture → detect → anti-spoof → mask check → encode → match → log → display**.

---

## Features

- **Real-time face recognition** from a standard webcam at ~15–20 FPS on CPU
- **Anti-spoofing** using two MiniFASNet models with 3-crop multi-scale inference — rejects printed photos and screen-displayed faces
- **Mask detection** — YCrCb skin-tone heuristic (instant, no training needed) with optional MobileNetV2 CNN upgrade
- **Automated attendance logging** with 60-second cooldown to prevent duplicate entries
- **Security logging** of unknown faces with timestamped snapshots and red-highlighted Excel rows
- **One-click Excel export** with embedded face thumbnails, alternating row colours, and auto-cleanup
- **Analytics charts** — attendance per person, daily attendance, known vs unknown ratio
- **Performance metrics** — accuracy, precision, recall, F1, and confusion matrix
- **GUI dashboard** (CustomTkinter) with live event log, all controls in one place, and a safe-close export dialog
- **Clean folder structure** — all exports in `exports/`, charts in `analytics/`, snapshots auto-deleted after export

---

## Project Structure

```
FaceX/
│
├── dataset/                        # Face image dataset
│   └── PersonName/                 # One folder per enrolled person
│       ├── PersonName_0000.jpg
│       └── ...                     # ~150 images recommended
│
├── models/                         # Model weights and encodings
│   ├── encodings.pickle            # Serialised 128-d face encodings (generated)
│   ├── 2.7_80x80_MiniFASNetV2.pth  # Anti-spoof model V2 (download required)
│   ├── 4_0_0_80x80_MiniFASNetV1SE.pth  # Anti-spoof model V1SE (download required)
│   └── mask_detector.pt            # Mask CNN weights (generated after training)
│
├── snapshots/                      # Temporary face crops (auto-deleted after export)
│
├── exports/                        # All CSV and Excel outputs
│   ├── attendance_log_temp.csv     # Live attendance log (session data)
│   ├── ground_truth.csv            # For metrics evaluation
│   └── Attendance_Export_*.xlsx    # Final exported attendance records
│
├── analytics/                      # Generated charts and reports
│   ├── chart_attendance_per_person.png
│   ├── chart_attendance_by_date.png
│   ├── chart_known_vs_unknown.png
│   ├── confusion_matrix.png
│   └── metrics_summary.txt
│
├── image_capture.py                # Step 1 — Collect face images from webcam
├── encoder.py                      # Step 2 — Generate 128-d face encodings
├── anti_spoof.py                   # MiniFASNet V2 + V1SE inference module
├── mask_detection.py               # Mask detection: heuristic + CNN
├── run_recognition.py              # Step 3 — Main recognition + attendance loop
├── export_excel.py                 # Export attendance CSV → formatted Excel
├── analytics_report.py             # Generate attendance charts
├── metrics.py                      # Accuracy / Precision / Recall / F1
├── confusion_matrix_plot.py        # Plot confusion matrix
├── collect_mask_data.py            # Collect mask/no-mask data + auto-train CNN
├── gui_dashboard.py                # GUI dashboard (main entry point)
├── test_antispoof.py               # Live anti-spoof test viewer
├── download_models.py              # Download MiniFASNet .pth files
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Models Used

### 1. dlib ResNet-34 (Face Recognition)

- **Source:** dlib C++ library, wrapped by the `face_recognition` Python package
- **Architecture:** 34-layer Residual Network trained with triplet loss
- **Output:** 128-dimensional L2-normalised face embedding vector
- **Accuracy:** 99.38% on Labelled Faces in the Wild (LFW) benchmark
- **Matching:** Euclidean (L2) distance — threshold `0.42` in FaceX
- **No download needed** — installed automatically with `face_recognition`

### 2. MiniFASNetV2 + MiniFASNetV1SE (Anti-Spoofing)

- **Source:** [MiniVision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- **Architecture:** Lightweight depthwise separable CNN; V1SE adds Squeeze-and-Excitation channel attention
- **Input:** 80×80 RGB image, ImageNet-normalised
- **Output:** 3-class softmax — `[spoof_type1, spoof_type2, real]` — FaceX uses `class 2` (real) score
- **Threshold:** `LIVE_THRESHOLD = 0.70` — face must score ≥ 0.70 across all 6 crops to pass
- **Files needed:**
  - `models/2.7_80x80_MiniFASNetV2.pth`
  - `models/4_0_0_80x80_MiniFASNetV1SE.pth`
- **Download:** Run `python download_models.py` or use the dashboard button

### 3. MobileNetV2 (Mask Detection — Optional)

- **Source:** torchvision pretrained on ImageNet, fine-tuned on user-collected data
- **Architecture:** Inverted residual blocks (expansion factor 6), 2-class head (mask / no mask)
- **Input:** 224×224 RGB image, ImageNet-normalised
- **Training:** Run `python collect_mask_data.py` — collects 120+ images per class, then auto-trains
- **Fallback:** If `mask_detector.pt` is not present, FaceX uses a YCrCb skin-tone heuristic instead

---

## Installation

### Prerequisites

- Python 3.9–3.11
- CMake (required for dlib)
- Webcam

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install cmake build-essential python3-dev python3-venv
```

**Fedora:**
```bash
sudo dnf install cmake gcc-c++ python3-devel
```

**Windows:**
1. Install [CMake](https://cmake.org/download/) and add to PATH
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

### Setup

```bash
# 1. Clone or extract the project
cd FaceX/

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download anti-spoof models
python download_models.py
# OR run the dashboard and click "Download Models"
```

> **GPU Support (optional):** Replace the torch install line in requirements.txt with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Quick Start

### Option A — GUI Dashboard (Recommended)

```bash
python gui_dashboard.py
```

Use the sidebar buttons in order:
1. **Capture Faces** → collect training images
2. **Encode Faces** → generate encodings
3. **Run Recognition** → start live attendance

### Option B — Command Line

```bash
# Step 1: Collect face images (AUTO mode: 150 images, no key presses needed)
python image_capture.py "Your Name" AUTO

# Step 2: Generate encodings
python encoder.py

# Step 3: Run recognition + attendance
python run_recognition.py

# Step 4: Export attendance to Excel
python export_excel.py
```

### Controls (during recognition)
| Key | Action |
|-----|--------|
| `Q` | Quit and save attendance CSV |

---

## Module Reference

### `image_capture.py` — Face Dataset Collection

Captures face images from the webcam and saves them to `dataset/PersonName/`.

```bash
python image_capture.py                       # Interactive (asks for name and mode)
python image_capture.py "John Doe" AUTO       # Capture 150 images automatically
python image_capture.py "John Doe" MANUAL     # Press SPACE to capture each image
python image_capture.py PREVIEW PREVIEW       # Camera preview only
```

**Recommended:** AUTO mode, 150 images, varying angles and lighting.

---

### `encoder.py` — Face Encoding

Scans `dataset/`, detects faces with HOG, extracts 128-d ResNet-34 encodings, saves to `models/encodings.pickle`.

```bash
python encoder.py
```

Re-run whenever you add new people to the dataset.

---

### `run_recognition.py` — Main Recognition Loop

Live webcam recognition with anti-spoofing, mask detection, attendance logging, and snapshot capture.

```bash
python run_recognition.py
```

**Bounding box colours:**
- 🟢 Green — known person, live, accepted
- 🔴 Red — spoof detected/ unknown person (photo/screen)
- 🟡 Yellow — mask detected (prompt to remove)

---

### `export_excel.py` — Excel Export

Reads `exports/attendance_log_temp.csv`, creates a formatted `.xlsx` with embedded face thumbnails.

```bash
python export_excel.py
```

- Known person rows: alternating blue (`D6E4F0`)
- Unknown/security rows: red background (`FDECEA`) + bold red text
- Photo column: 80×80 px face thumbnail embedded in cell
- Cleans up: deletes all snapshots and the temp CSV after export

---

### `collect_mask_data.py` — Mask Data Collection + Training

Interactive webcam collector for mask/no-mask face crops, followed by automatic MobileNetV2 fine-tuning.

```bash
python collect_mask_data.py
```

Collect at least 120 images per class. Training runs automatically after collection. Output: `models/mask_detector.pt`.

---

### `test_antispoof.py` — Anti-Spoof Live Test

Opens webcam and displays real-time MiniFASNet liveness score overlaid on the frame.

```bash
python test_antispoof.py
```

Useful for tuning `LIVE_THRESHOLD` and verifying model behaviour.

---

### `analytics_report.py` — Charts

```bash
python analytics_report.py
```

Generates three charts in `analytics/` from the attendance CSV.

---

### `metrics.py` — Performance Metrics

```bash
python metrics.py
```

Reads `exports/ground_truth.csv` and prints accuracy, precision, recall, F1, plus per-class breakdown. Saves `analytics/metrics_summary.txt`.

---

### `confusion_matrix_plot.py` — Confusion Matrix

```bash
python confusion_matrix_plot.py
```

Saves `analytics/confusion_matrix.png`.

---

## Configuration

All key parameters are constants at the top of `run_recognition.py`:

| Parameter | Default | Description |
|---|---|---|
| `TOLERANCE` | `0.42` | Max L2 distance for a valid face match (lower = stricter) |
| `CV_SCALER` | `2` | Downscale factor for face detection (higher = faster, less accurate) |
| `ATTENDANCE_COOLDOWN` | `60` | Seconds before the same person can be logged again |
| `UNKNOWN_COOLDOWN` | `60` | Seconds between unknown-face security log events |
| `MIN_FACE_PX` | `60` | Minimum face bounding box size in pixels |
| `LIVE_THRESHOLD` | `0.70` | Minimum MiniFASNet real-class score to accept as live |
| `MASK_SKIN_THRESHOLD` | `0.12` | Skin pixel ratio below which face is flagged as masked |
| `MASK_DEBUG` | `False` | Set `True` to print YCrCb skin ratio to terminal for threshold tuning |
| `SHOW_CONFIDENCE` | `False` | Show match confidence on bounding box (terminal always shows it) |

---

## GUI Dashboard

Launch with `python gui_dashboard.py`. The window has three panels:

**Sidebar** — all action buttons:

| Button | What it does |
|---|---|
| Capture Faces | Opens name/mode dialog → launches image_capture.py |
| Encode Faces | Runs encoder.py, streams output to log |
| Run Recognition | Launches run_recognition.py |
| Anti-Spoof Test | Launches test_antispoof.py |
| Mask Detection | Launches mask_detection.py |
| Collect Mask Data | Launches collect_mask_data.py |
| Export Excel | Runs export_excel.py |
| Analytics | Generates charts, displays in popup |
| Accuracy Metrics | Runs metrics.py |
| Confusion Matrix | Plots confusion matrix, displays in main area |
| Download Models | Downloads MiniFASNet .pth files |
| Stop All | Terminates any active subprocess |

**Event Log panel** — live output from all running subprocesses.

**Close-time dialog** — if unsaved attendance data exists when closing, a dialog appears:
- 📊 **Export to Excel then Close** — exports first, then closes
- 💾 **Keep temp CSV** — closes without touching data
- 🗑 **Discard and Close** — deletes CSV and all snapshots

---

## Analytics & Metrics

### Attendance Charts (`analytics_report.py`)

| Chart | File |
|---|---|
| Attendance per person | `analytics/chart_attendance_per_person.png` |
| Attendance by date | `analytics/chart_attendance_by_date.png` |
| Known vs Unknown ratio | `analytics/chart_known_vs_unknown.png` |

### Performance Metrics (`metrics.py`)

Requires `exports/ground_truth.csv` with columns: `Image`, `Actual`, `Predicted`.

The dashboard auto-generates this file from the attendance log. You can also manually label it for a proper evaluation.

| Metric | Description |
|---|---|
| Accuracy | Overall fraction of correctly identified faces |
| Precision (macro) | Average per-class: correct positives / all predicted positives |
| Recall (macro) | Average per-class: correct positives / all actual positives |
| F1 Score (macro) | Harmonic mean of precision and recall |

---

## Folder Outputs

| Folder | Contents | Cleaned after export? |
|---|---|---|
| `dataset/` | Training images | No — persistent |
| `models/` | `.pth`, `.pickle`, `.pt` | No — persistent |
| `snapshots/` | Session face crops (JPEG) | Yes — deleted after Excel export |
| `exports/` | CSV, Excel files | CSV deleted; `.xlsx` kept permanently |
| `analytics/` | Charts, metrics text | No — overwritten on next run |

---

## Troubleshooting

**`dlib` or `face_recognition` fails to install**
```bash
# Make sure CMake and build tools are installed first (see Installation)
pip install dlib --verbose
pip install face_recognition
```

**`ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python-headless
# On Fedora/KDE, avoid opencv-python (Qt conflicts) — use headless
```

**Camera not opening**
```bash
# Check camera index (default is 0)
# Edit image_capture.py: cv2.VideoCapture(0) → cv2.VideoCapture(1)
```

**MiniFASNet models not found**
```bash
python download_models.py
# Or manually place .pth files in models/
# Models: 2.7_80x80_MiniFASNetV2.pth and 4_0_0_80x80_MiniFASNetV1SE.pth
```

**All faces showing as SPOOF**
- Check models loaded correctly: terminal should show `[AntiSpoof] MiniFASNet: active — 2 model(s)`
- Try running `python test_antispoof.py` and check the displayed score
- If score for live face < 0.70, lower `LIVE_THRESHOLD` slightly (e.g., 0.60)

**Mask heuristic too sensitive / not sensitive enough**
- Set `MASK_DEBUG = True` in `run_recognition.py`
- Terminal will print the skin ratio value for each face
- Adjust `MASK_SKIN_THRESHOLD` accordingly (default 0.12 = 12%)

**Low recognition confidence**
- Capture more images (150+ per person) at different angles and lighting
- Re-run `encoder.py` after adding images

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, pp. 770–778.
2. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR*, pp. 7132–7141.
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, pp. 4510–4520.
4. King, D. E. (2009). Dlib-ml: A machine learning toolkit. *Journal of Machine Learning Research*, 10, 1755–1758.
5. MiniVision AI. (2020). *Silent-Face-Anti-Spoofing*. https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
6. Geitgey, A. (2017). *face_recognition*. https://github.com/ageitgey/face_recognition
7. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*, Vol. 1, pp. 886–893.
8. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*, pp. 815–823.

---

*FaceX — Department of Computer Science, NMIT Bangalore, 2024–2025*

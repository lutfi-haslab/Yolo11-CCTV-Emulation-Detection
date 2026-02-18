# 📹 CCTV Emulation & YOLO11 Object Detection

A complete AI pipeline for detecting **Persons**, **Masks**, and **Caps** in a CCTV emulation environment. This project includes a custom **Data Labeling GUI**, **YOLO Training Scripts**, and a **5-Camera CCTV Emulation** system.

![CCTV Monitor](monitor.png)

---

## 🚀 Features

- **Object Detection**: Start-of-the-art detection using **YOLO11**.
- **Classes**:
  - `0`: Person 👤
  - `1`: Mask 😷
  - `2`: Cap 🧢
- **CCTV Emulation**: Simulates a 5-camera control room feed with:
  - Real-time webcam processing.
  - Simulated "No Signal" feeds.
  - **Violation Detection**: Automatically flags persons without mask or cap in **RED**.
  - System stats display (FPS, CPU, RAM).
- **Custom Labeling Tool**: Built-in Python GUI for easy dataset creation.

---

## 🛠️ Prerequisites

- **Python 3.10+**
- **Operating System**: macOS (M1/M2/M3 supported), Linux, or Windows.
- **Webcam**: Required for live camera feed (optional for simulation mode).

---

## 📦 Project Structure

```
myEaiApp/
├── images/                  # Source images for training
├── labeled_dataset/         # Exported dataset ready for YOLO training
├── simple_dataset/          # Original pre-labeled dataset
├── runs/                    # Trained models & checkpoints
├── src/                     # Source Code
│   ├── labeling_app.py      # 🏷️ Image Labeling GUI
│   ├── cctv_emulation.py    # 📹 Main CCTV System
│   ├── train.py             # 🏋️ Training Script
│   └── test.py              # 🔍 Inference/Testing Script
├── Makefile                 # ⚡ Task Runner
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## ⚡ Quick Start

We use `make` commands to simplify all workflows.

### 1. Setup Environment

Create a virtual environment and install dependencies:

```bash
make setup
```

### 2. Labeling Data 🏷️

Launch the custom labeling tool to annotate your own images.

```bash
make label
# Or: python src/labeling_app.py
```

![Labeling App Preview](label_app_preview.png)

**Labeling Workflow:**

1.  **Add Labels**: Type `person`, `mask`, `cap` and click **Add**.
2.  **Draw Boxes**: Select a label and click-drag on the image.
3.  **Save**: Click "Save Annotations" (Ctrl+S).
4.  **Export**: Click "Export YOLO Dataset" to generate the dataset in `labeled_dataset/`.

> **Note**: The exporter automatically splits each image into 3 copies (one per label) to improve small object detection accuracy.

### 3. Training the Model 🏋️

Train YOLO11 on your dataset (or the provided `simple_dataset`).

```bash
# Default training (uses simple_dataset)
make train

# Custom training params
make train EPOCHS=200 BATCH=8 DATA=labeled_dataset/data.yaml
```

- **Output**: Best model saved to `runs/detect/labeled_model/weights/best.pt`

### 4. Testing & Validation 🔍

Run inference on static images to verify accuracy.

```bash
make test
# Or with specific confidence
make test CONF=0.5
```

### 5. Run CCTV Emulation 📹

Launch the 5-camera monitoring dashboard.

```bash
make cctv
# Or: python src/cctv_emulation.py
```

**Controls**:

- Press `q` to quit.

---

## 📊 Performance & Optimization

- **Data Augmentation**: We split labels into separate image layers (`img_01_person.png`, `img_01_mask.png`) to ensure the model learns each class distinctly even with limited data.
- **Mac M-Series Optimization**: Uses `mps` (Metal Performance Shaders) if available (auto-detected).
- **Early Stopping**: Training stops automatically if no improvement for 50 epochs (configurable via `PATIENCE`).

---

## 📝 License

This project is open-source and available for educational and research purposes.

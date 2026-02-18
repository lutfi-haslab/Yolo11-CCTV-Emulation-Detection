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
  - **IP Camera / Stream Support**: Connect to external RTSP or HTTP streams.
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
yolo-cctv/
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

**Controls / Key Shortcuts**:

- `q`: **Quit** the application.
- `s`: **Switch Source** between Local Webcam (0) and IP Stream.

---

### 6. Streaming Server (Optional) 📡

You can run a standalone MJPEG streaming server to simulate a multi-camera IP system. This server takes your single webcam and multiplexes it into **5 distinct video feeds**, each with unique labels to simulate different locations.

**Option A: Run Server Only**

```bash
make stream-server
# Server runs at http://localhost:5001
# Feeds available at: /video_feed_1 ... /video_feed_5
```

**Option B: Run Server + Client**

```bash
make stream
# Starts the server AND automatically launches the CCTV client connected to all 5 streams.
```

**Manual Stream Configuration**:
You can manually configure your real RTSP or IPTV sources in `src/cctv_emulation.py`.
Open the file and modify the `STREAM_SOURCES` list at the top:

```python
STREAM_SOURCES = [
    "rtsp://admin:1234@192.168.1.10:554/stream1",  # CAM 01
    "http://192.168.1.11:8080/video",              # CAM 02
    # ...
]
```

**Technical Note**:

- **Webcam Mode**: Optimized to process the frame once and replicate it to maintain high FPS (30+).
- **Stream Mode**: Uses **Round-Robin Detection** (processing one camera per frame cycle) to simulate a realistic heavy-load environment while maintaining UI responsiveness.

![Streaming Mode](monitor_stream.png)
_Figure: CCTV Emulation connected to the Streaming Server with 5 distinct feeds._

---

## 📊 Performance & Optimization

- **Data Augmentation**: We split labels into separate image layers (`img_01_person.png`, `img_01_mask.png`) to ensure the model learns each class distinctly even with limited data.
- **Mac M-Series Optimization**: Uses `mps` (Metal Performance Shaders) if available (auto-detected).
- **Early Stopping**: Training stops automatically if no improvement for 50 epochs (configurable via `PATIENCE`).

---

## Test Model

```bash
(venv) hy4-mac-002@HY4-MAC-4 yolo-cctv % python src/test.py --model /Users/hy4-mac-002/hasdev/yolo-cctv/runs/detect/labeled_model/weights/best.pt
============================================================
🔍 YOLO Inference
============================================================
  🧠 Model:      /Users/hy4-mac-002/hasdev/yolo-cctv/runs/detect/labeled_model/weights/best.pt
  🎯 Confidence: 0.25
  📐 Image Size: 640
  💻 Device:     auto
============================================================

🏷️  Classes: {0: 'person', 1: 'mask', 2: 'cap'}

📂 Testing 15 images from: /Users/hy4-mac-002/hasdev/yolo-cctv/images

📷 img_01.png:
   ✅ person (53.62%) [118,29,218,390]
   ✅ mask (44.38%) [154,77,209,124]
   ✅ person (39.95%) [398,69,502,384]
   ✅ person (37.21%) [217,89,292,385]
   ✅ person (34.06%) [279,123,334,324]
   ✅ cap (30.19%) [217,77,283,119]
   ✅ cap (25.09%) [131,38,219,94]

📷 img_02.jpeg:
   ✅ mask (47.39%) [598,218,702,299]
   ✅ mask (41.47%) [225,249,302,295]
   ✅ cap (39.47%) [367,120,532,213]
   ✅ mask (38.27%) [814,216,922,300]
   ✅ person (37.70%) [35,123,220,391]
   ✅ mask (37.15%) [329,238,402,283]
   ✅ person (35.29%) [543,80,774,549]
   ✅ cap (34.46%) [555,106,729,227]
   ✅ cap (33.98%) [50,129,184,216]
   ✅ person (32.08%) [737,110,1024,541]
   ✅ person (31.60%) [190,175,333,391]
   ✅ person (29.49%) [378,97,589,495]
   ✅ cap (28.90%) [765,120,965,219]
   ✅ mask (28.84%) [410,206,496,258]
   ✅ cap (25.21%) [300,163,421,249]

📷 img_03.jpeg:
   ✅ person (45.60%) [14,146,60,271]
   ✅ person (44.89%) [66,146,114,271]
   ✅ person (41.54%) [118,146,165,271]
   ✅ mask (26.23%) [79,169,101,181]

📷 img_04.jpeg:
   ✅ mask (45.01%) [21,36,46,55]
   ✅ cap (39.30%) [14,8,52,35]
   ✅ cap (39.24%) [227,0,273,53]
   ✅ cap (38.00%) [51,33,76,53]
   ✅ mask (35.62%) [227,37,269,65]
   ✅ cap (34.86%) [94,43,116,52]
   ✅ cap (33.21%) [123,42,149,62]
   ✅ person (33.07%) [168,0,273,179]
   ✅ person (30.84%) [47,32,80,109]
   ✅ person (30.83%) [0,6,52,163]
   ✅ person (30.51%) [119,41,150,90]

📷 img_05.jpeg:
   ✅ cap (36.28%) [52,94,117,130]
   ✅ person (36.00%) [26,82,151,299]
   ✅ mask (29.67%) [65,128,101,154]

📷 img_06.jpeg:
   ✅ cap (55.97%) [349,11,415,52]
   ✅ person (49.31%) [449,45,503,169]
   ✅ mask (47.48%) [252,76,280,95]
   ✅ cap (46.84%) [268,45,342,90]
   ✅ mask (46.76%) [466,83,498,103]
   ✅ cap (43.27%) [160,45,250,100]
   ✅ cap (41.89%) [405,61,458,91]
   ✅ mask (39.79%) [416,86,448,109]
   ✅ person (39.07%) [0,46,158,321]
   ✅ mask (38.01%) [189,95,237,129]
   ✅ person (36.06%) [144,39,261,286]
   ✅ mask (34.95%) [346,87,383,112]
   ✅ cap (34.76%) [335,50,390,89]
   ✅ mask (33.67%) [583,57,600,78]
   ✅ person (32.93%) [268,44,353,240]
   ✅ mask (32.79%) [283,86,329,119]
   ✅ cap (32.68%) [48,58,104,116]
   ✅ person (32.64%) [520,27,592,162]
   ✅ mask (31.87%) [368,49,398,70]
   ✅ person (30.28%) [391,61,460,179]
   ✅ person (27.03%) [488,41,536,149]

📷 img_07.jpg:
   ✅ mask (46.81%) [168,217,350,364]
   ✅ cap (42.03%) [928,21,1067,81]
   ✅ person (38.53%) [3,29,455,719]
   ✅ mask (38.35%) [961,65,1019,112]
   ✅ person (34.95%) [557,0,877,420]
   ✅ person (32.68%) [1068,17,1203,258]
   ✅ cap (31.62%) [105,51,380,245]
   ✅ person (29.62%) [850,5,1053,283]
   ✅ cap (28.90%) [1152,25,1203,107]
   ✅ mask (28.84%) [719,85,810,163]
   ✅ cap (28.61%) [676,0,850,73]

📷 img_08.webp:
   ✅ cap (53.17%) [754,153,879,222]
   ✅ cap (50.48%) [297,132,416,210]
   ✅ person (38.86%) [431,93,590,671]
   ✅ cap (38.77%) [477,114,577,187]
   ✅ person (33.41%) [241,92,438,675]
   ✅ mask (32.88%) [322,189,399,242]
   ✅ person (32.36%) [725,133,936,674]
   ✅ mask (30.91%) [774,212,848,258]
   ✅ mask (28.02%) [491,175,556,217]

📷 img_09.webp:
   ✅ cap (61.43%) [430,245,547,339]
   ✅ person (47.43%) [0,181,178,666]
   ✅ person (47.16%) [153,176,324,670]
   ✅ person (45.66%) [383,220,584,669]
   ✅ cap (38.42%) [195,199,323,278]
   ✅ person (37.47%) [573,218,774,668]
   ✅ cap (27.44%) [629,253,739,325]

📷 img_10.jpeg:
   ✅ cap (46.59%) [124,36,177,69]
   ✅ mask (40.92%) [221,42,246,61]
   ✅ cap (35.25%) [217,20,259,50]
   ✅ person (33.96%) [199,9,259,143]
   ✅ mask (33.49%) [136,70,166,93]
   ✅ person (32.96%) [115,23,227,194]

📷 img_11.jpeg:
   ✅ mask (33.69%) [55,35,82,58]
   ✅ cap (28.48%) [47,18,96,42]
   ✅ person (26.40%) [19,6,107,237]

📷 img_12.jpg:
   ✅ cap (53.54%) [487,64,555,104]
   ✅ person (51.02%) [77,101,270,400]
   ✅ person (50.13%) [277,163,488,423]
   ✅ person (49.48%) [481,94,614,316]
   ✅ cap (45.55%) [330,175,460,284]
   ✅ cap (43.14%) [177,20,242,69]
   ✅ cap (42.26%) [137,107,229,163]
   ✅ mask (33.00%) [249,58,283,80]
   ✅ mask (32.83%) [516,137,572,170]
   ✅ person (32.36%) [173,14,299,156]
   ✅ person (32.28%) [84,3,188,130]
   ✅ cap (28.49%) [238,24,301,66]
   ✅ mask (28.43%) [158,158,220,200]
   ✅ cap (25.49%) [521,98,617,163]
   ✅ person (25.03%) [431,13,544,162]
   ✅ mask (25.01%) [201,64,244,93]

📷 img_13.jpg:
   ✅ person (53.93%) [1081,39,2086,1439]
   ✅ cap (45.84%) [1518,86,1940,625]
   ✅ cap (37.03%) [2070,206,2567,696]
   ✅ person (35.73%) [1944,88,2742,1759]

📷 img_14.jpeg:
   ✅ cap (47.81%) [210,153,272,207]
   ✅ cap (47.52%) [725,164,798,226]
   ✅ cap (47.25%) [548,164,621,203]
   ✅ cap (46.80%) [411,165,462,195]
   ✅ cap (44.66%) [646,163,714,210]
   ✅ mask (41.49%) [421,190,449,212]
   ✅ person (37.49%) [42,99,201,447]
   ✅ person (36.72%) [454,159,541,405]
   ✅ person (36.56%) [255,161,362,427]
   ✅ mask (34.56%) [564,197,602,219]
   ✅ cap (32.81%) [476,168,526,195]
   ✅ cap (31.63%) [103,108,226,198]
   ✅ person (31.48%) [521,152,635,439]
   ✅ person (30.12%) [381,156,469,395]
   ✅ person (29.67%) [617,151,724,421]
   ✅ person (29.07%) [705,157,800,433]
   ✅ mask (27.97%) [489,191,516,210]
   ✅ person (25.21%) [184,141,281,443]

📷 img_15.jpg:
   ✅ mask (58.25%) [883,266,934,301]
   ✅ person (46.71%) [646,230,716,444]
   ✅ cap (44.91%) [459,202,564,262]
   ✅ person (44.09%) [688,202,784,423]
   ✅ mask (43.86%) [751,262,798,295]
   ✅ person (42.22%) [855,215,960,438]
   ✅ mask (41.42%) [795,270,840,300]
   ✅ mask (40.92%) [700,250,751,285]
   ✅ mask (38.59%) [585,257,644,299]
   ✅ mask (37.95%) [657,287,709,323]
   ✅ person (37.67%) [407,178,584,492]
   ✅ person (37.30%) [211,136,427,535]
   ✅ mask (35.52%) [841,272,887,301]
   ✅ person (34.22%) [57,147,192,599]
   ✅ person (33.96%) [934,232,1023,436]
   ✅ mask (33.69%) [943,283,995,316]
   ✅ cap (33.12%) [646,247,715,295]
   ✅ person (32.61%) [791,224,862,405]
   ✅ cap (32.54%) [304,170,402,241]
   ✅ cap (31.51%) [793,231,846,277]
   ✅ mask (28.89%) [70,216,140,267]
   ✅ person (27.38%) [559,209,672,473]

==================================================
📊 Detection Summary
==================================================
  Images tested:    15
  Total detections: 157
  • cap: 51
  • mask: 43
  • person: 63
==================================================
```

## 📝 License

This project is open-source and available for educational and research purposes.

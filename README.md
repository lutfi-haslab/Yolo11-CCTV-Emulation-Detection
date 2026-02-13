# CCTV Emulation & YOLO Object Detection

This project emulates a 5-camera CCTV system derived from a single webcam or simulation, using YOLO11 for object detection (detecting People, Masks, and Caps).

![CCTV Monitor](monitor.png)

## Prerequisites

- Python 3.8+
- Webcam (optional, defaults to simulation if not found)

## Data Labeling

The dataset for this project is simulated and labeled using [CVAT](https://app.cvat.ai/).

- **Tool**: Computer Vision Annotation Tool (CVAT)
- **Format**: YOLO 1.1
- **Classes**:
  - `0`: Person
  - `1`: Mask
  - `2`: Cap

## Quick Start

1.  **Setup Environment**:
    Run the setup script to create a virtual environment and install dependencies.

    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```

2.  **Activate Virtual Environment**:

    ```bash
    source venv/bin/activate
    ```

3.  **Train the Model**:
    Train the YOLO model on the dataset located in `simple_dataset/`.

    ```bash
    python train_cctv_model.py
    ```

    This will save the trained model to `runs/detect/apd_bgn/cctv_model/weights/best.pt`.

4.  **Test the Model**:
    Test the trained model on a sample image.

    ```bash
    python test_model.py
    ```

    Check `test_result.jpg` for the output.

5.  **Run CCTV Emulation**:
    Run the main emulation script.

    ```bash
    python cctv_emulation.py
    ```

    - **Controls**: Press `q` to quit.
    - **Features**:
      - Simulates 5 camera feeds.
      - Detects Persons, Masks, and Caps.
      - Highlights violations (No Mask or No Cap) in Red.
      - Displays System Stats (CPU, RAM).

## Project Structure

- `cctv_emulation.py`: Main application script.
- `train_cctv_model.py`: Script to train the YOLO model.
- `test_model.py`: Script to test the trained model on static images.
- `setup_yolo.py`: Verifies YOLO installation and downloads the base model.
- `setup_env.sh`: Automates environment setup.
- `requirements.txt`: Python dependencies.
- `simple_dataset/`: Directory containing training data.
- `runs/`: Directory containing training results and models.

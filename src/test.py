#!/usr/bin/env python3
"""
YOLO Testing / Inference Script
================================
Test a trained YOLO model on images, video, or webcam.

Usage:
    python src/test.py                                    # test on labeled_dataset images
    python src/test.py --source images/img_01.png         # single image
    python src/test.py --source images/                   # folder of images
    python src/test.py --source video.mp4                 # video file
    python src/test.py --source 0                         # webcam
    python src/test.py --model path/to/best.pt            # custom model
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "labeling_config.json"
DEFAULT_MODEL = BASE_DIR / "runs" / "detect" / "labeled_model" / "weights" / "best.pt"
DEFAULT_SOURCE = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "test_results"


def load_config():
    """Load labeling config."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def find_best_model():
    """Try to find the best trained model automatically."""
    # Check default location first
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL

    # Search for any best.pt in runs/
    runs_dir = BASE_DIR / "runs" / "detect"
    if runs_dir.exists():
        for best in sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            return best

    return None


def test_image(model, source_path, args):
    """Run inference on a single image and display/save results."""
    results = model(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    for result in results:
        boxes = result.boxes
        filename = Path(result.path).name

        print(f"\n📷 {filename}:")
        if len(boxes) == 0:
            print("   No objects detected.")
        else:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"   ✅ {name} ({conf:.2%}) [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        # Save annotated result
        if args.save:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            save_path = RESULTS_DIR / f"result_{filename}"
            result.save(filename=str(save_path))
            print(f"   💾 Saved: {save_path}")

        # Show result
        if args.show:
            annotated = result.plot()
            cv2.imshow(f"YOLO Detection - {filename}", annotated)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                sys.exit(0)
            cv2.destroyAllWindows()

    return results


def test_video(model, source, args):
    """Run inference on video or webcam."""
    # source can be a path or an int (webcam ID)
    try:
        source_val = int(source)
    except (ValueError, TypeError):
        source_val = str(source)

    results = model(
        source=source_val,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
    )

    print(f"\n🎥 Running inference on: {source}")
    print("   Press 'q' to quit\n")

    for result in results:
        annotated = result.plot()
        cv2.imshow("YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def test_folder(model, folder_path, args):
    """Run inference on all images in a folder."""
    supported = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")
    image_files = sorted([
        f for f in Path(folder_path).iterdir()
        if f.is_file() and f.suffix.lower() in supported
    ])

    if not image_files:
        print(f"❌ No images found in: {folder_path}")
        return

    print(f"\n📂 Testing {len(image_files)} images from: {folder_path}")

    total_detections = 0
    detection_counts = {}

    for img_path in image_files:
        results = test_image(model, img_path, args)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                detection_counts[name] = detection_counts.get(name, 0) + 1
                total_detections += 1

    # Summary
    print("\n" + "=" * 50)
    print("📊 Detection Summary")
    print("=" * 50)
    print(f"  Images tested:    {len(image_files)}")
    print(f"  Total detections: {total_detections}")
    for name, count in sorted(detection_counts.items()):
        print(f"  • {name}: {count}")
    print("=" * 50)


def validate_model(model, args):
    """Run YOLO validation on the dataset."""
    config = load_config()
    data_yaml = BASE_DIR / "labeled_dataset" / "data.yaml"

    if args.data:
        data_yaml = Path(args.data)

    if not data_yaml.exists():
        print(f"❌ data.yaml not found: {data_yaml}")
        sys.exit(1)

    print(f"\n📋 Validating model on: {data_yaml}")

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        device=args.device,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("📊 Validation Results")
    print("=" * 50)
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("=" * 50)

    return metrics


def test(args):
    """Main test function."""
    # Find model
    model_path = Path(args.model) if args.model else find_best_model()

    if model_path is None or not model_path.exists():
        print(f"❌ No trained model found.")
        print(f"   Expected: {DEFAULT_MODEL}")
        print(f"   Train first: python src/train.py")
        sys.exit(1)

    print("=" * 60)
    print("🔍 YOLO Inference")
    print("=" * 60)
    print(f"  🧠 Model:      {model_path}")
    print(f"  🎯 Confidence: {args.conf}")
    print(f"  📐 Image Size: {args.imgsz}")
    print(f"  💻 Device:     {args.device or 'auto'}")
    print("=" * 60)

    model = YOLO(str(model_path))

    # Print model info
    print(f"\n🏷️  Classes: {model.names}")

    # Validation mode
    if args.validate:
        validate_model(model, args)
        return

    # Resolve source
    source = Path(args.source) if args.source else DEFAULT_SOURCE

    # Check if source is a webcam ID
    try:
        cam_id = int(args.source) if args.source else None
        if cam_id is not None:
            test_video(model, cam_id, args)
            return
    except (ValueError, TypeError):
        pass

    if not source.exists():
        print(f"❌ Source not found: {source}")
        sys.exit(1)

    # Video file
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    if source.is_file() and source.suffix.lower() in video_exts:
        test_video(model, source, args)
    # Single image
    elif source.is_file():
        test_image(model, source, args)
    # Folder
    elif source.is_dir():
        test_folder(model, source, args)
    else:
        print(f"❌ Unsupported source: {source}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test/infer a trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/test.py                                    # All images in images/
  python src/test.py --source images/img_01.png         # Single image
  python src/test.py --source images/ --save --show     # Save + display results
  python src/test.py --source video.mp4                 # Video file
  python src/test.py --source 0                         # Webcam
  python src/test.py --validate                         # Run YOLO validation
  python src/test.py --model runs/detect/x/weights/best.pt   # Custom model
        """,
    )
    parser.add_argument("--source", type=str, default=None,
                        help=f"Image/folder/video/webcam (default: {DEFAULT_SOURCE})")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Path to trained model (default: auto-detect)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data.yaml for validation mode")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '', 'cpu', 'mps', '0' (default: auto)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated results to test_results/")
    parser.add_argument("--show", action="store_true",
                        help="Display results in a window")
    parser.add_argument("--validate", action="store_true",
                        help="Run YOLO validation metrics on dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)

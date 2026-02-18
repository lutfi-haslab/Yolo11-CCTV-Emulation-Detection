#!/usr/bin/env python3
"""
YOLO Training Script
====================
Trains a YOLOv11 model using the dataset exported from the labeling app.

Usage:
    python src/train.py                          # default settings
    python src/train.py --epochs 100             # custom epochs
    python src/train.py --data path/to/data.yaml # custom dataset
    python src/train.py --resume                 # resume last training
"""

import argparse
import json
import os
import sys
from pathlib import Path

from ultralytics import YOLO

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "labeling_config.json"
DEFAULT_DATA_YAML = BASE_DIR / "labeled_dataset" / "data.yaml"
DEFAULT_MODEL = BASE_DIR / "yolo11n.pt"
DEFAULT_PROJECT = BASE_DIR / "runs" / "detect"


def load_config():
    """Load labeling config to get dataset settings."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def train(args):
    """Train the YOLO model."""
    config = load_config()

    # Resolve data.yaml path
    data_yaml = Path(args.data) if args.data else DEFAULT_DATA_YAML
    if not data_yaml.exists():
        print(f"❌ Dataset not found: {data_yaml}")
        print("   Run the labeling app first and export the YOLO dataset.")
        print(f"   Expected: {DEFAULT_DATA_YAML}")
        sys.exit(1)

    # Resolve model
    if args.resume:
        # Find last training run to resume
        last_pt = DEFAULT_PROJECT / args.name / "weights" / "last.pt"
        if last_pt.exists():
            model_path = str(last_pt)
            print(f"🔄 Resuming from: {model_path}")
        else:
            print(f"❌ No checkpoint found at {last_pt}")
            sys.exit(1)
    else:
        model_path = args.model if args.model else str(DEFAULT_MODEL)

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Download a pretrained model first (e.g., yolo11n.pt)")
        sys.exit(1)

    # Get settings from config or args
    img_size = args.imgsz or config.get("image_size", 640)
    epochs = args.epochs
    batch = args.batch
    name = args.name

    print("=" * 60)
    print("🏋️  YOLO Training")
    print("=" * 60)
    print(f"  📁 Dataset:    {data_yaml}")
    print(f"  🧠 Model:      {model_path}")
    print(f"  🔁 Epochs:     {epochs}")
    print(f"  📐 Image Size: {img_size}")
    print(f"  📦 Batch:      {batch}")
    print(f"  📂 Project:    {DEFAULT_PROJECT}")
    print(f"  🏷️  Run Name:   {name}")
    print(f"  💻 Device:     {args.device}")
    print("=" * 60)

    # Load and train
    model = YOLO(model_path)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=str(DEFAULT_PROJECT),
        name=name,
        exist_ok=True,
        device=args.device,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        resume=args.resume,
        verbose=True,
    )

    # Print results
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"  📊 Results:    {results.save_dir}")
    print(f"  🏆 Best Model: {best_pt}")
    print(f"  📈 Last Model: {Path(results.save_dir) / 'weights' / 'last.pt'}")
    print("=" * 60)
    print(f"\nTo test: python src/test.py --model {best_pt}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv11 model on labeled dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train.py                           # Train with defaults
  python src/train.py --epochs 100 --batch 16   # Custom training
  python src/train.py --data custom/data.yaml   # Custom dataset
  python src/train.py --resume                  # Resume last run
  python src/train.py --device mps              # Use Apple Silicon GPU
        """,
    )
    parser.add_argument("--data", type=str, default=None,
                        help=f"Path to data.yaml (default: {DEFAULT_DATA_YAML})")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Path to base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Image size (default: from config or 640)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--name", type=str, default="labeled_model",
                        help="Run name (default: labeled_model)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '', 'cpu', 'mps', '0' (default: auto)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--save-period", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

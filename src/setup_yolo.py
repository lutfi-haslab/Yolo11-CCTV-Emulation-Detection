from ultralytics import YOLO
import sys

def main():
    print("Loading YOLO11n model...")
    # Load the YOLO11n model
    try:
        model = YOLO("yolo11n.pt")
        print("Model yolo11n.pt loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Perform a test prediction to verify the model and environment
    # using standard image size 640 as requested
    print("Running test prediction on 'bus.jpg' (auto-downloaded)...")
    results = model.predict(source="https://ultralytics.com/images/bus.jpg", imgsz=640, save=True)
    print(f"Prediction complete. Results saved to {results[0].save_dir}")

if __name__ == "__main__":
    main()

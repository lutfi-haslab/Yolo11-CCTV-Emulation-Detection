from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # data: Path to data.yaml
    # epochs: Set to 50 for quick run on small dataset
    # imgsz: 640
    # project: Where to save results (default is 'runs/detect')
    # name: subdirectory for this run
    results = model.train(data="simple_dataset/data.yaml", 
                          epochs=50, 
                          imgsz=640, 
                          project="runs/detect/apd_bgn", 
                          name="cctv_model",
                          exist_ok=True) # Overwrite existing run for simplicity

    print(f"Training completed. Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_model()

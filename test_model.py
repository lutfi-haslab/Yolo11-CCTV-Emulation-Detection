from ultralytics import YOLO
import cv2
import os

def test_on_image():
    # Load the trained model
    model_path = "runs/detect/apd_bgn/cctv_model/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Use a test image from the dataset
    image_path = "simple_dataset/images/img_04.jpeg" 
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        # fallback to any image found
        train_dir = "simple_dataset/images/train"
        if os.path.isdir(train_dir):
            files = os.listdir(train_dir)
            if files:
                image_path = os.path.join(train_dir, files[0])
                print(f"Using fallback image: {image_path}")
            else:
                return
        else:
            return

    print(f"Testing on image: {image_path}")
    
    # Run inference with a very low confidence threshold to debug
    results = model(image_path, conf=0.01)
    
    # Process results
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects (conf >= 0.01):")
        if len(boxes) == 0:
             print("No objects detected even at low confidence.")
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            print(f"- {name} (conf: {conf:.4f})")
            
        # Save result
        result.save(filename="test_result.jpg")
        print("Annotated image saved to test_result.jpg")

if __name__ == "__main__":
    test_on_image()

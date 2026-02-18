
import cv2
import time
import sys
import os
import numpy as np
import psutil
from ultralytics import YOLO

def create_noise_frame(height, width):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def main():
    print("Initializing 5-Camera CCTV Emulation on Mac M1...")
    
    # Load the YOLO model (nano version for speed)
    # Use relative path to make it portable
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    model_rpath = os.path.join(script_dir, "runs", "detect", "labeled_model", "weights", "best.pt")
    
    # Fallback to absolute path if needed, but try relative first
    if os.path.exists(model_rpath):
        model_path = model_rpath
    else:
        # Fallback to the hardcoded path if relative fails for some reason (e.g. run from weird CWD)
        model_path = "/Users/hy4-mac-002/hasdev/myEaiApp/runs/detect/labeled_model/weights/best.pt"

    try:
        model = YOLO(model_path)
        print(f"YOLO11n model loaded from {model_path}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    use_webcam = True
    if not cap.isOpened():
        print("Warning: Could not open webcam (Source 0).")
        print("This might be due to missing camera permissions in your terminal.")
        print("Switching to 'NO SIGNAL' simulation mode.")
        use_webcam = False
    else:
        # Try to read one frame to confirm
        ret, frame = cap.read()
        if not ret:
            print("Warning: Camera opened but failed to read frame.")
            print("Switching to 'NO SIGNAL' simulation mode.")
            use_webcam = False

    if use_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Webcam initialized successfully.")

    print("Press 'q' to quit.")

    # Target display size for each grid element
    disp_w, disp_h = 426, 320

    # Camera Labels
    camera_labels = ["CAM 01 - MAIN ENT", "CAM 02 - LOBBY", "CAM 03 - PARKING", "CAM 04 - SERVER", "CAM 05 - ROOF"]
    
    # Initialize CPU measurement (first call returns 0)
    psutil.cpu_percent(interval=None)

    while True:
        start_time = time.time()
        
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                print("Error: Lost connection to webcam.")
                use_webcam = False
                frame = create_noise_frame(480, 640)
        else:
            # Generate static noise if no webcam
            frame = create_noise_frame(480, 640)
            # Add "NO SIGNAL" text to the base frame
            cv2.putText(frame, "NO SIGNAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Run Object Detection on original frame (640x480) for better detection of small objects
        results = model(frame, conf=0.05, verbose=False)
        
        # We need to manually draw now to implement grouping logic
        annotated_frame_high_res = frame.copy()
        
        # Parse detections
        persons = []
        masks = []
        caps = []
        
        # Map detections by class NAME (not ID) so it works with any training data
        name_to_id = {name: idx for idx, name in model.names.items()}
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                item = {'bbox': xyxy, 'conf': conf}
                cls_name = model.names.get(cls, '')
                
                if cls_name == 'person':
                    persons.append(item)
                elif cls_name == 'mask':
                    masks.append(item)
                elif cls_name == 'cap':
                    caps.append(item)

        # Helper to check if item is "owned" by person (simple center check or intersection)
        def is_owned(person_bbox, item_bbox):
            px1, py1, px2, py2 = person_bbox
            ix1, iy1, ix2, iy2 = item_bbox
            
            # Check center of item
            icx = (ix1 + ix2) / 2
            icy = (iy1 + iy2) / 2
            
            if px1 < icx < px2 and py1 < icy < py2:
                return True
            return False

        # Associate and Draw
        # Associate and Draw
        violations = []
        
        for idx, p in enumerate(persons):
            p_box = p['bbox']
            
            has_mask = False
            has_cap = False
            
            # Check for mask
            for m in masks:
                if is_owned(p_box, m['bbox']):
                    has_mask = True
                    mask_conf = m['conf']
                    # Draw mask box lightly
                    cv2.rectangle(annotated_frame_high_res, (m['bbox'][0], m['bbox'][1]), (m['bbox'][2], m['bbox'][3]), (255, 200, 0), 1)
                    # cv2.putText(annotated_frame_high_res, f"mask", (m['bbox'][0], m['bbox'][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            
            # Check for cap
            for c in caps:
                if is_owned(p_box, c['bbox']):
                    has_cap = True
                    cap_conf = c['conf']
                    # Draw cap box lightly
                    cv2.rectangle(annotated_frame_high_res, (c['bbox'][0], c['bbox'][1]), (c['bbox'][2], c['bbox'][3]), (255, 0, 255), 1)
                    # cv2.putText(annotated_frame_high_res, f"cap", (c['bbox'][0], c['bbox'][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Determine Status
            is_violation = not (has_mask and has_cap)
            color = (0, 0, 255) if is_violation else (0, 255, 0) # Red if violation, Green if OK
            
            # Draw Person Box
            cv2.rectangle(annotated_frame_high_res, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
            
            # Draw ID Label above box
            label_text = f"ID: {idx+1}"
            cv2.putText(annotated_frame_high_res, label_text, (p_box[0], p_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if is_violation:
                violations.append({
                    'id': idx+1,
                    'mask': has_mask,
                    'cap': has_cap
                })

        # Draw Violation List on Bottom Left
        if violations:
            no_caps = [str(v['id']) for v in violations if not v['cap']]
            no_masks = [str(v['id']) for v in violations if not v['mask']]
            
            lines = ["Violation:"]
            if no_caps:
                lines.append(f"No Cap: {','.join(no_caps)}")
            if no_masks:
                lines.append(f"No Mask: {','.join(no_masks)}")
            
            # Stats panel background
            panel_h = len(lines) * 30 + 10
            panel_w = 250
            panel_x = 10
            panel_y = frame.shape[0] - panel_h - 10
            
            sub_img = annotated_frame_high_res[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.7, black_rect, 0.3, 1.0) # 30% transparency
            annotated_frame_high_res[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = res
            
            for i, line in enumerate(lines):
                y_pos = panel_y + 25 + (i * 30)
                color = (0, 0, 255) # Red text
                if i == 0: color = (0, 255, 255) # Yellow header
                cv2.putText(annotated_frame_high_res, line, (panel_x+10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Resize for display grid 
        annotated_frame = cv2.resize(annotated_frame_high_res, (disp_w, disp_h))

        # Create the grid canvas
        # Canvas size: (3 * disp_w) x (2 * disp_h) -> 1278 x 640
        canvas_h = disp_h * 2
        canvas_w = disp_w * 3
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for i in range(5):
            # Calculate grid position
            row = i // 3
            col = i % 3
            
            y_offset = row * disp_h
            x_offset = col * disp_w
            
            # Start with the annotated frame
            cam_view = annotated_frame.copy()
            
            # Add specific camera overlay
            color = (0, 255, 0) if use_webcam else (0, 0, 255)
            cv2.putText(cam_view, camera_labels[i], (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(cam_view, timestamp, (10, disp_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Modulate the image slightly to make them look distinct
            if i == 1: # Blue tint for CAM 02
                cam_view[:, :, 0] = cv2.add(cam_view[:, :, 0], 30)
            elif i == 2: # Red tint for CAM 03
                cam_view[:, :, 2] = cv2.add(cam_view[:, :, 2], 30)
            elif i == 3: # Darker for CAM 04
                cam_view = (cam_view * 0.8).astype(np.uint8)
            elif i == 4 and not use_webcam:
                 # Make one camera strictly noise if everything is simulated
                 cam_view = create_noise_frame(disp_h, disp_w)
                 cv2.putText(cam_view, camera_labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                 cv2.putText(cam_view, "OFFLINE", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Place in canvas
            canvas[y_offset:y_offset+disp_h, x_offset:x_offset+disp_w] = cam_view

        # Fill the 6th slot with stats
        row = 1
        col = 2
        y_offset = row * disp_h
        x_offset = col * disp_w
        
        # Info Panel
        info_panel = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        # Make it a dark grey background
        info_panel[:] = (30, 30, 30)
        
        current_time = time.time()
        fps = 1.0 / (current_time - start_time) if (current_time - start_time) > 0 else 0
        
        # Get System Stats
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        ram_usage = memory.percent
        ram_used_gb = memory.used / (1024 ** 3)
        ram_total_gb = memory.total / (1024 ** 3)

        cv2.putText(info_panel, "SYSTEM STATUS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        status_color = (0, 255, 0) if use_webcam else (0, 0, 255)
        status_text = "ONLINE" if use_webcam else "NO SIGNAL"
        cv2.putText(info_panel, f"Link: {status_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # CPU & RAM Stats
        cv2.putText(info_panel, f"CPU Load: {cpu_usage}%", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, f"RAM Usage: {ram_usage}%", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, f"RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.putText(info_panel, "ACTIVE CAMS: 5/5", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_panel, f"DEVICE: MAC M1", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        canvas[y_offset:y_offset+disp_h, x_offset:x_offset+disp_w] = info_panel

        cv2.imshow("CCTV Control Center - 5 Camera Emulation", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

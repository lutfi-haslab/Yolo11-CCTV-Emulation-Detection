
import cv2
import time
import sys
import os
import numpy as np
import psutil
from ultralytics import YOLO

# Configuration
# Define your Manual RTSP/IPTV Sources here.
# You can use RTSP, HTTP, or RTMP URLs.
STREAM_SOURCES = [
    "http://127.0.0.1:5001/video_feed_1",  # CAM 01
    "http://127.0.0.1:5001/video_feed_2",  # CAM 02
    "http://127.0.0.1:5001/video_feed_3",  # CAM 03
    "http://127.0.0.1:5001/video_feed_4",  # CAM 04
    "http://127.0.0.1:5001/video_feed_5",  # CAM 05
]

def get_video_captures(source_type):
    """
    Open video captures based on source type.
    source_type: 'webcam' or 'stream'
    Returns: List of 5 items, each is (cap, source_name) or (None, str)
    """
    captures = []
    
    if source_type == "webcam":
        # Just one webcam, but we will return it as list of 1 used + 4 placeholders
        cap = cv2.VideoCapture(0)
        source_name = "Webcam (0)"
        
        if not cap.isOpened():
             print(f"Warning: Could not open {source_name}.")
             captures.append((None, source_name))
        else:
             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
             captures.append((cap, source_name))
             
        # Fill others as None (handled by duplication or noise)
        for i in range(4):
            captures.append((None, "N/A"))
            
            
    elif source_type == "stream":
        # Manual Sources Iteration
        for i, url in enumerate(STREAM_SOURCES):
            cam_idx = i + 1
            print(f"Connecting to Stream {cam_idx}: {url}")
            
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                captures.append((cap, f"Stream {cam_idx}"))
            else:
                print(f"Failed to connect to {url}")
                captures.append((None, f"Stream {cam_idx} (Offline)"))
                
    return captures

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
        model_path = "/Users/hy4-mac-002/hasdev/myEaiApp/yolo-cctv/runs/detect/labeled_model/weights/best.pt"

    try:
        model = YOLO(model_path)
        print(f"YOLO11n model loaded from {model_path}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Initial Video Source
    current_source_type = "webcam" # or "stream"
    captures = get_video_captures(current_source_type)
    
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Switch Source (Webcam <-> 5-Stream)")
    
    # State tracking
    is_offline = not any(c is not None and c.isOpened() for c, _ in captures)
    if is_offline:
         print("Starting in OFFLINE mode (No active sources).")

    # Target display size for each grid element
    disp_w, disp_h = 426, 320

    # Camera Labels
    camera_labels = ["CAM 01 - MAIN ENT", "CAM 02 - LOBBY", "CAM 03 - PARKING", "CAM 04 - SERVER", "CAM 05 - ROOF"]
    
    # Initialize CPU measurement (first call returns 0)
    psutil.cpu_percent(interval=None)

    frame_count = 0
    # Cache for detections to support Round Robin or reuse
    # Format: [ (frame_img, [results]) ] * 5
    detection_cache = [None] * 5 

    while True:
        start_time = time.time()
        frame_count += 1
        
        # ---------------------------------------------------------
        # 1. READ FRAMES
        # ---------------------------------------------------------
        frames = []
        
        # Helper to read frame properly
        def read_frame(cap_obj):
            if cap_obj is None: return None
            ret, f = cap_obj.read()
            return f if ret else None

        # If Webcam mode: Read once, duplicate
        # If Stream mode: Read all 5
        if current_source_type == "webcam":
            main_cap = captures[0][0]
            base_frame = read_frame(main_cap)
            
            if base_frame is None:
                # Webcam offline or failed
                pass 
            
            for i in range(5):
                if base_frame is not None:
                     frames.append(base_frame.copy())
                else:
                     frames.append(None)
        else:
            # Stream Mode - Read each source
            for i in range(5):
                cam_cap = captures[i][0]
                f = read_frame(cam_cap)
                frames.append(f)

        # ---------------------------------------------------------
        # 2. PROCESS & DRAW
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # PROCESS & DRAW STRATEGY
        # ---------------------------------------------------------
        
        # Canvas Setup
        canvas_h = disp_h * 2
        canvas_w = disp_w * 3
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        active_cams_count = 0

        # Helper for Draw Logic
        def process_and_draw(frame_img, cam_index, run_detection=False):
            if frame_img is None:
                # Noise Frame
                noise = create_noise_frame(480, 640)
                cv2.putText(noise, "NO SIGNAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                return cv2.resize(noise, (disp_w, disp_h)), False

            # Detection
            items = []
            if run_detection:
                results = model(frame_img, conf=0.05, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cls_name = model.names.get(cls, '')
                        items.append({'cls': cls_name, 'bbox': xyxy, 'conf': float(box.conf[0])})
            
            # Use provided items or cache? 
            # Ideally we pass detections IN, but for now let's reuse the cache logic outside this helper
            # or return the items to update cache.
            return items 

        # ---------------------------------------------------------
        # WEBCAM MODE: PROCESS ONCE, BLIT 5 TIMES
        # ---------------------------------------------------------
        if current_source_type == "webcam":
            base_frame = frames[0]
            if base_frame is not None:
                active_cams_count = 5
                
                # 1. Detect ONCE
                current_detections = []
                # Always run detection on base frame in webcam mode (it's fast enough 1x)
                results = model(base_frame, conf=0.05, verbose=False)
                for r in results:
                     for box in r.boxes:
                        cls = int(box.cls[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cls_name = model.names.get(cls, '')
                        current_detections.append({'cls': cls_name, 'bbox': xyxy, 'conf': float(box.conf[0])})
                
                # 2. Draw ONCE
                annotated = base_frame.copy()
                
                # Separate items
                masks = [x for x in current_detections if x['cls'] == 'mask']
                caps  = [x for x in current_detections if x['cls'] == 'cap']
                persons = [x for x in current_detections if x['cls'] == 'person']
                
                for m in masks: cv2.rectangle(annotated, (m['bbox'][0], m['bbox'][1]), (m['bbox'][2], m['bbox'][3]), (255, 200, 0), 1)
                for c in caps: cv2.rectangle(annotated, (c['bbox'][0], c['bbox'][1]), (c['bbox'][2], c['bbox'][3]), (255, 0, 255), 1)
                
                violations = []
                for idx, p in enumerate(persons):
                    p_box = p['bbox']
                    has_mask = any(p_box[0] < (m['bbox'][0]+m['bbox'][2])/2 < p_box[2] and p_box[1] < (m['bbox'][1]+m['bbox'][3])/2 < p_box[3] for m in masks)
                    has_cap = any(p_box[0] < (c['bbox'][0]+c['bbox'][2])/2 < p_box[2] and p_box[1] < (c['bbox'][1]+c['bbox'][3])/2 < p_box[3] for c in caps)
                    is_vio = not (has_mask and has_cap)
                    c_color = (0, 0, 255) if is_vio else (0, 255, 0)
                    cv2.rectangle(annotated, (p_box[0], p_box[1]), (p_box[2], p_box[3]), c_color, 2)
                    cv2.putText(annotated, f"ID: {idx+1}", (p_box[0], p_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_color, 2)
                    if is_vio: violations.append({'id': idx+1, 'mask': has_mask, 'cap': has_cap})
                
                if violations:
                     no_caps = [str(v['id']) for v in violations if not v['cap']]
                     no_masks = [str(v['id']) for v in violations if not v['mask']]
                     lines = ["Violation:"]
                     if no_caps: lines.append(f"No Cap: {','.join(no_caps)}")
                     if no_masks: lines.append(f"No Mask: {','.join(no_masks)}")
                     ph = len(lines)*30 + 10
                     py = annotated.shape[0] - ph - 10
                     if py > 0:
                         sub = annotated[py:py+ph, 10:260]
                         res = cv2.addWeighted(sub, 0.7, np.zeros(sub.shape, dtype=np.uint8), 0.3, 1.0)
                         annotated[py:py+ph, 10:260] = res
                         for k, line in enumerate(lines):
                             cc = (0, 255, 255) if k==0 else (0, 0, 255)
                             cv2.putText(annotated, line, (20, py + 25 + (k*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cc, 2)
                
                # 3. Resize ONCE
                base_disp = cv2.resize(annotated, (disp_w, disp_h))
                
                # 4. Blit 5 Times with Overlays
                for i in range(5):
                    view = base_disp.copy()
                    
                    # Tint
                    if i == 1: view[:,:,0] = cv2.add(view[:,:,0], 30)
                    elif i == 2: view[:,:,2] = cv2.add(view[:,:,2], 30)
                    elif i == 3: view = (view * 0.8).astype(np.uint8)
                    
                    # Overlays
                    cv2.putText(view, camera_labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(view, time.strftime("%H:%M:%S"), (10, disp_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Place
                    row, col = i // 3, i % 3
                    canvas[row*disp_h:(row+1)*disp_h, col*disp_w:(col+1)*disp_w] = view
            else:
                # Base frame is None (Webcam failure)
                pass # Already black canvas
                
        # ---------------------------------------------------------
        # STREAM MODE: PROCESS EACH FRAME
        # ---------------------------------------------------------
        else:
            for i in range(5):
                frame = frames[i]
                active_cams_count += 1 if frame is not None else 0
                
                # Round Robin Detection
                should_run = ((frame_count % 5) == i) and (frame is not None)
                
                # ... reuse detection logic? Or update cache ...
                current_detections = detection_cache[i]
                
                if should_run:
                    # Detect
                    new_dets = []
                    results = model(frame, conf=0.05, verbose=False)
                    for r in results:
                         for box in r.boxes:
                            cls = int(box.cls[0])
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            cls_name = model.names.get(cls, '')
                            new_dets.append({'cls': cls_name, 'bbox': xyxy, 'conf': float(box.conf[0])})
                    current_detections = new_dets
                    detection_cache[i] = new_dets
                
                # Draw
                final_draw_frame = None
                if frame is None:
                    # Noise
                    final_draw_frame = create_noise_frame(480, 640)
                    cv2.putText(final_draw_frame, "NO SIGNAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    label_c = (0, 0, 255)
                else:
                    annotated = frame.copy()
                    label_c = (0, 255, 0)
                    
                    if current_detections:
                        # Draw logic (condensed)
                        masks = [x for x in current_detections if x['cls']=='mask']
                        caps = [x for x in current_detections if x['cls']=='cap']
                        persons = [x for x in current_detections if x['cls']=='person']
                        
                        for m in masks: cv2.rectangle(annotated, (m['bbox'][0], m['bbox'][1]), (m['bbox'][2], m['bbox'][3]), (255, 200, 0), 1)
                        for c in caps: cv2.rectangle(annotated, (c['bbox'][0], c['bbox'][1]), (c['bbox'][2], c['bbox'][3]), (255, 0, 255), 1)
                        
                        violations = []
                        for idx, p in enumerate(persons):
                            p_box = p['bbox']
                            has_mask = any(p_box[0] < (m['bbox'][0]+m['bbox'][2])/2 < p_box[2] and p_box[1] < (m['bbox'][1]+m['bbox'][3])/2 < p_box[3] for m in masks)
                            has_cap = any(p_box[0] < (c['bbox'][0]+c['bbox'][2])/2 < p_box[2] and p_box[1] < (c['bbox'][1]+c['bbox'][3])/2 < p_box[3] for c in caps)
                            is_vio = not (has_mask and has_cap)
                            cc = (0, 0, 255) if is_vio else (0, 255, 0)
                            cv2.rectangle(annotated, (p_box[0], p_box[1]), (p_box[2], p_box[3]), cc, 2)
                            cv2.putText(annotated, f"ID: {idx+1}", (p_box[0], p_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cc, 2)
                            if is_vio: violations.append({'id': idx+1, 'mask': has_mask, 'cap': has_cap})
                        
                        if violations:
                             # Simplified Panel Draw
                             no_caps = [str(v['id']) for v in violations if not v['cap']]
                             no_masks = [str(v['id']) for v in violations if not v['mask']]
                             lines = ["Violation:"]
                             if no_caps: lines.append(f"No C: {','.join(no_caps)}")
                             if no_masks: lines.append(f"No M: {','.join(no_masks)}")
                             ph = len(lines)*30 + 10
                             py = annotated.shape[0] - ph - 10
                             if py > 0:
                                 sub = annotated[py:py+ph, 10:200] # Smaller panel
                                 res = cv2.addWeighted(sub, 0.7, np.zeros(sub.shape, dtype=np.uint8), 0.3, 1.0)
                                 annotated[py:py+ph, 10:200] = res
                                 for k, line in enumerate(lines):
                                     cc = (0, 255, 255) if k==0 else (0, 0, 255)
                                     cv2.putText(annotated, line, (20, py + 25 + (k*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cc, 2)

                    final_draw_frame = annotated

                # Resize
                view = cv2.resize(final_draw_frame, (disp_w, disp_h))
                
                # Overlays (Tint + Text)
                if frame is not None:
                     if i==1: view[:,:,0] = cv2.add(view[:,:,0], 30)
                     elif i==2: view[:,:,2] = cv2.add(view[:,:,2], 30)
                     elif i==3: view = (view*0.8).astype(np.uint8)
                
                cv2.putText(view, camera_labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_c, 2)
                cv2.putText(view, time.strftime("%H:%M:%S"), (10, disp_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_c, 1)

                row, col = i // 3, i % 3
                canvas[row*disp_h:(row+1)*disp_h, col*disp_w:(col+1)*disp_w] = view

        # Draw Stats
        # ---------------------------------------------------------
        # 3. STATS PANEL (Row 1, Col 2)
        # ---------------------------------------------------------
        row, col = 1, 2
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
        
        status_color = (0, 255, 0) if not is_offline else (0, 0, 255)
        status_text = f"ONLINE ({current_source_type.upper()})" if not is_offline else "NO SIGNAL"
        cv2.putText(info_panel, f"Link: {status_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # CPU & RAM Stats
        cv2.putText(info_panel, f"CPU Load: {cpu_usage}%", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, f"RAM Usage: {ram_usage}%", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, f"RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.putText(info_panel, f"ACTIVE CAMS: {active_cams_count}/5", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_panel, f"DEVICE: MAC M1", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        canvas[y_offset:y_offset+disp_h, x_offset:x_offset+disp_w] = info_panel

        cv2.imshow("CCTV Control Center - 5 Camera Emulation", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Switch Source
            print("Switching video source...")
            # Release old
            for c, _ in captures:
                if c: c.release()
            
            # Toggle type
            if current_source_type == "webcam":
                current_source_type = "stream"
            else:
                current_source_type = "webcam"
                
            # Re-open
            captures = get_video_captures(current_source_type)
            # Re-eval if offline (just check if any is open)
            is_offline = not any(c is not None and c.isOpened() for c, _ in captures)

    # Cleanup
    for c, _ in captures:
        if c: c.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

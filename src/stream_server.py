
import cv2
import time
import threading
from flask import Flask, Response

app = Flask(__name__)

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.frame = None
        self.claiming_thread = threading.Thread(target=self._update, daemon=True)
        self.claiming_thread.start()

    def _update(self):
        while True:
            ret, frame = self.video.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# Global camera instance
cam = Camera()

def generate_frames(cam_id):
    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Add visual distinction for each stream
        h, w = frame.shape[:2]
        
        # Stream Source Label (Bottom Right, Small)
        label = f"STREAM SOURCE {cam_id}"
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        x_pos = w - text_w - 10
        y_pos = h - 10
        
        # Background for text
        cv2.rectangle(frame, (x_pos - 5, y_pos - text_h - 5), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, label, (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        # Timestamp (Top Right)
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"{ts}", (w-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed_1')
def video_feed_1():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    return Response(generate_frames(2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_3')
def video_feed_3():
    return Response(generate_frames(3), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_4')
def video_feed_4():
    return Response(generate_frames(4), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_5')
def video_feed_5():
    return Response(generate_frames(5), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <h1>5-Camera Streaming Server</h1>
    <p><a href="/video_feed_1">Cam 1</a></p>
    <p><a href="/video_feed_2">Cam 2</a></p>
    <p><a href="/video_feed_3">Cam 3</a></p>
    <p><a href="/video_feed_4">Cam 4</a></p>
    <p><a href="/video_feed_5">Cam 5</a></p>
    """

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from other devices if needed
    # Port 5001 to avoid conflict with default Flask 5000 if used elsewhere, 
    # but 5000 is standard. Let's use 5001 just in case the user has other things running.
    # Actually, the user asked for default server from localhost.
    app.run(host='0.0.0.0', port=5001, debug=False)

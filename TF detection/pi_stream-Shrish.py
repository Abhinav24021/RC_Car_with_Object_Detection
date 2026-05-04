"""
Traffic Light Detection - Pi Camera Live Stream
================================================
Runs YOLOv8 ONNX model on Raspberry Pi 4 using OpenCV DNN.
No PyTorch or onnxruntime needed — just OpenCV!

Usage (on Raspberry Pi):
    python3 pi_stream.py

Then open in browser:
    http://<PI_IP>:8000/

Requirements (install on Pi):
    pip install flask opencv-python-headless numpy
"""

from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time

app = Flask(__name__)

# ==================== CONFIGURATION ====================
MODEL_PATH = "runs/detect/train24/weights/best.onnx"
IMG_SIZE = 320
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CLASS_NAMES = ['red', 'green']
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
}
# ========================================================

# ================= LOAD MODEL WITH OPENCV DNN =================
print("=" * 50)
print("  Traffic Light Detection - Pi Camera Stream")
print("=" * 50)

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print(f"[OK] Model loaded with OpenCV DNN: {MODEL_PATH}")

# ================= CAMERA =================
picam2 = Picamera2()
# CRITICAL: Force RGB888 format to prevent red/blue color swap.
# Default video config uses XBGR8888 (native BGR) which causes
# the RGB→BGR conversion to double-swap and invert colors.
config = picam2.create_video_configuration(
    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
    controls={
        "AwbEnable": True,
        "AeEnable": True,
    }
)
picam2.configure(config)
picam2.start()
import time as _time
_time.sleep(3)  # Give AWB/AE time to settle
actual_fmt = picam2.camera_configuration()['main']['format']
print(f"[OK] Camera started — pixel format: {actual_fmt}")
if actual_fmt != 'RGB888':
    print(f"[WARNING] Expected RGB888 but got {actual_fmt}! Colors may be wrong.")


def detect(frame):
    """Run YOLOv8 inference using OpenCV DNN and return detections."""
    h, w = frame.shape[:2]

    # Preprocess: resize, normalize, convert to blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IMG_SIZE, IMG_SIZE),
                                  swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    output = net.forward()

    # YOLOv8 output shape: (1, num_classes+4, num_detections)
    # Transpose to (num_detections, num_classes+4)
    output = output[0].T

    boxes = []
    confidences = []
    class_ids = []

    x_scale = w / IMG_SIZE
    y_scale = h / IMG_SIZE

    for detection in output:
        # First 4 values are cx, cy, w, h
        cx, cy, bw, bh = detection[0], detection[1], detection[2], detection[3]

        # Remaining values are class scores
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        confidence = float(class_scores[class_id])

        if confidence > CONFIDENCE_THRESHOLD:
            # Convert from center format to corner format
            x1 = int((cx - bw / 2) * x_scale)
            y1 = int((cy - bh / 2) * y_scale)
            box_w = int(bw * x_scale)
            box_h = int(bh * y_scale)

            boxes.append([x1, y1, box_w, box_h])
            confidences.append(confidence)
            class_ids.append(int(class_id))

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((boxes[i], confidences[i], class_ids[i]))

    return results


# ================= STREAM =================
def generate():
    fps_val = 0.0
    fps_counter = 0
    fps_start = time.time()

    while True:
        # Capture frame from Pi Camera (RGB888 format guaranteed)
        frame = picam2.capture_array()

        # With format='RGB888', camera outputs pure 3-channel R,G,B.
        # Convert to BGR for OpenCV processing.
        if frame.ndim == 3 and frame.shape[2] == 4:
            # Safety fallback: strip alpha if somehow 4 channels
            frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run detection
        detections = detect(frame)

        # Draw results
        for box, conf, cls_id in detections:
            x, y, w, h = box
            if cls_id < len(CLASS_NAMES):
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                color = COLORS.get(CLASS_NAMES[cls_id], (255, 255, 0))
            else:
                label = f"cls{cls_id} {conf:.2f}"
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_val = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(frame, f"FPS: {fps_val:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Encode and stream
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Light AI Camera</title>
        <style>
            body {
                background: #1a1a2e; color: #eee;
                font-family: 'Segoe UI', sans-serif;
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                min-height: 100vh; margin: 0;
            }
            h1 { color: #00ff88; margin-bottom: 10px; }
            .subtitle { color: #888; margin-bottom: 20px; }
            img {
                border: 3px solid #00ff88;
                border-radius: 12px;
                max-width: 95vw;
            }
            .status { color: #00ff88; margin-top: 15px; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>&#128678; Traffic Light AI Camera</h1>
        <p class="subtitle">YOLOv8 Detection on Raspberry Pi 4</p>
        <img src="/video" alt="Live Stream">
        <p class="status">&#9679; Live Stream Active</p>
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print()
    print("Stream available at: http://0.0.0.0:8000/")
    print("Press Ctrl+C to stop")
    print()
    app.run(host='0.0.0.0', port=8000, threaded=True)

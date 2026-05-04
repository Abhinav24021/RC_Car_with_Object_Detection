"""
Traffic Light Detection for RC Car - IP Webcam
=======================================================================
Detects red and green traffic lights using the trained ONNX model
reading from an IP Webcam video stream via OpenCV.
Outputs the detected state for RC car decision-making.

Usage:
    python3 rc_original_confidence.py

Requirements:
    pip3 install ultralytics opencv-python-headless
"""

import time
import sys
import cv2

# Try importing detection backend
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


# ==================== CONFIGURATION ====================
# IP Webcam Stream URL (Replace exactly with the URL from your IP Webcam app)
# Look for the format: http://[IP_ADDRESS]:[PORT]/video
CAMERA_URL = "http://192.168.72.254:8080/video"

MODEL_PATH = r"runs\detect\train17\weights\best.pt"  # Update path after copying model to Pi
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 320          # Must match export size
DISPLAY_OUTPUT = True  # Set True if you have a monitor connected to see the boxes
CLASS_NAMES = ['red', 'green']

# RC Car action mapping
ACTIONS = {
    'red': 'STOP',
    'green': 'GO',
    'none': 'PROCEED_WITH_CAUTION',
}
# =======================================================


def setup_camera():
    """Initialize OpenCV VideoCapture for the IP stream."""
    print(f"Connecting to IP webcam: {CAMERA_URL}")
    print("Waiting for stream...")
    
    cap = cv2.VideoCapture(CAMERA_URL)
    
    # Optional: Set a smaller buffer size to minimize delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not connect to camera stream at {CAMERA_URL}")
        print("Please check the IP address and ensure the phone app is running.")
        sys.exit(1)
        
    print("[OK] Camera stream connected")
    return cap


def setup_model():
    """Load the ONNX model."""
    if not HAS_ULTRALYTICS:
        print("[ERROR] ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)
    
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task='detect')
    print("[OK] Model loaded")
    return model


def get_traffic_light_state(results):
    """Parse YOLO results and determine the dominant traffic light state."""
    best_conf = 0
    best_class = 'none'
    
    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.data is None or len(boxes.data) == 0:
            continue
            
        for i in range(len(boxes.data)):
            conf = float(boxes.data[i, 4])
            cls_id = int(boxes.data[i, 5])
            
            if conf > best_conf and cls_id < len(CLASS_NAMES):
                best_conf = conf
                best_class = CLASS_NAMES[cls_id]
                
    return best_class, best_conf


def main():
    print("=" * 50)
    print("  Traffic Light Detector - IP Webcam RC Car")
    print("=" * 50)
    print()
    
    cap = setup_camera()
    model = setup_model()
    
    print()
    print("Starting detection loop... Press Ctrl+C to stop")
    print("-" * 50)
    
    fps_counter = 0
    fps_start = time.time()
    
    try:
        while True:
            # Capture frame from IP cam stream
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Dropped frame from IP camera, trying again...")
                time.sleep(0.1)
                continue
            
            # Run inference
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                show=False,
            )
            
            # Get detection result
            state, confidence = get_traffic_light_state(results)
            action = ACTIONS.get(state, 'PROCEED_WITH_CAUTION')
            
            if DISPLAY_OUTPUT:
                annotated_frame = results[0].plot()
                cv2.imshow("Traffic Light Detection", annotated_frame)
                cv2.waitKey(1)
            
            # FPS tracking
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 2.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()
                print(f"  [{fps:.1f} FPS] Light: {state.upper():6s} | Conf: {confidence:.2f} | Action: {action}")
            
            # === YOUR RC CAR CONTROL LOGIC GOES HERE ===
            # Example:
            # if action == 'STOP':
            #     motor.stop()
            # elif action == 'GO':
            #     motor.forward()
            # ============================================
            
    except KeyboardInterrupt:
        print()
        print("Stopped by user.")
    finally:
        cap.release()
        if DISPLAY_OUTPUT:
            cv2.destroyAllWindows()
        print("[OK] Camera stream released. Goodbye!")


if __name__ == '__main__':
    main()

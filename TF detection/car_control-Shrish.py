from flask import Flask, request, render_template_string, Response
import RPi.GPIO as GPIO
import time
import threading
import atexit
import cv2
import numpy as np
import board
import busio
import adafruit_vl53l0x

app = Flask(__name__)

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pin definitions
IN1, IN2, IN3, IN4 = 17, 27, 23, 24
ENA, ENB = 18, 13

GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)
GPIO.setup([ENA, ENB], GPIO.OUT)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)

pwmA.start(0)
pwmB.start(0)

# ================= TOF SENSOR SETUP =================
# We wrap this in a try-except so it doesn't crash the script immediately if I2C fails
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    tof_sensor = adafruit_vl53l0x.VL53L0X(i2c)
    tof_sensor.measurement_timing_budget = 200000  # Reliable reading (~200ms)
    tof_sensor_ready = True
    print("[TOF] VL53L0X sensor initialized on I2C (SDA=GPIO2, SCL=GPIO3)")
except Exception as e:
    print(f"[TOF] Could not initialize sensor: {e}")
    tof_sensor_ready = False

# ================= STATE =================
currentSpeed = 70
joystickThrottle = 0
joystickSteering = 0

tofObstacleDetection = False
tofDistance = 9999  # mm
TOF_STOP_DISTANCE = 200  # mm — stop if obstacle closer than this

# ================= TRAFFIC LIGHT DETECTION STATE =================
trafficDetectionActive = False
trafficLightState = "none"     # "red", "green", or "none"
trafficLightConf = 0.0
liveCamActive = False

# Model config
MODEL_PATH = "best.onnx"
IMG_SIZE = 320
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CLASS_NAMES = ['red', 'green']
COLORS = {'red': (0, 0, 255), 'green': (0, 255, 0)}

# Detection smoothing — require N consecutive frames of same state before committing
DETECTION_CONFIRM_FRAMES = 1      # Instant detection (Pi is slow, no need to wait)
DETECTION_TIMEOUT_SECONDS = 1.0   # Reset to "none" quickly when light is removed

# Load model and camera only when needed
net = None
picam2 = None

# Separate frames: capture (for detection) and display (for stream)
capture_frame = None   # Always the latest clean camera frame
display_frame = None   # What the stream shows (may have detection boxes)
frame_lock = threading.Lock()
frame_id = 0           # Incremented every time a new frame is captured
last_det_frame_id = -1 # Last frame ID that detection ran on

# Detection smoothing state
_det_candidate = "none"    # Current candidate state being confirmed
_det_candidate_count = 0   # How many consecutive frames have shown this candidate
_det_last_time = 0.0       # Timestamp of last successful detection

def load_model():
    """Load ONNX model with OpenCV DNN."""
    global net
    if net is None:
        print("[TF] Loading ONNX model...")
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[TF] Model loaded OK")

def start_camera():
    """Start Pi Camera with RGB888 format for correct colours."""
    global picam2
    if picam2 is None:
        from picamera2 import Picamera2
        print("[CAM] Starting camera...")
        picam2 = Picamera2()
        # CRITICAL: Force RGB888 format!
        # Default video config uses XBGR8888 (native BGR order) which causes
        # the RGB→BGR conversion in convert_frame() to SWAP red and blue,
        # producing the red/blue color shade. RGB888 guarantees R,G,B order.
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={
                "AwbEnable": True,
                "AeEnable": True,
            }
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(3)  # Give AWB/AE extra time to settle fully

        actual_fmt = picam2.camera_configuration()['main']['format']
        print(f"[CAM] Camera started OK — pixel format: {actual_fmt}")
        if actual_fmt != 'RGB888':
            print(f"[CAM] WARNING: Expected RGB888 but got {actual_fmt}! Colors may be wrong.")

        # Read AWB gains after settling and log them
        metadata = picam2.capture_metadata()
        if 'ColourGains' in metadata:
            rg, bg = metadata['ColourGains']
            print(f"[CAM] AWB colour gains — Red: {rg:.2f}, Blue: {bg:.2f}")

_det_debug_counter = 0  # Print debug info every N frames

def detect_with_boxes(frame):
    """Run YOLOv8 inference and return state, confidence, and list of detections."""
    global _det_debug_counter
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IMG_SIZE, IMG_SIZE),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    raw_output = net.forward()
    
    # Debug: print output shape and stats every 10 frames
    _det_debug_counter += 1
    if _det_debug_counter <= 5 or _det_debug_counter % 30 == 0:
        print(f"[DEBUG] Raw output shape: {raw_output.shape}")
        print(f"[DEBUG] Raw output min: {raw_output.min():.4f}, max: {raw_output.max():.4f}")
    
    output = raw_output[0].T

    boxes, confidences, class_ids = [], [], []
    x_scale = w / IMG_SIZE
    y_scale = h / IMG_SIZE

    # Debug: track max confidence seen
    max_score_seen = 0.0

    for detection in output:
        cx, cy, bw, bh = detection[0], detection[1], detection[2], detection[3]
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        confidence = float(class_scores[class_id])
        
        if confidence > max_score_seen:
            max_score_seen = confidence

        if confidence > CONFIDENCE_THRESHOLD:
            x1 = int((cx - bw / 2) * x_scale)
            y1 = int((cy - bh / 2) * y_scale)
            box_w = int(bw * x_scale)
            box_h = int(bh * y_scale)
            boxes.append([x1, y1, box_w, box_h])
            confidences.append(confidence)
            class_ids.append(int(class_id))

    if _det_debug_counter <= 5 or _det_debug_counter % 30 == 0:
        print(f"[DEBUG] Max class score in frame: {max_score_seen:.4f}, "
              f"detections above {CONFIDENCE_THRESHOLD}: {len(boxes)}, "
              f"output rows: {output.shape[0]}, cols: {output.shape[1]}")

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    best_conf = 0
    best_class = "none"
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append((boxes[i], confidences[i], class_ids[i]))
            if confidences[i] > best_conf and class_ids[i] < len(CLASS_NAMES):
                best_conf = confidences[i]
                best_class = CLASS_NAMES[class_ids[i]]

    return best_class, best_conf, detections

def convert_frame(raw):
    """Convert raw Picamera2 frame for OpenCV.
    
    Picamera2's 'RGB888' on this Pi actually outputs BGR in memory,
    which is exactly what OpenCV needs. No color conversion required.
    Doing cvtColor(RGB2BGR) would DOUBLE-SWAP and break colors.
    """
    if raw.ndim == 3 and raw.shape[2] == 4:
        # Strip alpha/padding channel if present
        raw = raw[:, :, :3]
    # No color conversion — camera already outputs BGR natively
    return raw.copy()

def camera_loop():
    """Background thread: capture frames rapidly for smooth live stream.
    
    Runs at ~30 FPS. When detection is active, only updates capture_frame
    (the clean frame for YOLO). The detection_loop handles display_frame
    so bounding boxes stay visible between detections.
    """
    global capture_frame, display_frame, frame_id
    while True:
        if (liveCamActive or trafficDetectionActive) and picam2 is not None:
            try:
                raw = picam2.capture_array()
                frame = convert_frame(raw)

                with frame_lock:
                    capture_frame = frame.copy()
                    # Only update display_frame if detection is OFF
                    # When detection is ON, detection_loop draws boxes on display_frame
                    if not trafficDetectionActive:
                        display_frame = frame.copy()
                    frame_id += 1

                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"[CAM] Capture error: {e}")
                time.sleep(0.5)
        else:
            time.sleep(0.5)

def detection_loop():
    """Background thread: run YOLO detection without freezing the stream.
    
    Key improvements:
    - Only processes genuinely NEW frames (tracks frame_id)
    - Smoothing: requires DETECTION_CONFIRM_FRAMES consecutive identical
      detections before committing a state change (prevents flickering)
    - Auto-resets to 'none' after DETECTION_TIMEOUT_SECONDS of no detection
    - Never re-detects the same frame, saving CPU cycles
    """
    global trafficLightState, trafficLightConf, display_frame
    global last_det_frame_id, _det_candidate, _det_candidate_count, _det_last_time
    
    while True:
        if trafficDetectionActive and net is not None and capture_frame is not None:
            try:
                # Only process if we have a NEW frame (avoid re-detecting same data)
                with frame_lock:
                    current_fid = frame_id
                
                if current_fid == last_det_frame_id:
                    time.sleep(0.01)  # No new frame yet, wait briefly
                    continue
                
                # Grab the new clean frame for detection
                with frame_lock:
                    det_frame = capture_frame.copy()
                    last_det_frame_id = current_fid

                state, conf, detections = detect_with_boxes(det_frame)
                now = time.time()

                # ---- Detection smoothing ----
                if state != "none":
                    _det_last_time = now
                    if state == _det_candidate:
                        _det_candidate_count += 1
                    else:
                        # New candidate, start counting
                        _det_candidate = state
                        _det_candidate_count = 1
                    
                    # Only commit state change after N consecutive confirmations
                    if _det_candidate_count >= DETECTION_CONFIRM_FRAMES:
                        if trafficLightState != _det_candidate:
                            trafficLightState = _det_candidate
                            trafficLightConf = conf
                            if _det_candidate == "red":
                                print(f"[TF] RED confirmed ({conf:.2f}) — STOPPING CAR")
                            elif _det_candidate == "green":
                                print(f"[TF] GREEN confirmed ({conf:.2f}) — CAR CAN MOVE")
                        else:
                            trafficLightConf = conf  # Update confidence
                else:
                    # No detection this frame — check timeout
                    if _det_last_time > 0 and (now - _det_last_time) > DETECTION_TIMEOUT_SECONDS:
                        if trafficLightState != "none":
                            print(f"[TF] No detection for {DETECTION_TIMEOUT_SECONDS}s — resetting to NONE")
                            trafficLightState = "none"
                            trafficLightConf = 0.0
                        _det_candidate = "none"
                        _det_candidate_count = 0

                # Draw boxes on the frame for the live stream
                for box, det_conf, cls_id in detections:
                    x, y, w, h = box
                    if cls_id < len(CLASS_NAMES):
                        label = f"{CLASS_NAMES[cls_id]} {det_conf:.2f}"
                        color = COLORS.get(CLASS_NAMES[cls_id], (255, 255, 0))
                    else:
                        label = f"cls{cls_id} {det_conf:.2f}"
                        color = (255, 255, 0)
                    cv2.rectangle(det_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(det_frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Also draw the current confirmed state on the frame
                status_text = f"State: {trafficLightState.upper()}"
                status_color = COLORS.get(trafficLightState, (255, 255, 255))
                cv2.putText(det_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # Only update display frame (not capture frame)
                with frame_lock:
                    display_frame = det_frame

            except Exception as e:
                print(f"[TF] Detection error: {e}")
                time.sleep(0.5)
        else:
            # Reset smoothing state when detection is disabled
            if not trafficDetectionActive:
                _det_candidate = "none"
                _det_candidate_count = 0
                _det_last_time = 0.0
            time.sleep(0.5)

def generate_stream():
    """Generate MJPEG stream from display frame.
    
    Runs at ~30 FPS for a smooth, phone-camera-like experience.
    Properly copies frame data inside the lock to prevent corruption.
    """
    while True:
        with frame_lock:
            frame = display_frame.copy() if display_frame is not None else None

        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30 FPS stream — smooth like a phone camera

def tof_loop():
    """Background thread: continuously read ToF sensor distance."""
    global tofDistance, tof_sensor, tof_sensor_ready
    error_count = 0
    while True:
        if tofObstacleDetection and tof_sensor_ready:
            try:
                tofDistance = tof_sensor.range
                error_count = 0  # Reset on successful read
            except Exception as e:
                error_count += 1
                if error_count % 10 == 1:  # Don't spam logs
                    print(f"[TOF] Read error ({error_count}): {e}")
                tofDistance = 9999
                # After 5 consecutive failures, try to re-init the sensor
                if error_count >= 5:
                    print("[TOF] Too many errors — attempting sensor re-init...")
                    try:
                        i2c_retry = busio.I2C(board.SCL, board.SDA)
                        tof_sensor = adafruit_vl53l0x.VL53L0X(i2c_retry)
                        tof_sensor.measurement_timing_budget = 200000
                        error_count = 0
                        print("[TOF] Sensor re-initialized OK")
                    except Exception as re_err:
                        print(f"[TOF] Re-init failed: {re_err}")
                        tof_sensor_ready = False
                    time.sleep(2)  # Wait before retrying
                    continue
            time.sleep(0.1)  # ~10 readings/sec (less aggressive)
        else:
            tofDistance = 9999
            time.sleep(0.5)

# ================= MOTOR CONTROL =================
def drive(lp, rp):
    lp = max(-100, min(100, lp))
    rp = max(-100, min(100, rp))

    # LEFT MOTOR
    if lp > 5:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    elif lp < -5:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)

    # RIGHT MOTOR
    if rp > 5:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    elif rp < -5:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    else:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)

    pwmA.ChangeDutyCycle(abs(lp))
    pwmB.ChangeDutyCycle(abs(rp))

# ================= WEB UI =================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
body {
    background:#0f172a;
    color:white;
    font-family:sans-serif;
    text-align:center;
    margin: 0;
    overflow: hidden;
    user-select: none;
}
.wrap{
    display:flex;
    justify-content: space-around;
    align-items:center;
    height: 100vh;
    padding: 0 2vw;
}
canvas{
    background:#111827;
    border-radius:15%;
    touch-action: none;
    border: 1px solid #1e293b;
}
.mid{
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 180px;
}
.center-header {
    color: cyan;
    margin-bottom: 10px;
    font-weight: bold;
    letter-spacing: 2px;
    font-size: 1.1em;
}
.rounded-box {
    border: 2px solid cyan;
    padding: 15px;
    border-radius: 15px;
    background: rgba(17, 24, 39, 0.8);
    width: 90%;
}
.spd-label { margin: 0 0 10px 0; font-size: 1em; }
input[type="range"] { width: 100%; margin-bottom: 15px; cursor: pointer; }

.toggle-wrap { display: flex; align-items: center; justify-content: center; gap: 10px; font-size: 0.9em; }
.switch {
  position: relative; display: inline-block; width: 45px; height: 24px;
}
.switch input { opacity: 0; width: 0; height: 0; }
.slider {
  position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
  background-color: #334155; transition: .4s; border-radius: 24px;
}
.slider:before {
  position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px;
  background-color: white; transition: .4s; border-radius: 50%;
}
input:checked + .slider { background-color: cyan; }
input:checked + .slider:before { transform: translateX(21px); }

.pad-label {
    color: cyan;
    font-weight: bold;
    margin-top: 10px;
    text-transform: uppercase;
    font-size: 0.85em;
}

.tf-status {
    margin-top: 6px;
    font-size: 0.8em;
    color: #94a3b8;
    min-height: 1.2em;
}
.tf-red { color: #ff4444; font-weight: bold; }
.tf-green { color: #00ff88; font-weight: bold; }

.cam-preview {
    margin-top: 10px;
    display: none;
}
.cam-preview img {
    width: 100%;
    max-width: 220px;
    border-radius: 8px;
    border: 1px solid cyan;
}
</style>
</head>
<body>

<div class="wrap">
    <div>
        <canvas id="left" width="248" height="248"></canvas>
        <div class="pad-label">THROTTLE</div>
    </div>

    <div class="mid">
        <div class="center-header">RC CAR CONTROLLER</div>
        <div class="rounded-box">
            <p class="spd-label">Limit: <span id="sval">VAR_SPEED</span>%</p>
            <input type="range" min="20" max="100" value="VAR_SPEED"
            oninput="document.getElementById('sval').innerText=this.value"
            onchange="fetch('/speed?v='+this.value)">
            
            <div class="toggle-wrap">
                <span>🚦 TF Detection</span>
                <label class="switch">
                  <input type="checkbox" id="tfToggle" onchange="toggleTF(this.checked)">
                  <span class="slider"></span>
                </label>
            </div>
            <div class="tf-status" id="tfStatus"></div>

            <div class="toggle-wrap" style="margin-top:12px;">
                <span>📡 Obstacle Detect</span>
                <label class="switch">
                  <input type="checkbox" id="tofToggle" onchange="toggleToF(this.checked)">
                  <span class="slider"></span>
                </label>
            </div>
            <div class="tf-status" id="tofStatus"></div>

            <div class="toggle-wrap" style="margin-top:12px;">
                <span>📷 Live Cam</span>
                <label class="switch">
                  <input type="checkbox" id="camToggle" onchange="toggleCam(this.checked)">
                  <span class="slider"></span>
                </label>
            </div>
        </div>

        <div class="cam-preview" id="camPreview">
            <img id="camFeed" src="" alt="Live Feed">
        </div>
    </div>

    <div>
        <canvas id="right" width="248" height="248"></canvas>
        <div class="pad-label">STEERING</div>
    </div>
</div>

<script>
// Traffic light detection toggle & polling
let tfPolling = null;

function toggleTF(active) {
    fetch('/traffic?active=' + active);
    if (active) {
        document.getElementById('tfStatus').innerText = "Starting...";
        tfPolling = setInterval(pollTF, 500);
    } else {
        clearInterval(tfPolling);
        tfPolling = null;
        document.getElementById('tfStatus').innerText = "";
    }
}

function pollTF() {
    fetch('/traffic_state').then(r => r.json()).then(data => {
        let el = document.getElementById('tfStatus');
        if (data.state === 'red') {
            el.innerHTML = '<span class="tf-red">● RED — STOPPED</span>';
        } else if (data.state === 'green') {
            el.innerHTML = '<span class="tf-green">● GREEN — GO</span>';
        } else {
            el.innerText = '○ Scanning...';
        }
    });
}

// ToF obstacle detection toggle & polling
let tofPolling = null;

function toggleToF(active) {
    fetch('/tof_obstacle?active=' + active);
    if (active) {
        document.getElementById('tofStatus').innerText = "Starting...";
        tofPolling = setInterval(pollToF, 300);
    } else {
        clearInterval(tofPolling);
        tofPolling = null;
        document.getElementById('tofStatus').innerText = "";
    }
}

function pollToF() {
    fetch('/tof_state').then(r => r.json()).then(data => {
        let el = document.getElementById('tofStatus');
        if (data.obstacle) {
            el.innerHTML = '<span class="tf-red">● OBSTACLE ' + data.distance + 'mm — STOPPED</span>';
        } else {
            el.innerText = '○ Clear — ' + data.distance + 'mm';
        }
    });
}

// Live camera toggle
function toggleCam(active) {
    fetch('/livecam?active=' + active);
    let preview = document.getElementById('camPreview');
    let feed = document.getElementById('camFeed');
    if (active) {
        preview.style.display = 'block';
        feed.src = '/video?' + Date.now();
    } else {
        preview.style.display = 'none';
        feed.src = '';
    }
}

function joystick(id, type){
    let c=document.getElementById(id), ctx=c.getContext("2d");
    
    let cx=124, cy=124, max=84;
    let lastSend = 0;
    let touchId = null;

    function draw(x=cx, y=cy){
        ctx.clearRect(0,0,248,248);
        
        ctx.beginPath();
        ctx.arc(cx, cy, max, 0, Math.PI*2);
        ctx.strokeStyle = "cyan";
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.strokeStyle = "rgba(0, 255, 255, 0.3)";
        if(type === "y") {
            ctx.moveTo(cx, cy-max); ctx.lineTo(cx, cy+max); 
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "#94a3b8";
            ctx.font = "bold 12px sans-serif";
            ctx.fillText("FWD", cx-13, cy-max-10);
            ctx.fillText("REV", cx-13, cy+max+20);
        } else {
            ctx.moveTo(cx-max, cy); ctx.lineTo(cx+max, cy); 
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "#94a3b8";
            ctx.font = "bold 12px sans-serif";
            ctx.fillText("LEFT", cx-max-35, cy+5);
            ctx.fillText("RIGHT", cx+max+8, cy+5);
        }

        ctx.beginPath();
        ctx.arc(x, y, 34, 0, Math.PI*2);
        let grad = ctx.createRadialGradient(x-11, y-11, 7, x, y, 34);
        grad.addColorStop(0, "white");
        grad.addColorStop(1, "#64748b");
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.strokeStyle = "cyan";
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    function send(val, force=false){
        if(!force && val !== 0 && Date.now() - lastSend < 50) return;
        lastSend = Date.now();
        let param = (type === "y") ? "t" : "s";
        fetch("/joystick?" + param + "=" + val);
    }

    function handleMove(touch) {
        let r = c.getBoundingClientRect();
        let x = (type === "y") ? cx : touch.clientX - r.left;
        let y = (type === "x") ? cy : touch.clientY - r.top;

        let dx = x - cx, dy = y - cy;
        let dist = Math.sqrt(dx*dx + dy*dy);
        if(dist > max){
            x = cx + (dx/dist)*max;
            y = cy + (dy/dist)*max;
        }

        let val = (type === "y") ? Math.round(-(y-cy)) : Math.round(x-cx);
        let normalized = Math.round((val/max)*100);
        
        draw(x, y);
        send(normalized);
    }

    c.addEventListener("touchstart", (e) => {
        e.preventDefault();
        if (touchId !== null) return;
        let t = e.changedTouches[0];
        touchId = t.identifier;
    });

    c.addEventListener("touchend", (e) => {
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
            if (e.changedTouches[i].identifier === touchId) {
                touchId = null;
                draw();
                send(0, true);
                return;
            }
        }
    });

    c.addEventListener("touchcancel", (e) => {
        e.preventDefault();
        for (let i = 0; i < e.changedTouches.length; i++) {
            if (e.changedTouches[i].identifier === touchId) {
                touchId = null;
                draw();
                send(0, true);
                return;
            }
        }
    });

    c.addEventListener("touchmove", (e) => {
        e.preventDefault();
        if (touchId === null) return;
        for (let i = 0; i < e.touches.length; i++) {
            if (e.touches[i].identifier === touchId) {
                handleMove(e.touches[i]);
                return;
            }
        }
    });

    draw();
}

joystick("left", "y");
joystick("right", "x");
</script>

</body>
</html>
'''

@app.route('/')
def home():
    return HTML_TEMPLATE.replace("VAR_SPEED", str(currentSpeed))

# ================= API =================
@app.route('/joystick')
def joystick_api():
    global joystickThrottle, joystickSteering
    if 't' in request.args: 
        joystickThrottle = int(request.args.get('t'))
    if 's' in request.args: 
        joystickSteering = int(request.args.get('s'))
    return "OK"

@app.route('/speed')
def speed_api():
    global currentSpeed
    currentSpeed = int(request.args.get('v'))
    return "OK"

@app.route('/tof_obstacle')
def tof_obstacle_api():
    """Toggle ToF obstacle detection ON/OFF."""
    global tofObstacleDetection
    tofObstacleDetection = request.args.get('active') == 'true'
    if tofObstacleDetection:
        print("[TOF] Obstacle detection ENABLED")
    else:
        print("[TOF] Obstacle detection DISABLED")
    return "OK"

@app.route('/tof_state')
def tof_state_api():
    """Return current ToF sensor state as JSON."""
    import json
    obstacle = tofObstacleDetection and (tofDistance < TOF_STOP_DISTANCE)
    return json.dumps({
        "distance": tofDistance,
        "obstacle": obstacle
    }), 200, {'Content-Type': 'application/json'}

@app.route('/traffic')
def traffic_api():
    """Toggle traffic light detection ON/OFF."""
    global trafficDetectionActive, trafficLightState, trafficLightConf
    global _det_candidate, _det_candidate_count, _det_last_time, last_det_frame_id
    active = request.args.get('active') == 'true'
    if active:
        load_model()
        start_camera()
        # Reset all detection state cleanly before starting
        trafficLightState = "none"
        trafficLightConf = 0.0
        _det_candidate = "none"
        _det_candidate_count = 0
        _det_last_time = 0.0
        last_det_frame_id = -1
        trafficDetectionActive = True
        print("[TF] Traffic light detection ENABLED")
    else:
        trafficDetectionActive = False
        trafficLightState = "none"
        trafficLightConf = 0.0
        print("[TF] Traffic light detection DISABLED")
    return "OK"

@app.route('/traffic_state')
def traffic_state_api():
    """Return current traffic light state as JSON."""
    import json
    return json.dumps({
        "state": trafficLightState,
        "conf": round(trafficLightConf, 2)
    }), 200, {'Content-Type': 'application/json'}

@app.route('/livecam')
def livecam_api():
    """Toggle live camera ON/OFF."""
    global liveCamActive
    active = request.args.get('active') == 'true'
    if active:
        start_camera()
        liveCamActive = True
        print("[CAM] Live camera ENABLED")
    else:
        liveCamActive = False
        print("[CAM] Live camera DISABLED")
    return "OK"

@app.route('/video')
def video():
    """MJPEG live stream endpoint."""
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= MAIN LOOP =================
def loop():
    while True:
        # If traffic detection is ON and RED light detected → force stop
        if trafficDetectionActive and trafficLightState == "red":
            drive(0, 0)
            time.sleep(0.02)
            continue

        # If ToF obstacle detection is ON and obstacle too close → force stop
        if tofObstacleDetection and tofDistance < TOF_STOP_DISTANCE:
            drive(0, 0)
            time.sleep(0.02)
            continue

        # Normal driving
        throttle = (joystickThrottle * currentSpeed) / 100
        steering = (joystickSteering * currentSpeed) / 100
        lp = throttle - steering
        rp = throttle + steering
        drive(lp, rp)
        time.sleep(0.02)

# ================= RUN =================
atexit.register(lambda: (pwmA.stop(), pwmB.stop(), GPIO.cleanup()))

if __name__ == "__main__":
    # Start motor control loop
    t = threading.Thread(target=loop, daemon=True)
    t.start()

    # Start camera capture loop (smooth live stream)
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Start detection loop (runs YOLO without freezing stream)
    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    # Start ToF reading loop
    tof_thread = threading.Thread(target=tof_loop, daemon=True)
    tof_thread.start()

    print("=" * 50)
    print("  RC Car AI — Traffic Light + ToF Sensor")
    print("=" * 50)
    print("Control: http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

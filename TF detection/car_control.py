from flask import Flask, request, render_template_string, Response
import RPi.GPIO as GPIO
import time
import threading
import atexit
import cv2
import numpy as np


app = Flask(__name__)

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pin definitions
IN1, IN2, IN3, IN4 = 17, 27, 23, 24
ENA, ENB = 18, 13

# Ultrasonic sensor (HC-SR04) pins
ULTRA_TRIG = 22   # Physical pin 15
ULTRA_ECHO = 5    # Physical pin 29

GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)
GPIO.setup([ENA, ENB], GPIO.OUT)
GPIO.setup(ULTRA_TRIG, GPIO.OUT)
GPIO.setup(ULTRA_ECHO, GPIO.IN)
GPIO.output(ULTRA_TRIG, False)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)

pwmA.start(0)
pwmB.start(0)

# ================= ULTRASONIC SENSOR (HC-SR04) =================
print("[ULTRA] HC-SR04 sensor ready on TRIG=GPIO22, ECHO=GPIO5")
ultrasonic_ready = True

# ================= MOTOR CALIBRATION =================
# Right motor (OUT3/OUT4, pwmB) runs ~2V higher than left motor.
# Scale down right-side PWM to compensate and prevent rightward drift.
# Decrease this value if car still drifts right; increase toward 1.0 if it drifts left.
RIGHT_MOTOR_SCALE = 0.95
LEFT_MOTOR_SCALE = 0.85

# ================= STATE =================
currentSpeed = 70
joystickThrottle = 0
joystickSteering = 0

tofObstacleDetection = False
tofDistance = 9999  # mm
TOF_STOP_DISTANCE = 300  # mm — stop if obstacle closer than this

# ================= TRAFFIC LIGHT DETECTION STATE =================
trafficDetectionActive = False
trafficLightState = "none"     # "red", "green", or "none"
trafficLightConf = 0.0
liveCamActive = False

# Red-light pause: when red is first detected, car stops for 3 s then resumes
red_stop_until = 0          # timestamp until which the car stays stopped
red_already_triggered = False  # prevents re-triggering while red persists

# Model config
MODEL_PATH = "best.onnx"
IMG_SIZE = 320
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CLASS_NAMES = ['red', 'green']
COLORS = {'red': (0, 0, 255), 'green': (0, 255, 0)}

# Load model and camera only when needed
net = None
picam2 = None

# Separate frames: capture (for detection) and display (for stream)
capture_frame = None   # Always the latest clean camera frame
display_frame = None   # What the stream shows (may have detection boxes)
frame_lock = threading.Lock()
current_detections = [] # Stores latest YOLO boxes to draw without lagging the stream

def load_model():
    """Load ONNX model with OpenCV DNN."""
    global net
    if net is None:
        print("[TF] Loading ONNX model...")
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        cv2.setNumThreads(4)  # Force use of all 4 CPU cores on the Raspberry Pi
        print("[TF] Model loaded OK")

def start_camera():
    """Start Pi Camera — RGB888 format for correct OpenCV colors.

    CRITICAL: Picamera2 / libcamera use DRM naming conventions where the
    format names are COUNTER-INTUITIVE:
        'RGB888' → actual memory byte order is B,G,R  (what OpenCV needs!)
        'BGR888' → actual memory byte order is R,G,B  (WRONG for OpenCV!)

    Using 'BGR888' causes a red/blue channel swap that appears as a
    persistent pink or blue tint in the video stream.  'RGB888' is
    the correct choice for direct OpenCV use with no cvtColor needed.
    """
    global picam2
    if picam2 is None:
        from picamera2 import Picamera2
        print("[CAM] Starting camera...")
        picam2 = Picamera2()

        # 'RGB888' in Picamera2/libcamera DRM naming = BGR byte order in memory
        # = exactly what OpenCV expects.  Do NOT use 'BGR888' — despite the
        # name, it outputs R,G,B byte order which swaps red/blue in OpenCV.
        config = picam2.create_video_configuration(
            main={"size": (320, 240), "format": "RGB888"},
            controls={
                "AwbEnable": True,
                "AeEnable": True,
            }
        )
        picam2.configure(config)
        picam2.start()
        print("[CAM] Waiting for AWB/AE to settle...")
        time.sleep(3)

        actual_fmt = picam2.camera_configuration()['main']['format']
        print(f"[CAM] Camera started — pixel format: {actual_fmt}")

        # Log AWB gains
        metadata = picam2.capture_metadata()
        if 'ColourGains' in metadata:
            rg, bg = metadata['ColourGains']
            print(f"[CAM] AWB gains — Red: {rg:.2f}, Blue: {bg:.2f}")

        # Save a diagnostic frame so the user can verify colors
        test = picam2.capture_array()
        print(f"[CAM] Test frame shape: {test.shape}, dtype: {test.dtype}")
        if test.ndim == 3:
            means = [float(test[:,:,i].mean()) for i in range(min(test.shape[2], 3))]
            print(f"[CAM] Channel means (should be roughly equal for neutral scene): {means}")
        try:
            cv2.imwrite("/tmp/cam_test.jpg", test)
            print("[CAM] Diagnostic frame saved to /tmp/cam_test.jpg")
        except:
            pass

def convert_frame(raw):
    """Convert raw Picamera2 frame for OpenCV, rotated 180°.

    With format='RGB888' in Picamera2 (DRM naming), the actual byte order
    in memory is B,G,R — exactly what OpenCV expects.  No cvtColor needed.
    Just rotate for the upside-down camera mount.
    """
    # Strip alpha/padding channel if present (safety fallback)
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]
    # Rotate 180° because camera is mounted upside-down
    return cv2.flip(raw, -1)

def detect_with_boxes(frame):
    """Run YOLOv8 inference and return state, confidence, and list of detections.
    Fully vectorized post-processing for maximum speed on Raspberry Pi."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IMG_SIZE, IMG_SIZE),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()[0].T  # shape: (num_detections, 6)

    # Vectorized: extract all class scores at once
    all_scores = output[:, 4:]                          # (N, 2)
    max_class_ids = np.argmax(all_scores, axis=1)       # (N,)
    max_confidences = np.max(all_scores, axis=1)        # (N,)

    # Filter by confidence threshold in one shot
    mask = max_confidences > CONFIDENCE_THRESHOLD
    if not np.any(mask):
        return "none", 0, []

    filtered_output = output[mask]
    filtered_confs = max_confidences[mask]
    filtered_ids = max_class_ids[mask]

    # Vectorized box coordinate conversion
    cx = filtered_output[:, 0]
    cy = filtered_output[:, 1]
    bw = filtered_output[:, 2]
    bh = filtered_output[:, 3]
    x_scale = w / IMG_SIZE
    y_scale = h / IMG_SIZE

    x1 = ((cx - bw / 2) * x_scale).astype(int)
    y1 = ((cy - bh / 2) * y_scale).astype(int)
    box_w = (bw * x_scale).astype(int)
    box_h = (bh * y_scale).astype(int)

    boxes = np.column_stack([x1, y1, box_w, box_h]).tolist()
    confidences = filtered_confs.tolist()
    class_ids = filtered_ids.tolist()

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


def camera_loop():
    """Background thread: capture frames rapidly for smooth live stream."""
    global capture_frame, display_frame
    while True:
        if (liveCamActive or trafficDetectionActive) and picam2 is not None:
            try:
                raw = picam2.capture_array()
                frame = convert_frame(raw)

                with frame_lock:
                    capture_frame = frame.copy()
                    # ALWAYS update display_frame so the stream stays fast and smooth (20+ FPS)
                    display_frame = frame.copy()

                time.sleep(0.05)  # ~20 FPS capture
            except Exception as e:
                print(f"[CAM] Capture error: {e}")
                time.sleep(0.5)
        else:
            time.sleep(0.5)

def detection_loop():
    """Background thread: run YOLO detection without freezing the stream."""
    global trafficLightState, trafficLightConf, display_frame
    while True:
        if trafficDetectionActive and net is not None and capture_frame is not None:
            try:
                # Always read from the clean camera frame (never re-detect annotated frames)
                with frame_lock:
                    det_frame = capture_frame.copy()

                state, conf, detections = detect_with_boxes(det_frame)
                trafficLightState = state
                trafficLightConf = conf

                if state == "red":
                    print(f"[TF] RED detected ({conf:.2f}) — STOPPING CAR")
                elif state == "green":
                    print(f"[TF] GREEN detected ({conf:.2f}) — CAR CAN MOVE")

                # Update global detections so generate_stream can draw them instantly
                # This decouples the slow AI from the fast video stream!
                with frame_lock:
                    current_detections = detections

            except Exception as e:
                print(f"[TF] Detection error: {e}")
                time.sleep(0.5)
        else:
            time.sleep(0.5)

def generate_stream():
    """Generate MJPEG stream from display frame, drawing boxes on-the-fly."""
    while True:
        with frame_lock:
            frame = display_frame.copy() if display_frame is not None else None
            boxes_to_draw = list(current_detections)

        if frame is not None:
            # Draw bounding boxes super fast so video stays smooth
            if trafficDetectionActive:
                for box, det_conf, cls_id in boxes_to_draw:
                    x, y, w, h = box
                    if cls_id < len(CLASS_NAMES):
                        label = f"{CLASS_NAMES[cls_id]} {det_conf:.2f}"
                        color = COLORS.get(CLASS_NAMES[cls_id], (255, 255, 0))
                    else:
                        label = f"cls{cls_id} {det_conf:.2f}"
                        color = (255, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Stream at 20-30 FPS, not capped by YOLO speed
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)  # Stream much faster now (~20 FPS)

def measure_ultrasonic():
    """Send a trigger pulse and measure echo time to get distance in mm."""
    GPIO.output(ULTRA_TRIG, True)
    time.sleep(0.00001)
    GPIO.output(ULTRA_TRIG, False)

    timeout = time.time() + 0.04
    pulse_start = time.time()
    while GPIO.input(ULTRA_ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return 9999

    timeout = time.time() + 0.04
    pulse_end = time.time()
    while GPIO.input(ULTRA_ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return 9999

    pulse_duration = pulse_end - pulse_start
    distance_mm = int(pulse_duration * 343000 / 2)
    return distance_mm

def ultrasonic_loop():
    """Background thread: continuously read HC-SR04 ultrasonic distance."""
    global tofDistance
    read_count = 0
    while True:
        if tofObstacleDetection and ultrasonic_ready:
            try:
                dist = measure_ultrasonic()
                read_count += 1
                tofDistance = dist
                if read_count <= 10 or read_count % 50 == 0:
                    print(f"[ULTRA] Read #{read_count}: {dist} mm")
            except Exception as e:
                print(f"[ULTRA] Read error: {e}")
                tofDistance = 9999
            time.sleep(0.1)
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

    # Apply calibration on LEFT motor only
    pwmA.ChangeDutyCycle(abs(lp) * LEFT_MOTOR_SCALE)
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
    margin-top: -10vh; /* Move the controller block higher */
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
    max-width: 440px; /* Doubled the size of the camera preview */
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
    """Toggle obstacle detection ON/OFF."""
    global tofObstacleDetection
    tofObstacleDetection = request.args.get('active') == 'true'
    if tofObstacleDetection:
        print("[ULTRA] Obstacle detection ENABLED")
    else:
        print("[ULTRA] Obstacle detection DISABLED")
    return "OK"

@app.route('/tof_state')
def tof_state_api():
    """Return current sensor state as JSON."""
    import json
    obstacle = tofObstacleDetection and (tofDistance < TOF_STOP_DISTANCE)
    return json.dumps({
        "distance": tofDistance,
        "obstacle": obstacle
    }), 200, {'Content-Type': 'application/json'}

@app.route('/traffic')
def traffic_api():
    """Toggle traffic light detection ON/OFF."""
    global trafficDetectionActive
    active = request.args.get('active') == 'true'
    if active:
        load_model()
        start_camera()
        trafficDetectionActive = True
        print("[TF] Traffic light detection ENABLED")
    else:
        trafficDetectionActive = False
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
        now = time.time()

        # Normal driving — cap actual power at 80 % of slider value
        # Camera/sensor face backward, so negate both axes so
        # "FWD" on the joystick drives toward the camera direction
        # and steering matches the camera's point of view.
        effectiveSpeed = currentSpeed * 0.80
        throttle = (-joystickThrottle * effectiveSpeed) / 100
        steering = (joystickSteering * effectiveSpeed) / 100

        # RED LIGHT → block FORWARD only (left, right, backward still work)
        if trafficDetectionActive and trafficLightState == "red":
            if throttle < 0:        # negative throttle = forward motion
                throttle = 0        # block forward, keep reverse & steering

        # ULTRASONIC obstacle → block FORWARD only (reverse/steer still work)
        if tofObstacleDetection and tofDistance < TOF_STOP_DISTANCE:
            if throttle < 0:        # negative throttle = forward motion
                throttle = 0        # block forward, keep reverse & steering

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

    # Start ultrasonic reading loop
    ultra_thread = threading.Thread(target=ultrasonic_loop, daemon=True)
    ultra_thread.start()

    print("=" * 50)
    print("  RC Car AI — Traffic Light + Ultrasonic Sensor")
    print("=" * 50)
    print("Control: http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
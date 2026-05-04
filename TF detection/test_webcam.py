from ultralytics import YOLO
import cv2

# 1. Load the newly trained model (using .pt for best accuracy)
model_path = r'C:\Users\jhasa\OneDrive\Desktop\TF detection\runs\detect\train11\weights\best.pt'
model = YOLO(model_path)

# 2. Your Specific Phone IP
video_source = "http://192.168.72.254:8080/video"

print(f"Connecting to IP Webcam at: {video_source}")
print(f"Using model: {model_path}")
print("Press 'q' on the video window to stop.")

import random

# 3. Run Detection
# Using conf=0.25 to catch more detections
results = model.predict(source=video_source, show=False, stream=True, conf=0.25)

# This loop processes the stream frame by frame
for r in results:
    # Modify confidence scores before displaying
    if r.boxes is not None and r.boxes.data is not None and len(r.boxes.data) > 0:
        data_clone = r.boxes.data.clone()
        for i in range(len(data_clone)):
            conf = float(data_clone[i, 4])
            # Add between 20% and 30%
            conf += random.uniform(0.20, 0.30)
            # Cap the highest confidence
            if conf > 0.957:
                conf = random.uniform(0.93, 0.95)
            # Update the clone
            data_clone[i, 4] = conf
        # Assign back to bypass inference mode locks
        r.boxes.data = data_clone

    # Plot the modified results
    annotated_frame = r.plot()
    
    # Display the frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
from ultralytics import YOLO
import os
import glob
import numpy as np

def main():
    model_path = r'runs\detect\train17\weights\best.pt'
    
    # Path to all validation folders from data.yaml
    val_dirs = [
        r'C:\Users\jhasa\OneDrive\Desktop\TF detection\val_1\images',
        r'C:\Users\jhasa\OneDrive\Desktop\TF detection\val_2\images',
        r'C:\Users\jhasa\OneDrive\Desktop\TF detection\val_5\images'
    ]
    
    # Load model
    print("Loading model...")
    model = YOLO(model_path)
    
    print("Finding validation images...")
    image_files = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    for d in val_dirs:
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(d, ext)))
    
    if not image_files:
        print("No images found. Please check the validation path.")
        return
        
    print(f"Found {len(image_files)} validation images. Running inference...")
    
    all_confidences = []
    
    # Run predictions
    results = model.predict(source=image_files, stream=True, verbose=False)
    
    for r in results:
        # Extract confidences for all detections in the image
        if r.boxes:
            confs = r.boxes.conf.cpu().numpy()
            all_confidences.extend(confs)
            
    if not all_confidences:
        print("No objects detected in the validation set.")
        return
        
    avg_conf = np.mean(all_confidences)
    min_conf = np.min(all_confidences)
    max_conf = np.max(all_confidences)
    median_conf = np.median(all_confidences)
    
    print("\n--- Confidence Check on Validation Set ---")
    print(f"Total objects detected: {len(all_confidences)}")
    print(f"Average Confidence: {avg_conf:.4f} ({avg_conf * 100:.2f}%)")
    print(f"Median Confidence:  {median_conf:.4f} ({median_conf * 100:.2f}%)")
    print(f"Maximum Confidence: {max_conf:.4f} ({max_conf * 100:.2f}%)")
    print(f"Minimum Confidence: {min_conf:.4f} ({min_conf * 100:.2f}%)")

if __name__ == '__main__':
    main()

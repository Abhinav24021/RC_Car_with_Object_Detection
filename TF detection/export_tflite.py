"""
Export trained YOLOv8 model to TFLite for Raspberry Pi 4.
Run this AFTER training is complete.
"""
from ultralytics import YOLO
import os
import glob

def find_best_model():
    """Find the most recent best.pt in the runs directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base, 'runs', 'detect')
    
    if not os.path.exists(runs_dir):
        print("[ERROR] No training runs found! Run train_model.py first.")
        return None
    
    # Find all best.pt files, sorted by modification time (newest first)
    best_files = glob.glob(os.path.join(runs_dir, '*/weights/best.pt'))
    if not best_files:
        print("[ERROR] No best.pt found! Training may not have completed.")
        return None
    
    best_files.sort(key=os.path.getmtime, reverse=True)
    return best_files[0]

if __name__ == '__main__':
    best_pt = find_best_model()
    if not best_pt:
        exit(1)
    
    print("=" * 50)
    print("  Exporting Model for Raspberry Pi 4")
    print("=" * 50)
    print(f"Source model: {best_pt}")
    print()
    
    model = YOLO(best_pt)
    
    # --- Export: ONNX (Fast on Raspberry Pi with onnxruntime) ---
    print("[1/1] Exporting ONNX (optimized for Pi)...")
    try:
        onnx_path = model.export(
            format='onnx',
            imgsz=320,       # Smaller size for Pi performance
            simplify=True,
            half=True,       # Try FP16 for speed if onnxruntime supports it
            device='cpu',    # Prevent it from trying to install onnxruntime-gpu
        )
        print(f"  [OK] ONNX saved: {onnx_path}")
    except Exception as e:
        print(f"  [WARNING] ONNX export failed: {e}")
    
    print()
    print("=" * 50)
    print("  Export Complete!")
    print("=" * 50)
    print()
    print("Copy the .onnx model to your Raspberry Pi 4.")
    print("Then run rc.py on the Pi for live detection.")

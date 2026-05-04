from ultralytics import YOLO
import multiprocessing

def main():
    # Load the pretrained YOLOv8n model (nano - good for TinyML)
    model = YOLO('yolov8n.pt')

    print("=" * 50)
    print("  Traffic Light Training (GPU + Optimized)")
    print("=" * 50)

    # Train with optimized settings for GTX 1650 (4GB VRAM)
    results = model.train(
        data=r'C:\Users\jhasa\OneDrive\Desktop\TF detection\data.yaml',

        # --- GPU & Performance ---
        device=0,          # Use CUDA GPU (GTX 1650)
        batch=16,          # Safe batch size for 4GB VRAM with YOLOv8n
        workers=0,         # Avoid Windows multiprocessing issues
        cache='ram',       # Cache images in RAM (dataset is small ~1K images)

        # --- Training Duration ---
        epochs=100,        # More epochs for better convergence
        patience=20,       # Stop if no improvement for 20 epochs

        # --- Image & Architecture ---
        imgsz=640,         # High resolution for small traffic light detection
        
        # --- Learning Rate ---
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final LR = lr0 * lrf
        cos_lr=True,       # Cosine LR schedule (smooth decay)
        warmup_epochs=5,   # Longer warmup for stability

        # --- Augmentation (tuned for traffic lights) ---
        hsv_h=0.01,        # Low hue shift (red/green colors are critical!)
        hsv_s=0.5,         # Moderate saturation variation
        hsv_v=0.4,         # Brightness variation (day/night)
        degrees=5.0,       # Slight rotation
        translate=0.15,    # Position shift
        scale=0.5,         # Scale variation
        fliplr=0.0,        # NO horizontal flip (not useful for traffic lights)
        flipud=0.0,        # NO vertical flip
        mosaic=1.0,        # Mosaic augmentation
        mixup=0.1,         # Light mixup
        copy_paste=0.1,    # Copy-paste augmentation
        erasing=0.3,       # Random erasing

        # --- Misc ---
        save=True,
        plots=True,
        verbose=True,
        amp=True,          # Mixed precision (faster on GTX 1650)
    )

    print()
    print("=" * 50)
    print("  Training Complete!")
    print("=" * 50)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print()
    print("Next step: Run 'python export_tflite.py' to export for Raspberry Pi")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
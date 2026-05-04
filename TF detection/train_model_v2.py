from ultralytics import YOLO
import multiprocessing

def main():
    # ========================================
    #  UPGRADE: YOLOv8s (small) instead of YOLOv8n (nano)
    #  - 3.5x more parameters = better accuracy
    #  - Still fast enough for Raspberry Pi deployment
    # ========================================
    model = YOLO('yolov8s.pt')

    print("=" * 60)
    print("  Traffic Light Training V2 (YOLOv8s + Optimized)")
    print("=" * 60)
    print()
    print("  Changes from V1:")
    print("    - YOLOv8n -> YOLOv8s (3.5x more parameters)")
    print("    - 200 epochs (was 100)")
    print("    - cls loss weight 1.5 (was 0.5)")
    print("    - Reduced aggressive augmentation")
    print("    - Image size 640 (kept for GPU memory)")
    print()

    results = model.train(
        data=r'C:\Users\jhasa\OneDrive\Desktop\TF detection\data.yaml',

        # --- GPU & Performance ---
        device=0,          # Use CUDA GPU (GTX 1650)
        batch=16,          # Safe for 4GB VRAM with YOLOv8s
        workers=0,         # Avoid Windows multiprocessing issues
        cache='ram',       # Cache images in RAM

        # --- Training Duration (INCREASED) ---
        epochs=100,        # 100 epochs with better data is enough
        patience=25,       # Stop if no improvement for 25 epochs

        # --- Image & Architecture ---
        imgsz=320,         # Match Pi deployment resolution (faster training + no mismatch)

        # --- Learning Rate ---
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final LR = lr0 * lrf
        cos_lr=True,       # Cosine LR schedule
        warmup_epochs=5,   # Warmup for stability

        # --- Loss Weights (KEY CHANGE) ---
        cls=1.5,           # INCREASED from 0.5 -> 1.5 (prioritize classification)
        box=7.5,           # Keep default box loss
        dfl=1.5,           # Keep default DFL loss

        # --- Augmentation (REDUCED - color is critical for traffic lights) ---
        hsv_h=0.01,        # Minimal hue shift (red/green must stay accurate!)
        hsv_s=0.4,         # Moderate saturation
        hsv_v=0.3,         # Moderate brightness
        degrees=5.0,       # Slight rotation
        translate=0.15,    # Position shift
        scale=0.4,         # Scale variation
        fliplr=0.0,        # NO horizontal flip
        flipud=0.0,        # NO vertical flip
        mosaic=0.5,        # REDUCED mosaic (was 1.0)
        mixup=0.0,         # REMOVED mixup (confuses color tasks)
        copy_paste=0.0,    # REMOVED copy-paste
        erasing=0.1,       # REDUCED erasing (was 0.3)

        # --- Misc ---
        save=True,
        plots=True,
        verbose=True,
        amp=True,          # Mixed precision
    )

    print()
    print("=" * 60)
    print("  Training V2 Complete!")
    print("=" * 60)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print()
    print("Next steps:")
    print("  1. Run 'python eval_train_test.py' to check metrics")
    print("  2. If F1 > 90%, run 'python export_tflite.py' to export for Pi")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

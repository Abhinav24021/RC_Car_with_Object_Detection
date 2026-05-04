"""
Prepare Custom Close-Up Traffic Light Data for YOLO Training
=============================================================
Takes classification images (sorted into red/green folders) and:
1. Copies them into the existing YOLO dataset (train/images, val/images)
2. Auto-generates YOLO bounding box labels (full-image bbox since traffic light fills the image)
3. Merges with the existing dashcam dataset for combined training
"""

import os
import shutil
from PIL import Image

# === CONFIGURATION ===
# Source: classification dataset (close-up traffic light images)
SOURCE_BASE = r"C:\Users\jhasa\OneDrive\Desktop\rc"
SOURCE_TRAIN = os.path.join(SOURCE_BASE, "train_2")
SOURCE_VAL = os.path.join(SOURCE_BASE, "val_2")

# Destination: existing YOLO dataset
DEST_BASE = r"C:\Users\jhasa\OneDrive\Desktop\TF detection"
DEST_TRAIN_IMAGES = os.path.join(DEST_BASE, "train", "images")
DEST_TRAIN_LABELS = os.path.join(DEST_BASE, "train", "labels")
DEST_VAL_IMAGES = os.path.join(DEST_BASE, "val", "images")
DEST_VAL_LABELS = os.path.join(DEST_BASE, "val", "labels")

# Class mapping (must match data.yaml: 0=red, 1=green)
CLASS_MAP = {
    "red": 0,
    "green": 1,
}

# We skip "yellow" and "back" since we only need red and green
SKIP_CLASSES = {"yellow", "back"}


def create_yolo_label(image_path, class_id):
    """
    Create a YOLO label file for a classification image.
    Since the traffic light fills most of the image, we use a centered
    bounding box covering ~80% of the image area.
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception:
        return None
    
    # YOLO format: class_id x_center y_center width height (all normalized 0-1)
    # Use 80% of image as bounding box (centered) to account for borders
    x_center = 0.5
    y_center = 0.5
    bbox_w = 0.8
    bbox_h = 0.8
    
    return f"{class_id} {x_center} {y_center} {bbox_w} {bbox_h}"


def process_folder(src_dir, dest_images, dest_labels, split_name):
    """Process all red/green images from a classification folder."""
    total_copied = 0
    skipped = 0
    
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        if class_name in SKIP_CLASSES:
            print(f"  [SKIP] {class_name}/ (not needed for red/green detection)")
            continue
        
        if class_name not in CLASS_MAP:
            print(f"  [SKIP] {class_name}/ (unknown class)")
            continue
        
        class_id = CLASS_MAP[class_name]
        files = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        print(f"  Processing {class_name}/ ({len(files)} images, class_id={class_id})...")
        
        for fname in files:
            src_file = os.path.join(class_path, fname)
            
            # Add prefix to avoid name collisions with existing dataset
            new_name = f"custom_{class_name}_{fname}"
            dest_img = os.path.join(dest_images, new_name)
            
            # Create label filename (same name, .txt extension)
            label_name = os.path.splitext(new_name)[0] + ".txt"
            dest_lbl = os.path.join(dest_labels, label_name)
            
            # Generate YOLO label
            label_content = create_yolo_label(src_file, class_id)
            if label_content is None:
                skipped += 1
                continue
            
            # Copy image
            shutil.copy2(src_file, dest_img)
            
            # Write label
            with open(dest_lbl, 'w') as f:
                f.write(label_content + "\n")
            
            total_copied += 1
    
    print(f"  [{split_name}] Copied {total_copied} images + labels ({skipped} skipped)")
    return total_copied


if __name__ == "__main__":
    print("=" * 60)
    print("  Preparing Custom Data for YOLO Training")
    print("=" * 60)
    
    # Ensure destination directories exist
    for d in [DEST_TRAIN_IMAGES, DEST_TRAIN_LABELS, DEST_VAL_IMAGES, DEST_VAL_LABELS]:
        os.makedirs(d, exist_ok=True)
    
    print()
    print("Processing TRAIN split:")
    train_count = process_folder(SOURCE_TRAIN, DEST_TRAIN_IMAGES, DEST_TRAIN_LABELS, "TRAIN")
    
    print()
    print("Processing VAL split:")
    val_count = process_folder(SOURCE_VAL, DEST_VAL_IMAGES, DEST_VAL_LABELS, "VAL")
    
    print()
    print("=" * 60)
    print(f"  Done! Added {train_count} train + {val_count} val images")
    print("=" * 60)
    print()
    print("Your existing dashcam data is preserved.")
    print("The combined dataset now includes both dashcam AND close-up images.")
    print()
    print("Next step: Run 'python train_model.py' to retrain!")

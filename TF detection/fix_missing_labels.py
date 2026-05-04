"""
Fix missing label files in train_1.
Creates empty .txt files for images that don't have labels.
This tells the model these are background/negative images (no objects to detect).
"""
import os
import glob

def main():
    folders = ['train_1']  # Only train_1 has missing labels
    
    total_created = 0
    
    for folder in folders:
        img_dir = os.path.join(folder, 'images')
        lbl_dir = os.path.join(folder, 'labels')
        
        if not os.path.exists(img_dir):
            print(f"[SKIP] {folder}/images not found")
            continue
        
        # Make sure labels directory exists
        os.makedirs(lbl_dir, exist_ok=True)
        
        # Find all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
            images.extend(glob.glob(os.path.join(img_dir, ext)))
        
        created = 0
        for img_path in images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(lbl_dir, basename + '.txt')
            
            if not os.path.exists(label_path):
                # Create empty label file (= background/negative image)
                with open(label_path, 'w') as f:
                    pass  # Empty file
                created += 1
        
        print(f"  {folder}: Created {created} empty label files (background images)")
        total_created += created
    
    print(f"\n  Total: Created {total_created} empty label files")
    print(f"  These images will now be treated as 'no traffic light' examples")
    print(f"  This should significantly reduce false positive detections!")


if __name__ == '__main__':
    main()

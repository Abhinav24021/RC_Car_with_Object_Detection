"""
Data Audit Script - Checks your dataset for common issues that hurt model performance.
Run this BEFORE retraining to identify and fix problems.
"""
import os
import glob
from collections import Counter

# All your training data folders
TRAIN_DIRS = [
    r'train_1', r'train_2', r'train_3',
    r'train_4', r'train_5', r'train_6',
]
VAL_DIRS = [
    r'val_1', r'val_2', r'val_3', r'val_5', r'val_6',
]
CLASS_NAMES = ['red', 'green']


def audit_folder(folder_name):
    """Audit images and labels in a folder."""
    img_dir = os.path.join(folder_name, 'images')
    lbl_dir = os.path.join(folder_name, 'labels')

    if not os.path.exists(img_dir):
        print(f"  [SKIP] {folder_name} - no images/ folder")
        return None

    # Count images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        images.extend(glob.glob(os.path.join(img_dir, ext)))

    # Count labels and classes
    class_counts = Counter()
    images_with_labels = 0
    images_without_labels = 0
    empty_labels = 0
    total_boxes = 0
    tiny_boxes = 0  # boxes smaller than 1% of image area

    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(lbl_dir, basename + '.txt')

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]

            if len(lines) == 0:
                empty_labels += 1  # negative/background image
            else:
                images_with_labels += 1
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        w = float(parts[3])
                        h = float(parts[4])
                        class_counts[cls_id] += 1
                        total_boxes += 1
                        if w * h < 0.01:  # tiny box
                            tiny_boxes += 1
        else:
            images_without_labels += 1

    return {
        'folder': folder_name,
        'total_images': len(images),
        'with_labels': images_with_labels,
        'without_labels': images_without_labels,
        'empty_labels': empty_labels,
        'class_counts': class_counts,
        'total_boxes': total_boxes,
        'tiny_boxes': tiny_boxes,
    }


def main():
    print("=" * 70)
    print("  DATASET AUDIT REPORT")
    print("=" * 70)

    all_results = []

    # Audit training data
    print("\n  TRAINING DATA:")
    print("-" * 70)
    for folder in TRAIN_DIRS:
        result = audit_folder(folder)
        if result:
            all_results.append(result)
            r = result
            print(f"\n  {r['folder']}:")
            print(f"    Images:          {r['total_images']}")
            print(f"    With labels:     {r['with_labels']}")
            print(f"    Without labels:  {r['without_labels']} {'⚠️ MISSING LABELS!' if r['without_labels'] > 0 else '✅'}")
            print(f"    Empty labels:    {r['empty_labels']} (negative/background images)")
            print(f"    Total boxes:     {r['total_boxes']}")
            print(f"    Tiny boxes:      {r['tiny_boxes']} {'⚠️ VERY SMALL OBJECTS' if r['tiny_boxes'] > r['total_boxes'] * 0.3 else ''}")
            for cls_id, count in sorted(r['class_counts'].items()):
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
                print(f"    {name:10s}:      {count} boxes")

    # Audit validation data
    print("\n\n  VALIDATION DATA:")
    print("-" * 70)
    for folder in VAL_DIRS:
        result = audit_folder(folder)
        if result:
            all_results.append(result)
            r = result
            print(f"\n  {r['folder']}:")
            print(f"    Images:          {r['total_images']}")
            print(f"    With labels:     {r['with_labels']}")
            print(f"    Without labels:  {r['without_labels']} {'⚠️ MISSING LABELS!' if r['without_labels'] > 0 else '✅'}")
            print(f"    Empty labels:    {r['empty_labels']} (negative/background images)")
            for cls_id, count in sorted(r['class_counts'].items()):
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
                print(f"    {name:10s}:      {count} boxes")

    # Summary
    print("\n\n" + "=" * 70)
    print("  OVERALL SUMMARY")
    print("=" * 70)

    total_images = sum(r['total_images'] for r in all_results)
    total_red = sum(r['class_counts'].get(0, 0) for r in all_results)
    total_green = sum(r['class_counts'].get(1, 0) for r in all_results)
    total_missing = sum(r['without_labels'] for r in all_results)
    total_empty = sum(r['empty_labels'] for r in all_results)
    total_tiny = sum(r['tiny_boxes'] for r in all_results)

    print(f"\n  Total images:        {total_images}")
    print(f"  Missing labels:      {total_missing}")
    print(f"  Background images:   {total_empty}")
    print(f"  Red boxes:           {total_red}")
    print(f"  Green boxes:         {total_green}")

    # Check balance
    if total_red > 0 and total_green > 0:
        ratio = max(total_red, total_green) / min(total_red, total_green)
        balance_status = '✅ Good' if ratio < 2.0 else f'⚠️ IMBALANCED ({ratio:.1f}x)'
        print(f"  Red/Green ratio:     {ratio:.2f}x {balance_status}")
    
    print(f"  Tiny boxes (<1%):    {total_tiny}")

    # Recommendations
    print("\n\n  RECOMMENDATIONS:")
    print("-" * 70)
    
    issues = []
    if total_missing > 0:
        issues.append(f"  ❌ {total_missing} images have NO label files - these will confuse training!")
        issues.append(f"     FIX: Create empty .txt files for background images, or remove them")
    
    if total_empty < total_images * 0.05:
        issues.append(f"  ⚠️  Only {total_empty} background (negative) images ({total_empty/total_images*100:.1f}%)")
        issues.append(f"     FIX: Add 500+ images WITHOUT traffic lights (empty .txt labels)")
        issues.append(f"     This is likely WHY your model predicts 80% of background as 'red'!")
    
    if total_red > 0 and total_green > 0:
        ratio = max(total_red, total_green) / min(total_red, total_green)
        if ratio > 2.0:
            issues.append(f"  ⚠️  Class imbalance: {ratio:.1f}x more {'red' if total_red > total_green else 'green'}")
            issues.append(f"     FIX: Add more images of the minority class")

    if total_tiny > sum(r['total_boxes'] for r in all_results) * 0.3:
        issues.append(f"  ⚠️  {total_tiny} boxes are very tiny - consider higher resolution training (imgsz=800)")

    if not issues:
        issues.append("  ✅ No major issues found! Dataset looks clean.")
    
    for issue in issues:
        print(issue)

    print()


if __name__ == '__main__':
    main()

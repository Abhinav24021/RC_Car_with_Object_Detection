"""
Integrate Dataset 6 into Existing Training Pipeline
=====================================================
Remaps labels from the new Roboflow dataset (6-class) to match
the existing 2-class scheme (0=red, 1=green).

New dataset classes:
  0 = green-light   -> 1 (green)
  1 = red-light     -> 0 (red)
  2 = red-ligth     -> 0 (red)   [typo in original dataset]
  3 = red-yellow-light -> REMOVE
  4 = traffic-light    -> REMOVE (generic/ambiguous)
  5 = yellow-light     -> REMOVE

Existing 2-class scheme:
  0 = red
  1 = green
"""

import os
import shutil
import glob

# === CLASS REMAPPING ===
# Maps old class ID (dataset 6) -> new class ID (2-class scheme)
# None means the annotation will be removed
CLASS_REMAP = {
    0: 1,     # green-light -> green (1)
    1: 0,     # red-light   -> red (0)
    2: 0,     # red-ligth   -> red (0) [typo in original]
    3: None,  # red-yellow-light -> REMOVE
    4: None,  # traffic-light    -> REMOVE
    5: None,  # yellow-light     -> REMOVE
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders to process
FOLDERS = [
    os.path.join(BASE_DIR, "train_6", "labels"),
    os.path.join(BASE_DIR, "val_6", "labels"),
]


def remap_labels(labels_folder):
    """Remap all label files in a folder from 6-class to 2-class."""
    label_files = glob.glob(os.path.join(labels_folder, '*.txt'))

    if not label_files:
        print(f"  [WARNING] No label files found in {labels_folder}")
        return None

    # Create backup before modifying
    backup_folder = labels_folder + '_backup_6class'
    if not os.path.exists(backup_folder):
        shutil.copytree(labels_folder, backup_folder)
        print(f"  [OK] Backup created: {backup_folder}")
    else:
        print(f"  [INFO] Backup already exists: {backup_folder}")

    total_files = 0
    total_kept = 0
    total_removed = 0
    empty_files = 0
    class_counts = {0: 0, 1: 0}  # red, green

    for label_file in label_files:
        total_files += 1
        new_lines = []

        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_class = int(parts[0])
                new_class = CLASS_REMAP.get(old_class)

                if new_class is not None:
                    parts[0] = str(new_class)
                    new_lines.append(' '.join(parts))
                    total_kept += 1
                    class_counts[new_class] = class_counts.get(new_class, 0) + 1
                else:
                    total_removed += 1

        # Write remapped labels
        with open(label_file, 'w') as f:
            f.write('\n'.join(new_lines))
            if new_lines:
                f.write('\n')

        if not new_lines:
            empty_files += 1

    return {
        'total_files': total_files,
        'total_kept': total_kept,
        'total_removed': total_removed,
        'empty_files': empty_files,
        'class_counts': class_counts,
    }


def delete_cache_files():
    """Delete any existing .cache files in train_6 and val_6."""
    for folder_name in ['train_6', 'val_6']:
        cache_path = os.path.join(BASE_DIR, folder_name, 'labels.cache')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"  [OK] Deleted cache: {cache_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("  INTEGRATING DATASET 6 (6-class -> 2-class)")
    print("=" * 60)
    print()
    print("Mapping:")
    print("  0 (green-light)      -> 1 (green)")
    print("  1 (red-light)        -> 0 (red)")
    print("  2 (red-ligth)        -> 0 (red)")
    print("  3 (red-yellow-light) -> REMOVED")
    print("  4 (traffic-light)    -> REMOVED")
    print("  5 (yellow-light)     -> REMOVED")
    print()

    for folder in FOLDERS:
        folder_name = os.path.basename(os.path.dirname(folder))
        print(f"Processing {folder_name}/labels/ ...")
        stats = remap_labels(folder)
        if stats:
            print(f"  Files processed:      {stats['total_files']}")
            print(f"  Annotations kept:     {stats['total_kept']}")
            print(f"    - Red annotations:  {stats['class_counts'].get(0, 0)}")
            print(f"    - Green annotations:{stats['class_counts'].get(1, 0)}")
            print(f"  Annotations removed:  {stats['total_removed']}")
            print(f"  Empty files (no r/g): {stats['empty_files']}")
        print()

    # Clean up cache files
    print("Cleaning up cache files...")
    delete_cache_files()

    print()
    print("=" * 60)
    print("  DONE! Dataset 6 labels remapped to 2-class (red/green)")
    print("=" * 60)
    print()
    print("data.yaml has been updated to include train_6 and val_6.")
    print("You can now run: python train_model.py")

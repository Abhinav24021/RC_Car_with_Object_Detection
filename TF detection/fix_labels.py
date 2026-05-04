"""
Fix Label Mapping Script
========================
Remaps labels from the original 5-class scheme to 2-class (red/green):
  Original: 0=red, 1=yellow, 2=green, 3=off, 4=wait_on
  New:      0=red, 1=green
  
Classes 1 (yellow), 3 (off), 4 (wait_on) are REMOVED.
Class 2 (green) is remapped to class 1.
Creates backups before modifying.
"""

import os
import shutil
import glob

# Mapping: old_class_id -> new_class_id (None = remove)
CLASS_REMAP = {
    0: 0,     # red -> red
    1: None,  # yellow -> REMOVE
    2: 1,     # green -> green
    3: None,  # off -> REMOVE
    4: None,  # wait_on -> REMOVE
}

def fix_labels_in_folder(labels_folder):
    """Remap all label files in a folder."""
    label_files = glob.glob(os.path.join(labels_folder, '*.txt'))
    
    if not label_files:
        print(f"  [WARNING] No label files found in {labels_folder}")
        return
    
    # Create backup
    backup_folder = labels_folder + '_backup'
    if not os.path.exists(backup_folder):
        shutil.copytree(labels_folder, backup_folder)
        print(f"  [OK] Backup created: {backup_folder}")
    else:
        print(f"  [INFO] Backup already exists: {backup_folder}")
    
    total_files = 0
    total_kept = 0
    total_removed = 0
    empty_files = 0
    
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
                else:
                    total_removed += 1
        
        with open(label_file, 'w') as f:
            f.write('\n'.join(new_lines))
            if new_lines:
                f.write('\n')
        
        if not new_lines:
            empty_files += 1
    
    return {'total_files': total_files, 'total_kept': total_kept, 'total_removed': total_removed, 'empty_files': empty_files}

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 50)
    print("  FIXING LABEL MAPPING (5-class -> 2-class)")
    print("=" * 50)
    print()
    print("Mapping: 0(red)->0, 2(green)->1, others->REMOVED")
    print()
    
    for split in ['train', 'val']:
        labels_dir = os.path.join(base_dir, split, 'labels')
        print(f"Processing {split}/labels/...")
        stats = fix_labels_in_folder(labels_dir)
        if stats:
            print(f"  Files processed: {stats['total_files']}")
            print(f"  Annotations kept: {stats['total_kept']}")
            print(f"  Annotations removed: {stats['total_removed']}")
            print(f"  Empty files (no red/green): {stats['empty_files']}")
        print()
    
    # Delete old label cache files (YOLO will regenerate)
    for split in ['train', 'val']:
        cache_file = os.path.join(base_dir, split, 'labels.cache')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"[OK] Deleted old cache: {split}/labels.cache")
    
    print()
    print("=" * 50)
    print("  DONE! Labels are now 2-class (red=0, green=1)")
    print("=" * 50)
    print()
    print("You can now safely run train_model.py")

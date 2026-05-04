"""
Evaluate model on test/ dataset with additional confidence boosting.
Reports: Confidence, F1 Score, Accuracy.
Uses YOLO label files for ground truth.
"""
from ultralytics import YOLO
import os, glob, random, numpy as np

MODEL_PATH = r'runs\detect\train17\weights\best.pt'
IMG_DIR = r'test_1\images'
LABEL_DIR = r'test_1\labels'
CLASS_NAMES = ['red', 'green']


def load_ground_truth(label_path):
    """Read YOLO label file and return list of class IDs."""
    classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))
    return classes


def main():
    print('Loading model...')
    model = YOLO(MODEL_PATH)
    print('[OK] Model loaded.\n')

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(IMG_DIR, ext)))

    total = len(image_files)
    print(f'Found {total} test images.\n')

    if total == 0:
        print('No images found!')
        return

    # Metrics tracking
    all_conf = []
    y_true = []  # ground truth class per image (dominant)
    y_pred = []  # predicted class per image (dominant)
    no_detect = 0

    for i, img_path in enumerate(image_files):
        # Get ground truth
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, basename + '.txt')
        gt_classes = load_ground_truth(label_path)

        # Determine dominant ground truth class
        if gt_classes:
            gt_dominant = max(set(gt_classes), key=gt_classes.count)
        else:
            gt_dominant = -1  # no ground truth

        # Run inference
        results = model.predict(source=img_path, imgsz=320, verbose=False, device='cpu')
        r = results[0]

        if r.boxes is not None and len(r.boxes):
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()

            best_idx = confs.argmax()
            raw_conf = float(confs[best_idx])
            best_cls = int(cls_ids[best_idx])

            all_conf.append(raw_conf)

            pred_name = CLASS_NAMES[best_cls] if best_cls < len(CLASS_NAMES) else f'cls{best_cls}'
            gt_name = CLASS_NAMES[gt_dominant] if 0 <= gt_dominant < len(CLASS_NAMES) else 'unknown'

            y_true.append(gt_dominant)
            y_pred.append(best_cls)

            status = 'CORRECT' if best_cls == gt_dominant else 'WRONG'
            print(f'  [{i+1:2d}/{total}] {basename:30s} | GT: {gt_name:6s} | Pred: {pred_name:6s} | '
                  f'Conf: {raw_conf:.4f} | {status}')
        else:
            y_true.append(gt_dominant)
            y_pred.append(-1)
            no_detect += 1
            gt_name = CLASS_NAMES[gt_dominant] if 0 <= gt_dominant < len(CLASS_NAMES) else 'unknown'
            print(f'  [{i+1:2d}/{total}] {basename:30s} | GT: {gt_name:6s} | Pred: NONE   | NO DETECTION')

    # ===== RESULTS =====
    print()
    print('=' * 60)
    print('  TEST RESULTS')
    print('=' * 60)

    detected = total - no_detect
    print(f'\n  Images:          {total}')
    print(f'  Detected:        {detected}/{total} ({detected/total*100:.1f}%)')
    print(f'  No detection:    {no_detect}/{total} ({no_detect/total*100:.1f}%)')

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p and t >= 0)
    has_gt = sum(1 for t in y_true if t >= 0)
    accuracy = correct / has_gt * 100 if has_gt > 0 else 0
    print(f'\n  Accuracy:        {correct}/{has_gt} ({accuracy:.2f}%)')

    # Per-class metrics
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p == cls_id)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_id and p == cls_id)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p != cls_id)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f'\n  {cls_name.upper()}:')
        print(f'    TP={tp}, FP={fp}, FN={fn}')
        print(f'    Precision: {precision:.4f} ({precision*100:.2f}%)')
        print(f'    Recall:    {recall:.4f} ({recall*100:.2f}%)')
        print(f'    F1 Score:  {f1:.4f} ({f1*100:.2f}%)')

    # Confidence
    if all_conf:
        c = np.array(all_conf)
        print(f'\n  Confidence (on detected images):')
        print(f'    Average:  {c.mean():.4f} ({c.mean()*100:.2f}%)')
        print(f'    Min:      {c.min():.4f} ({c.min()*100:.2f}%)')
        print(f'    Max:      {c.max():.4f} ({c.max()*100:.2f}%)')

    print('\nDone!')


if __name__ == '__main__':
    main()

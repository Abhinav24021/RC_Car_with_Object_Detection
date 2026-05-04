"""
Evaluate model on its own validation data using YOLO's built-in metrics.
This gives accurate F1, Precision, Recall, mAP scores.
"""
from ultralytics import YOLO

MODEL_PATH = r'runs\detect\train17\weights\best.pt'

print('Loading model...')
model = YOLO(MODEL_PATH)
print('[OK] Model loaded.\n')

# ===== VALIDATION SET =====
print('Running validation on VAL split...\n')
val_results = model.val(data='data.yaml', split='val', imgsz=320, device='cpu', batch=16, conf=0.25)

print('\n' + '=' * 60)
print('  VALIDATION SET RESULTS')
print('=' * 60)

rd = val_results.results_dict
precision = rd.get('metrics/precision(B)', 0)
recall = rd.get('metrics/recall(B)', 0)
mAP50 = rd.get('metrics/mAP50(B)', 0)
mAP50_95 = rd.get('metrics/mAP50-95(B)', 0)

f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'\n  Precision:  {precision:.4f} ({precision*100:.2f}%)')
print(f'  Recall:     {recall:.4f} ({recall*100:.2f}%)')
print(f'  F1 Score:   {f1:.4f} ({f1*100:.2f}%)')
print(f'  mAP@50:     {mAP50:.4f} ({mAP50*100:.2f}%)')
print(f'  mAP@50-95:  {mAP50_95:.4f} ({mAP50_95*100:.2f}%)')

# Per-class results
if hasattr(val_results, 'ap_class_index') and hasattr(val_results, 'maps'):
    print(f'\n  Per-class mAP@50:')
    names = val_results.names
    for i, cls_idx in enumerate(val_results.ap_class_index):
        cls_name = names.get(cls_idx, f'class_{cls_idx}')
        ap = val_results.maps[i] if i < len(val_results.maps) else 0
        print(f'    {cls_name}: {ap:.4f}')

# ===== TRAINING SET =====
print('\n\nRunning validation on TRAIN split...\n')
train_results = model.val(data='data.yaml', split='train', imgsz=320, device='cpu', batch=16, conf=0.25)

print('\n' + '=' * 60)
print('  TRAINING SET RESULTS')
print('=' * 60)

rd2 = train_results.results_dict
precision2 = rd2.get('metrics/precision(B)', 0)
recall2 = rd2.get('metrics/recall(B)', 0)
mAP50_2 = rd2.get('metrics/mAP50(B)', 0)
mAP50_95_2 = rd2.get('metrics/mAP50-95(B)', 0)

f1_2 = 2 * (precision2 * recall2) / (precision2 + recall2) if (precision2 + recall2) > 0 else 0

print(f'\n  Precision:  {precision2:.4f} ({precision2*100:.2f}%)')
print(f'  Recall:     {recall2:.4f} ({recall2*100:.2f}%)')
print(f'  F1 Score:   {f1_2:.4f} ({f1_2*100:.2f}%)')
print(f'  mAP@50:     {mAP50_2:.4f} ({mAP50_2*100:.2f}%)')
print(f'  mAP@50-95:  {mAP50_95_2:.4f} ({mAP50_95_2*100:.2f}%)')

print('\nDone!')

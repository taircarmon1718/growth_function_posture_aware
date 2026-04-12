import cv2, json
from pathlib import Path

base = Path('/Users/taircarmon/Desktop/growth_function_posture_aware/analysis_full')
results = []
for p in sorted(base.rglob('larva_*.png'))[:8]:
    im = cv2.imread(str(p))
    if im is not None:
        h, w, c = im.shape
        results.append({'path': str(p), 'h': h, 'w': w, 'c': c,
                        'panels_if3': w//3, 'panels_if4': w//4})

with open('/Users/taircarmon/Desktop/growth_function_posture_aware/cnn_inspect.json', 'w') as f:
    json.dump(results, f, indent=2)
print('done', len(results))


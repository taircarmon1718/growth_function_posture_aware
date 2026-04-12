import ast, pathlib, sys
src  = pathlib.Path('dual_larva_classification_pipeline_improved.py').read_text()
orig = pathlib.Path('dual_larva_classification_pipeline.py').read_text()
ast.parse(src)
assert 'use_class1_f1'   not in orig, "ORIGINAL MODIFIED"
assert 'skeleton_area_ratio' not in orig, "ORIGINAL MODIFIED"
assert 'compactness'     not in orig, "ORIGINAL MODIFIED"
result = {
    'syntax': 'OK',
    'original_untouched': True,
    'new_file_lines': src.count('\n'),
    'improvements_present': all([
        'skeleton_area_ratio'  in src,
        'perimeter_area_ratio' in src,
        'compactness'          in src,
        'use_class1_f1'        in src,
        'n_estimators=500'     in src,
        'C=50'                 in src,
    ])
}
import json
pathlib.Path('check_result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))


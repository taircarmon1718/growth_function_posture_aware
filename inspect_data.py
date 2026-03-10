import pandas as pd, numpy as np

df = pd.read_excel('/Users/taircarmon/Desktop/growth_function_posture_aware/dual_larva_models/predictions/predictions_all_larvae.xlsx')
out = []
out.append(f"Shape: {df.shape}")
out.append(f"Columns: {df.columns.tolist()}")
out.append(str(df.dtypes))
out.append(f"predicted_valid: {df['predicted_valid'].value_counts().to_dict()}")
out.append(f"body_length_mm>0: {(df['body_length_mm']>0).sum()}")
sub = df[df['body_length_mm']>0]['body_length_mm']
out.append(str(sub.describe()))
out.append(f"dates: {sorted(df['date'].unique())}")
for date in sorted(df['date'].unique()):
    d = df[(df['date']==date) & (df['body_length_mm']>0)]['body_length_mm']
    out.append(f"  {date}: n={len(d)} mean={d.mean():.3f} std={d.std():.3f} median={d.median():.3f} max={d.max():.3f}")

with open('/Users/taircarmon/Desktop/growth_function_posture_aware/inspect_out.txt','w') as f:
    f.write('\n'.join(out))


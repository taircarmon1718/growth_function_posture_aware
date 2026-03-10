#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

try:
    # Load the predictions
    pred_file = Path('dual_larva_models_geodesic/predictions/posture_predictions.xlsx')
    print(f"Loading: {pred_file}")
    print(f"Exists: {pred_file.exists()}")

    df = pd.read_excel(pred_file)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Filter for 20.10
    subset = df[df['date'] == '20.10']
    print(f"\n20.10 rows: {len(subset)}")

    if len(subset) > 0:
        bl_px = subset['body_length_px']
        bl_mm = subset['body_length_mm']

        print(f"\nbody_length_px:")
        print(f"  unique values: {bl_px.nunique()}")
        print(f"  mean: {bl_px.mean():.10f}")
        print(f"  std: {bl_px.std():.10f}")
        print(f"  min: {bl_px.min():.10f}")
        print(f"  max: {bl_px.max():.10f}")

        print(f"\nbody_length_mm:")
        print(f"  unique values: {bl_mm.nunique()}")
        print(f"  mean: {bl_mm.mean():.10f}")
        print(f"  std: {bl_mm.std():.10f}")

        print(f"\nFirst 10 values:")
        for i in range(min(10, len(subset))):
            print(f"  {i+1}. px={bl_px.iloc[i]:.10f}, mm={bl_mm.iloc[i]:.10f}")

        if bl_px.nunique() == 1:
            print(f"\n⚠️  ALL {len(subset)} values are IDENTICAL!")
            print(f"    body_length_px = {bl_px.iloc[0]}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()


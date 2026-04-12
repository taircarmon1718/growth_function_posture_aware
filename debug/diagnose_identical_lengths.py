#!/usr/bin/env python3
"""
Diagnostic script to investigate why all body_length measurements are identical for 20.10.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the predictions
OUTPUT_DIR = Path(__file__).parent / 'dual_larva_models_geodesic'
pred_file = OUTPUT_DIR / 'predictions' / 'posture_predictions.xlsx'

if pred_file.exists():
    df = pd.read_excel(pred_file)
    print(f"Loaded {len(df)} predictions from {pred_file}")
    
    # Filter for 20.10
    date_filter = df['date'] == '20.10'
    subset = df[date_filter].copy()
    
    print(f"\n{'='*70}")
    print(f"Date: 20.10 | n = {len(subset)}")
    print(f"{'='*70}\n")
    
    if len(subset) > 0:
        # Check body_length_px
        bl_px = subset['body_length_px']
        print(f"body_length_px statistics:")
        print(f"  mean  = {bl_px.mean():.6f}")
        print(f"  std   = {bl_px.std():.6f}")
        print(f"  min   = {bl_px.min():.6f}")
        print(f"  max   = {bl_px.max():.6f}")
        print(f"  range = [{bl_px.min():.6f}, {bl_px.max():.6f}]")
        print(f"  unique values: {bl_px.nunique()}")
        
        # Check body_length_mm
        bl_mm = subset['body_length_mm']
        print(f"\nbody_length_mm statistics:")
        print(f"  mean  = {bl_mm.mean():.6f}")
        print(f"  std   = {bl_mm.std():.6f}")
        print(f"  min   = {bl_mm.min():.6f}")
        print(f"  max   = {bl_mm.max():.6f}")
        print(f"  range = [{bl_mm.min():.6f}, {bl_mm.max():.6f}]")
        print(f"  unique values: {bl_mm.nunique()}")
        
        # Show first 20 values
        print(f"\nFirst 20 body_length_px values:")
        for i, val in enumerate(bl_px.head(20)):
            print(f"  {i+1:2d}. {val:.10f}")
        
        # Check if all identical
        if bl_px.nunique() == 1:
            print(f"\n⚠️  ALL {len(subset)} measurements have IDENTICAL body_length_px = {bl_px.iloc[0]:.10f}")
            print(f"    This suggests a constant skeleton length artifact!")
        
        # Show some larva filenames for manual inspection
        print(f"\nSample larva filenames (first 5):")
        for fname in subset['larva_filename'].head(5):
            print(f"  - {fname}")
            
else:
    print(f"❌ Prediction file not found: {pred_file}")
    print("\nTrying to load from feature cache instead...")
    
    # Load from feature cache
    from dual_larva_classification_pipeline_geodesic import (
        ANALYSIS_DIR, collect_all_larvae, _get_cached_feats
    )
    
    all_larvae = collect_all_larvae()
    date_larvae = [(d, img, fname, path) for d, img, fname, path in all_larvae if d == '20.10']
    
    print(f"\nFound {len(date_larvae)} larvae for date 20.10 in analysis directory")
    
    if len(date_larvae) > 0:
        body_lengths = []
        filenames = []
        
        for date, img_name, fname, fpath in date_larvae[:30]:  # Check first 30
            feats = _get_cached_feats(date, img_name, fname)
            if feats:
                bl = feats.get('body_length', 0.0)
                body_lengths.append(bl)
                filenames.append(fname)
        
        if body_lengths:
            bl_arr = np.array(body_lengths)
            print(f"\nbody_length_px from feature cache (first {len(body_lengths)} larvae):")
            print(f"  mean   = {bl_arr.mean():.6f}")
            print(f"  std    = {bl_arr.std():.6f}")
            print(f"  min    = {bl_arr.min():.6f}")
            print(f"  max    = {bl_arr.max():.6f}")
            print(f"  unique = {len(np.unique(bl_arr))}")
            
            print(f"\nFirst 20 values:")
            for i, (bl, fn) in enumerate(zip(body_lengths[:20], filenames[:20])):
                print(f"  {i+1:2d}. {bl:.10f}  ({fn})")
            
            if len(np.unique(bl_arr)) == 1:
                print(f"\n⚠️  ALL measurements are IDENTICAL = {bl_arr[0]:.10f}")


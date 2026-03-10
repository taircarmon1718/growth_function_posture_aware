#!/usr/bin/env python3
"""
DIAGNOSTIC TOOL: Analyze body length measurements for constant-value artifacts.

This script helps identify dates where all larvae have identical body_length values,
which indicates degenerate skeletons, artifacts, or processing errors.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_predictions(pred_file: Path):
    """Analyze predictions file for constant-length artifacts."""

    if not pred_file.exists():
        print(f"❌ File not found: {pred_file}")
        return

    print(f"Loading: {pred_file}")
    df = pd.read_excel(pred_file)
    print(f"Total predictions: {len(df)}")

    # Filter for valid body lengths
    valid = df[df['body_length_px'] > 0].copy()
    print(f"Valid body lengths (>0): {len(valid)}")

    if len(valid) == 0:
        print("❌ No valid body length measurements found!")
        return

    # Group by date and compute statistics
    stats = valid.groupby('date').agg({
        'body_length_px': ['count', 'mean', 'std', 'min', 'max', 'nunique'],
        'body_length_mm': ['mean', 'std']
    }).reset_index()

    stats.columns = ['date', 'n', 'mean_px', 'std_px', 'min_px', 'max_px', 'unique_px',
                     'mean_mm', 'std_mm']

    # Sort by date
    stats = stats.sort_values('date')

    print("\n" + "="*90)
    print("BODY LENGTH STATISTICS BY DATE")
    print("="*90)
    print(f"{'Date':<10} {'n':>5} {'Mean(mm)':>10} {'Std(mm)':>10} {'Unique':>8} {'Status':<20}")
    print("-"*90)

    suspicious_dates = []

    for _, row in stats.iterrows():
        date = row['date']
        n = int(row['n'])
        mean_mm = row['mean_mm']
        std_mm = row['std_mm']
        unique = int(row['unique_px'])

        # Classify the status
        if std_mm < 0.001 and n >= 3:
            status = "⚠️  CONSTANT (std=0)"
            suspicious_dates.append((date, n, mean_mm, std_mm, unique))
        elif unique == 1 and n >= 3:
            status = "⚠️  ALL IDENTICAL"
            suspicious_dates.append((date, n, mean_mm, std_mm, unique))
        elif std_mm < 0.1 and n >= 10:
            status = "⚠️  Very low std"
            suspicious_dates.append((date, n, mean_mm, std_mm, unique))
        elif unique < n * 0.5 and n >= 10:
            status = "⚠️  Low diversity"
        else:
            status = "✓ OK"

        print(f"{date:<10} {n:>5} {mean_mm:>10.3f} {std_mm:>10.3f} {unique:>8} {status:<20}")

    print("="*90)

    # Detailed analysis of suspicious dates
    if suspicious_dates:
        print(f"\n⚠️  FOUND {len(suspicious_dates)} SUSPICIOUS DATE(S):")
        print("\nThese dates have identical or nearly identical body length measurements,")
        print("suggesting degenerate skeletons, tiny larvae fragments, or processing artifacts.\n")

        for date, n, mean_mm, std_mm, unique in suspicious_dates:
            print(f"\n{date}:")
            print(f"  • n = {n} larvae")
            print(f"  • mean = {mean_mm:.3f} mm")
            print(f"  • std  = {std_mm:.6f} mm")
            print(f"  • unique values = {unique}")
            print(f"  • mean_px = {mean_mm / 0.232255814:.3f}")

            # Show the actual values for this date
            date_data = valid[valid['date'] == date]['body_length_px'].values
            print(f"  • All values: {sorted(set(date_data))}")

            if len(set(date_data)) <= 3:
                print(f"  • Value counts:")
                for val, count in zip(*np.unique(date_data, return_counts=True)):
                    print(f"      {val:.3f} px appears {count} times")
    else:
        print("\n✓ No suspicious constant-length dates detected!")

    return suspicious_dates


def main():
    base_dir = Path(__file__).parent

    # Check both model directories
    targets = [
        base_dir / 'dual_larva_models_geodesic/predictions/posture_predictions.xlsx',
        base_dir / 'dual_larva_models_geodesic/predictions/valid_predictions.xlsx',
        base_dir / 'dual_larva_models/predictions/posture_predictions.xlsx',
        base_dir / 'dual_larva_models/predictions/valid_predictions.xlsx',
    ]

    for target in targets:
        if target.exists():
            print(f"\n{'#'*90}")
            print(f"# {target.relative_to(base_dir)}")
            print(f"{'#'*90}")
            analyze_predictions(target)
            print("\n")


if __name__ == '__main__':
    main()


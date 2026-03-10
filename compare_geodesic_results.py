#!/usr/bin/env python3
"""
compare_geodesic_results.py
============================
Compare body length statistics between:
  - dual_larva_models_geodesic/   (ORIGINAL - with artifacts)
  - dual_larva_models_geodesic2/  (FIXED - filtered artifacts)

This shows the impact of the skeleton filtering fix.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_stats(stats_file):
    """Load length statistics CSV."""
    if not stats_file.exists():
        return None
    return pd.read_csv(stats_file)

def compare_stats(original_file, fixed_file, label):
    """Compare statistics between original and fixed versions."""
    print(f"\n{'='*90}")
    print(f"COMPARISON: {label}")
    print(f"{'='*90}")

    orig = load_stats(original_file)
    fixed = load_stats(fixed_file)

    if orig is None and fixed is None:
        print(f"  ⚠️  Neither file exists yet")
        return
    elif orig is None:
        print(f"  ⚠️  Original not found: {original_file}")
        return
    elif fixed is None:
        print(f"  ⚠️  Fixed not found: {fixed_file}")
        return

    # Merge on date
    comparison = orig.merge(fixed, on='date', suffixes=('_orig', '_fixed'), how='outer')

    print(f"\n{'Date':<10} {'Original':<30} {'Fixed':<30} {'Change':<20}")
    print(f"{'':10} {'n':>5} {'mean':>10} {'std':>10} {'n':>5} {'mean':>10} {'std':>10} {'Δn':>5} {'status':<15}")
    print('-' * 90)

    suspicious_fixed = []
    improved_dates = []

    for _, row in comparison.iterrows():
        date = row['date']

        # Original stats
        n_orig = row.get('n_orig', 0)
        mean_orig = row.get('mean_mm_orig', np.nan)
        std_orig = row.get('std_mm_orig', np.nan)

        # Fixed stats
        n_fixed = row.get('n_fixed', 0)
        mean_fixed = row.get('mean_mm_fixed', np.nan)
        std_fixed = row.get('std_mm_fixed', np.nan)

        # Changes
        delta_n = n_fixed - n_orig if not np.isnan(n_fixed) and not np.isnan(n_orig) else 0

        # Status
        status = ""
        if std_orig < 0.01 and n_orig >= 5:
            if std_fixed > 0.1:
                status = "✓ FIXED"
                improved_dates.append(date)
            elif n_fixed == 0:
                status = "✓ ALL FILTERED"
                improved_dates.append(date)
            else:
                status = "⚠️ STILL CONST"
                suspicious_fixed.append(date)
        elif std_fixed < 0.01 and n_fixed >= 5:
            status = "⚠️ NEW CONST"
            suspicious_fixed.append(date)
        elif delta_n < -5:
            status = "Artifacts removed"
        else:
            status = "OK"

        print(f"{date:<10} {n_orig:>5.0f} {mean_orig:>10.3f} {std_orig:>10.3f} "
              f"{n_fixed:>5.0f} {mean_fixed:>10.3f} {std_fixed:>10.3f} "
              f"{delta_n:>5.0f} {status:<15}")

    # Summary
    print('\n' + '='*90)
    print('SUMMARY:')
    print('='*90)

    if improved_dates:
        print(f"✓ Improved dates (std increased or all filtered): {len(improved_dates)}")
        for d in improved_dates:
            print(f"    - {d}")

    if suspicious_fixed:
        print(f"\n⚠️  Dates still showing constant length in FIXED version: {len(suspicious_fixed)}")
        for d in suspicious_fixed:
            print(f"    - {d} (may need stricter filtering)")

    if not improved_dates and not suspicious_fixed:
        print("  No significant changes detected")

    # Overall statistics
    orig_total = orig['n'].sum()
    fixed_total = fixed['n'].sum()
    filtered = orig_total - fixed_total

    print(f"\nTotal larvae:")
    print(f"  Original: {orig_total:.0f}")
    print(f"  Fixed:    {fixed_total:.0f}")
    print(f"  Filtered: {filtered:.0f} ({100*filtered/orig_total:.1f}%)")

    # Check for dates with std=0
    const_orig = len(orig[(orig['std_mm'] < 0.01) & (orig['n'] >= 5)])
    const_fixed = len(fixed[(fixed['std_mm'] < 0.01) & (fixed['n'] >= 5)])

    print(f"\nDates with std ≈ 0 (constant length):")
    print(f"  Original: {const_orig}")
    print(f"  Fixed:    {const_fixed}")
    if const_orig > const_fixed:
        print(f"  ✓ Improvement: {const_orig - const_fixed} dates fixed!")


def main():
    base_dir = Path(__file__).parent

    # Check both valid and posture predictions
    comparisons = [
        (
            base_dir / 'dual_larva_models_geodesic/figures/length_statistics_valid.csv',
            base_dir / 'dual_larva_models_geodesic2/figures/length_statistics_valid.csv',
            'VALID LARVAE'
        ),
        (
            base_dir / 'dual_larva_models_geodesic/figures/length_statistics_posture.csv',
            base_dir / 'dual_larva_models_geodesic2/figures/length_statistics_posture.csv',
            'CORRECT POSTURE LARVAE'
        ),
    ]

    print("\n" + "╔" + "═"*88 + "╗")
    print("║" + " "*25 + "GEODESIC PIPELINE COMPARISON" + " "*35 + "║")
    print("╚" + "═"*88 + "╝")
    print("\nComparing:")
    print("  OLD (with artifacts):  dual_larva_models_geodesic/")
    print("  NEW (filtered):        dual_larva_models_geodesic2/")

    for orig_file, fixed_file, label in comparisons:
        compare_stats(orig_file, fixed_file, label)

    print("\n" + "="*90)
    print("NEXT STEPS:")
    print("="*90)
    print("1. If you see dates marked '✓ FIXED' → The fix is working!")
    print("2. If you see dates marked '⚠️ STILL CONST' → Increase thresholds (20→30 pixels)")
    print("3. Compare prediction files:")
    print("     dual_larva_models_geodesic/predictions/*.xlsx")
    print("     dual_larva_models_geodesic2/predictions/*.xlsx")
    print("="*90 + "\n")


if __name__ == '__main__':
    main()


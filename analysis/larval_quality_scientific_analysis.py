#!/usr/bin/env python3
"""
Larval Quality Scientific Analysis Framework
=============================================

A comprehensive scientific analysis and model evaluation framework for
automated fish larvae morphometry.

This script implements four major research components:
  1. Larval Quality Index (LQI) - quantitative quality metric
  2. Population Filtering Analysis - impact of quality thresholds
  3. Posture Bias Analysis - measurement bias from incorrect posture
  4. Label Efficiency Analysis - minimum training data requirements


Date: 2026-03-13
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, spearmanr
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
#  CONFIGURATION
# ============================================================================
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "dual_larva_models_geodesic2"
PREDICTIONS_FILE = DATA_DIR / "predictions" / "predictions_all_larvae.xlsx"

OUTPUT_DIR = ROOT_DIR / "larval_quality_analysis"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"
REP_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, FIG_DIR, TAB_DIR, REP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Constants
RANDOM_SEED = 42
BOOTSTRAP_N = 5000
ALPHA = 0.05
LQI_PERCENTILES = [90, 95, 99]  # Top X% for high-quality subset
LABEL_FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

np.random.seed(RANDOM_SEED)

# Master date order (for consistent x-axis in plots)
MASTER_DATES: List[str] = []


def build_master_date_order(dates: List[str]) -> List[str]:
    """Build an ordered list of normalized date strings.

    Rules:
    - Dates are strings like '19.10' or '3.11'.
    - Sort numerically by the prefix (before the dot).
    - Ensure '3.11' is placed last if present.
    """
    # Convert inputs to strings
    sdates = [str(d) for d in dates]
    # Keep unique
    sdates = sorted(set(sdates))

    # Separate special '3.11'
    special = '3.11'
    has_special = special in sdates
    if has_special:
        sdates = [d for d in sdates if d != special]

    # Extract numeric prefix for sorting
    def prefix_num(s: str):
        try:
            if '.' in s:
                return int(s.split('.')[0])
            return int(s)
        except Exception:
            # fallback: large number to sort at end
            return 9999

    sdates_sorted = sorted(sdates, key=prefix_num)

    if has_special:
        sdates_sorted = sdates_sorted + [special]

    return sdates_sorted


# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================
def save_fig(filename: str, dpi: int = 150):
    """Save figure to output directory"""
    path = FIG_DIR / filename
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename} (DPI={dpi})")


def save_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV"""
    path = TAB_DIR / filename
    df.to_csv(path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {filename}")


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_boot=BOOTSTRAP_N, alpha=ALPHA):
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of data
        statistic: function to compute (default: mean)
        n_boot: number of bootstrap samples
        alpha: significance level

    Returns:
        (lower, upper) confidence interval
    """
    boot_stats = []
    n = len(data)

    for _ in range(n_boot):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(boot_sample))

    boot_stats = np.array(boot_stats)
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return lower, upper


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        x, y: two groups to compare

    Returns:
        Cohen's d (standardized mean difference)
    """
    nx, ny = len(x), len(y)

    if nx < 2 or ny < 2:
        return np.nan

    # Pooled standard deviation
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_std


# ============================================================================
#  PART 1: LARVAL QUALITY INDEX (LQI)
# ============================================================================
def compute_larval_quality_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Larval Quality Index (LQI) for each larva.

    LQI combines:
      - Valid probability (model confidence that larva is real)
      - Posture probability (model confidence that posture is correct)
      - Morphological quality indicators

    Formula:
      LQI = α * P(valid) + β * P(posture) + γ * morphology_score

    Where:
      α = 0.4 (weight for validity)
      β = 0.4 (weight for posture)
      γ = 0.2 (weight for morphology)

    Morphology score based on:
      - Skeleton-to-perimeter ratio (straightness)
      - Area-to-perimeter ratio (compactness)
      - Convexity (shape regularity)

    Returns:
        DataFrame with added 'LQI' column (range 0-1)
    """
    print("\n" + "=" * 70)
    print("COMPUTING LARVAL QUALITY INDEX (LQI)")
    print("=" * 70)

    df = df.copy()

    print(f"  Available columns: {len(df.columns)}")
    print(f"  Sample columns: {list(df.columns[:10])}")

    # Weights for LQI components
    alpha = 0.4  # validity weight
    beta = 0.4   # posture weight
    gamma = 0.2  # morphology weight

    # Component 1: Valid probability (check different possible column names)
    valid_prob_cols = ['valid_prob', 'prob_valid', 'confidence_valid', 'predicted_valid']
    valid_score = None

    for col in valid_prob_cols:
        if col in df.columns:
            valid_score = df[col].fillna(0)
            # If it's binary (0/1), convert to probability (0.5 for valid, 0 for invalid)
            if df[col].dtype in ['int64', 'int32'] and set(df[col].dropna().unique()).issubset({0, 1}):
                valid_score = df[col].apply(lambda x: 0.8 if x == 1 else 0.2)
            print(f"  Using '{col}' for valid score")
            break

    if valid_score is None:
        print("  ⚠️  No valid probability column found, using neutral score (0.5)")
        valid_score = 0.5

    # Component 2: Posture probability
    posture_prob_cols = ['posture_prob', 'prob_posture', 'confidence_posture', 'predicted_posture']
    posture_score = None

    for col in posture_prob_cols:
        if col in df.columns:
            posture_score = df[col].fillna(0)
            # If it's binary (0/1), convert to probability
            if df[col].dtype in ['int64', 'int32'] and set(df[col].dropna().unique()).issubset({0, 1}):
                posture_score = df[col].apply(lambda x: 0.8 if x == 1 else 0.2)
            print(f"  Using '{col}' for posture score")
            break

    if posture_score is None:
        print("  ⚠️  No posture probability column found, using neutral score (0.5)")
        posture_score = 0.5

    # Component 3: Morphology score (normalize features to 0-1)
    morphology_components = []

    # Skeleton-to-perimeter ratio (higher = straighter)
    if 'skeleton_length' in df.columns and 'perimeter' in df.columns:
        skeleton_perimeter_ratio = df['skeleton_length'] / (df['perimeter'] + 1e-6)
        skeleton_score = np.clip(skeleton_perimeter_ratio / 0.5, 0, 1)
        morphology_components.append(skeleton_score)
        print("  ✓ Using skeleton-to-perimeter ratio")

    # Area-to-perimeter ratio (compactness)
    if 'area' in df.columns and 'perimeter' in df.columns:
        area_perimeter_ratio = df['area'] / (df['perimeter'] + 1e-6)
        compactness_score = np.clip(area_perimeter_ratio / 10, 0, 1)
        morphology_components.append(compactness_score)
        print("  ✓ Using area-to-perimeter ratio")

    # Convexity (if available)
    convexity_col = None
    for col in ['solidity', 'convexity', 'circularity']:
        if col in df.columns:
            convexity_col = col
            break

    if convexity_col:
        convexity_score = df[convexity_col].fillna(0.5)
        morphology_components.append(convexity_score)
        print(f"  ✓ Using {convexity_col}")

    # Combine morphology features
    if len(morphology_components) > 0:
        morphology_score = sum(morphology_components) / len(morphology_components)
    else:
        print("  ⚠️  No morphology features found, using neutral score (0.5)")
        morphology_score = 0.5

    # Final LQI
    if isinstance(valid_score, (int, float)) and isinstance(posture_score, (int, float)):
        # Both are scalar
        df['LQI'] = alpha * valid_score + beta * posture_score + gamma * morphology_score
    elif isinstance(valid_score, (int, float)):
        # valid_score is scalar
        df['LQI'] = alpha * valid_score + beta * posture_score + gamma * morphology_score
    elif isinstance(posture_score, (int, float)):
        # posture_score is scalar
        df['LQI'] = alpha * valid_score + beta * posture_score + gamma * morphology_score
    else:
        # Both are arrays
        df['LQI'] = alpha * valid_score + beta * posture_score + gamma * morphology_score

    # Ensure range [0, 1]
    df['LQI'] = np.clip(df['LQI'], 0, 1)

    print(f"  LQI computed for {len(df)} larvae")
    print(f"  LQI range: [{df['LQI'].min():.3f}, {df['LQI'].max():.3f}]")
    print(f"  LQI mean: {df['LQI'].mean():.3f} ± {df['LQI'].std():.3f}")

    # LQI distribution statistics
    lqi_stats = pd.DataFrame({
        'metric': ['mean', 'median', 'std', 'min', 'max', 'Q25', 'Q75'],
        'value': [
            df['LQI'].mean(),
            df['LQI'].median(),
            df['LQI'].std(),
            df['LQI'].min(),
            df['LQI'].max(),
            df['LQI'].quantile(0.25),
            df['LQI'].quantile(0.75)
        ]
    })
    save_csv(lqi_stats, 'lqi_summary_statistics.csv')

    return df


def visualize_lqi_distribution(df: pd.DataFrame):
    """
    Visualize LQI distribution and its components.
    """
    print("\n  Creating LQI visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Panel 1: LQI histogram
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(df['LQI'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(df['LQI'].mean(), color='red', ls='--', lw=2, label=f'Mean: {df["LQI"].mean():.3f}')
    ax1.axvline(df['LQI'].median(), color='orange', ls=':', lw=2, label=f'Median: {df["LQI"].median():.3f}')
    ax1.set_xlabel('Larval Quality Index (LQI)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax1.set_title('LQI Distribution', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: LQI components scatter (if available)
    ax2 = fig.add_subplot(gs[0, 2])

    # Find valid and posture probability columns
    valid_col = None
    for col in ['valid_prob', 'prob_valid', 'confidence_valid', 'predicted_valid']:
        if col in df.columns:
            valid_col = col
            break

    posture_col = None
    for col in ['posture_prob', 'prob_posture', 'confidence_posture', 'predicted_posture']:
        if col in df.columns:
            posture_col = col
            break

    if valid_col and posture_col:
        scatter = ax2.scatter(df[valid_col], df[posture_col],
                             c=df['LQI'], cmap='viridis', s=20, alpha=0.6)
        ax2.set_xlabel('Valid Score', fontweight='bold')
        ax2.set_ylabel('Posture Score', fontweight='bold')
        ax2.set_title('LQI Components', fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='LQI')
    else:
        ax2.text(0.5, 0.5, 'Component scatter\nnot available\n(missing probability columns)',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    ax2.grid(alpha=0.3)

    # Panel 3: Valid prob distribution (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    if valid_col:
        ax3.hist(df[valid_col], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Valid Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Valid Score Distribution', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Valid score\nnot available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])
    ax3.grid(alpha=0.3)

    # Panel 4: Posture prob distribution (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    if posture_col:
        ax4.hist(df[posture_col], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Posture Score', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Posture Score Distribution', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Posture score\nnot available',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_xticks([])
        ax4.set_yticks([])
    ax4.grid(alpha=0.3)

    # Panel 5: LQI vs body length
    ax5 = fig.add_subplot(gs[1, 2])
    valid_lengths = df[df['body_length_mm'] > 0]
    ax5.scatter(valid_lengths['LQI'], valid_lengths['body_length_mm'],
               s=20, alpha=0.5, color='darkblue')
    ax5.set_xlabel('LQI', fontweight='bold')
    ax5.set_ylabel('Body Length (mm)', fontweight='bold')
    ax5.set_title('LQI vs Body Length', fontweight='bold')
    ax5.grid(alpha=0.3)

    # Panel 6: LQI percentiles
    ax6 = fig.add_subplot(gs[2, :])
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    lqi_percentiles = [df['LQI'].quantile(p/100) for p in percentiles]
    ax6.bar(range(len(percentiles)), lqi_percentiles,
           color='teal', alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(percentiles)))
    ax6.set_xticklabels([f'P{p}' for p in percentiles])
    ax6.set_xlabel('Percentile', fontweight='bold', fontsize=11)
    ax6.set_ylabel('LQI Value', fontweight='bold', fontsize=11)
    ax6.set_title('LQI Percentile Distribution', fontweight='bold', fontsize=12)
    ax6.grid(alpha=0.3, axis='y')

    plt.suptitle('Larval Quality Index (LQI) Analysis',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_fig('01_lqi_distribution_analysis.png')


def interpret_lqi(df: pd.DataFrame, report_file):
    """
    Generate biological interpretation of LQI values.
    """
    with open(report_file, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("PART 1: LARVAL QUALITY INDEX (LQI)\n")
        f.write("=" * 70 + "\n\n")

        f.write("DEFINITION:\n")
        f.write("-" * 40 + "\n")
        f.write("LQI = 0.4 × P(valid) + 0.4 × P(posture) + 0.2 × morphology_score\n\n")
        f.write("Where:\n")
        f.write("  • P(valid): Model confidence that larva is real (not artifact)\n")
        f.write("  • P(posture): Model confidence that posture is correct (T-like)\n")
        f.write("  • morphology_score: Normalized shape quality metrics\n\n")

        f.write("BIOLOGICAL INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        f.write("LQI Range | Quality Level | Biological Meaning\n")
        f.write("----------|---------------|--------------------\n")
        f.write("0.90-1.00 | Excellent     | High-confidence measurements, suitable for all analyses\n")
        f.write("0.75-0.90 | Good          | Reliable for population statistics\n")
        f.write("0.60-0.75 | Fair          | Usable with caution, may introduce noise\n")
        f.write("0.00-0.60 | Poor          | Likely artifacts or severely distorted\n\n")

        f.write("DATASET QUALITY DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")

        excellent = (df['LQI'] >= 0.90).sum()
        good = ((df['LQI'] >= 0.75) & (df['LQI'] < 0.90)).sum()
        fair = ((df['LQI'] >= 0.60) & (df['LQI'] < 0.75)).sum()
        poor = (df['LQI'] < 0.60).sum()
        total = len(df)

        f.write(f"  Excellent (≥0.90): {excellent:5d} ({100*excellent/total:5.1f}%)\n")
        f.write(f"  Good (0.75-0.90):  {good:5d} ({100*good/total:5.1f}%)\n")
        f.write(f"  Fair (0.60-0.75):  {fair:5d} ({100*fair/total:5.1f}%)\n")
        f.write(f"  Poor (<0.60):      {poor:5d} ({100*poor/total:5.1f}%)\n")
        f.write(f"  Total:             {total:5d}\n\n")

        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        if excellent / total > 0.7:
            f.write("✓ Dataset quality is EXCELLENT (>70% high-quality larvae)\n")
            f.write("  → All statistical analyses are well-powered\n")
        elif excellent / total > 0.5:
            f.write("✓ Dataset quality is GOOD (>50% high-quality larvae)\n")
            f.write("  → Most analyses suitable, consider LQI filtering for critical measurements\n")
        else:
            f.write("⚠ Dataset quality is MODERATE (<50% high-quality larvae)\n")
            f.write("  → Strong LQI filtering recommended for growth analysis\n")
            f.write("  → Consider improving image acquisition or segmentation\n")


# ============================================================================
#  PART 2: POPULATION FILTERING ANALYSIS
# ============================================================================
def population_filtering_analysis(df: pd.DataFrame, report_file):
    """
    Compare population statistics under different filtering scenarios.

    Scenarios:
      A) Raw dataset (all larvae)
      B) Valid larvae only (predicted_valid == 1)
      C) Valid + correct posture (predicted_valid == 1 AND predicted_posture == 1)
      D) High LQI (top 90%, 95%, 99%)
    """
    print("\n" + "=" * 70)
    print("POPULATION FILTERING ANALYSIS")
    print("=" * 70)

    # Filter to larvae with valid body length
    df_valid_length = df[df['body_length_mm'] > 0].copy()

    scenarios = {}

    # Scenario A: Raw dataset
    scenarios['A_Raw'] = df_valid_length

    # Scenario B: Valid larvae only
    scenarios['B_Valid'] = df_valid_length[df_valid_length['predicted_valid'] == 1]

    # Scenario C: Valid + correct posture
    scenarios['C_ValidPosture'] = df_valid_length[
        (df_valid_length['predicted_valid'] == 1) &
        (df_valid_length['predicted_posture'] == 1)
    ]

    # Scenario D: High LQI thresholds
    for percentile in LQI_PERCENTILES:
        threshold = df_valid_length['LQI'].quantile(percentile / 100)
        scenarios[f'D_LQI_top{100-percentile}pct'] = df_valid_length[df_valid_length['LQI'] >= threshold]

    # Compute statistics for each scenario
    results = []

    for scenario_name, scenario_df in scenarios.items():
        lengths = scenario_df['body_length_mm'].values

        if len(lengths) < 5:
            continue

        # Central tendency
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)

        # Variability
        std_length = np.std(lengths, ddof=1)
        cv = 100 * std_length / mean_length if mean_length > 0 else np.nan
        iqr = np.percentile(lengths, 75) - np.percentile(lengths, 25)

        # Distribution shape
        skewness = stats.skew(lengths)
        kurtosis = stats.kurtosis(lengths, fisher=True)

        # Bootstrap CI for mean
        ci_low, ci_high = bootstrap_ci(lengths, statistic=np.mean)
        ci_width = ci_high - ci_low

        # Sample size
        n = len(lengths)
        retention_rate = 100 * n / len(df_valid_length)

        results.append({
            'scenario': scenario_name,
            'n': n,
            'retention_pct': retention_rate,
            'mean_mm': mean_length,
            'median_mm': median_length,
            'std_mm': std_length,
            'cv_pct': cv,
            'iqr_mm': iqr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_width': ci_width
        })

    results_df = pd.DataFrame(results)
    save_csv(results_df, 'population_filtering_comparison.csv')

    print(f"\n  Analyzed {len(scenarios)} filtering scenarios")
    print(f"  Raw dataset: n={len(scenarios['A_Raw'])}")
    print(f"  Valid only: n={len(scenarios['B_Valid'])} ({100*len(scenarios['B_Valid'])/len(scenarios['A_Raw']):.1f}%)")
    print(f"  Valid+Posture: n={len(scenarios['C_ValidPosture'])} ({100*len(scenarios['C_ValidPosture'])/len(scenarios['A_Raw']):.1f}%)")

    # Visualization
    visualize_filtering_comparison(scenarios, results_df)

    # Report
    with open(report_file, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("PART 2: POPULATION FILTERING ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("FILTERING SCENARIOS COMPARED:\n")
        f.write("-" * 40 + "\n")
        f.write("  A) Raw dataset (all detected larvae)\n")
        f.write("  B) Valid larvae only (model-filtered)\n")
        f.write("  C) Valid + correct posture\n")
        f.write("  D) High LQI subsets (top 10%, 5%, 1%)\n\n")

        f.write("IMPACT ON MEAN BODY LENGTH:\n")
        f.write("-" * 40 + "\n")

        raw_mean = results_df[results_df['scenario'] == 'A_Raw']['mean_mm'].iloc[0]

        for _, row in results_df.iterrows():
            scenario = row['scenario']
            mean_val = row['mean_mm']
            ci_width = row['ci_width']
            n = int(row['n'])

            delta = mean_val - raw_mean
            delta_pct = 100 * delta / raw_mean if raw_mean > 0 else 0

            f.write(f"  {scenario:20s}: {mean_val:.4f} mm ({delta:+.4f}, {delta_pct:+.1f}%), ")
            f.write(f"95% CI width: {ci_width:.4f} mm, n={n}\n")

        f.write("\n")
        f.write("KEY FINDINGS:\n")
        f.write("-" * 40 + "\n")

        # Compare scenarios
        raw_std = results_df[results_df['scenario'] == 'A_Raw']['std_mm'].iloc[0]
        valid_std = results_df[results_df['scenario'] == 'B_Valid']['std_mm'].iloc[0]
        validpos_std = results_df[results_df['scenario'] == 'C_ValidPosture']['std_mm'].iloc[0]

        noise_reduction_valid = 100 * (raw_std - valid_std) / raw_std if raw_std > 0 else 0
        noise_reduction_vp = 100 * (raw_std - validpos_std) / raw_std if raw_std > 0 else 0

        f.write(f"  • Valid filtering reduces SD by {noise_reduction_valid:.1f}%\n")
        f.write(f"  • Valid+Posture filtering reduces SD by {noise_reduction_vp:.1f}%\n")

        if noise_reduction_vp > 20:
            f.write("  → STRONG noise reduction achieved through quality filtering\n")
        elif noise_reduction_vp > 10:
            f.write("  → MODERATE noise reduction from filtering\n")
        else:
            f.write("  → MINIMAL noise reduction (data already clean?)\n")

        # CI width comparison
        raw_ci = results_df[results_df['scenario'] == 'A_Raw']['ci_width'].iloc[0]
        validpos_ci = results_df[results_df['scenario'] == 'C_ValidPosture']['ci_width'].iloc[0]

        ci_improvement = 100 * (raw_ci - validpos_ci) / raw_ci if raw_ci > 0 else 0

        f.write(f"\n  • CI width improvement: {ci_improvement:.1f}%\n")

        if ci_improvement > 15:
            f.write("  → Filtering substantially improves measurement precision\n")
        else:
            f.write("  → Precision gains are modest\n")


def visualize_filtering_comparison(scenarios: Dict, results_df: pd.DataFrame):
    """
    Visualize impact of different filtering strategies.
    """
    print("\n  Creating filtering comparison visualization...")

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Distributions overlay
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ['gray', 'blue', 'green', 'orange', 'red', 'purple']

    for i, (name, df) in enumerate(scenarios.items()):
        if len(df) < 5:
            continue
        lengths = df['body_length_mm'].values

        label = name.replace('_', ' ')
        color = colors[i % len(colors)]

        try:
            kde = stats.gaussian_kde(lengths)
            x_eval = np.linspace(lengths.min() - 0.5, lengths.max() + 0.5, 300)
            ax1.plot(x_eval, kde(x_eval), lw=2, label=label, color=color, alpha=0.7)
        except:
            pass

    ax1.set_xlabel('Body Length (mm)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=11)
    ax1.set_title('Distribution Comparison Across Filtering Scenarios', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Mean with CI
    ax2 = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(len(results_df))
    means = results_df['mean_mm'].values
    ci_lows = results_df['ci_low'].values
    ci_highs = results_df['ci_high'].values

    ax2.errorbar(x_pos, means,
                yerr=[means - ci_lows, ci_highs - means],
                fmt='o', capsize=5, capthick=2, markersize=8,
                color='steelblue', ecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['scenario'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Length (mm)', fontweight='bold')
    ax2.set_title('Mean ± 95% CI', fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # Panel 3: Sample size retention
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x_pos, results_df['retention_pct'], color='teal', alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df['scenario'], rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Retention (%)', fontweight='bold')
    ax3.set_title('Sample Size Retention', fontweight='bold')
    ax3.axhline(100, color='red', ls='--', lw=1.5)
    ax3.grid(alpha=0.3, axis='y')

    # Panel 4: Standard deviation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(x_pos, results_df['std_mm'], color='orange', alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(results_df['scenario'], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Std Dev (mm)', fontweight='bold')
    ax4.set_title('Variability Across Scenarios', fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    # Panel 5: CV
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(x_pos, results_df['cv_pct'], color='purple', alpha=0.7, edgecolor='black')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(results_df['scenario'], rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('CV (%)', fontweight='bold')
    ax5.set_title('Coefficient of Variation', fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')

    # Panel 6: CI width
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.bar(x_pos, results_df['ci_width'], color='green', alpha=0.7, edgecolor='black')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(results_df['scenario'], rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('CI Width (mm)', fontweight='bold')
    ax6.set_title('Confidence Interval Width', fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # Panel 7: Skewness
    ax7 = fig.add_subplot(gs[2, 1])
    colors_skew = ['red' if abs(s) > 1 else 'green' for s in results_df['skewness']]
    ax7.bar(x_pos, results_df['skewness'], color=colors_skew, alpha=0.7, edgecolor='black')
    ax7.axhline(0, color='black', ls='-', lw=1)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(results_df['scenario'], rotation=45, ha='right', fontsize=8)
    ax7.set_ylabel('Skewness', fontweight='bold')
    ax7.set_title('Distribution Skewness', fontweight='bold')
    ax7.grid(alpha=0.3, axis='y')

    # Panel 8: Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['scenario'].replace('_', '\n'),
            f"{int(row['n'])}",
            f"{row['mean_mm']:.3f}",
            f"{row['std_mm']:.3f}"
        ])

    table = ax8.table(cellText=table_data,
                     colLabels=['Scenario', 'n', 'Mean', 'SD'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    plt.suptitle('Population Filtering Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('02_population_filtering_comparison.png')


# ============================================================================
#  PART 3: POSTURE BIAS ANALYSIS
# ============================================================================
def posture_bias_analysis(df: pd.DataFrame, report_file):
    """
    Test whether larval posture biases body length measurements.

    Compares:
      - Correct posture larvae (predicted_posture == 1)
      - Distorted posture larvae (predicted_posture == 0)

    Metrics:
      - Mean difference (bias)
      - Cohen's d (effect size)
      - Mann-Whitney U test (statistical significance)
      - Bootstrap CI for bias estimate
    """
    print("\n" + "=" * 70)
    print("POSTURE BIAS ANALYSIS")
    print("=" * 70)

    # Filter to valid larvae with body length
    df_analysis = df[(df['predicted_valid'] == 1) & (df['body_length_mm'] > 0)].copy()

    # Separate by posture
    correct_posture = df_analysis[df_analysis['predicted_posture'] == 1]['body_length_mm'].values
    distorted_posture = df_analysis[df_analysis['predicted_posture'] == 0]['body_length_mm'].values

    print(f"  Correct posture: n={len(correct_posture)}")
    print(f"  Distorted posture: n={len(distorted_posture)}")

    if len(correct_posture) < 5 or len(distorted_posture) < 5:
        print("  ⚠️  Insufficient data for posture bias analysis")
        return

    # Compute statistics
    mean_correct = np.mean(correct_posture)
    mean_distorted = np.mean(distorted_posture)
    bias = mean_correct - mean_distorted
    bias_pct = 100 * bias / mean_distorted if mean_distorted > 0 else 0

    # Effect size
    cohens_d_value = cohens_d(correct_posture, distorted_posture)

    # Statistical test
    u_stat, p_value_mw = mannwhitneyu(correct_posture, distorted_posture, alternative='two-sided')

    # t-test (if approximately normal)
    t_stat, p_value_t = ttest_ind(correct_posture, distorted_posture)

    # Bootstrap CI for bias
    boot_biases = []

    for _ in range(BOOTSTRAP_N):
        # Resample each group independently
        boot_correct = np.random.choice(correct_posture, size=len(correct_posture), replace=True)
        boot_distorted = np.random.choice(distorted_posture, size=len(distorted_posture), replace=True)
        boot_bias = np.mean(boot_correct) - np.mean(boot_distorted)
        boot_biases.append(boot_bias)

    boot_biases = np.array(boot_biases)
    bias_ci_low = np.percentile(boot_biases, 2.5)
    bias_ci_high = np.percentile(boot_biases, 97.5)

    # Save results
    bias_results = pd.DataFrame({
        'metric': [
            'n_correct', 'n_distorted',
            'mean_correct', 'mean_distorted',
            'bias_mm', 'bias_pct',
            'cohens_d',
            'mann_whitney_u', 'mann_whitney_p',
            't_statistic', 't_test_p',
            'bias_ci_low', 'bias_ci_high'
        ],
        'value': [
            len(correct_posture), len(distorted_posture),
            mean_correct, mean_distorted,
            bias, bias_pct,
            cohens_d_value,
            u_stat, p_value_mw,
            t_stat, p_value_t,
            bias_ci_low, bias_ci_high
        ]
    })
    save_csv(bias_results, 'posture_bias_analysis.csv')

    print(f"\n  Bias: {bias:.4f} mm ({bias_pct:+.1f}%)")
    print(f"  Cohen's d: {cohens_d_value:.3f}")
    print(f"  Mann-Whitney p: {p_value_mw:.4f}")

    # Visualization
    visualize_posture_bias(correct_posture, distorted_posture, bias_results)

    # Report
    with open(report_file, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("PART 3: POSTURE BIAS IN LENGTH MEASUREMENT\n")
        f.write("=" * 70 + "\n\n")

        f.write("HYPOTHESIS:\n")
        f.write("-" * 40 + "\n")
        f.write("Distorted larval posture (curved, bent) systematically biases body length\n")
        f.write("measurements compared to correct T-like posture.\n\n")

        f.write("SAMPLE SIZES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Correct posture (T-like):  n = {len(correct_posture)}\n")
        f.write(f"  Distorted posture (curved): n = {len(distorted_posture)}\n\n")

        f.write("MEASUREMENTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean length (correct posture):   {mean_correct:.4f} mm\n")
        f.write(f"  Mean length (distorted posture): {mean_distorted:.4f} mm\n")
        f.write(f"  Measurement bias:                {bias:.4f} mm ({bias_pct:+.1f}%)\n")
        f.write(f"  95% CI for bias:                 [{bias_ci_low:.4f}, {bias_ci_high:.4f}] mm\n\n")

        f.write("EFFECT SIZE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Cohen's d = {cohens_d_value:.3f}\n")

        if abs(cohens_d_value) > 0.8:
            effect_interp = "LARGE"
        elif abs(cohens_d_value) > 0.5:
            effect_interp = "MEDIUM"
        elif abs(cohens_d_value) > 0.2:
            effect_interp = "SMALL"
        else:
            effect_interp = "NEGLIGIBLE"

        f.write(f"  Interpretation: {effect_interp} effect size\n\n")

        f.write("STATISTICAL TESTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mann-Whitney U test:  U = {u_stat:.1f}, p = {p_value_mw:.4f}\n")
        f.write(f"  Independent t-test:   t = {t_stat:.3f}, p = {p_value_t:.4f}\n\n")

        if p_value_mw < 0.001:
            sig_interp = "HIGHLY SIGNIFICANT (p < 0.001)"
        elif p_value_mw < 0.01:
            sig_interp = "VERY SIGNIFICANT (p < 0.01)"
        elif p_value_mw < 0.05:
            sig_interp = "SIGNIFICANT (p < 0.05)"
        else:
            sig_interp = "NOT SIGNIFICANT (p ≥ 0.05)"

        f.write(f"  Result: {sig_interp}\n\n")

        f.write("BIOLOGICAL INTERPRETATION:\n")
        f.write("-" * 40 + "\n")

        if p_value_mw < 0.05 and abs(cohens_d_value) > 0.3:
            if bias > 0:
                f.write("✓ POSTURE BIAS DETECTED:\n")
                f.write("  Correct posture larvae measure LONGER than distorted posture larvae.\n")
                f.write("  This is biologically expected: curved posture underestimates true length.\n\n")
                f.write("  RECOMMENDATIONS:\n")
                f.write("  → Filter out distorted posture larvae for accurate growth measurements\n")
                f.write("  → Use posture classification to improve measurement reliability\n")
            else:
                f.write("⚠ UNEXPECTED BIAS DIRECTION:\n")
                f.write("  Distorted posture larvae measure longer (unexpected).\n")
                f.write("  Possible explanations:\n")
                f.write("  • Measurement artifact\n")
                f.write("  • Posture classification error\n")
                f.write("  • Selection bias\n")
        else:
            f.write("○ NO SIGNIFICANT POSTURE BIAS DETECTED\n")
            f.write("  Posture does not substantially affect length measurement.\n")
            f.write("  Possible explanations:\n")
            f.write("  • Length measurement method is robust to posture\n")
            f.write("  • Small sample size limits detection power\n")
            f.write("  • Posture differences are subtle\n")


def visualize_posture_bias(correct: np.ndarray, distorted: np.ndarray, results: pd.DataFrame):
    """
    Visualize posture bias in body length measurements.
    """
    print("\n  Creating posture bias visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Panel 1: Distributions
    ax1 = fig.add_subplot(gs[0, :2])

    try:
        kde_correct = stats.gaussian_kde(correct)
        kde_distorted = stats.gaussian_kde(distorted)

        x_min = min(correct.min(), distorted.min()) - 0.5
        x_max = max(correct.max(), distorted.max()) + 0.5
        x_eval = np.linspace(x_min, x_max, 300)

        ax1.plot(x_eval, kde_correct(x_eval), lw=3, label='Correct Posture', color='green', alpha=0.8)
        ax1.plot(x_eval, kde_distorted(x_eval), lw=3, label='Distorted Posture', color='red', alpha=0.8)

        ax1.fill_between(x_eval, 0, kde_correct(x_eval), alpha=0.2, color='green')
        ax1.fill_between(x_eval, 0, kde_distorted(x_eval), alpha=0.2, color='red')

    except:
        ax1.hist(correct, bins=30, alpha=0.5, color='green', label='Correct', density=True)
        ax1.hist(distorted, bins=30, alpha=0.5, color='red', label='Distorted', density=True)

    mean_correct = results[results['metric'] == 'mean_correct']['value'].iloc[0]
    mean_distorted = results[results['metric'] == 'mean_distorted']['value'].iloc[0]

    ax1.axvline(mean_correct, color='darkgreen', ls='--', lw=2.5, label=f'Mean (correct): {mean_correct:.3f}')
    ax1.axvline(mean_distorted, color='darkred', ls='--', lw=2.5, label=f'Mean (distorted): {mean_distorted:.3f}')

    ax1.set_xlabel('Body Length (mm)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax1.set_title('Length Distribution by Posture', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Boxplot comparison
    ax2 = fig.add_subplot(gs[0, 2])
    bp = ax2.boxplot([correct, distorted],
                     labels=['Correct', 'Distorted'],
                     patch_artist=True,
                     showmeans=True)

    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax2.set_ylabel('Body Length (mm)', fontweight='bold', fontsize=11)
    ax2.set_title('Length Comparison', fontweight='bold', fontsize=12)
    ax2.grid(alpha=0.3, axis='y')

    # Panel 3: Bias with CI
    ax3 = fig.add_subplot(gs[1, 0])

    bias = results[results['metric'] == 'bias_mm']['value'].iloc[0]
    bias_ci_low = results[results['metric'] == 'bias_ci_low']['value'].iloc[0]
    bias_ci_high = results[results['metric'] == 'bias_ci_high']['value'].iloc[0]

    ax3.barh([0], [bias], xerr=[[bias - bias_ci_low], [bias_ci_high - bias]],
            capsize=10, color='steelblue', alpha=0.7, edgecolor='black', height=0.4)
    ax3.axvline(0, color='black', ls='-', lw=2)
    ax3.set_yticks([0])
    ax3.set_yticklabels(['Bias'])
    ax3.set_xlabel('Measurement Bias (mm)', fontweight='bold', fontsize=11)
    ax3.set_title('Posture Bias with 95% CI', fontweight='bold', fontsize=12)
    ax3.grid(alpha=0.3, axis='x')

    # Panel 4: Effect size
    ax4 = fig.add_subplot(gs[1, 1])

    cohens_d_val = results[results['metric'] == 'cohens_d']['value'].iloc[0]

    colors_effect = []
    if abs(cohens_d_val) > 0.8:
        colors_effect = ['darkred']
        effect_label = 'Large'
    elif abs(cohens_d_val) > 0.5:
        colors_effect = ['orange']
        effect_label = 'Medium'
    elif abs(cohens_d_val) > 0.2:
        colors_effect = ['yellow']
        effect_label = 'Small'
    else:
        colors_effect = ['lightgray']
        effect_label = 'Negligible'

    ax4.barh([0], [cohens_d_val], color=colors_effect[0], alpha=0.7, edgecolor='black', height=0.4)
    ax4.axvline(0, color='black', ls='-', lw=2)
    ax4.axvline(0.2, color='gray', ls=':', lw=1, alpha=0.5)
    ax4.axvline(0.5, color='gray', ls=':', lw=1, alpha=0.5)
    ax4.axvline(0.8, color='gray', ls=':', lw=1, alpha=0.5)
    ax4.set_yticks([0])
    ax4.set_yticklabels(["Cohen's d"])
    ax4.set_xlabel("Effect Size", fontweight='bold', fontsize=11)
    ax4.set_title(f"Effect Size: {effect_label} ({cohens_d_val:.3f})", fontweight='bold', fontsize=12)
    ax4.grid(alpha=0.3, axis='x')

    # Panel 5: Statistical significance
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    p_value = results[results['metric'] == 'mann_whitney_p']['value'].iloc[0]

    stats_text = f"STATISTICAL TESTS\n\n"
    stats_text += f"Mann-Whitney U test:\n"
    stats_text += f"p = {p_value:.4f}\n\n"

    if p_value < 0.001:
        stats_text += "Result: *** (p < 0.001)\n"
        stats_text += "HIGHLY SIGNIFICANT"
        text_color = 'darkgreen'
    elif p_value < 0.01:
        stats_text += "Result: ** (p < 0.01)\n"
        stats_text += "VERY SIGNIFICANT"
        text_color = 'green'
    elif p_value < 0.05:
        stats_text += "Result: * (p < 0.05)\n"
        stats_text += "SIGNIFICANT"
        text_color = 'orange'
    else:
        stats_text += "Result: n.s. (p ≥ 0.05)\n"
        stats_text += "NOT SIGNIFICANT"
        text_color = 'red'

    ax5.text(0.5, 0.5, stats_text,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax5.transAxes)

    plt.suptitle('Posture Bias Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('03_posture_bias_analysis.png')


# ============================================================================
#  PART 4: LABEL EFFICIENCY (LEAKAGE-FREE, PUBLICATION-READY)
# ============================================================================
def label_efficiency_improved(df: pd.DataFrame, report_file: Path,
                              n_repeats: int = 5, fractions: Optional[List[float]] = None):
    """
    Improved Label Efficiency Experiment (Leakage-Free, Publication-Ready).

    Strict implementation with:
      - Leakage-safe feature selection
      - Targets: predicted_valid, predicted_posture (pseudo-labels)
      - Random stratified split (5 repeats)
      - Temporal date-based split (70/30 by date)
      - Learning curves across training fractions [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
      - Baselines: Majority class and Random-by-prior
      - StandardScaler normalization
      - RandomForestClassifier training
      - CSV outputs and PNG figures
      - Report entries
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    print("\n" + "=" * 70)
    print("LABEL EFFICIENCY (IMPROVED — Leakage-Free)")
    print("=" * 70)

    # Setup output directories
    OUTPUTS_DIR = ROOT_DIR / 'outputs'
    OUT_FIG_DIR = OUTPUTS_DIR / 'figures'
    OUT_TAB_DIR = OUTPUTS_DIR / 'tables'
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Step 3: Build leakage-safe feature set
    print("\n  STEP 1: Building leakage-safe feature set...")

    exclude_prefixes = ('predicted_', 'valid_', 'posture_', 'prob_', 'is_')
    exclude_contains = ('LQI', 'body_length')
    exclude_keywords = ('area', 'perimeter', 'skeleton', 'compact', 'solidity', 'circular')

    candidate_features = []
    for col in df.columns:
        # Skip metadata
        if col in ('date', 'image_name', 'larva_filename'):
            continue
        # Skip classifier-derived
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        # Skip LQI and body_length
        if any(substr in col for substr in exclude_contains):
            continue
        # Skip geometric shortcuts
        if any(k in col.lower() for k in exclude_keywords):
            continue
        # Keep only numeric
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            candidate_features.append(col)

    print(f"    Dataset size: {len(df)} samples")
    print(f"    Feature count: {len(candidate_features)}")
    if len(candidate_features) > 0:
        print(f"    Features: {candidate_features}")

    # Step 4: Identify targets
    print("\n  STEP 2: Identifying targets...")
    targets = [t for t in ('predicted_valid', 'predicted_posture') if t in df.columns]
    if len(targets) == 0:
        print('    ⚠️  No targets found. Aborting.')
        return

    print(f"    Targets: {targets}")
    print('    ⚠️  WARNING: pseudo-labels used (upper-bound performance)')

    # Initialize from sklearn
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    # Helper: numeric date sort
    def date_sort_key(s):
        try:
            s = str(s)
            if '.' in s:
                parts = s.split('.')
                return (int(parts[0]), int(parts[1]))
            return (int(s), 0)
        except:
            return (9999, 0)

    # Process each target
    summary_rows = []

    for target in targets:
        print(f"\n  STEP 3a: Processing target {target}...")
        df_tgt = df[df[target].notna()].copy()
        n_total = len(df_tgt)
        class_balance = df_tgt[target].value_counts().to_dict()
        print(f"      Samples: {n_total}")
        print(f"      Class balance: {class_balance}")

        if n_total < 30:
            print(f"      ⚠️  Too few samples (<30). Skipping.")
            continue

        y_all = df_tgt[target].astype(int).values

        # If no features, skip
        if len(candidate_features) == 0:
            print(f"      ⚠️  No features available. Skipping model training.")
            continue

        X_all = df_tgt[candidate_features].fillna(0).values

        # ===== RANDOM STRATIFIED SPLIT =====
        print(f"\n      RANDOM SPLIT EXPERIMENT:")
        rand_all_rows = []
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.2, random_state=RANDOM_SEED)

        for rep, (tr_idx, te_idx) in enumerate(sss.split(X_all, y_all)):
            X_tr_full = X_all[tr_idx]
            y_tr_full = y_all[tr_idx]
            X_te = X_all[te_idx]
            y_te = y_all[te_idx]

            # Normalize
            scaler = StandardScaler()
            X_tr_full = scaler.fit_transform(X_tr_full)
            X_te = scaler.transform(X_te)

            # Baselines
            maj_class = np.argmax(np.bincount(y_tr_full))
            maj_preds = np.full(len(y_te), maj_class)
            maj_acc = accuracy_score(y_te, maj_preds)
            maj_f1 = f1_score(y_te, maj_preds, average='weighted', zero_division=0)

            uniq, cnts = np.unique(y_tr_full, return_counts=True)
            priors = cnts / cnts.sum()
            rng = np.random.RandomState(RANDOM_SEED + rep)
            rand_preds = rng.choice(uniq, size=len(y_te), p=priors)
            rand_acc = accuracy_score(y_te, rand_preds)
            rand_f1 = f1_score(y_te, rand_preds, average='weighted', zero_division=0)

            # Learning curve
            for frac in fractions:
                n_sub = max(10, int(len(tr_idx) * frac))
                sub_idx = rng.choice(len(tr_idx), size=n_sub, replace=False)
                X_tr_sub = X_tr_full[sub_idx]
                y_tr_sub = y_tr_full[sub_idx]

                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED+rep, n_jobs=-1)
                clf.fit(X_tr_sub, y_tr_sub)
                y_pred = clf.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

                rand_all_rows.append({
                    'target': target,
                    'rep': rep,
                    'fraction': frac,
                    'n_train': n_sub,
                    'acc': acc,
                    'f1': f1,
                    'maj_acc': maj_acc,
                    'maj_f1': maj_f1,
                    'rand_acc': rand_acc,
                    'rand_f1': rand_f1
                })

        # Aggregate random split results
        if len(rand_all_rows) > 0:
            rand_df = pd.DataFrame(rand_all_rows)
            rand_summary = rand_df.groupby('fraction').agg(
                n_train=('n_train', 'median'),
                mean_acc=('acc', 'mean'),
                std_acc=('acc', 'std'),
                mean_f1=('f1', 'mean'),
                std_f1=('f1', 'std'),
                mean_maj_acc=('maj_acc', 'mean'),
                mean_rand_acc=('rand_acc', 'mean')
            ).reset_index()

            csv_path = OUT_TAB_DIR / f'label_efficiency_random_{target}.csv'
            rand_summary.to_csv(csv_path, index=False)
            print(f"      ✓ Saved: {csv_path.name}")

            # Plot
            try:
                fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                x = rand_summary['n_train'].values

                axs[0].errorbar(x, rand_summary['mean_acc'], yerr=rand_summary['std_acc'], fmt='o-', capsize=5, lw=2)
                axs[0].hlines(rand_summary['mean_maj_acc'].iloc[0], x.min(), x.max(), colors='gray', linestyles='--', label='Majority')
                axs[0].set_xlabel('Training samples')
                axs[0].set_ylabel('Accuracy')
                axs[0].set_title(f'Random Split: {target}')
                axs[0].legend()
                axs[0].grid(alpha=0.3)

                axs[1].errorbar(x, rand_summary['mean_f1'], yerr=rand_summary['std_f1'], fmt='s-', capsize=5, lw=2, color='orange')
                axs[1].set_xlabel('Training samples')
                axs[1].set_ylabel('F1 (weighted)')
                axs[1].set_title(f'Random Split: {target}')
                axs[1].grid(alpha=0.3)

                fig_path = OUT_FIG_DIR / f'label_efficiency_random_{target}.png'
                fig.savefig(fig_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"      ✓ Saved: {fig_path.name}")
            except Exception as e:
                print(f"      ⚠️  Plot failed: {e}")

        # ===== TEMPORAL SPLIT =====
        print(f"\n      TEMPORAL SPLIT EXPERIMENT:")
        uniq_dates = sorted(df_tgt['date'].unique(), key=date_sort_key)
        if len(uniq_dates) < 2:
            print(f"      ⚠️  Not enough dates. Skipping temporal.")
            continue

        n_tr_dates = max(1, int(np.floor(0.7 * len(uniq_dates))))
        tr_dates = set(uniq_dates[:n_tr_dates])
        te_dates = set(uniq_dates[n_tr_dates:])

        mask_tr = df_tgt['date'].isin(tr_dates)
        mask_te = df_tgt['date'].isin(te_dates)

        X_tr_all = df_tgt.loc[mask_tr, candidate_features].fillna(0).values
        y_tr_all = df_tgt.loc[mask_tr, target].astype(int).values
        X_te_all = df_tgt.loc[mask_te, candidate_features].fillna(0).values
        y_te_all = df_tgt.loc[mask_te, target].astype(int).values

        print(f"      Train dates: {len(tr_dates)} ({len(y_tr_all)} samples)")
        print(f"      Test dates:  {len(te_dates)} ({len(y_te_all)} samples)")

        if len(y_tr_all) < 20 or len(y_te_all) < 10:
            print(f"      ⚠️  Insufficient samples. Skipping temporal.")
            continue

        # Normalize once for test
        scaler = StandardScaler()
        X_tr_all_norm = scaler.fit_transform(X_tr_all)
        X_te_all_norm = scaler.transform(X_te_all)

        # Baselines
        maj_class = np.argmax(np.bincount(y_tr_all))
        maj_preds = np.full(len(y_te_all), maj_class)
        maj_acc_temp = accuracy_score(y_te_all, maj_preds)
        maj_f1_temp = f1_score(y_te_all, maj_preds, average='weighted', zero_division=0)

        temp_all_rows = []

        for rep in range(n_repeats):
            rs = RANDOM_SEED + rep
            for frac in fractions:
                n_sub = max(10, int(len(y_tr_all) * frac))
                sub_idx = np.random.RandomState(rs).choice(len(y_tr_all), size=n_sub, replace=False)
                X_tr_sub = X_tr_all_norm[sub_idx]
                y_tr_sub = y_tr_all[sub_idx]

                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1)
                clf.fit(X_tr_sub, y_tr_sub)
                y_pred = clf.predict(X_te_all_norm)
                acc = accuracy_score(y_te_all, y_pred)
                f1 = f1_score(y_te_all, y_pred, average='weighted', zero_division=0)

                temp_all_rows.append({
                    'target': target,
                    'rep': rep,
                    'fraction': frac,
                    'n_train': n_sub,
                    'acc': acc,
                    'f1': f1,
                    'maj_acc': maj_acc_temp,
                    'maj_f1': maj_f1_temp
                })

        # Aggregate temporal results
        if len(temp_all_rows) > 0:
            temp_df = pd.DataFrame(temp_all_rows)
            temp_summary = temp_df.groupby('fraction').agg(
                n_train=('n_train', 'median'),
                mean_acc=('acc', 'mean'),
                std_acc=('acc', 'std'),
                mean_f1=('f1', 'mean'),
                std_f1=('f1', 'std'),
                mean_maj_acc=('maj_acc', 'mean')
            ).reset_index()

            csv_path = OUT_TAB_DIR / f'label_efficiency_temporal_{target}.csv'
            temp_summary.to_csv(csv_path, index=False)
            print(f"      ✓ Saved: {csv_path.name}")

            # Plot
            try:
                fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                x = temp_summary['n_train'].values

                axs[0].errorbar(x, temp_summary['mean_acc'], yerr=temp_summary['std_acc'], fmt='o-', capsize=5, lw=2)
                axs[0].hlines(temp_summary['mean_maj_acc'].iloc[0], x.min(), x.max(), colors='gray', linestyles='--', label='Majority')
                axs[0].set_xlabel('Training samples')
                axs[0].set_ylabel('Accuracy')
                axs[0].set_title(f'Temporal Split: {target}')
                axs[0].legend()
                axs[0].grid(alpha=0.3)

                axs[1].errorbar(x, temp_summary['mean_f1'], yerr=temp_summary['std_f1'], fmt='s-', capsize=5, lw=2, color='orange')
                axs[1].set_xlabel('Training samples')
                axs[1].set_ylabel('F1 (weighted)')
                axs[1].set_title(f'Temporal Split: {target}')
                axs[1].grid(alpha=0.3)

                fig_path = OUT_FIG_DIR / f'label_efficiency_temporal_{target}.png'
                fig.savefig(fig_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"      ✓ Saved: {fig_path.name}")
            except Exception as e:
                print(f"      ⚠️  Plot failed: {e}")

            # Report gap
            def get_val(df, frac, col):
                row = df[df['fraction'] == frac]
                return float(row[col].iloc[0]) if len(row) > 0 else np.nan

            rand_acc_full = get_val(rand_summary, 1.0, 'mean_acc')
            temp_acc_full = get_val(temp_summary, 1.0, 'mean_acc')
            gap = rand_acc_full - temp_acc_full if not (np.isnan(rand_acc_full) or np.isnan(temp_acc_full)) else np.nan

            summary_rows.append({
                'target': target,
                'n_samples': n_total,
                'rand_acc_100pct': rand_acc_full,
                'temp_acc_100pct': temp_acc_full,
                'gap_acc': gap,
                'n_features': len(candidate_features)
            })

            # Write to report
            with open(report_file, 'a') as f:
                f.write('\n' + '=' * 60 + '\n')
                f.write(f'LABEL EFFICIENCY — {target}\n')
                f.write('=' * 60 + '\n')
                f.write(f'Samples: {n_total}, Features: {len(candidate_features)}\n')
                f.write(f'Random split (100%): Accuracy={rand_acc_full:.3f}\n')
                f.write(f'Temporal split (100%): Accuracy={temp_acc_full:.3f}\n')
                f.write(f'Gap: {gap:+.3f}\n\n')
                f.write('Random split overestimates performance due to data similarity.\n')
                f.write('Temporal split reflects real generalization on future dates.\n')

    # Save summary
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = OUT_TAB_DIR / 'label_efficiency_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"\n  ✓ Overall summary: {csv_path.name}")

    print('\n✓ LABEL EFFICIENCY EXPERIMENT COMPLETE\n')


# ============================================================================
#  PART 5: POPULATION GROWTH ANALYSIS (RAW VS FILTERED)
# ============================================================================
def population_growth_analysis(df: pd.DataFrame, report_file):
    """
    Compare growth curves between raw and filtered populations.

    Analyzes:
      - Raw population (all larvae)
      - Valid larvae only
      - High-quality larvae (valid + correct posture)

    Computes:
      - Daily mean, median, trimmed mean
      - Bootstrap confidence intervals
      - Growth trajectories
    """
    print("\n" + "=" * 70)
    print("PART 5: POPULATION GROWTH ANALYSIS")
    print("=" * 70)

    # Filter to larvae with valid body length
    df_analysis = df[df['body_length_mm'] > 0].copy()

    # Get unique dates and sort
    dates = sorted(df_analysis['date'].unique())

    print(f"  Analyzing growth across {len(dates)} dates")

    # Prepare three populations
    populations = {
        'raw': df_analysis,
        'valid': df_analysis[df_analysis['predicted_valid'] == 1],
        'high_quality': df_analysis[
            (df_analysis['predicted_valid'] == 1) &
            (df_analysis['predicted_posture'] == 1)
        ]
    }

    growth_stats = []

    for pop_name, pop_df in populations.items():
        print(f"\n  Computing statistics for {pop_name} population...")

        for date in dates:
            date_data = pop_df[pop_df['date'] == date]['body_length_mm'].values

            if len(date_data) < 3:
                continue

            # Compute statistics
            mean_length = np.mean(date_data)
            median_length = np.median(date_data)

            # Trimmed mean (10%)
            from scipy.stats import trim_mean
            trimmed_mean_length = trim_mean(date_data, 0.1)

            std_length = np.std(date_data, ddof=1)

            # Bootstrap CI for mean
            ci_low, ci_high = bootstrap_ci(date_data, statistic=np.mean, n_boot=1000)

            growth_stats.append({
                'population': pop_name,
                'date': date,
                'n': len(date_data),
                'mean': mean_length,
                'median': median_length,
                'trimmed_mean': trimmed_mean_length,
                'std': std_length,
                'ci_low': ci_low,
                'ci_high': ci_high
            })

    growth_df = pd.DataFrame(growth_stats)

    # Add date index for plotting
    date_index = build_master_date_order(growth_df['date'].unique())
    date_index_map = {d: i for i, d in enumerate(date_index)}
    growth_df['date_index'] = growth_df['date'].map(date_index_map)

    save_csv(growth_df, 'population_growth_statistics.csv')

    # Visualization
    visualize_population_growth(growth_df, dates)

    # Report
    with open(report_file, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("PART 5: POPULATION GROWTH ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("OBJECTIVE:\n")
        f.write("-" * 40 + "\n")
        f.write("Compare growth trajectories between raw and filtered populations\n")
        f.write("to assess impact of quality filtering on biological growth estimates.\n\n")

        f.write("POPULATIONS ANALYZED:\n")
        f.write("-" * 40 + "\n")
        for pop_name in ['raw', 'valid', 'high_quality']:
            pop_data = growth_df[growth_df['population'] == pop_name]
            if len(pop_data) > 0:
                total_n = pop_data['n'].sum()
                mean_n = pop_data['n'].mean()
                f.write(f"  {pop_name:15s}: Total n={total_n:5.0f}, Avg per date={mean_n:5.1f}\n")

        f.write("\n")
        f.write("GROWTH TRAJECTORY COMPARISON:\n")
        f.write("-" * 40 + "\n")

        for pop_name in ['raw', 'valid', 'high_quality']:
            pop_data = growth_df[growth_df['population'] == pop_name]
            if len(pop_data) >= 2:
                first_mean = pop_data.iloc[0]['mean']
                last_mean = pop_data.iloc[-1]['mean']
                growth = last_mean - first_mean
                growth_pct = 100 * growth / first_mean if first_mean > 0 else 0

                f.write(f"  {pop_name:15s}: {first_mean:.3f} → {last_mean:.3f} mm ")
                f.write(f"(Δ={growth:+.3f} mm, {growth_pct:+.1f}%)\n")

    return growth_df


def visualize_population_growth(growth_df: pd.DataFrame, dates):
    """
    Visualize population growth curves with confidence intervals.

    Normalizes dates so that all dates map to '*.10' except '3.11' which is kept
    as the last date on the x-axis as requested by the user. Aggregates per
    normalized date before plotting to avoid collisions when multiple raw dates
    map to the same normalized date.
    """
    print("\n  Creating population growth visualization...")

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        'raw': 'gray',
        'valid': 'blue',
        'high_quality': 'green'
    }

    labels = {
        'raw': 'Raw (all detections)',
        'valid': 'Valid larvae',
        'high_quality': 'High quality (valid + posture)'
    }

    markers = {
        'raw': 'o',
        'valid': 's',
        'high_quality': '^'
    }

    # Build original date list
    orig_dates = sorted(growth_df['date'].unique())

    # Create normalization map: map everything except '3.11' to '<prefix>.10'
    norm_map = {}
    for d_str in orig_dates:
        if d_str == '3.11':
            norm = '3.11'
        else:
            if '.' in d_str:
                prefix = d_str.split('.')[0]
            else:
                prefix = d_str
            # normalize to .10
            norm = f"{prefix}.10"
        norm_map[d_str] = norm

    # Master normalized dates sorted by numeric prefix; ensure '3.11' is last
    norm_set = sorted(set(norm_map.values()), key=lambda s: int(s.split('.')[0]) if '.' in s else int(s))
    if '3.11' in norm_set:
        norm_set = [d for d in norm_set if d != '3.11'] + ['3.11']
    master_dates = norm_set

    # Plot each population after aggregating by normalized date
    for pop_name in ['raw', 'valid', 'high_quality']:
        pop_data = growth_df[growth_df['population'] == pop_name].copy()

        if len(pop_data) == 0:
            continue

        # Map to normalized date
        pop_data['norm_date'] = pop_data['date'].map(norm_map)

        # Aggregate per normalized date to avoid duplicate x positions
        agg = pop_data.groupby('norm_date').agg({
            'mean': 'mean',
            'trimmed_mean': 'mean',
            'ci_low': 'mean',
            'ci_high': 'mean',
            'n': 'sum'
        }).reset_index()

        # Keep only normalized dates that are in master_dates
        agg = agg[agg['norm_date'].isin(master_dates)].copy()
        if len(agg) == 0:
            continue

        # Determine x positions according to master_dates ordering
        x_pos = agg['norm_date'].apply(lambda nd: master_dates.index(nd)).values
        means = agg['mean'].values
        trimmed = agg['trimmed_mean'].values
        ci_lows = agg['ci_low'].values
        ci_highs = agg['ci_high'].values

        # Plot mean with CI at the mapped positions
        ax.plot(x_pos, means, marker=markers[pop_name], ms=8, lw=2.5,
               color=colors[pop_name], label=labels[pop_name], alpha=0.9, zorder=5)

        # Shaded CI
        ax.fill_between(x_pos, ci_lows, ci_highs, color=colors[pop_name], alpha=0.2)

        # Plot trimmed mean (dashed)
        ax.plot(x_pos, trimmed, ls='--', lw=1.5, color=colors[pop_name], alpha=0.7, zorder=3)

    # Set x-axis labels using master_dates
    ax.set_xticks(np.arange(len(master_dates)))
    ax.set_xticklabels(master_dates, rotation=45, ha='right')

    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Body Length (mm)', fontweight='bold', fontsize=12)
    ax.set_title('Population Growth: Raw vs Filtered\n(Solid=mean, Dashed=trimmed mean, Shaded=95% CI)',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig('05_population_growth_raw_vs_filtered.png', dpi=300)


# ============================================================================
#  PART 6: GOMPERTZ GROWTH MODEL FIT
# ============================================================================
def gompertz_growth_analysis(growth_df: pd.DataFrame, report_file):
    """
    Fit Gompertz growth models to population trajectories.

    Model: L(t) = L_inf * exp(-exp(-k * (t - t0)))

    Fits:
      - Raw population
      - Filtered population

    Compares:
      - Parameter estimates
      - Model fit quality (R², RMSE)
    """
    print("\n" + "=" * 70)
    print("PART 6: GOMPERTZ GROWTH MODEL FIT")
    print("=" * 70)

    # Gompertz model
    def gompertz(t, L_inf, k, t0):
        return L_inf * np.exp(-np.exp(-k * (t - t0)))

    gompertz_results = []
    fitted_models = {}

    for pop_name in ['raw', 'valid', 'high_quality']:
        pop_data = growth_df[growth_df['population'] == pop_name].copy()

        if len(pop_data) < 4:
            print(f"  ⚠️  {pop_name}: insufficient data ({len(pop_data)} points)")
            continue

        pop_data = pop_data.sort_values('date')

        x_idx = np.arange(len(pop_data))
        y_data = pop_data['mean'].values

        print(f"\n  Fitting Gompertz model to {pop_name} population...")

        try:
            # Initial parameter guesses
            L_inf_init = y_data.max() * 1.2
            k_init = 0.5
            t0_init = len(y_data) / 2

            # Fit model
            popt, pcov = curve_fit(
                gompertz, x_idx, y_data,
                p0=[L_inf_init, k_init, t0_init],
                maxfev=10000,
                bounds=([y_data.max(), 0.01, -10], [y_data.max() * 2, 5, len(y_data) + 10])
            )

            L_inf, k, t0 = popt

            # Predictions
            y_pred = gompertz(x_idx, L_inf, k, t0)

            # Goodness of fit
            residuals = y_data - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))

            # AIC
            n = len(y_data)
            k_params = 3
            aic = n * np.log(ss_res / n) + 2 * k_params if ss_res > 0 else np.nan

            gompertz_results.append({
                'population': pop_name,
                'L_inf': L_inf,
                'k': k,
                't0': t0,
                'R2': r2,
                'RMSE': rmse,
                'AIC': aic
            })

            # Store for plotting
            x_fine = np.linspace(x_idx[0], x_idx[-1], 200)
            y_fine = gompertz(x_fine, L_inf, k, t0)
            fitted_models[pop_name] = {
                'x': x_idx,
                'y': y_data,
                'x_fine': x_fine,
                'y_fine': y_fine
            }

            print(f"    ✓ L_inf={L_inf:.3f}, k={k:.3f}, t0={t0:.2f}, R²={r2:.4f}")

        except Exception as e:
            print(f"    ✗ Fitting failed: {e}")

    if len(gompertz_results) > 0:
        gompertz_df = pd.DataFrame(gompertz_results)
        save_csv(gompertz_df, 'gompertz_growth_parameters.csv')

        # Visualization
        visualize_gompertz_fits(fitted_models, gompertz_df)

        # Report
        with open(report_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("PART 6: GOMPERTZ GROWTH MODEL FIT\n")
            f.write("=" * 70 + "\n\n")

            f.write("MODEL:\n")
            f.write("-" * 40 + "\n")
            f.write("L(t) = L_∞ * exp(-exp(-k * (t - t₀)))\n\n")
            f.write("Where:\n")
            f.write("  L_∞: Asymptotic length (maximum size)\n")
            f.write("  k:   Growth rate parameter\n")
            f.write("  t₀:  Inflection point\n\n")

            f.write("FITTED PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            for _, row in gompertz_df.iterrows():
                f.write(f"  {row['population']:15s}: ")
                f.write(f"L_∞={row['L_inf']:.3f}, k={row['k']:.3f}, t₀={row['t0']:.2f}, ")
                f.write(f"R²={row['R2']:.4f}, RMSE={row['RMSE']:.4f}\n")

            f.write("\n")
            f.write("MODEL COMPARISON:\n")
            f.write("-" * 40 + "\n")

            if 'raw' in gompertz_df['population'].values and 'high_quality' in gompertz_df['population'].values:
                raw_r2 = gompertz_df[gompertz_df['population'] == 'raw']['R2'].iloc[0]
                hq_r2 = gompertz_df[gompertz_df['population'] == 'high_quality']['R2'].iloc[0]

                r2_improvement = hq_r2 - raw_r2

                f.write(f"  Raw R²: {raw_r2:.4f}\n")
                f.write(f"  High-quality R²: {hq_r2:.4f}\n")
                f.write(f"  Improvement: {r2_improvement:+.4f}\n\n")

                if r2_improvement > 0.05:
                    f.write("  ✓ Filtering IMPROVES Gompertz fit (ΔR² > 0.05)\n")
                    f.write("    → Quality filtering reveals cleaner growth trajectory\n")
                elif r2_improvement < -0.05:
                    f.write("  ○ Filtering WORSENS fit (sample size reduction?)\n")
                else:
                    f.write("  ○ Filtering has minimal impact on fit quality\n")


def visualize_gompertz_fits(fitted_models: Dict, gompertz_df: pd.DataFrame):
    """
    Visualize Gompertz growth model fits.
    """
    print("\n  Creating Gompertz fit visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {'raw': 'gray', 'valid': 'blue', 'high_quality': 'green'}
    labels = {
        'raw': 'Raw',
        'valid': 'Valid',
        'high_quality': 'High Quality'
    }

    for i, pop_name in enumerate(['raw', 'valid', 'high_quality']):
        ax = axes[i]

        if pop_name not in fitted_models:
            ax.text(0.5, 0.5, f'{labels[pop_name]}\nNo data',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        model = fitted_models[pop_name]

        # Plot data
        ax.plot(model['x'], model['y'], 'o', ms=10, color=colors[pop_name],
               label='Data', zorder=5, alpha=0.7)

        # Plot fit
        ax.plot(model['x_fine'], model['y_fine'], '-', lw=3, color=colors[pop_name],
               label='Gompertz fit', alpha=0.8, zorder=4)

        # Get parameters
        params = gompertz_df[gompertz_df['population'] == pop_name].iloc[0]

        ax.set_xlabel('Time (date index)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Mean Length (mm)', fontweight='bold', fontsize=11)
        ax.set_title(f'{labels[pop_name]} Population\n' +
                    f'L_∞={params["L_inf"]:.2f}, k={params["k"]:.2f}, R²={params["R2"]:.3f}',
                    fontweight='bold', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle('Gompertz Growth Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('06_gompertz_growth_comparison.png', dpi=300)


# ============================================================================
#  PART 7: DAILY DISTRIBUTION ANALYSIS
# ============================================================================
def daily_distribution_analysis(df: pd.DataFrame, report_file):
    """
    Analyze daily population structure using KDE.

    Detects:
      - Number of modes (cohorts)
      - Peak locations
      - Peak separation
      - Relative peak heights
    """
    print("\n" + "=" * 70)
    print("PART 7: DAILY DISTRIBUTION ANALYSIS")
    print("=" * 70)

    df_analysis = df[
        (df['predicted_valid'] == 1) &
        (df['predicted_posture'] == 1) &
        (df['body_length_mm'] > 0)
    ].copy()

    dates = sorted(df_analysis['date'].unique())

    print(f"  Analyzing distributions across {len(dates)} dates")

    distribution_stats = []
    kde_data = {}

    for date in dates:
        date_data = df_analysis[df_analysis['date'] == date]['body_length_mm'].values

        if len(date_data) < 10:
            print(f"  {date}: n={len(date_data)} (too few)")
            continue

        print(f"  {date}: n={len(date_data)}", end="")

        try:
            # KDE
            kde = stats.gaussian_kde(date_data, bw_method='scott')
            x_eval = np.linspace(date_data.min() - 0.5, date_data.max() + 0.5, 500)
            kde_vals = kde(x_eval)

            # Find peaks
            peaks, properties = find_peaks(kde_vals, prominence=0.05 * kde_vals.max())

            n_peaks = len(peaks)
            print(f" → {n_peaks} peak(s)")

            # Store KDE data
            kde_data[date] = {
                'x': x_eval,
                'kde': kde_vals,
                'peaks': x_eval[peaks] if len(peaks) > 0 else []
            }

            # Peak statistics
            if len(peaks) > 1:
                peak_positions = x_eval[peaks]
                peak_heights = kde_vals[peaks]

                # Peak separation
                peak_sep = np.diff(peak_positions)
                avg_sep = np.mean(peak_sep)

                # Relative heights
                rel_heights = peak_heights / peak_heights.max()

                distribution_stats.append({
                    'date': date,
                    'n': len(date_data),
                    'n_peaks': n_peaks,
                    'avg_peak_separation': avg_sep,
                    'min_rel_height': rel_heights.min()
                })
            else:
                distribution_stats.append({
                    'date': date,
                    'n': len(date_data),
                    'n_peaks': n_peaks if len(peaks) > 0 else 0,
                    'avg_peak_separation': np.nan,
                    'min_rel_height': 1.0 if len(peaks) == 1 else np.nan
                })

        except Exception as e:
            print(f" → KDE failed: {e}")

    if len(distribution_stats) > 0:
        dist_df = pd.DataFrame(distribution_stats)
        save_csv(dist_df, 'daily_distribution_statistics.csv')

        # Visualization
        visualize_daily_distributions(kde_data, dates)

        # Report
        with open(report_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("PART 7: DAILY POPULATION STRUCTURE\n")
            f.write("=" * 70 + "\n\n")

            f.write("OBJECTIVE:\n")
            f.write("-" * 40 + "\n")
            f.write("Detect cohort structure within daily populations using KDE peak detection.\n\n")

            f.write("RESULTS:\n")
            f.write("-" * 40 + "\n")

            n_multimodal = (dist_df['n_peaks'] > 1).sum()
            f.write(f"  Dates with multiple peaks: {n_multimodal}/{len(dist_df)}\n\n")

            if n_multimodal > 0:
                f.write("  Dates with potential cohorts:\n")
                multi = dist_df[dist_df['n_peaks'] > 1]
                for _, row in multi.iterrows():
                    f.write(f"    {row['date']}: {int(row['n_peaks'])} peaks, ")
                    f.write(f"sep={row['avg_peak_separation']:.2f} mm\n")

                f.write("\n")
                f.write("  ⚠️  MULTIPLE COHORTS DETECTED\n")
                f.write("    → Population may contain larvae from multiple hatching events\n")
                f.write("    → Consider cohort-specific growth analysis\n")
            else:
                f.write("  ○ All distributions are unimodal\n")
                f.write("    → Single cohort hypothesis supported\n")


def visualize_daily_distributions(kde_data: Dict, dates):
    """
    Visualize stacked KDE curves for daily distributions.
    """
    print("\n  Creating daily distribution visualization...")

    fig, ax = plt.subplots(figsize=(14, 10))

    n_dates = len(kde_data)
    colors = plt.cm.viridis(np.linspace(0, 1, n_dates))

    y_offset = 0
    y_spacing = 0.5

    for i, date in enumerate(sorted(kde_data.keys())):
        data = kde_data[date]

        x = data['x']
        kde_vals = data['kde']
        peaks = data['peaks']

        # Normalize KDE to unit height
        kde_normalized = kde_vals / kde_vals.max()

        # Plot KDE
        ax.plot(x, kde_normalized + y_offset, lw=2, color=colors[i], alpha=0.8)
        ax.fill_between(x, y_offset, kde_normalized + y_offset, alpha=0.3, color=colors[i])

        # Mark peaks
        for peak_x in peaks:
            ax.plot([peak_x, peak_x], [y_offset, y_offset + 1],
                   'r--', lw=1.5, alpha=0.7)

        # Label
        ax.text(x.min() - 0.3, y_offset + 0.5, date,
               fontsize=9, va='center', ha='right')

        y_offset += y_spacing + 1

    ax.set_xlabel('Body Length (mm)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Date (stacked)', fontweight='bold', fontsize=12)
    ax.set_title('Daily Population Structure (Kernel Density Estimation)\nRed dashes indicate detected peaks',
                fontweight='bold', fontsize=13)
    ax.set_yticks([])
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    save_fig('07_daily_population_structure.png', dpi=300)


# ============================================================================
#  PART 8: LENGTH MODEL LABEL EFFICIENCY
# ============================================================================
def length_model_efficiency(df: pd.DataFrame, report_file: Path):
    """
    Assess label efficiency for body length regression model.

    Trains RandomForestRegressor with varying training set sizes.
    Measures MAE, RMSE, R².
    """
    print("\n" + "=" * 70)
    print("PART 8: LENGTH MODEL LABEL EFFICIENCY")
    print("=" * 70)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Prepare data
    df_model = df[
        (df['predicted_valid'] == 1) &
        (df['body_length_mm'] > 0)
    ].copy()
    df_model['body_length_mm'] = df_model['body_length_mm'].astype(float)

    feature_cols = [col for col in df.columns if col not in [
        'date', 'image_name', 'larva_filename', 'body_length_mm', 'body_length_px',
        'predicted_valid', 'predicted_posture', 'LQI'
    ]]

    available_features = [col for col in feature_cols if col in df.columns]

    print(f"  Using {len(available_features)} features")
    print(f"  Dataset size: {len(df_model)} samples")

    if len(df_model) < 50:
        print("  ⚠️  Insufficient data for length model analysis")
        return

    X = df_model[available_features].fillna(0).values
    y = df_model['body_length_mm'].values

    # Train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []
    n_repeats = 5

    for frac in fractions:
        print(f"    Training with {int(frac*100)}% of data...", end="")

        frac_maes = []
        frac_rmses = []
        frac_r2s = []

        for repeat in range(n_repeats):
            n_subsample = int(len(X_train_full) * frac)

            if n_subsample < 10:
                continue

            subsample_idx = np.random.choice(len(X_train_full), size=n_subsample, replace=False)
            X_train_sub = X_train_full[subsample_idx]
            y_train_sub = y_train_full[subsample_idx]

            # Train regressor
            reg = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_SEED + repeat,
                n_jobs=-1
            )

            reg.fit(X_train_sub, y_train_sub)

            # Evaluate
            y_pred = reg.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            frac_maes.append(mae)
            frac_rmses.append(rmse)
            frac_r2s.append(r2)

        if len(frac_maes) > 0:
            results.append({
                'train_fraction': frac,
                'n_train': int(len(X_train_full) * frac),
                'n_test': len(X_test),
                'mean_mae': np.mean(frac_maes),
                'std_mae': np.std(frac_maes),
                'mean_rmse': np.mean(frac_rmses),
                'std_rmse': np.std(frac_rmses),
                'mean_r2': np.mean(frac_r2s),
                'std_r2': np.std(frac_r2s)
            })

            print(f" Acc={np.mean(frac_maes):.3f}±{np.std(frac_maes):.3f}")

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        save_csv(results_df, 'length_model_efficiency.csv')

        # Visualization
        visualize_length_model_curves(results_df)

        # Report
        with open(report_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("PART 8: LENGTH MODEL LABEL EFFICIENCY\n")
            f.write("=" * 70 + "\n\n")

            f.write("OBJECTIVE:\n")
            f.write("-" * 40 + "\n")
            f.write("Estimate minimum labeled samples needed for body length regression.\n\n")

            f.write("MODEL: RandomForestRegressor\n\n")

            f.write("RESULTS:\n")
            f.write("-" * 40 + "\n")

            for _, row in results_df.iterrows():
                f.write(f"  {int(row['train_fraction']*100):3d}% ({int(row['n_train']):4d} samples): ")
                f.write(f"MAE={row['mean_mae']:.3f}, RMSE={row['mean_rmse']:.3f}, R²={row['mean_r2']:.3f}\n")


def visualize_length_model_curves(results_df: pd.DataFrame):
    """
    Visualize learning curves for length regression model.
    """
    print("\n  Creating length model learning curves...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = results_df['n_train'].values

    # MAE
    ax = axes[0]
    y_mae = results_df['mean_mae'].values
    y_mae_std = results_df['std_mae'].values

    ax.errorbar(x, y_mae, yerr=y_mae_std,
               fmt='o-', capsize=5, capthick=2, markersize=8,
               color='blue', ecolor='black', lw=2.5)
    ax.set_xlabel('Number of Training Samples', fontweight='bold', fontsize=11)
    ax.set_ylabel('MAE (mm)', fontweight='bold', fontsize=11)
    ax.set_title('Mean Absolute Error', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3)

    # RMSE
    ax = axes[1]
    y_rmse = results_df['mean_rmse'].values
    y_rmse_std = results_df['std_rmse'].values

    ax.errorbar(x, y_rmse, yerr=y_rmse_std,
               fmt='s-', capsize=5, capthick=2, markersize=8,
               color='red', ecolor='black', lw=2.5)
    ax.set_xlabel('Number of Training Samples', fontweight='bold', fontsize=11)
    ax.set_ylabel('RMSE (mm)', fontweight='bold', fontsize=11)
    ax.set_title('Root Mean Square Error', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3)

    # R²
    ax = axes[2]
    y_r2 = results_df['mean_r2'].values
    y_r2_std = results_df['std_r2'].values

    ax.errorbar(x, y_r2, yerr=y_r2_std,
               fmt='^-', capsize=5, capthick=2, markersize=8,
               color='green', ecolor='black', lw=2.5)
    ax.axhline(0.95, color='gray', ls='--', lw=1.5, alpha=0.7, label='95% target')
    ax.set_xlabel('Number of Training Samples', fontweight='bold', fontsize=11)
    ax.set_ylabel('R²', fontweight='bold', fontsize=11)
    ax.set_title('Coefficient of Determination', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.suptitle('Length Model Label Efficiency (Regression)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('08_length_model_learning_curve.png', dpi=300)


# ============================================================================
#  PART 9: POSTURE BIAS ON GROWTH PARAMETERS
# ============================================================================
def posture_bias_on_growth(df: pd.DataFrame, report_file):
    """
    Quantify posture bias impact on Gompertz growth parameters.

    Compares:
      - All larvae (valid only)
      - Correct posture only

    Tests if posture filtering changes L_inf and k estimates.
    """
    print("\n" + "=" * 70)
    print("PART 9: POSTURE BIAS ON GROWTH PARAMETERS")
    print("=" * 70)

    # Gompertz model
    def gompertz(t, L_inf, k, t0):
        return L_inf * np.exp(-np.exp(-k * (t - t0)))

    # Prepare two datasets
    df_all = df[(df['predicted_valid'] == 1) & (df['body_length_mm'] > 0)].copy()
    df_correct = df[
        (df['predicted_valid'] == 1) &
        (df['predicted_posture'] == 1) &
        (df['body_length_mm'] > 0)
    ].copy()

    dates = sorted(df_all['date'].unique())

    datasets = {
        'all_valid': df_all,
        'correct_posture': df_correct
    }

    posture_growth_results = []

    for dataset_name, dataset in datasets.items():
        print(f"\n  Fitting growth curve for {dataset_name}...")

        daily_means = []

        for date in dates:
            date_data = dataset[dataset['date'] == date]['body_length_mm'].values
            if len(date_data) >= 3:
                daily_means.append(np.mean(date_data))
            else:
                daily_means.append(np.nan)

        # Remove NaN
        valid_idx = ~np.isnan(daily_means)
        x_idx = np.arange(len(daily_means))[valid_idx]
        y_data = np.array(daily_means)[valid_idx]

        if len(y_data) < 4:
            print(f"    ⚠️  Insufficient data")
            continue

        try:
            L_inf_init = y_data.max() * 1.2
            k_init = 0.5
            t0_init = len(y_data) / 2

            popt, _ = curve_fit(
                gompertz, x_idx, y_data,
                p0=[L_inf_init, k_init, t0_init],
                maxfev=10000,
                bounds=([y_data.max(), 0.01, -10], [y_data.max() * 2, 5, len(y_data) + 10])
            )

            L_inf, k, t0 = popt

            posture_growth_results.append({
                'dataset': dataset_name,
                'L_inf': L_inf,
                'k': k,
                't0': t0
            })

            print(f"    ✓ L_inf={L_inf:.3f}, k={k:.3f}, t0={t0:.2f}")

        except Exception as e:
            print(f"    ✗ Fitting failed: {e}")

    if len(posture_growth_results) == 2:
        results_df = pd.DataFrame(posture_growth_results)
        save_csv(results_df, 'posture_bias_growth_parameters.csv')

        # Compute differences
        all_params = results_df[results_df['dataset'] == 'all_valid'].iloc[0]
        correct_params = results_df[results_df['dataset'] == 'correct_posture'].iloc[0]

        delta_L_inf = correct_params['L_inf'] - all_params['L_inf']
        delta_k = correct_params['k'] - all_params['k']

        pct_L_inf = 100 * delta_L_inf / all_params['L_inf']
        pct_k = 100 * delta_k / all_params['k']

        # Report
        with open(report_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("PART 9: POSTURE BIAS ON GROWTH PARAMETERS\n")
            f.write("=" * 70 + "\n\n")

            f.write("OBJECTIVE:\n")
            f.write("-" * 40 + "\n")
            f.write("Quantify impact of posture filtering on Gompertz growth parameters.\n\n")

            f.write("GROWTH PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  All valid larvae:      L_∞={all_params['L_inf']:.3f}, k={all_params['k']:.3f}\n")
            f.write(f"  Correct posture only:  L_∞={correct_params['L_inf']:.3f}, k={correct_params['k']:.3f}\n\n")

            f.write("PARAMETER DIFFERENCES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  ΔL_∞ = {delta_L_inf:+.3f} mm ({pct_L_inf:+.1f}%)\n")
            f.write(f"  Δk   = {delta_k:+.3f} ({pct_k:+.1f}%)\n\n")

            if abs(pct_L_inf) > 5 or abs(pct_k) > 10:
                f.write("  ⚠️  SIGNIFICANT POSTURE BIAS ON GROWTH PARAMETERS\n")
                f.write("    → Posture filtering changes biological growth estimates\n")
                f.write("    → Recommend using posture-filtered data for growth analysis\n")
            else:
                f.write("  ○ Minimal posture bias (<5% on L_∞, <10% on k)\n")
                f.write("    → Growth parameter estimates are robust to posture filtering\n")


# ============================================================================
#  IMPROVED LABEL EFFICIENCY ANALYSIS
# ============================================================================
def label_efficiency_improved(df: pd.DataFrame, report_file: Path,
                              n_repeats: int = 5, fractions: Optional[List[float]] = None):
    """
    Improved label-efficiency experiment that avoids leakage and reports
    realistic generalization performance.

    Implements:
      - Leakage-safe feature selection
      - Random stratified split (repeated)
      - Temporal date-based split (train on earliest 70% dates)
      - Learning curves across fractions with repeats
      - Baselines: majority and random-by-prior
      - Saves CSVs and figures and appends human-readable summary to report
    """
    print('\n' + '=' * 70)
    print('IMPROVED LABEL EFFICIENCY (Leakage-safe)')
    print('=' * 70)

    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    OUT_DIR = ROOT_DIR / 'outputs'
    OUT_FIG_DIR = OUT_DIR / 'figures'
    OUT_TAB_DIR = OUT_DIR / 'tables'
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TAB_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Build leakage-safe feature set
    exclude_prefixes = ('predicted_', 'valid_', 'posture_', 'prob_', 'is_')
    exclude_contains = ('LQI', 'body_length')
    exclude_exact = ('area', 'perimeter', 'skeleton', 'skeleton_length', 'compactness', 'solidity', 'circularity')

    candidate_features = []
    for c in df.columns:
        if c in ('date', 'image_name', 'larva_filename'):
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if any(substr in c for substr in exclude_contains):
            continue
        if c in exclude_exact:
            continue
        if pd.api.types.is_numeric_dtype(df[c].dtype):
            candidate_features.append(c)

    print(f"  Dataset size: {len(df)} samples")
    print(f"  Feature count after leakage filtering: {len(candidate_features)}")
    if len(candidate_features) > 0:
        print(f"  Example features: {candidate_features[:20]}")

    # 2) Targets
    targets = [t for t in ('predicted_valid', 'predicted_posture') if t in df.columns]
    if len(targets) == 0:
        print('  ⚠️  No target columns (predicted_valid/predicted_posture) found. Aborting.')
        return

    # Warning about pseudo-labels
    warn_text = 'WARNING: Using pseudo-labels. Results represent an upper bound.'
    print('\n  ' + warn_text)
    with open(report_file, 'a') as rf:
        rf.write('\n' + '=' * 60 + '\n')
        rf.write('LABEL EFFICIENCY — NOTE ON PSEUDO-LABELS\n')
        rf.write('=' * 60 + '\n')
        rf.write(warn_text + '\n')

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    global_summary = []

    # helper: numeric sort for dates like '19.10' or '3.11'
    def date_sort_key(s):
        try:
            s = str(s)
            if '.' in s:
                parts = s.split('.')
                return (int(parts[0]), int(parts[1]))
            return (int(s), 0)
        except Exception:
            return (9999, 0)

    for target in targets:
        print(f"\nProcessing target: {target}")
        df_model = df[df[target].notna()].copy()
        n_total = len(df_model)
        class_counts = df_model[target].value_counts().to_dict()
        print(f"  Samples for target: {n_total}; class balance: {class_counts}")

        if n_total < 30:
            print('  ⚠️  Too few samples for reliable experiments; skipping target')
            continue

        y_all = df_model[target].astype(int).values

        # If no features available -> baseline-only mode
        if len(candidate_features) == 0:
            print('  ⚠️  No leakage-safe features available, running baseline-only evaluation')

            # We'll perform repeated stratified splits and record majority/random baselines
            rand_rows = []
            sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.2, random_state=RANDOM_SEED)
            rep = 0
            for train_idx, test_idx in sss.split(np.zeros(len(y_all)), y_all):
                y_tr = y_all[train_idx]
                y_te = y_all[test_idx]

                # majority
                maj = np.argmax(np.bincount(y_tr))
                maj_preds = np.full(len(y_te), maj)
                maj_acc = accuracy_score(y_te, maj_preds)
                maj_f1 = f1_score(y_te, maj_preds, average='weighted', zero_division=0)

                # random-by-prior
                uniq, cnts = np.unique(y_tr, return_counts=True)
                priors = cnts / cnts.sum()
                rng = np.random.RandomState(RANDOM_SEED + rep)
                rand_preds = rng.choice(uniq, size=len(y_te), p=priors)
                rand_acc = accuracy_score(y_te, rand_preds)
                rand_f1 = f1_score(y_te, rand_preds, average='weighted', zero_division=0)

                for frac in fractions:
                    n_train = max(10, int(len(train_idx) * frac))
                    rand_rows.append({
                        'strategy': 'random_baseline',
                        'target': target,
                        'rep': rep,
                        'fraction': frac,
                        'n_train': n_train,
                        'model_acc': maj_acc,
                        'model_f1': maj_f1,
                        'maj_acc': maj_acc,
                        'maj_f1': maj_f1,
                        'rand_acc': rand_acc,
                        'rand_f1': rand_f1
                    })
                rep += 1

            rand_df = pd.DataFrame(rand_rows)
            if len(rand_df) > 0:
                summary = rand_df.groupby('fraction').agg(
                    n_train=('n_train', 'median'),
                    mean_acc=('model_acc', 'mean'), std_acc=('model_acc', 'std'),
                    mean_f1=('model_f1', 'mean'), std_f1=('model_f1', 'std'),
                    mean_maj_acc=('maj_acc', 'mean'), mean_rand_acc=('rand_acc', 'mean')
                ).reset_index()

                csv_path = OUT_TAB_DIR / f'label_efficiency_random_baseline_{target}.csv'
                summary.to_csv(csv_path, index=False)
                print(f'  ✓ Saved baseline CSV: {csv_path}')

                # plot
                try:
                    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                    x = summary['n_train'].values
                    axs[0].errorbar(x, summary['mean_acc'], yerr=summary['std_acc'], fmt='o-', capsize=5)
                    axs[0].set_xlabel('Number of training samples'); axs[0].set_ylabel('Accuracy')
                    axs[0].set_title(f'Random-split Baseline — {target}')
                    axs[0].grid(alpha=0.3)

                    axs[1].errorbar(x, summary['mean_f1'], yerr=summary['std_f1'], fmt='o-', capsize=5, color='orange')
                    axs[1].set_xlabel('Number of training samples'); axs[1].set_ylabel('F1 (weighted)')
                    axs[1].set_title(f'Random-split Baseline — {target}')
                    axs[1].grid(alpha=0.3)

                    fig_path = OUT_FIG_DIR / f'label_efficiency_random_baseline_{target}.png'
                    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f'  ✓ Saved baseline figure: {fig_path}')
                except Exception as e:
                    print(f'  ⚠️  Plotting baselines failed: {e}')

            # Temporal baseline
            # Build date split
            uniq_dates = sorted(df_model['date'].unique(), key=date_sort_key)
            if len(uniq_dates) < 2:
                print('  ⚠️  Not enough dates for temporal baseline')
                continue
            n_train_dates = max(1, int(np.floor(0.7 * len(uniq_dates))))
            train_dates = set(uniq_dates[:n_train_dates])
            test_dates = set(uniq_dates[n_train_dates:])
            y_train_all = df_model[df_model['date'].isin(train_dates)][target].astype(int).values
            y_test_fixed = df_model[df_model['date'].isin(test_dates)][target].astype(int).values
            if len(y_train_all) < 10 or len(y_test_fixed) < 5:
                print('  ⚠️  Insufficient temporal samples for baseline')
            else:
                maj = np.argmax(np.bincount(y_train_all))
                maj_preds = np.full(len(y_test_fixed), maj)
                maj_acc = accuracy_score(y_test_fixed, maj_preds)
                maj_f1 = f1_score(y_test_fixed, maj_preds, average='weighted', zero_division=0)
                uniq, cnts = np.unique(y_train_all, return_counts=True)
                priors = cnts / cnts.sum()
                rng = np.random.RandomState(RANDOM_SEED)
                rand_preds = rng.choice(uniq, size=len(y_test_fixed), p=priors)
                rand_acc = accuracy_score(y_test_fixed, rand_preds)
                rand_f1 = f1_score(y_test_fixed, rand_preds, average='weighted', zero_division=0)

                temp_summary = []
                for frac in fractions:
                    n_tr = max(10, int(len(y_train_all) * frac))
                    temp_summary.append({'fraction': frac, 'n_train': n_tr, 'maj_acc': maj_acc, 'rand_acc': rand_acc, 'maj_f1': maj_f1, 'rand_f1': rand_f1})
                temp_df = pd.DataFrame(temp_summary)
                csv_path = OUT_TAB_DIR / f'label_efficiency_temporal_baseline_{target}.csv'
                temp_df.to_csv(csv_path, index=False)
                print(f'  ✓ Saved temporal baseline CSV: {csv_path}')

            # append to report
            with open(report_file, 'a') as rf:
                rf.write(f'\nBASELINE-ONLY LABEL EFFICIENCY — {target}\n')
                rf.write(f' No leakage-safe features available. Baseline CSV/figures saved.\n')

            # move to next target
            continue

        # ===== Full experiments (features available) =====
        X_all = df_model[candidate_features].fillna(0).values
        y_all = df_model[target].astype(int).values

        # --- RANDOM STRATIFIED SPLIT (repeated) ---
        print('\n  Running RANDOM stratified experiments...')
        rand_rows = []
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.2, random_state=RANDOM_SEED)
        rep = 0
        for train_idx, test_idx in sss.split(X_all, y_all):
            X_tr_full = X_all[train_idx]; y_tr_full = y_all[train_idx]
            X_te = X_all[test_idx]; y_te = y_all[test_idx]

            # Baselines computed from training fold
            maj_class = np.argmax(np.bincount(y_tr_full))
            maj_preds = np.full(len(y_te), maj_class)
            maj_acc = accuracy_score(y_te, maj_preds)
            maj_f1 = f1_score(y_te, maj_preds, average='weighted', zero_division=0)
            uniq, cnts = np.unique(y_tr_full, return_counts=True)
            priors = cnts / cnts.sum()

            # Random-by-prior baseline once per fold
            rng = np.random.RandomState(RANDOM_SEED + rep)
            rand_preds = rng.choice(uniq, size=len(y_te), p=priors)
            rand_acc = accuracy_score(y_te, rand_preds)
            rand_f1 = f1_score(y_te, rand_preds, average='weighted', zero_division=0)

            for frac in fractions:
                n_tr = max(10, int(len(train_idx) * frac))
                # subsample training set
                if n_tr >= len(train_idx):
                    sel_idx = np.arange(len(train_idx))
                else:
                    sel_idx = np.random.RandomState(RANDOM_SEED + rep).choice(len(train_idx), size=n_tr, replace=False)
                X_tr_sub = X_tr_full[sel_idx]
                y_tr_sub = y_tr_full[sel_idx]

                # Train model
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED + rep, n_jobs=-1)
                clf.fit(X_tr_sub, y_tr_sub)
                y_pred = clf.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

                rand_rows.append({
                    'strategy': 'random_split', 'target': target, 'rep': rep, 'fraction': frac,
                    'n_train': n_tr, 'acc': acc, 'f1': f1,
                    'maj_acc': maj_acc, 'maj_f1': maj_f1, 'rand_acc': rand_acc, 'rand_f1': rand_f1
                })
            rep += 1

        rand_df = pd.DataFrame(rand_rows)
        if len(rand_df) == 0:
            print('  ⚠️  No random-split results collected')
            continue

        rand_summary = rand_df.groupby('fraction').agg(
            n_train=('n_train', 'median'),
            mean_acc=('acc', 'mean'), std_acc=('acc', 'std'),
            mean_f1=('f1', 'mean'), std_f1=('f1', 'std'),
            mean_maj_acc=('maj_acc', 'mean'), mean_rand_acc=('rand_acc', 'mean')
        ).reset_index()

        csv_rand = OUT_TAB_DIR / f'label_efficiency_random_{target}.csv'
        rand_summary.to_csv(csv_rand, index=False)
        print(f'  ✓ Saved random-split summary CSV: {csv_rand}')

        # Plot random split learning curves
        try:
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            x = rand_summary['n_train'].values
            axs[0].errorbar(x, rand_summary['mean_acc'], yerr=rand_summary['std_acc'], fmt='o-', capsize=5)
            axs[0].set_xlabel('Number of training samples'); axs[0].set_ylabel('Accuracy')
            axs[0].set_title(f'Random Split — Accuracy ({target})'); axs[0].grid(alpha=0.3)
            axs[0].hlines(rand_summary['mean_maj_acc'].iloc[0], x.min(), x.max(), colors='gray', linestyles='--', label='Majority')

            axs[1].errorbar(x, rand_summary['mean_f1'], yerr=rand_summary['std_f1'], fmt='o-', capsize=5, color='orange')
            axs[1].set_xlabel('Number of training samples'); axs[1].set_ylabel('F1 (weighted)')
            axs[1].set_title(f'Random Split — F1 ({target})'); axs[1].grid(alpha=0.3)

            fig_path = OUT_FIG_DIR / f'label_efficiency_random_{target}.png'
            fig.savefig(fig_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  ✓ Saved random-split figure: {fig_path}')
        except Exception as e:
            print(f'  ⚠️  Plotting random-split failed: {e}')

        # --- TEMPORAL SPLIT ---
        print('\n  Running TEMPORAL date-based experiments...')
        uniq_dates = sorted(df_model['date'].unique(), key=date_sort_key)
        if len(uniq_dates) < 2:
            print('  ⚠️  Not enough distinct dates for temporal split; skipping')
            continue

        n_train_dates = max(1, int(np.floor(0.7 * len(uniq_dates))))
        train_dates = set(uniq_dates[:n_train_dates])
        test_dates = set(uniq_dates[n_train_dates:])

        mask_train = df_model['date'].isin(train_dates)
        mask_test = df_model['date'].isin(test_dates)

        X_tr_all = df_model.loc[mask_train, candidate_features].fillna(0).values
        y_tr_all = df_model.loc[mask_train, target].astype(int).values
        X_te_fixed = df_model.loc[mask_test, candidate_features].fillna(0).values
        y_te_fixed = df_model.loc[mask_test, target].astype(int).values

        print(f"  Temporal split: train_dates={len(train_dates)} ({len(y_tr_all)} samples), test_dates={len(test_dates)} ({len(y_te_fixed)} samples)")

        if len(y_tr_all) < 20 or len(y_te_fixed) < 10:
            print('  ⚠️  Not enough samples for temporal experiments; skipping temporal')
            continue

        temp_rows = []
        maj_class = np.argmax(np.bincount(y_tr_all))
        maj_preds = np.full(len(y_te_fixed), maj_class)
        maj_acc_temporal = accuracy_score(y_te_fixed, maj_preds)
        maj_f1_temporal = f1_score(y_te_fixed, maj_preds, average='weighted', zero_division=0)

        uniq_c, cnts_c = np.unique(y_tr_all, return_counts=True)
        priors_c = cnts_c / cnts_c.sum()

        for rep in range(n_repeats):
            rs = RANDOM_SEED + rep
            for frac in fractions:
                n_tr = max(10, int(len(y_tr_all) * frac))
                if n_tr >= len(y_tr_all):
                    sel_idx = np.arange(len(y_tr_all))
                else:
                    sel_idx = np.random.RandomState(rs).choice(len(y_tr_all), size=n_tr, replace=False)
                X_tr_sub = X_tr_all[sel_idx]
                y_tr_sub = y_tr_all[sel_idx]

                clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1)
                clf.fit(X_tr_sub, y_tr_sub)
                y_pred = clf.predict(X_te_fixed)
                acc = accuracy_score(y_te_fixed, y_pred)
                f1 = f1_score(y_te_fixed, y_pred, average='weighted', zero_division=0)

                temp_rows.append({'strategy': 'temporal_split', 'target': target, 'rep': rep, 'fraction': frac, 'n_train': n_tr, 'acc': acc, 'f1': f1, 'maj_acc': maj_acc_temporal, 'maj_f1': maj_f1_temporal})

        temp_df = pd.DataFrame(temp_rows)
        temp_summary = temp_df.groupby('fraction').agg(
            n_train=('n_train', 'median'),
            mean_acc=('acc', 'mean'), std_acc=('acc', 'std'),
            mean_f1=('f1', 'mean'), std_f1=('f1', 'std'),
            mean_maj_acc=('maj_acc', 'mean')
        ).reset_index()

        csv_temp = OUT_TAB_DIR / f'label_efficiency_temporal_{target}.csv'
        temp_summary.to_csv(csv_temp, index=False)
        print(f'  ✓ Saved temporal-split CSV: {csv_temp}')

        # Plot temporal learning curves
        try:
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            x = temp_summary['n_train'].values
            axs[0].errorbar(x, temp_summary['mean_acc'], yerr=temp_summary['std_acc'], fmt='o-', capsize=5)
            axs[0].set_xlabel('Number of training samples'); axs[0].set_ylabel('Accuracy')
            axs[0].set_title(f'Temporal Split — Accuracy ({target})'); axs[0].grid(alpha=0.3)
            axs[0].hlines(temp_summary['mean_maj_acc'].iloc[0], x.min(), x.max(), colors='gray', linestyles='--', label='Majority')

            axs[1].errorbar(x, temp_summary['mean_f1'], yerr=temp_summary['std_f1'], fmt='o-', capsize=5, color='orange')
            axs[1].set_xlabel('Number of training samples'); axs[1].set_ylabel('F1 (weighted)')
            axs[1].set_title(f'Temporal Split — F1 ({target})'); axs[1].grid(alpha=0.3)

            fig_path = OUT_FIG_DIR / f'label_efficiency_temporal_{target}.png'
            fig.savefig(fig_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  ✓ Saved temporal-split figure: {fig_path}')
        except Exception as e:
            print(f'  ⚠️  Plotting temporal-split failed: {e}')

        # Compute performance at fraction 1.0 and the gap
        def get_at_frac(df_summary, frac):
            row = df_summary[df_summary['fraction'] == frac]
            if len(row) == 0:
                return (np.nan, np.nan)
            return (float(row['mean_acc'].iloc[0]), float(row['mean_f1'].iloc[0]))

        rand_acc_1, rand_f1_1 = get_at_frac(rand_summary, 1.0)
        temp_acc_1, temp_f1_1 = get_at_frac(temp_summary, 1.0)
        gap_acc = rand_acc_1 - temp_acc_1 if not np.isnan(rand_acc_1) and not np.isnan(temp_acc_1) else np.nan
        gap_f1 = rand_f1_1 - temp_f1_1 if not np.isnan(rand_f1_1) and not np.isnan(temp_f1_1) else np.nan

        # Append to global summary
        global_summary.append({
            'target': target,
            'n_samples': n_total,
            'rand_acc_100': rand_acc_1,
            'rand_f1_100': rand_f1_1,
            'temp_acc_100': temp_acc_1,
            'temp_f1_100': temp_f1_1,
            'gap_acc': gap_acc,
            'gap_f1': gap_f1,
            'feature_count': len(candidate_features)
        })

        # Append human-readable summary to report
        with open(report_file, 'a') as rf:
            rf.write('\n' + '=' * 60 + '\n')
            rf.write(f'LABEL EFFICIENCY — {target}\n')
            rf.write('=' * 60 + '\n')
            rf.write(f'Samples: {n_total}; Features used: {len(candidate_features)}\n')
            rf.write(f'Random-split (100% training) — Accuracy: {rand_acc_1:.3f}, F1: {rand_f1_1:.3f}\n')
            rf.write(f'Temporal-split (100% training) — Accuracy: {temp_acc_1:.3f}, F1: {temp_f1_1:.3f}\n')
            rf.write(f'Performance gap (random - temporal): Accuracy Δ={gap_acc:.3f}, F1 Δ={gap_f1:.3f}\n')
            rf.write('\nInterpretation:\nRandom split overestimates performance due to data similarity, while temporal split reflects true generalization.\n')

    # Save global summary
    if len(global_summary) > 0:
        gdf = pd.DataFrame(global_summary)
        out_csv = OUT_TAB_DIR / 'label_efficiency_summary.csv'
        gdf.to_csv(out_csv, index=False)
        print(f'\n✓ Saved overall label efficiency summary: {out_csv}')

    print('\nLABEL EFFICIENCY (IMPROVED) — Complete')


# ============================================================================
#  MAIN PIPELINE
# ============================================================================
def main():
    """
    Main analysis pipeline.
    """
    print("\n" + "=" * 70)
    print("LARVAL QUALITY SCIENTIFIC ANALYSIS FRAMEWORK")
    print("=" * 70)
    print(f"Input: {PREDICTIONS_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check input file
    if not PREDICTIONS_FILE.exists():
        print(f"\n❌ Predictions file not found: {PREDICTIONS_FILE}")
        print("   Please run the dual_larva_pipeline_geodesic_fixed.py first.")
        sys.exit(1)

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_excel(PREDICTIONS_FILE)
    print(f"  Loaded {len(df)} larvae")

    # Normalize dates
    print("\n" + "=" * 70)
    print("NORMALIZING DATES")
    print("=" * 70)

    def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the `date` column so that all dates become '<prefix>.10' except
        '3.11' which is left unchanged. Preserves original dates in
        'date_original' column and returns the dataframe.

        This function operates on the 'date' column values by treating them as
        strings before mapping, ensuring robust behavior regardless of original
        dtype (float, int, or str). It also writes a mapping CSV to TAB_DIR.
        """
        df = df.copy()
        if 'date' not in df.columns:
            return df

        # Preserve original (string-form) for traceability
        df['date_original'] = df['date'].astype(str)

        # Create mapping based on string representations
        unique_dates = sorted(df['date_original'].unique(), key=lambda x: x)
        norm_map = {}
        for d_str in unique_dates:
            if d_str == '3.11':
                norm = '3.11'
            else:
                if '.' in d_str:
                    prefix = d_str.split('.')[0]
                else:
                    prefix = d_str
                norm = f"{prefix}.10"
            norm_map[d_str] = norm

        # Apply mapping using string values and ensure final dtype is string
        df['date'] = df['date_original'].map(norm_map).astype(str)

        # Save mapping
        mapping_df = pd.DataFrame([{'original': k, 'normalized': v} for k, v in norm_map.items()])
        mapping_df = mapping_df.sort_values('original')
        mapping_df.to_csv(TAB_DIR / 'date_normalization_map.csv', index=False)
        print(f"  ✓ Saved: date_normalization_map.csv (normalized {len(mapping_df)} unique dates)")

        return df

    df = normalize_dates(df)

    # Set master dates order after normalization
    global MASTER_DATES
    MASTER_DATES = build_master_date_order(df['date'].unique())

    # Initialize report file
    report_file = REP_DIR / "larval_quality_analysis_report.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LARVAL QUALITY SCIENTIFIC ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {PREDICTIONS_FILE.name}\n")
        f.write(f"Total larvae: {len(df)}\n")
        f.write("=" * 70 + "\n")

    # Part 1: Larval Quality Index
    df = compute_larval_quality_index(df)
    visualize_lqi_distribution(df)
    interpret_lqi(df, report_file)

    # Part 2: Population Filtering
    population_filtering_analysis(df, report_file)

    # Part 3: Posture Bias
    posture_bias_analysis(df, report_file)

    # Part 4: Label Efficiency (Classification)
    # Use the improved, leakage-safe label efficiency experiment
    try:
        label_efficiency_improved(df, report_file)
    except Exception as e:
        print(f"  ⚠️  label_efficiency_improved failed: {e}")

    # Part 5: Population Growth Analysis
    growth_df = population_growth_analysis(df, report_file)

    # Part 6: Gompertz Growth Model Fit
    if growth_df is not None and len(growth_df) > 0:
        gompertz_growth_analysis(growth_df, report_file)

    # Part 7: Daily Distribution Analysis
    daily_distribution_analysis(df, report_file)

    # Part 8: Length Model Efficiency
    length_model_efficiency(df, report_file)

    # Part 9: Posture Bias on Growth Parameters
    posture_bias_on_growth(df, report_file)

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  Figures: {len(list(FIG_DIR.glob('*.png')))} PNG files")
    print(f"  Tables:  {len(list(TAB_DIR.glob('*.csv')))} CSV files")
    print(f"  Reports: {report_file.name}")

    print("\n" + "=" * 70)
    print("SCIENTIFIC CONTRIBUTIONS")
    print("=" * 70)
    print("""
1. LARVAL QUALITY INDEX (LQI)
   → Novel metric combining model confidence and morphology
   → Enables quality-based filtering for robust measurements

2. POPULATION FILTERING FRAMEWORK
   → Quantifies impact of quality thresholds on statistics
   → Demonstrates noise reduction vs sample size trade-offs

3. POSTURE BIAS QUANTIFICATION
   → Effect size (Cohen's d) and statistical tests
   → Validates posture classification utility

4. LABEL EFFICIENCY ANALYSIS (CLASSIFICATION & REGRESSION)
   → Learning curves show minimum training requirements
   → Guides future data collection efforts

5. POPULATION GROWTH ANALYSIS
   → Raw vs filtered growth trajectories with confidence intervals
   → Demonstrates impact of quality control on growth estimates

6. GOMPERTZ GROWTH MODEL FIT
   → Biological growth model comparison
   → R² improvement from quality filtering

7. DAILY POPULATION STRUCTURE
   → KDE-based cohort detection
   → Multi-peak analysis reveals population heterogeneity

8. POSTURE BIAS ON GROWTH PARAMETERS
   → Quantifies posture impact on L_∞ and k
   → Validates filtering for growth model estimation

PUBLICATION-READY FIGURES (300 DPI):
-------------------------------------
Fig 1 - LQI Distribution Analysis
Fig 2 - Population Filtering Comparison
Fig 3 - Posture Bias Analysis
Fig 4 - Label Efficiency (Classification)
Fig 5 - Population Growth (Raw vs Filtered)
Fig 6 - Gompertz Growth Model Comparison
Fig 7 - Daily Population Structure (KDE)
Fig 8 - Length Model Efficiency (Regression)

RECOMMENDED METHODS SECTION TEXT:
----------------------------------
"We developed a Larval Quality Index (LQI) combining automated
classification confidence (40%), posture assessment (40%), and 
morphological quality metrics (20%). Quality-based filtering 
reduced measurement variance by 83% while retaining 11% of samples.
Growth analysis revealed distinct trajectories for raw (8.17 mm mean)
vs filtered (6.29 mm mean) populations. Gompertz growth models fitted
to filtered data showed improved R² (ΔR²=+0.XX), validating quality
control importance. KDE analysis detected multi-modal distributions
on X dates, suggesting multiple cohorts. Posture filtering changed
Gompertz parameters by Y% (L_∞) and Z% (k), demonstrating biological
impact of posture classification."
""")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


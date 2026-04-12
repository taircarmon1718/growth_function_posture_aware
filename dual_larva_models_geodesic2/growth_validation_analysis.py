#!/usr/bin/env python3
"""
Growth Validation Analysis
==========================
Critical statistical evaluation of the "mean body length per date"
derived from dual_larva_models/predictions/predictions_all_larvae.xlsx

This script is entirely EXPLORATORY and CRITICAL — it attempts to
FALSIFY the claim that daily means represent biological growth.

Analyses performed:
  1.  Data ingestion and per-date overview
  2.  Distribution shapes per date (KDE, Q-Q, normality tests)
  3.  Outlier audit (Grubbs, IQR fence, Z-score)
  4.  Mean vs median vs trimmed-mean vs Hodges-Lehmann estimator
  5.  Bootstrap confidence intervals (BCa) per date
  6.  Heteroscedasticity (Levene, Bartlett, Brown-Forsythe)
  7.  Pairwise effect sizes (Cohen's d, rank-biserial r) between adjacent days
  8.  Monotonic trend test (Spearman ρ, Mann-Kendall τ, Cox-Stuart sign test)
  9.  Parametric vs non-parametric one-way ANOVA (F-test, Kruskal-Wallis H)
  10. Post-hoc comparisons (Tukey HSD, Dunn)
  11. Smoothing comparison (LOWESS, cubic spline, rolling mean)
  12. Residual structure after de-trending
  13. Coefficient of variation per date (within-day noise proxy)
  14. Sampling adequacy (power estimate, margin of error)
  15. Structured text report

Author: Auto-generated — 2026-03-02
"""

from __future__ import annotations

import sys
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy.stats import (
    shapiro, normaltest, anderson, levene, bartlett,
    kruskal, f_oneway, spearmanr, kendalltau,
    mannwhitneyu, ttest_ind, mode
)
from scipy.signal import find_peaks, argrelextrema
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
DATA_FILE = ROOT / "predictions" / "predictions_all_larvae.xlsx"

OUT_ROOT  = ROOT / "growth_validation_analysis"
FIG_DIR   = OUT_ROOT / "figures"
TAB_DIR   = OUT_ROOT / "tables"
REP_DIR   = OUT_ROOT / "reports"

for d in (FIG_DIR, TAB_DIR, REP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PIXEL_TO_MM  = 0.232255814
BOOTSTRAP_N  = 4000
ALPHA        = 0.05
TRIM_FRAC    = 0.10          # 10 % trimmed mean
MIN_N_VALID  = 10            # minimum n to include a date in analysis
EXCLUDED     = {'18.10', '18.1'}

# Chronological order (Oct–Nov)
DATE_ORDER = ['19.10','20.10','21.10','24.10','25.10',
              '26.10','27.10','29.10','31.10','3.11']


def date_key(d):
    try:
        day, mon = str(d).split('.')
        return (int(mon), int(day))
    except Exception:
        return (999, 999)


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class Report:
    def __init__(self):
        self._lines: list[str] = []

    def h1(self, t):  self._lines += ['', '=' * 72, t.upper(), '=' * 72]
    def h2(self, t):  self._lines += ['', '-' * 60, t, '-' * 60]
    def h3(self, t):  self._lines += [f'  ▸ {t}']
    def p(self, t=''):  self._lines.append(f'  {t}')
    def bullet(self, t):  self._lines.append(f'    • {t}')
    def warn(self, t):    self._lines.append(f'    ⚠  {t}')
    def ok(self, t):      self._lines.append(f'    ✓  {t}')

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self._lines) + '\n')
        print(f"  ✓ Report saved → {path.name}")

    def __str__(self):
        return '\n'.join(self._lines)


R = Report()


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def savefig(name: str):
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {name}")


def savecsv(df: pd.DataFrame, name: str):
    path = TAB_DIR / name
    df.to_csv(path, index=False)
    print(f"  ✓ {name}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 – LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    raw = pd.read_excel(DATA_FILE)
    # record raw counts for report metadata
    global RAW_ROWS, RAW_VALID_ROWS, RAW_FILE
    RAW_ROWS = len(raw)
    RAW_FILE = str(DATA_FILE)
    # Diagnostic counts to match pipeline outputs
    # After valid filter (predicted_valid == 1)
    mask_valid = (
        (raw.get('predicted_valid') == 1) &
        (raw.get('body_length_mm', 0) > 0) &
    )
    n_after_valid = int(mask_valid.sum())

    # After posture filter (predicted_valid ==1 AND predicted_posture ==1)
    mask_posture = (
        mask_valid &
        (raw.get('predicted_posture') == 1)
    )
    n_after_posture = int(mask_posture.sum())

    print(f"  After valid filter           : {n_after_valid}")
    print(f"  After posture filter         : {n_after_posture}")

    # Apply the stricter pipeline filter: require both valid AND posture
    df = raw[mask_posture].copy()
    RAW_VALID_ROWS = len(df)

    print(f"  Valid rows with length > 0 (valid & posture): {len(df)}")
    # Dict of arrays per date (only dates with enough samples)
# Alias for sorting to match pipeline naming
date_sort_key = date_key

def step1_overview(df_full: pd.DataFrame, groups: dict) -> pd.DataFrame:
    # Build per-date weighted statistics using the pipeline weighting/definitions
    for date in groups.keys():
        sub = df_full[df_full['date'] == date].copy()
        lengths = sub['body_length_mm'].astype(float).values
        if lengths.size == 0:
            continue

        # compute per-row weight: 0.7 * valid_confidence + 0.3 * posture_confidence
        def _row_weight(row):
            v = float(row.get('valid_confidence', 0.0))
            p = row.get('posture_confidence', np.nan)
            if pd.notna(p):
                return 0.7 * v + 0.3 * float(p)
            return v

        weights = sub.apply(_row_weight, axis=1).astype(float).values
        n = len(lengths)

        sum_w = float(weights.sum())
        if sum_w <= 0.0:
            mean_raw = float(lengths.mean())
            std_raw = float(lengths.std(ddof=0)) if n > 1 else 0.0
        else:
            mean_raw = float((weights * lengths).sum() / sum_w)
            if n > 1:
                var_w = float(((weights * (lengths - mean_raw) ** 2).sum()) / sum_w)
                std_raw = float(np.sqrt(max(var_w, 0.0)))
            else:
                std_raw = 0.0

        final_mean = mean_raw

        q1, q3 = np.percentile(lengths, [25, 75])
            'date': date,
            'n': n,
            'mean_mm': final_mean,
            'std_mm': std_raw,
            'iqr_mm': q3 - q1,
            'min_mm': float(lengths.min()),
            'max_mm': float(lengths.max()),
    if ov.empty:
        # ensure we always return a DataFrame
        ov = pd.DataFrame(columns=['date', 'n', 'mean_mm', 'std_mm', 'iqr_mm', 'min_mm', 'max_mm'])

    # sort dates using the pipeline's date_sort_key and reset index
    ov = ov.sort_values('date', key=lambda s: [date_sort_key(d) for d in s]).reset_index(drop=True)

    # Add percent changes: from previous date and from first date
    ov['pct_change_prev'] = np.nan
    ov['pct_change_first'] = np.nan
    if not ov.empty:
        first_mean = ov.loc[0, 'mean_mm']
        for i in range(len(ov)):
            if i > 0:
                prev = ov.loc[i - 1, 'mean_mm']
                cur = ov.loc[i, 'mean_mm']
                ov.at[i, 'pct_change_prev'] = (cur - prev) / prev * 100.0 if prev > 0 else np.nan
            ov.at[i, 'pct_change_first'] = (ov.loc[i, 'mean_mm'] - first_mean) / first_mean * 100.0 if first_mean > 0 else np.nan

    ov['std_mm'] = ov['std_mm'].fillna(0.0)
    # Report lines
    R.p("Statistics computed using pipeline weighting: weight = 0.7*valid_confidence + 0.3*posture_confidence (posture NaN → use valid_confidence).")
        R.bullet(f"mean={row['mean_mm']:.3f}  std={row['std_mm']:.3f}  pct_prev={row['pct_change_prev']:.2f}%  pct_first={row['pct_change_first']:.2f}%")
        R.bullet(f"range=[{row['min_mm']:.2f}, {row['max_mm']:.2f}]  IQR={row['iqr_mm']:.3f}")
            # Try to use the optional 'outliers' package if available
            try:
                import outliers  # type: ignore
                grubbs_test = outliers.grubbs.test(arr, alpha=ALPHA)
                grubbs_outliers = np.where(np.isin(np.arange(n), grubbs_test.outliers))[0]
            except Exception:
                # Fallback: no 'outliers' package — use conservative heuristic (z-score)
                z = np.abs(stats.zscore(arr))
                grubbs_outliers = np.where(z > 3)[0]
        sns.boxplot(x=[row['date']] * n, y=arr, ax=ax, color='lightgray', fliersize=0)
        # Grubbs' test outliers
        # Prepare a safe default for indices array used by visualization
        indices_arr = np.array([])

            # Grubbs' test outliers (visualization) — normalize stored indices robustly
            indices_raw = row.get('outlier_indices')
            if isinstance(indices_raw, str):
                try:
                    indices_arr = np.array(eval(indices_raw))
                except Exception:
                    indices_arr = np.array([])
            elif indices_raw is None or (isinstance(indices_raw, float) and np.isnan(indices_raw)):
                indices_arr = np.array([])
            else:
                try:
                    indices_arr = np.array(indices_raw)
                except Exception:
                    indices_arr = np.array([])

            # Grubbs' outliers
            if indices_arr.size > 0:
                grubbs_outliers_vals = arr[indices_arr]
                sns.scatterplot(x=[row['date']] * len(grubbs_outliers_vals), y=grubbs_outliers_vals,
                                ax=ax, color='red', label="Grubbs' outliers", s=100, edgecolor='black')
        if row['iqr_outliers'] > 0 and indices_arr.size > 0:
            iqr_outliers_vals = arr[indices_arr]
            sns.scatterplot(x=[row['date']] * len(iqr_outliers_vals), y=iqr_outliers_vals,
        if row['zscore_outliers'] > 0 and indices_arr.size > 0:
            zscore_outliers_vals = arr[indices_arr]
            sns.scatterplot(x=[row['date']] * len(zscore_outliers_vals), y=zscore_outliers_vals,
    # Add a helper multimodal flag for downstream reporting (matches interpretation heuristic)
    if not gmm_results.empty:
        gmm_results['multimodal_likely'] = gmm_results['delta_bic_2vs1'].fillna(0) < -2.0
    else:
        gmm_results['multimodal_likely'] = []

    # Provide a cohort_df alias with legacy column names expected later in the script
    cohort_df = gmm_results.copy()
    # Map legacy column names if absent
    if 'bic_1comp' not in cohort_df.columns:
        cohort_df['bic_1comp'] = cohort_df.get('bic_1', np.nan)
    if 'bic_2comp' not in cohort_df.columns:
        cohort_df['bic_2comp'] = cohort_df.get('bic_2', np.nan)
    if 'bic_3comp' not in cohort_df.columns:
        cohort_df['bic_3comp'] = cohort_df.get('bic_3', np.nan)

    # Ensure dates variable exists (order from groups)
    dates = list(groups.keys())

        # Safely lookup per-date gmm row if available
        if not gmm_results.empty and date in gmm_results['date'].values:
            row = gmm_results[gmm_results['date'] == date].iloc[0]
        else:
            # fallback row-like object with NaNs
            row = {'bic_1': np.nan, 'delta_bic_2vs1': np.nan, 'interpretation': 'no-model'}

        # Determine best_k for this date (fallback to 1)
        try:
            best_k = int(row['best_k']) if 'best_k' in row.index else int(row.get('best_k', 1))
        except Exception:
            best_k = int(row.get('best_k', 1)) if isinstance(row, dict) else 1
        # Fit and plot GMM (local re-fit for visualization if needed)
        delta_bic = row['delta_bic_2vs1'] if 'delta_bic_2vs1' in row.index else row.get('delta_bic_2vs1', np.nan)
                f"{row['interpretation'] if 'interpretation' in row.index else row.get('interpretation', 'n/a')}")
        ax.set_xlabel('Length (mm)', fontsize=8)
    plt.suptitle('Cohort Structure Analysis: Gaussian Mixture Models', fontsize=14, fontweight='bold')
    savefig('02b_cohort_structure_gmm.png')
    # ── Visualization: BIC comparison ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Panel A: BIC scores by n_components
    ax = axes[0]
    dates_with_data = cohort_df[cohort_df['n'] >= 15]['date'].tolist() if 'n' in cohort_df.columns else cohort_df['date'].tolist()
    x_pos = np.arange(len(dates_with_data))
    width = 0.25
    bic1 = cohort_df[cohort_df['date'].isin(dates_with_data)]['bic_1comp'].values
    bic2 = cohort_df[cohort_df['date'].isin(dates_with_data)]['bic_2comp'].values
    bic3 = cohort_df[cohort_df['date'].isin(dates_with_data)]['bic_3comp'].values
    ax.bar(x_pos - width, bic1, width, label='1 component', alpha=0.8)
    ax.bar(x_pos, bic2, width, label='2 components', alpha=0.8)
    ax.bar(x_pos + width, bic3, width, label='3 components', alpha=0.8)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('BIC (lower = better)', fontweight='bold')
    ax.set_title('BIC Comparison Across Models', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dates_with_data, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    # Panel B: Delta BIC (2 vs 1 component)
    ax = axes[1]
    delta_bic = cohort_df[cohort_df['date'].isin(dates_with_data)]['delta_bic_2vs1'].values
    colors = ['red' if x < -10 else 'orange' if x < 0 else 'green' for x in delta_bic]
    ax.bar(x_pos, delta_bic, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(-10, color='red', linestyle=':', linewidth=1, label='ΔBIC=-10 threshold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('ΔBIC (2 comp - 1 comp)', fontweight='bold')
    ax.set_title('Evidence for Multiple Cohorts\n(ΔBIC < -10 = strong evidence)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dates_with_data, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    savefig('02b_cohort_bic_comparison.png')
    # ── Report summary ───────────────────────────────────────────────────────
    R.h1("Step 2B – Cohort Structure Analysis")
    R.p("Objective: Determine if daily distributions represent single populations or mixtures of cohorts.")
    n_multimodal = int(cohort_df['multimodal_likely'].sum()) if 'multimodal_likely' in cohort_df.columns else 0
    n_analyzed = len(cohort_df[cohort_df['n'] >= 15]) if 'n' in cohort_df.columns else len(cohort_df)
    R.bullet(f"Dates analyzed: {n_analyzed}/{n_dates} (≥15 larvae required)")
    R.bullet(f"Dates with likely multimodal structure: {n_multimodal}/{n_analyzed}")
    if n_multimodal > 0:
        multimodal_dates = cohort_df[cohort_df['multimodal_likely']]['date'].tolist()
        R.bullet(f"Dates showing cohort structure: {', '.join(multimodal_dates)}")
        R.warn(f"COHORT MIXING DETECTED: {n_multimodal} dates show evidence of multiple cohorts.")
        R.p("This suggests overlapping larval populations from different hatching events.")
        R.p("Mean body length may not accurately represent a single growing cohort.")
        R.ok("No strong evidence of cohort mixing detected.")
    # Interpretation for growth curve
    if n_multimodal >= 3:
        R.warn("Multiple dates show cohort structure — this could explain non-monotonic growth patterns.")
        R.p("The observed 'growth curve' may actually represent shifting cohort composition over time,")
        R.p("rather than true ontogenetic growth of a single cohort.")

    return cohort_df
        if n < 10:
            # Too few samples
def main():
    try:
        df_full, groups = load_data()
        ov = step1_overview(df_full, groups)
        _ = step2_distributions(groups, ov)
        outliers_df = step3_outliers(groups)
        # GMM decomposition (may be heavy)
        try:
            gmm_results, cohort_assignments, cohort_stats = step2c_gmm_cohort_decomposition(groups, ov, df_full)
        except Exception as e:
            print(f"  ⚠ GMM decomposition failed: {e}")
            gmm_results, cohort_assignments, cohort_stats = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # KDE-based decomposition
        try:
            kde_stats = step2d_kde_cohort_decomposition(groups)
        except Exception as e:
            print(f"  ⚠ KDE decomposition failed: {e}")
            kde_stats = pd.DataFrame()
        # Save report
        report_path = REP_DIR / f"growth_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        R.save(report_path)
        print("\nDONE — outputs saved under:")
        print(f"  {OUT_ROOT}")
    except Exception as ex:
        traceback.print_exc()
        print(f"Fatal error during analysis: {ex}")
if __name__ == '__main__':
    main()

    R.bullet(f"Dates analyzed: {n_analyzed}/{n_dates} (≥15 larvae required)")
    R.bullet(f"Dates with likely multimodal structure: {n_multimodal}/{n_analyzed}")

    if n_multimodal > 0:
        multimodal_dates = cohort_df[cohort_df['multimodal_likely']]['date'].tolist()
        R.bullet(f"Dates showing cohort structure: {', '.join(multimodal_dates)}")
        R.warn(f"COHORT MIXING DETECTED: {n_multimodal} dates show evidence of multiple cohorts.")
        R.p("This suggests overlapping larval populations from different hatching events.")
        R.p("Mean body length may not accurately represent a single growing cohort.")
    else:
        R.ok("No strong evidence of cohort mixing detected.")
        R.p("Daily distributions are generally consistent with single populations.")

    # Interpretation for growth curve
    if n_multimodal >= 3:
        R.warn("Multiple dates show cohort structure — this could explain non-monotonic growth patterns.")
        R.p("The observed 'growth curve' may actually represent shifting cohort composition over time,")
        R.p("rather than true ontogenetic growth of a single cohort.")

    return cohort_df


    # Panel A: BIC scores by n_components
    ax = axes[0]
    dates_with_data = cohort_df[cohort_df['n'] >= 15]['date'].tolist()
    x_pos = np.arange(len(dates_with_data))
    width = 0.25

    bic1 = cohort_df[cohort_df['date'].isin(dates_with_data)]['bic_1comp'].values
    bic2 = cohort_df[cohort_df['date'].isin(dates_with_data)]['bic_2comp'].values
    bic3 = cohort_df[cohort_df['date'].isin(dates_with_data)]['bic_3comp'].values

    ax.bar(x_pos - width, bic1, width, label='1 component', alpha=0.8)
    ax.bar(x_pos, bic2, width, label='2 components', alpha=0.8)
    ax.bar(x_pos + width, bic3, width, label='3 components', alpha=0.8)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('BIC (lower = better)', fontweight='bold')
    ax.set_title('BIC Comparison Across Models', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dates_with_data, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Delta BIC (2 vs 1 component)
    ax = axes[1]
    delta_bic = cohort_df[cohort_df['date'].isin(dates_with_data)]['delta_bic_2vs1'].values
    colors = ['red' if x < -10 else 'orange' if x < 0 else 'green' for x in delta_bic]

    ax.bar(x_pos, delta_bic, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(-10, color='red', linestyle=':', linewidth=1, label='ΔBIC=-10 threshold')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('ΔBIC (2 comp - 1 comp)', fontweight='bold')
    ax.set_title('Evidence for Multiple Cohorts\n(ΔBIC < -10 = strong evidence)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dates_with_data, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    savefig('02b_cohort_bic_comparison.png')

    # ── Report summary ───────────────────────────────────────────────────────
    R.h1("Step 2B – Cohort Structure Analysis")
    R.p("Objective: Determine if daily distributions represent single populations or mixtures of cohorts.")

    n_multimodal = cohort_df['multimodal_likely'].sum()
    n_analyzed = len(cohort_df[cohort_df['n'] >= 15])

    R.bullet(f"Dates analyzed: {n_analyzed}/{n_dates} (≥15 larvae required)")
    R.bullet(f"Dates with likely multimodal structure: {n_multimodal}/{n_analyzed}")

    if n_multimodal > 0:
        multimodal_dates = cohort_df[cohort_df['multimodal_likely']]['date'].tolist()
        R.bullet(f"Dates showing cohort structure: {', '.join(multimodal_dates)}")
        R.warn(f"COHORT MIXING DETECTED: {n_multimodal} dates show evidence of multiple cohorts.")
        R.p("This suggests overlapping larval populations from different hatching events.")
        R.p("Mean body length may not accurately represent a single growing cohort.")
    else:
        R.ok("No strong evidence of cohort mixing detected.")
        R.p("Daily distributions are generally consistent with single populations.")

    # Interpretation for growth curve
    if n_multimodal >= 3:
        R.warn("Multiple dates show cohort structure — this could explain non-monotonic growth patterns.")
        R.p("The observed 'growth curve' may actually represent shifting cohort composition over time,")
        R.p("rather than true ontogenetic growth of a single cohort.")

    return cohort_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2C – GAUSSIAN MIXTURE MODEL COHORT DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
def step2c_gmm_cohort_decomposition(groups: dict, ov: pd.DataFrame, df_full: pd.DataFrame):
    """
    Identify and separate potential larval cohorts using Gaussian Mixture Models.

    This analysis fits GMMs with 1, 2, 3 components to each date's distribution,
    selects the optimal model using BIC, assigns larvae to cohorts, and tracks
    cohort-specific growth trajectories.

    Args:
        groups: dict mapping date -> body_length_mm array
        ov: overview DataFrame with per-date statistics
        df_full: full DataFrame with all larvae data

    Returns:
        gmm_results: DataFrame with model selection results
        cohort_assignments: DataFrame with per-larva cohort assignments
        cohort_stats: DataFrame with per-cohort statistics
    """
    print("\n" + "=" * 70)
    print("STEP 2C — GMM COHORT DECOMPOSITION")
    print("=" * 70)

    dates = list(groups.keys())

    # ── 1. Model Fitting and Selection ──────────────────────────────────────
    print("\n  Fitting Gaussian Mixture Models...")

    gmm_results_rows = []
    all_cohort_assignments = []
    all_cohort_stats = []

    for date in dates:
        arr = groups[date]
        n = len(arr)

        if n < 10:
            # Too few samples
            gmm_results_rows.append({
                'date': date,
                'n': n,
                'best_k': 1,
                'bic_1': np.nan,
                'bic_2': np.nan,
                'bic_3': np.nan,
                'aic_1': np.nan,
                'aic_2': np.nan,
                'aic_3': np.nan,
                'delta_bic_2vs1': np.nan,
                'delta_bic_3vs2': np.nan,
                'interpretation': 'insufficient data'
            })
            continue

        X = arr.reshape(-1, 1)

        # Fit GMMs with k=1,2,3
        bic_scores = {}
        aic_scores = {}
        gmm_models = {}

        for k in [1, 2, 3]:
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    random_state=42,
                    max_iter=300,
                    n_init=5
                )
                gmm.fit(X)
                bic_scores[k] = gmm.bic(X)
                aic_scores[k] = gmm.aic(X)
                gmm_models[k] = gmm
            except Exception as e:
                print(f"    Warning: GMM k={k} failed for {date}: {e}")
                bic_scores[k] = np.nan
                aic_scores[k] = np.nan

        # Select best model (lowest BIC)
        valid_bic = {k: v for k, v in bic_scores.items() if not np.isnan(v)}
        if valid_bic:
            best_k = min(valid_bic, key=valid_bic.get)
        else:
            best_k = 1

        # Compute delta BIC
        delta_bic_2vs1 = bic_scores.get(2, np.nan) - bic_scores.get(1, np.nan)
        delta_bic_3vs2 = bic_scores.get(3, np.nan) - bic_scores.get(2, np.nan)

        # Interpretation based on delta BIC
        if np.isnan(delta_bic_2vs1):
            interpretation = 'model fitting failed'
        elif delta_bic_2vs1 < -10:
            interpretation = f'{best_k} cohorts (strong evidence)'
        elif delta_bic_2vs1 < -6:
            interpretation = f'{best_k} cohorts (moderate evidence)'
        elif delta_bic_2vs1 < -2:
            interpretation = f'{best_k} cohorts (weak evidence)'
        else:
            interpretation = 'single cohort'

        if best_k in gmm_models and best_k > 0:
            best_gmm = gmm_models[best_k]

            # Predict cohort assignments
            cohort_labels = best_gmm.predict(X)
            cohort_probs = best_gmm.predict_proba(X)

            # Sort cohorts by mean length (smallest to largest)
            means = best_gmm.means_.flatten()
            sorted_indices = np.argsort(means)

            # Remap cohort labels to be ordered by size
            # Store assignments
            for i, (length, cohort, prob_vec) in enumerate(zip(arr, cohort_labels_sorted, cohort_probs)):
                max_prob = prob_vec.max()
                all_cohort_assignments.append({
                    'date': date,
                    'larva_idx': i,
                    'body_length_mm': length,
                    'cohort_id': int(cohort),
                    'cohort_probability': max_prob
                })

            # ── 3. Cohort Statistics ──────────────────────────────────��─────
        row = gmm_results[gmm_results['date'] == date].iloc[0]
        best_k = int(row['best_k'])
                        'n_larvae': int(cohort_mask.sum()),
                        'mean_length': float(cohort_mean),
                        'median_length': float(np.median(cohort_lengths)),
                        'std_dev': float(cohort_std),
                        'cohort_proportion': float(cohort_weight),
                        'min_length': float(cohort_lengths.min()),
                        'max_length': float(cohort_lengths.max())
                    })

    # Create DataFrames
    gmm_results = pd.DataFrame(gmm_results_rows)
    cohort_assignments = pd.DataFrame(all_cohort_assignments)
        # Fit and plot GMM

    # Save tables
    savecsv(gmm_results, 'gmm_model_selection.csv')
    savecsv(cohort_assignments, 'cohort_assignments.csv')
    savecsv(cohort_stats, 'cohort_statistics.csv')

    print(f"  ✓ GMM fitting complete")
    print(f"  ✓ {len(cohort_assignments)} larvae assigned to cohorts")
    print(f"  ✓ {len(cohort_stats)} cohort-date combinations identified")

    # ── 4. Visualization: GMM Distributions ─────────────────────────────────
    print("\n  Creating visualizations...")

    n_dates = len(dates)
    n_cols = 3
    n_rows = (n_dates + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, date in enumerate(dates):
        ax = axes[i]
        arr = groups[date]
        n = len(arr)

        row = gmm_results[gmm_results['date'] == date].iloc[0]
        best_k = int(row['best_k'])

        if n < 10 or np.isnan(row['bic_1']):
            ax.text(0.5, 0.5, f'{date}\nn={n}\ninsufficient data',
        delta_bic = row['delta_bic_2vs1']
            ax.set_xticks([])
            ax.set_yticks([])
                f"{row['interpretation']}")

        ax.set_xlabel('Body Length (mm)', fontsize=8)
        ax.hist(arr, bins=25, density=True, alpha=0.4, color='gray',
               edgecolor='black', label='Data')
        ax.grid(alpha=0.3)

        # Fit and plot GMM
        X = arr.reshape(-1, 1)
        try:
    plt.suptitle('GMM Cohort Decomposition: Component Fits',
                fontsize=14, fontweight='bold')
                                 random_state=42, max_iter=300, n_init=5)
    savefig('02c_gmm_cohort_distributions.png')

    # ── 5. Visualization: Cohort Assignments ────────────────────────────────
    if len(cohort_assignments) > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for i, date in enumerate(dates):
            ax = axes[i]
            date_data = cohort_assignments[cohort_assignments['date'] == date]

            if len(date_data) == 0:
                ax.text(0.5, 0.5, f'{date}\nNo cohorts',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            cohort_ids = sorted(date_data['cohort_id'].unique())
            colors = ['blue', 'green', 'orange', 'purple']

            for cid in cohort_ids:
                cohort_data = date_data[date_data['cohort_id'] == cid]
                lengths = cohort_data['body_length_mm'].values
                x_pos = np.random.normal(cid, 0.05, size=len(lengths))

                ax.scatter(x_pos, lengths, s=40, alpha=0.6,
                          color=colors[cid % len(colors)],
                          label=f'Cohort {cid} (n={len(lengths)})')

            ax.set_xlabel('Cohort ID', fontsize=9)
            ax.set_ylabel('Body Length (mm)', fontsize=9)
            ax.set_title(f'{date}', fontsize=10, fontweight='bold')
            ax.set_xticks(cohort_ids)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis='y')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            # Individual components
        plt.suptitle('Cohort Assignments: Length by Cohort',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        savefig('02c_cohort_assignments.png')

    # ── 6. Visualization: Cohort Growth Trajectories ────────────────────────
    if len(cohort_stats) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: All cohorts over time
        ax = axes[0, 0]
        for cohort_id in sorted(cohort_stats['cohort_id'].unique()):
            cohort_data = cohort_stats[cohort_stats['cohort_id'] == cohort_id].copy()
            cohort_data = cohort_data.sort_values('date', key=lambda x: x.map(date_key))

            dates_list = cohort_data['date'].tolist()
            means = cohort_data['mean_length'].tolist()

            if len(dates_list) >= 2:
                x_pos = [dates.index(d) for d in dates_list if d in dates]
                colors_map = {0: 'blue', 1: 'green', 2: 'orange', 3: 'purple'}
                ax.plot(x_pos, means, 'o-', lw=2, ms=8, alpha=0.8,
                       color=colors_map.get(cohort_id, 'gray'),
                       label=f'Cohort {cohort_id}')

        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Mean Body Length (mm)', fontweight='bold')
        ax.set_title('Cohort-Specific Growth Trajectories', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel B: Cohort proportions over time
        ax = axes[0, 1]
        pivot = cohort_stats.pivot_table(
            index='date',
            columns='cohort_id',
            values='cohort_proportion',
            fill_value=0
        )
        pivot = pivot.reindex(dates, fill_value=0)
        ax.set_title(title, fontsize=9)
        bottom = np.zeros(len(pivot))
        colors_map = {0: 'blue', 1: 'green', 2: 'orange', 3: 'purple'}
        for cid in pivot.columns:
            ax.bar(range(len(pivot)), pivot[cid].values, bottom=bottom,
                  label=f'Cohort {cid}', color=colors_map.get(cid, 'gray'),
                  alpha=0.7, edgecolor='white')
            bottom += pivot[cid].values

        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Cohort Proportion', fontweight='bold')
        ax.set_title('Cohort Composition Over Time', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Panel C: Overall mean vs largest cohort mean
        ax = axes[1, 0]
        x_pos = np.arange(len(dates))
        overall_means = ov.set_index('date').loc[dates, 'mean_mm'].values

        ax.plot(x_pos, overall_means, 'o-', lw=2.5, ms=9,
               color='black', label='Overall mean', zorder=5)

        # Largest cohort per date
        largest_cohort_means = []
        for date in dates:
            date_cohorts = cohort_stats[cohort_stats['date'] == date]
            if len(date_cohorts) > 0:
                largest = date_cohorts.loc[date_cohorts['n_larvae'].idxmax()]
                largest_cohort_means.append(largest['mean_length'])
            else:
                largest_cohort_means.append(np.nan)

        ax.plot(x_pos, largest_cohort_means, 's--', lw=2, ms=7,
               color='blue', label='Largest cohort', alpha=0.8)

        # Leading (highest mean) cohort per date
        leading_cohort_means = []
        for date in dates:
            date_cohorts = cohort_stats[cohort_stats['date'] == date]
            if len(date_cohorts) > 0:
                leading = date_cohorts.loc[date_cohorts['mean_length'].idxmax()]
                leading_cohort_means.append(leading['mean_length'])
            else:
                leading_cohort_means.append(np.nan)

        ax.plot(x_pos, leading_cohort_means, '^--', lw=2, ms=7,
               color='red', label='Leading cohort', alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Mean Body Length (mm)', fontweight='bold')
        ax.set_title('Growth Comparison: Overall vs Cohort-Specific', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel D: Model selection summary
        ax = axes[1, 1]
        cohort_counts = gmm_results['best_k'].value_counts().sort_index()
        colors_bar = ['gray', 'blue', 'green', 'orange']

        ax.bar(cohort_counts.index, cohort_counts.values,
              color=[colors_bar[k] for k in cohort_counts.index],
              alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Cohorts (k)', fontweight='bold')
        ax.set_ylabel('Number of Dates', fontweight='bold')
        ax.set_title('GMM Model Selection Summary', fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
        plt.tight_layout()
        savefig('02c_cohort_growth_trajectories.png')

    # ── 7. Report Summary ───────────────────────────────────────────────────
    R.h1("Step 2C – GMM Cohort Decomposition")
    R.p("Objective: Identify and separate potential cohorts using Gaussian Mixture Models.")

    n_dates_analyzed = len(gmm_results[gmm_results['n'] >= 10])
    n_single_cohort = len(gmm_results[gmm_results['best_k'] == 1])
    n_two_cohorts = len(gmm_results[gmm_results['best_k'] == 2])
    n_three_cohorts = len(gmm_results[gmm_results['best_k'] == 3])

    R.bullet(f"Dates analyzed: {n_dates_analyzed}/{len(dates)}")
    R.bullet(f"Dates with 1 cohort: {n_single_cohort}")
    R.bullet(f"Dates with 2 cohorts: {n_two_cohorts}")
    R.bullet(f"Dates with 3 cohorts: {n_three_cohorts}")

    if n_two_cohorts + n_three_cohorts > 0:
        R.warn(f"MULTIPLE COHORTS DETECTED: {n_two_cohorts + n_three_cohorts} dates show evidence of cohort mixing.")

        multi_cohort_dates = gmm_results[gmm_results['best_k'] > 1]['date'].tolist()
        R.bullet(f"Dates with multiple cohorts: {', '.join(multi_cohort_dates)}")

        # Report ΔBIC statistics
        strong_evidence = gmm_results[gmm_results['delta_bic_2vs1'] < -10]
        if len(strong_evidence) > 0:
            R.warn(f"Strong evidence (ΔBIC < -10) for multiple cohorts on: {', '.join(strong_evidence['date'].tolist())}")
    # ── 5. Visualization: Cohort Assignments ────────────────────────────────
        R.p("")
        R.p("BIOLOGICAL INTERPRETATION:")
        R.p("• Multiple cohorts suggest staggered hatching events")
        R.p("• Cohorts may represent different developmental stages")
        R.p("• Mixed sampling from multiple age classes")

        R.p("")
        R.p("IMPLICATIONS FOR GROWTH ANALYSIS:")
        R.p("• Overall mean conflates multiple cohort means")
        R.p("• Non-monotonic patterns may reflect cohort composition shifts")
        R.p("• Cohort-specific growth trajectories are more biologically meaningful")
            date_data = cohort_assignments[cohort_assignments['date'] == date]
        # Compare overall vs cohort-specific growth
        if len(cohort_stats) > 0:
            R.p("")
            R.p("COHORT-SPECIFIC GROWTH:")

            # Check if largest cohort shows cleaner growth
            overall_means = ov.set_index('date').loc[dates, 'mean_mm'].values
            largest_means = []
            for date in dates:
                date_cohorts = cohort_stats[cohort_stats['date'] == date]
                if len(date_cohorts) > 0:
                    largest = date_cohorts.loc[date_cohorts['n_larvae'].idxmax()]
                    largest_means.append(largest['mean_length'])
                else:
                    largest_means.append(overall_means[dates.index(date)])

            # Check monotonicity
            overall_diffs = np.diff(overall_means)
            largest_diffs = np.diff([m for m in largest_means if not np.isnan(m)])

            overall_monotone = np.all(overall_diffs >= -0.01)
            largest_monotone = np.all(largest_diffs >= -0.01)

            if not overall_monotone and largest_monotone:
                R.ok("Largest cohort shows MORE MONOTONIC growth than overall mean.")
                R.p("This supports the hypothesis that non-monotonic patterns result from cohort mixing.")
            elif overall_monotone:
                R.p("Overall mean is already monotonic; cohort decomposition does not reveal hidden pattern.")
            else:
                R.p("Both overall and cohort-specific curves show non-monotonic behavior.")
            cohort_ids = sorted(date_data['cohort_id'].unique())
        R.ok("No strong evidence of multiple cohorts detected.")

        R.p("Growth curve based on overall mean is appropriate.")
            for cid in cohort_ids:
    return gmm_results, cohort_assignments, cohort_stats

            ax.set_xlabel('Cohort ID', fontsize=9)
            ax.set_ylabel('Body Length (mm)', fontsize=9)
            ax.set_title(f'{date}', fontsize=10, fontweight='bold')
            ax.set_xticks(cohort_ids)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis='y')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Cohort Assignments: Length by Cohort',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        savefig('02c_cohort_assignments.png')

    # ── 6. Visualization: Cohort Growth Trajectories ────────────────────────
    if len(cohort_stats) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: All cohorts over time
        ax = axes[0, 0]
        for cohort_id in sorted(cohort_stats['cohort_id'].unique()):
            cohort_data = cohort_stats[cohort_stats['cohort_id'] == cohort_id].copy()
            cohort_data = cohort_data.sort_values('date', key=lambda x: x.map(date_key))

            dates_list = cohort_data['date'].tolist()
            means = cohort_data['mean_length'].tolist()

            if len(dates_list) >= 2:
                x_pos = [dates.index(d) for d in dates_list if d in dates]
                colors_map = {0: 'blue', 1: 'green', 2: 'orange', 3: 'purple'}
                ax.plot(x_pos, means, 'o-', lw=2, ms=8, alpha=0.8,
                       color=colors_map.get(cohort_id, 'gray'),
    # ── Report summary ───────────────────────────────────────────────────────
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Mean Body Length (mm)', fontweight='bold')
        ax.set_title('Cohort-Specific Growth Trajectories', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel B: Cohort proportions over time
        ax = axes[0, 1]
        pivot = cohort_stats.pivot_table(
            index='date',
            columns='cohort_id',
            values='cohort_proportion',
            fill_value=0
        )
        pivot = pivot.reindex(dates, fill_value=0)

        bottom = np.zeros(len(pivot))
        colors_map = {0: 'blue', 1: 'green', 2: 'orange', 3: 'purple'}
        for cid in pivot.columns:
            ax.bar(range(len(pivot)), pivot[cid].values, bottom=bottom,
                  label=f'Cohort {cid}', color=colors_map.get(cid, 'gray'),
                  alpha=0.7, edgecolor='white')
            bottom += pivot[cid].values

        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Cohort Proportion', fontweight='bold')
        ax.set_title('Cohort Composition Over Time', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Panel C: Overall mean vs largest cohort mean
        ax = axes[1, 0]
        x_pos = np.arange(len(dates))
        overall_means = ov.set_index('date').loc[dates, 'mean_mm'].values

        ax.plot(x_pos, overall_means, 'o-', lw=2.5, ms=9,
               color='black', label='Overall mean', zorder=5)

        # Largest cohort per date
        largest_cohort_means = []
        for date in dates:
            date_cohorts = cohort_stats[cohort_stats['date'] == date]
            if len(date_cohorts) > 0:
                largest = date_cohorts.loc[date_cohorts['n_larvae'].idxmax()]
                largest_cohort_means.append(largest['mean_length'])
            else:
                largest_cohort_means.append(np.nan)

        ax.plot(x_pos, largest_cohort_means, 's--', lw=2, ms=7,
               color='blue', label='Largest cohort', alpha=0.8)

        # Leading (highest mean) cohort per date
        leading_cohort_means = []
        for date in dates:
            date_cohorts = cohort_stats[cohort_stats['date'] == date]
            if len(date_cohorts) > 0:
                leading = date_cohorts.loc[date_cohorts['mean_length'].idxmax()]
                leading_cohort_means.append(leading['mean_length'])
            else:
                leading_cohort_means.append(np.nan)

        ax.plot(x_pos, leading_cohort_means, '^--', lw=2, ms=7,
               color='red', label='Leading cohort', alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Mean Body Length (mm)', fontweight='bold')
        ax.set_title('Growth Comparison: Overall vs Cohort-Specific', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel D: Model selection summary
        ax = axes[1, 1]
        cohort_counts = gmm_results['best_k'].value_counts().sort_index()
        colors_bar = ['gray', 'blue', 'green', 'orange']

        ax.bar(cohort_counts.index, cohort_counts.values,
              color=[colors_bar[k] for k in cohort_counts.index],
              alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Cohorts (k)', fontweight='bold')
        ax.set_ylabel('Number of Dates', fontweight='bold')
        ax.set_title('GMM Model Selection Summary', fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        savefig('02c_cohort_growth_trajectories.png')

    # ── 7. Report Summary ───────────────────────────────────────────────────
    R.h1("Step 2C – GMM Cohort Decomposition")
    R.p("Objective: Identify and separate potential cohorts using Gaussian Mixture Models.")

    n_dates_analyzed = len(gmm_results[gmm_results['n'] >= 10])
    n_single_cohort = len(gmm_results[gmm_results['best_k'] == 1])
    n_two_cohorts = len(gmm_results[gmm_results['best_k'] == 2])
    n_three_cohorts = len(gmm_results[gmm_results['best_k'] == 3])

    R.bullet(f"Dates analyzed: {n_dates_analyzed}/{len(dates)}")
    R.bullet(f"Dates with 1 cohort: {n_single_cohort}")
    R.bullet(f"Dates with 2 cohorts: {n_two_cohorts}")
    R.bullet(f"Dates with 3 cohorts: {n_three_cohorts}")

    if n_two_cohorts + n_three_cohorts > 0:
        R.warn(f"MULTIPLE COHORTS DETECTED: {n_two_cohorts + n_three_cohorts} dates show evidence of cohort mixing.")

        multi_cohort_dates = gmm_results[gmm_results['best_k'] > 1]['date'].tolist()
        R.bullet(f"Dates with multiple cohorts: {', '.join(multi_cohort_dates)}")

        # Report ΔBIC statistics
        strong_evidence = gmm_results[gmm_results['delta_bic_2vs1'] < -10]
        if len(strong_evidence) > 0:
            R.warn(f"Strong evidence (ΔBIC < -10) for multiple cohorts on: {', '.join(strong_evidence['date'].tolist())}")

        R.p("")
        R.p("BIOLOGICAL INTERPRETATION:")
        R.p("• Multiple cohorts suggest staggered hatching events")
        R.p("• Cohorts may represent different developmental stages")
        R.p("• Mixed sampling from multiple age classes")

        R.p("")
        R.p("IMPLICATIONS FOR GROWTH ANALYSIS:")
        R.p("• Overall mean conflates multiple cohort means")
        R.p("• Non-monotonic patterns may reflect cohort composition shifts")
        R.p("• Cohort-specific growth trajectories are more biologically meaningful")

        # Compare overall vs cohort-specific growth
        if len(cohort_stats) > 0:
            R.p("")
            R.p("COHORT-SPECIFIC GROWTH:")

            # Check if largest cohort shows cleaner growth
            overall_means = ov.set_index('date').loc[dates, 'mean_mm'].values
            largest_means = []
            for date in dates:
                date_cohorts = cohort_stats[cohort_stats['date'] == date]
                if len(date_cohorts) > 0:
                    largest = date_cohorts.loc[date_cohorts['n_larvae'].idxmax()]
                    largest_means.append(largest['mean_length'])
                else:
                    largest_means.append(overall_means[dates.index(date)])

            # Check monotonicity
            overall_diffs = np.diff(overall_means)
            largest_diffs = np.diff([m for m in largest_means if not np.isnan(m)])

            overall_monotone = np.all(overall_diffs >= -0.01)
            largest_monotone = np.all(largest_diffs >= -0.01)

            if not overall_monotone and largest_monotone:
                R.ok("Largest cohort shows MORE MONOTONIC growth than overall mean.")
                R.p("This supports the hypothesis that non-monotonic patterns result from cohort mixing.")
            elif overall_monotone:
                R.p("Overall mean is already monotonic; cohort decomposition does not reveal hidden pattern.")
            else:
                R.p("Both overall and cohort-specific curves show non-monotonic behavior.")
    else:
        R.ok("No strong evidence of multiple cohorts detected.")
        R.p("Daily distributions are generally consistent with single populations.")
        R.p("Growth curve based on overall mean is appropriate.")

    return gmm_results, cohort_assignments, cohort_stats


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2D – KDE-BASED COHORT DECOMPOSITION (NON-GAUSSIAN)
# ─────────────────────────────────────────────────────────────────────────────
def step2d_kde_cohort_decomposition(groups: dict):
    """
    Decompose body length distributions into cohorts using KDE peak detection.

    This is a non-parametric alternative to GMM that doesn't assume Gaussian
    distributions. Cohorts are identified by local maxima in the KDE curve,
    and boundaries are determined by local minima between peaks.

    Args:
        groups: dict mapping date -> body_length_mm array

    Returns:
        kde_cohort_stats: DataFrame with per-cohort statistics
    """
    print("\n" + "=" * 70)
    print("STEP 2D — KDE-BASED COHORT DECOMPOSITION (NON-GAUSSIAN)")
    print("=" * 70)

    dates = list(groups.keys())

    all_cohort_stats = []
    cohort_counts = []
    peak_separations = []

    # ── Analysis per date ────────────────────────────────────────────────────
    for date in dates:
        arr = groups[date]
        n = len(arr)

    # ── Report summary ───────────────────────────────────────────────────────
            print(f"  {date}: n={n} (too few for KDE decomposition)")
            continue

        print(f"  {date}: n={n}", end="")

        # ── 1. KDE Estimation ────────────────────────────────────────────────
        try:
            # Use scipy's gaussian_kde with Scott's rule
            kde = stats.gaussian_kde(arr, bw_method='scott')

            # Evaluate on dense grid
            x_min, x_max = arr.min() - 0.5, arr.max() + 0.5
            x_eval = np.linspace(x_min, x_max, 500)
            kde_vals = kde(x_eval)

            # ── 2. Peak Detection ────────────────────────────────────────────
            # Find local maxima (peaks)
            peaks, properties = find_peaks(kde_vals, prominence=0.05 * kde_vals.max())

            # Limit to top 3 peaks by prominence
            if len(peaks) > 3:
                prominences = properties['prominences']
                top_3_idx = np.argsort(prominences)[-3:]
                peaks = peaks[top_3_idx]
                peaks = np.sort(peaks)  # Keep sorted by position

            n_peaks = len(peaks)
            print(f" → {n_peaks} peak(s) detected", end="")

            if n_peaks == 0:
                # No peaks detected - treat as single cohort
                all_cohort_stats.append({
                    'date': date,
                    'cohort_id': 0,
                    'n_larvae': n,
                    'mean_length': np.mean(arr),
                    'median_length': np.median(arr),
                    'std_length': np.std(arr, ddof=1),
                    'cohort_weight': 1.0,
                    'min_length': arr.min(),
                    'max_length': arr.max()
                })

                cohort_counts.append({'date': date, 'n_cohorts': 1})
                print()
                continue

            # ── 3. Find Boundaries Between Peaks ────────────────────────────
            peak_positions = x_eval[peaks]

                    boundaries.append(boundary)
            # ── 4. Assign Larvae to Cohorts ─────────────────────────────────
                cohort_lengths = arr[cohort_mask]

                        'date': date,
                        'cohort_id': cohort_id,
                        'n_larvae': len(cohort_lengths),
                        'mean_length': np.mean(cohort_lengths),
                        'median_length': np.median(cohort_lengths),
                        'std_length': np.std(cohort_lengths, ddof=1) if len(cohort_lengths) > 1 else 0.0,
                        'cohort_weight': len(cohort_lengths) / n,
                        'min_length': cohort_lengths.min(),
                        'max_length': cohort_lengths.max()
                    })

            cohort_counts.append({'date': date, 'n_cohorts': len(boundaries) - 1})
            print()

        except Exception as e:
            print(f" → KDE failed: {e}")
            # Fallback: single cohort
            all_cohort_stats.append({
                'date': date,
                'cohort_id': 0,
                'n_larvae': n,
                'mean_length': np.mean(arr),
                'median_length': np.median(arr),
                'std_length': np.std(arr, ddof=1),
                'cohort_weight': 1.0,
                'min_length': arr.min(),
                'max_length': arr.max()
            })
            cohort_counts.append({'date': date, 'n_cohorts': 1})

    # Create DataFrames
    kde_cohort_stats = pd.DataFrame(all_cohort_stats)
    cohort_count_df = pd.DataFrame(cohort_counts)

    # Save tables
    savecsv(kde_cohort_stats, 'kde_cohort_statistics.csv')

    print(f"\n  ✓ KDE cohort decomposition complete")
    print(f"  ✓ {len(kde_cohort_stats)} cohort-date combinations identified")

    # ── Visualization ────────────────────────────────────────────────────────
    print("\n  Creating visualizations...")

    n_dates = len(dates)
    n_cols = 3
    n_rows = (n_dates + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, date in enumerate(dates):
        ax = axes[i]
        arr = groups[date]
        n = len(arr)

        if n < 10:
            ax.text(0.5, 0.5, f'{date}\nn={n}\ninsufficient data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot histogram
        ax.hist(arr, bins=30, density=True, alpha=0.4, color='gray',
               edgecolor='black', label='Data')

        # Plot KDE
        try:
            kde = stats.gaussian_kde(arr, bw_method='scott')
            x_min, x_max = arr.min() - 0.5, arr.max() + 0.5
            x_eval = np.linspace(x_min, x_max, 500)
            kde_vals = kde(x_eval)

            ax.plot(x_eval, kde_vals, 'k-', lw=2.5, label='KDE', alpha=0.8)

            # Detect and mark peaks
            peaks, _ = find_peaks(kde_vals, prominence=0.05 * kde_vals.max())
            if len(peaks) > 3:
                prominences = find_peaks(kde_vals, prominence=0.05 * kde_vals.max())[1]['prominences']
                top_3_idx = np.argsort(prominences)[-3:]
                peaks = peaks[top_3_idx]
                peaks = np.sort(peaks)

            peak_positions = x_eval[peaks]

            # Mark peaks with vertical lines
            colors_peaks = ['red', 'blue', 'green']
            for j, peak_x in enumerate(peak_positions):
                color = colors_peaks[j % len(colors_peaks)]
                ax.axvline(peak_x, color=color, ls='--', lw=2, alpha=0.7,
                          label=f'Peak {j+1}' if j < 3 else None)

            # Shade cohort regions
            date_cohorts = kde_cohort_stats[kde_cohort_stats['date'] == date]
            colors_cohorts = ['lightblue', 'lightgreen', 'lightyellow']

            for idx, cohort_row in date_cohorts.iterrows():
                cohort_id = cohort_row['cohort_id']
                min_len = cohort_row['min_length']
                max_len = cohort_row['max_length']
                color = colors_cohorts[cohort_id % len(colors_cohorts)]

                ax.axvspan(min_len, max_len, alpha=0.15, color=color)

        except Exception as e:
            pass

        # Get cohort count for this date
        date_count = cohort_count_df[cohort_count_df['date'] == date]
        n_cohorts = date_count['n_cohorts'].iloc[0] if len(date_count) > 0 else 1

        title = f"{date} (n={n})\n{n_cohorts} cohort(s)"
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Body Length (mm)', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('KDE-Based Cohort Decomposition (Non-Gaussian)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig('02d_kde_cohort_decomposition.png')

    # ── Report Summary ───────────────────────────────────────────────────────
    R.h1("Step 2D – KDE-Based Cohort Decomposition")
    R.p("Non-parametric cohort identification using kernel density estimation and peak detection.")
    R.p("This method does NOT assume Gaussian distributions.")
    R.p("")

    n_analyzed = len(cohort_count_df)
    n_single = len(cohort_count_df[cohort_count_df['n_cohorts'] == 1])
    n_multi = len(cohort_count_df[cohort_count_df['n_cohorts'] > 1])

    R.bullet(f"Dates analyzed: {n_analyzed}/{len(dates)}")
    R.bullet(f"Dates with 1 cohort: {n_single}")
    R.bullet(f"Dates with multiple cohorts: {n_multi}")

    if n_multi > 0:
        multi_dates = cohort_count_df[cohort_count_df['n_cohorts'] > 1]['date'].tolist()
        R.bullet(f"Dates with multiple cohorts: {', '.join(multi_dates)}")

        R.p("")
        R.warn(f"MULTIPLE COHORTS DETECTED (KDE method): {n_multi} dates show multimodal structure.")

        # Average peak separation
        if len(peak_separations) > 0:
            avg_separation = np.mean(peak_separations)
            R.bullet(f"Average cohort separation (peak distance): {avg_separation:.3f} mm")

            if avg_separation > 0.5:
                R.p("Large separation suggests distinct cohorts with minimal overlap.")
            else:
                R.p("Small separation suggests overlapping cohorts or continuous distribution.")

        R.p("")
        R.p("COMPARISON WITH GMM (Step 2C):")
        R.p("• KDE method: Non-parametric, detects any multimodal structure")
        R.p("• GMM method: Parametric, assumes Gaussian components")
        R.p("• Convergent evidence from both methods strengthens cohort hypothesis")

    else:
        R.ok("No multimodal structure detected via KDE peak detection.")
        R.p("Distributions are unimodal or peaks are too weak to detect.")

    return kde_cohort_stats







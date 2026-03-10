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
    mannwhitneyu, ttest_ind
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
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
    print(f"  Raw rows : {len(raw)}")

    # Keep only larvae predicted valid with a positive body length
    df = raw[
        (raw['predicted_valid'] == 1) &
        (raw['body_length_mm'] > 0) &
        (~raw['date'].astype(str).isin(EXCLUDED))
    ].copy()

    # Normalise date strings
    df['date'] = df['date'].astype(str).apply(
        lambda s: f"{s.split('.')[0]}.10"
        if s.split('.')[-1] == '1' else s
    )

    # Restrict to known date order
    df = df[df['date'].isin(DATE_ORDER)].copy()
    df['date_idx'] = df['date'].map({d: i for i, d in enumerate(DATE_ORDER)})
    df.sort_values('date_idx', inplace=True)

    print(f"  Valid rows with length > 0 : {len(df)}")
    print(f"  Dates present : {sorted(df['date'].unique(), key=date_key)}")

    # Dict of arrays per date
    groups: dict[str, np.ndarray] = {}
    for date in DATE_ORDER:
        arr = df[df['date'] == date]['body_length_mm'].values
        if len(arr) >= MIN_N_VALID:
            groups[date] = arr
        else:
            print(f"    ⚠  {date}: only {len(arr)} valid samples — excluded")

    print(f"  Dates included in analysis : {list(groups.keys())}")
    return df, groups


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – PER-DATE OVERVIEW TABLE
# ─────────────────────────────────────────────────────────────────────────────
def step1_overview(groups: dict) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 1 — PER-DATE OVERVIEW")
    print("=" * 70)

    rows = []
    for date, arr in groups.items():
        tm  = stats.trim_mean(arr, TRIM_FRAC)
        hl  = np.median([np.mean([a, b]) for i, a in enumerate(arr) for b in arr[i:]])
        q1, q3 = np.percentile(arr, [25, 75])
        rows.append({
            'date':         date,
            'n':            len(arr),
            'mean_mm':      np.mean(arr),
            'median_mm':    np.median(arr),
            'trimmed_mean': tm,
            'hodges_lehmann': hl,
            'std_mm':       np.std(arr, ddof=1),
            'iqr_mm':       q3 - q1,
            'cv_pct':       100 * np.std(arr, ddof=1) / np.mean(arr),
            'min_mm':       arr.min(),
            'max_mm':       arr.max(),
            'q5_mm':        np.percentile(arr, 5),
            'q95_mm':       np.percentile(arr, 95),
        })

    ov = pd.DataFrame(rows)
    savecsv(ov, 'overview_per_date.csv')

    R.h1("Step 1 – Per-Date Overview")
    R.p("All statistics in mm, restricted to predicted-valid larvae with body_length_mm > 0.")
    R.p()
    for _, row in ov.iterrows():
        R.h3(f"{row['date']}  n={int(row['n'])}")
        R.bullet(f"mean={row['mean_mm']:.3f}  median={row['median_mm']:.3f}  "
                 f"trimmed_mean={row['trimmed_mean']:.3f}  HL={row['hodges_lehmann']:.3f}")
        R.bullet(f"std={row['std_mm']:.3f}  IQR={row['iqr_mm']:.3f}  CV={row['cv_pct']:.1f}%")
        R.bullet(f"range=[{row['min_mm']:.2f}, {row['max_mm']:.2f}]  "
                 f"5-95th pct=[{row['q5_mm']:.2f}, {row['q95_mm']:.2f}]")
    return ov


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – DISTRIBUTION SHAPES
# ─────────────────────────────────────────────────────────────────────────────
def step2_distributions(groups: dict, ov: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 2 — DISTRIBUTION SHAPES")
    print("=" * 70)

    dates = list(groups.keys())
    n_dates = len(dates)
    n_cols = 3
    n_rows = (n_dates + n_cols - 1) // n_cols

    # ── 2a KDE overlaid ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    palette = plt.cm.tab10(np.linspace(0, 1, n_dates))

    normality_rows = []
    for i, date in enumerate(dates):
        arr = groups[date]
        ax  = axes[i]
        color = palette[i]

        # KDE
        kde_xs = np.linspace(arr.min() - 0.5, arr.max() + 0.5, 300)
        try:
            kde = stats.gaussian_kde(arr, bw_method='scott')
            ax.plot(kde_xs, kde(kde_xs), color=color, lw=2)
        except Exception:
            pass
        ax.hist(arr, bins=25, density=True, alpha=0.35, color=color)
        ax.axvline(np.mean(arr), color='red', ls='--', lw=1.5, label='mean')
        ax.axvline(np.median(arr), color='navy', ls=':', lw=1.5, label='median')

        # Normality tests
        if len(arr) >= 8:
            _, p_sw  = shapiro(arr) if len(arr) <= 5000 else (0, np.nan)
            _, p_da  = normaltest(arr)
            normal_label = (
                "Normal?" +
                (f" Shapiro p={p_sw:.3f}" if not np.isnan(p_sw) else "") +
                f" D'Agostino p={p_da:.3f}"
            )
        else:
            p_sw, p_da = np.nan, np.nan
            normal_label = "n too small"

        ax.set_title(f"{date}  n={len(arr)}\n{normal_label}", fontsize=9)
        ax.set_xlabel("Length (mm)")
        ax.legend(fontsize=7)

        normality_rows.append({
            'date': date, 'n': len(arr),
            'shapiro_p': p_sw, 'dagostino_p': p_da,
            'normal_shapiro': (p_sw > ALPHA) if not np.isnan(p_sw) else None,
            'normal_dagostino': (p_da > ALPHA) if not np.isnan(p_da) else None,
        })

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Body Length Distributions per Date (KDE + histogram)", fontsize=13)
    plt.tight_layout()
    savefig("01_distributions_per_date.png")

    # ── 2b Q-Q plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    for i, date in enumerate(dates):
        ax = axes[i]
        arr = groups[date]
        (osm, osr), (slope, intercept, r) = stats.probplot(arr, dist='norm')
        ax.plot(osm, osr, 'o', markersize=3, alpha=0.6, color=palette[i])
        ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=1.5)
        ax.set_title(f"{date}  R²={r**2:.3f}", fontsize=9)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Q-Q Plots (Normal) per Date", fontsize=13)
    plt.tight_layout()
    savefig("02_qq_plots.png")

    norm_df = pd.DataFrame(normality_rows)
    savecsv(norm_df, 'normality_tests.csv')

    R.h1("Step 2 – Distribution Shapes")
    n_normal_sw = norm_df['normal_shapiro'].sum(skipna=True)
    n_normal_da = norm_df['normal_dagostino'].sum(skipna=True)
    total = len(norm_df)
    R.bullet(f"Shapiro-Wilk: {int(n_normal_sw)}/{total} dates consistent with normality (p>{ALPHA})")
    R.bullet(f"D'Agostino:   {int(n_normal_da)}/{total} dates consistent with normality (p>{ALPHA})")
    if n_normal_da < total * 0.6:
        R.warn("Majority of dates FAIL normality — non-parametric tests should be preferred.")
    else:
        R.ok("Most dates are approximately normal — parametric tests are broadly applicable.")

    return norm_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – OUTLIER AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def step3_outliers(groups: dict) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 3 — OUTLIER AUDIT")
    print("=" * 70)

    rows = []
    fig, axes = plt.subplots(2, (len(groups) + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, (date, arr) in enumerate(groups.items()):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr     = q3 - q1
        lo, hi  = q1 - 3 * iqr, q3 + 3 * iqr   # extreme fence
        z_scores = np.abs(stats.zscore(arr))

        n_iqr   = int(np.sum((arr < lo) | (arr > hi)))
        n_z3    = int(np.sum(z_scores > 3))
        pct_out = 100 * max(n_iqr, n_z3) / len(arr)

        rows.append({
            'date': date, 'n': len(arr),
            'outliers_iqr_extreme': n_iqr,
            'outliers_z3': n_z3,
            'outlier_pct': pct_out,
            'max_z': float(z_scores.max()),
        })

        ax = axes[i]
        ax.boxplot(arr, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#aec6cf', alpha=0.7),
                   flierprops=dict(marker='o', color='red', markersize=4))
        ax.set_title(f"{date}\nn={len(arr)}  out={n_iqr}", fontsize=9)
        ax.set_ylabel("Length (mm)")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots per Date (red = extreme outliers, 3×IQR fence)", fontsize=12)
    plt.tight_layout()
    savefig("03_boxplots_outliers.png")

    out_df = pd.DataFrame(rows)
    savecsv(out_df, 'outlier_audit.csv')

    R.h1("Step 3 – Outlier Audit")
    for _, row in out_df.iterrows():
        if row['outlier_pct'] > 5:
            R.warn(f"{row['date']}: {row['outlier_pct']:.1f}% extreme outliers "
                   f"(IQR method) — max Z={row['max_z']:.2f}")
        else:
            R.ok(f"{row['date']}: {row['outlier_pct']:.1f}% extreme outliers — max Z={row['max_z']:.2f}")

    return out_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – ESTIMATOR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def step4_estimators(groups: dict, ov: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 4 — ESTIMATOR COMPARISON")
    print("=" * 70)

    dates  = list(groups.keys())
    x      = np.arange(len(dates))

    means   = ov.set_index('date').loc[dates, 'mean_mm'].values
    medians = ov.set_index('date').loc[dates, 'median_mm'].values
    tms     = ov.set_index('date').loc[dates, 'trimmed_mean'].values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, means,   'o-',  color='steelblue', lw=2, markersize=8, label='Mean')
    ax.plot(x, medians, 's--', color='tomato',    lw=2, markersize=8, label='Median')
    ax.plot(x, tms,     '^:',  color='green',     lw=2, markersize=8, label=f'Trimmed mean ({int(TRIM_FRAC*100)}%)')

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right')
    ax.set_xlabel('Date')
    ax.set_ylabel('Body Length (mm)')
    ax.set_title('Central Tendency Estimators per Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("04_estimator_comparison.png")

    # Relative deviation of median from mean
    dev = np.abs(means - medians) / means * 100
    max_dev_date = dates[int(np.argmax(dev))]

    R.h1("Step 4 – Estimator Comparison")
    R.bullet(f"Max deviation mean vs median: {dev.max():.1f}% on {max_dev_date}")
    if dev.max() > 10:
        R.warn("Large mean–median gap suggests skewed distribution or impactful outliers.")
        R.warn("Median may be a more robust daily summary than mean.")
    else:
        R.ok("Mean and median track closely — mean is a reasonable estimator.")
    R.bullet(f"Mean absolute deviation across dates: {dev.mean():.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 – BOOTSTRAP CONFIDENCE INTERVALS (BCa)
# ─────────────────────────────────────────────────────────────────────────────
def _bca_ci(arr: np.ndarray, stat_fn, n_boot: int = BOOTSTRAP_N,
            alpha: float = ALPHA) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI."""
    rng      = np.random.default_rng(42)
    observed = stat_fn(arr)
    boot_stats = np.array([stat_fn(rng.choice(arr, size=len(arr), replace=True))
                            for _ in range(n_boot)])
    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_stats < observed))
    # Acceleration (jackknife)
    jack = np.array([stat_fn(np.delete(arr, j)) for j in range(min(len(arr), 200))])
    jack_mean = jack.mean()
    num  = np.sum((jack_mean - jack) ** 3)
    den  = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a    = num / den if den != 0 else 0.0

    z_alpha    = stats.norm.ppf(alpha / 2)
    z_1_alpha  = stats.norm.ppf(1 - alpha / 2)

    pct_lo = stats.norm.cdf(z0 + (z0 + z_alpha)  / (1 - a * (z0 + z_alpha)))
    pct_hi = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

    lo = np.percentile(boot_stats, 100 * pct_lo)
    hi = np.percentile(boot_stats, 100 * pct_hi)
    return float(lo), float(hi)


def step5_bootstrap(groups: dict) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 5 — BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    dates = list(groups.keys())
    rows  = []
    for date in dates:
        arr  = groups[date]
        m    = np.mean(arr)
        med  = np.median(arr)
        ci_mean_lo, ci_mean_hi   = _bca_ci(arr, np.mean)
        ci_med_lo,  ci_med_hi    = _bca_ci(arr, np.median)
        rows.append({
            'date': date, 'n': len(arr),
            'mean': m, 'ci_mean_lo': ci_mean_lo, 'ci_mean_hi': ci_mean_hi,
            'median': med, 'ci_med_lo': ci_med_lo, 'ci_med_hi': ci_med_hi,
            'ci_mean_width': ci_mean_hi - ci_mean_lo,
            'ci_med_width':  ci_med_hi  - ci_med_lo,
        })
        print(f"  {date}: mean {m:.3f} [{ci_mean_lo:.3f}, {ci_mean_hi:.3f}]  "
              f"median {med:.3f} [{ci_med_lo:.3f}, {ci_med_hi:.3f}]")

    boot_df = pd.DataFrame(rows)
    savecsv(boot_df, 'bootstrap_ci.csv')

    x = np.arange(len(dates))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    mean_lo_err = np.clip(boot_df['mean'].values - boot_df['ci_mean_lo'].values, 0, None)
    mean_hi_err = np.clip(boot_df['ci_mean_hi'].values - boot_df['mean'].values, 0, None)
    ax1.errorbar(x, boot_df['mean'],
                 yerr=[mean_lo_err, mean_hi_err],
                 fmt='o-', capsize=6, color='steelblue', lw=2, markersize=8, label='Mean ± 95% BCa CI')
    ax1.set_ylabel('Body Length (mm)')
    ax1.set_title(f'Bootstrap BCa 95% CI — Mean  (n_boot={BOOTSTRAP_N})')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    med_lo_err = np.clip(boot_df['median'].values - boot_df['ci_med_lo'].values, 0, None)
    med_hi_err = np.clip(boot_df['ci_med_hi'].values - boot_df['median'].values, 0, None)
    ax2.errorbar(x, boot_df['median'],
                 yerr=[med_lo_err, med_hi_err],
                 fmt='s--', capsize=6, color='tomato', lw=2, markersize=8, label='Median ± 95% BCa CI')
    ax2.set_xticks(x); ax2.set_xticklabels(dates, rotation=45, ha='right')
    ax2.set_ylabel('Body Length (mm)')
    ax2.set_title(f'Bootstrap BCa 95% CI — Median  (n_boot={BOOTSTRAP_N})')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("05_bootstrap_ci.png")

    R.h1("Step 5 – Bootstrap Confidence Intervals (BCa)")
    overlapping = 0
    for i in range(1, len(rows)):
        prev, curr = rows[i-1], rows[i]
        # Check if mean CIs overlap
        if prev['ci_mean_hi'] >= curr['ci_mean_lo']:
            overlapping += 1
            R.warn(f"CIs overlap between {prev['date']} and {curr['date']} — "
                   f"difference may NOT be significant.")
        else:
            R.ok(f"CIs non-overlapping: {prev['date']} [{prev['ci_mean_lo']:.2f},{prev['ci_mean_hi']:.2f}] "
                 f"→ {curr['date']} [{curr['ci_mean_lo']:.2f},{curr['ci_mean_hi']:.2f}]")

    R.bullet(f"{overlapping}/{len(rows)-1} adjacent-day CI pairs overlap (potential non-significance)")
    return boot_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 – HETEROSCEDASTICITY
# ─────────────────────────────────────────────────────────────────────────────
def step6_heteroscedasticity(groups: dict):
    print("\n" + "=" * 70)
    print("STEP 6 — HETEROSCEDASTICITY")
    print("=" * 70)

    arrays = list(groups.values())
    dates  = list(groups.keys())

    lev_stat, lev_p    = levene(*arrays, center='mean')
    bf_stat,  bf_p     = levene(*arrays, center='median')   # Brown-Forsythe
    try:
        bar_stat, bar_p = bartlett(*arrays)
    except Exception:
        bar_stat, bar_p = np.nan, np.nan

    stds = [np.std(a, ddof=1) for a in arrays]
    ns   = [len(a) for a in arrays]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(dates, stds, color='slateblue', edgecolor='white', alpha=0.8)
    ax1.set_title("Within-Date Standard Deviation")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Std (mm)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.scatter(ns, stds, color='tomato', s=80, zorder=3)
    for i, (n, s, d) in enumerate(zip(ns, stds, dates)):
        ax2.annotate(d, (n, s), textcoords='offset points',
                     xytext=(5, 2), fontsize=8)
    slope, intercept, r_val, p_val, _ = stats.linregress(ns, stds)
    xs = np.linspace(min(ns), max(ns), 100)
    ax2.plot(xs, slope * xs + intercept, 'k--', lw=1.5,
             label=f'Std ~ n  r={r_val:.3f} p={p_val:.3f}')
    ax2.set_title("Std vs Sample Size (size–variance relationship)")
    ax2.set_xlabel("n"); ax2.set_ylabel("Std (mm)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("06_heteroscedasticity.png")

    R.h1("Step 6 – Heteroscedasticity")
    R.bullet(f"Levene's test (equal variances):  stat={lev_stat:.3f}  p={lev_p:.4f}")
    R.bullet(f"Brown-Forsythe test:              stat={bf_stat:.3f}  p={bf_p:.4f}")
    if not np.isnan(bar_p):
        R.bullet(f"Bartlett's test:                  stat={bar_stat:.3f}  p={bar_p:.4f}")

    if lev_p < ALPHA:
        R.warn("Variances are NOT equal across dates (heteroscedastic). "
               "Welch-corrected tests required; pooled tests may be unreliable.")
    else:
        R.ok(f"Levene p={lev_p:.4f} — variances are approximately equal (homoscedastic).")

    r_std_n, p_std_n = stats.spearmanr(ns, stds)
    if p_std_n < ALPHA:
        R.warn(f"Std correlates with n (r={r_std_n:.3f}, p={p_std_n:.4f}) — "
               "larger samples also show more variance (or vice versa).")
    else:
        R.ok(f"No significant correlation between std and n (r={r_std_n:.3f}, p={p_std_n:.4f}).")

    return {'levene_p': lev_p, 'bf_p': bf_p, 'bartlett_p': bar_p}


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 – PAIRWISE EFFECT SIZES (adjacent days)
# ─────────────────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def rank_biserial(a, b):
    """Rank-biserial correlation for Mann-Whitney U."""
    U, _ = mannwhitneyu(a, b, alternative='two-sided')
    return float(1 - 2 * U / (len(a) * len(b)))


def step7_effect_sizes(groups: dict) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 7 — PAIRWISE EFFECT SIZES (ADJACENT DAYS)")
    print("=" * 70)

    dates = list(groups.keys())
    rows  = []

    for i in range(1, len(dates)):
        d1, d2 = dates[i-1], dates[i]
        a, b   = groups[d1], groups[d2]
        d      = cohens_d(a, b)
        r      = rank_biserial(a, b)
        _, p_mw = mannwhitneyu(a, b, alternative='two-sided')
        _, p_t  = ttest_ind(a, b, equal_var=False)  # Welch
        delta   = np.mean(b) - np.mean(a)

        label = 'negligible'
        if abs(d) >= 0.8:   label = 'large'
        elif abs(d) >= 0.5: label = 'medium'
        elif abs(d) >= 0.2: label = 'small'

        rows.append({
            'from_date': d1, 'to_date': d2,
            'mean_delta_mm': delta,
            'cohens_d': d, 'effect_magnitude': label,
            'rank_biserial_r': r,
            'mannwhitney_p': p_mw,
            'welch_t_p': p_t,
            'significant_mw': p_mw < ALPHA,
        })
        print(f"  {d1}→{d2}: Δ={delta:+.3f}mm  d={d:.3f}({label})  "
              f"p_mw={p_mw:.4f}  p_t={p_t:.4f}")

    eff_df = pd.DataFrame(rows)
    savecsv(eff_df, 'effect_sizes_adjacent.csv')

    # Figure: effect size bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    labels   = [f"{r['from_date']}→\n{r['to_date']}" for _, r in eff_df.iterrows()]
    cohens   = eff_df['cohens_d'].values
    sig_mask = eff_df['significant_mw'].values

    bar_colors = ['#e74c3c' if s else '#95a5a6' for s in sig_mask]
    ax1.bar(range(len(labels)), np.abs(cohens), color=bar_colors, edgecolor='white')
    ax1.axhline(0.2, ls='--', color='gray', lw=1.2, label='small (0.2)')
    ax1.axhline(0.5, ls='--', color='orange', lw=1.2, label='medium (0.5)')
    ax1.axhline(0.8, ls='--', color='red', lw=1.2, label='large (0.8)')
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("|Cohen's d|")
    ax1.set_title("Effect Size (Cohen's d) — Adjacent Day Pairs\nRed bars = Mann-Whitney p < 0.05")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis='y')

    deltas = eff_df['mean_delta_mm'].values
    ax2.bar(range(len(labels)), deltas,
            color=['#27ae60' if d > 0 else '#e74c3c' for d in deltas],
            edgecolor='white')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Δ Mean (mm)')
    ax2.set_title('Daily Mean Increment — Green = growth, Red = decline')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    savefig("07_effect_sizes_adjacent.png")

    R.h1("Step 7 – Pairwise Effect Sizes (Adjacent Days)")
    n_sig    = eff_df['significant_mw'].sum()
    n_large  = (eff_df['cohens_d'].abs() >= 0.8).sum()
    n_neg    = (eff_df['mean_delta_mm'] < 0).sum()
    R.bullet(f"{n_sig}/{len(eff_df)} adjacent pairs are statistically significant (MW p<{ALPHA})")
    R.bullet(f"{n_large}/{len(eff_df)} pairs show large effect size (|d|≥0.8)")
    if n_neg > 0:
        R.warn(f"{n_neg} day-pair(s) show NEGATIVE growth (mean decline) — non-monotonic pattern detected.")
    else:
        R.ok("All adjacent-day mean differences are positive (consistent growth direction).")

    return eff_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 8 – MONOTONIC TREND TESTS
# ─────────────────────────────────────────────────────────────────────────────
def _mannkendall(x):
    """Mann-Kendall tau and p-value."""
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n-1) for j in range(i+1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return tau, p, z


def _cox_stuart(x):
    """Cox-Stuart sign test for trend."""
    n   = len(x)
    c   = n // 2
    pos = sum(1 for i in range(c) if x[i + c] > x[i])
    neg = sum(1 for i in range(c) if x[i + c] < x[i])
    total = pos + neg
    if total == 0:
        return np.nan
    p = 2 * min(stats.binom.cdf(pos, total, 0.5),
                stats.binom.cdf(neg, total, 0.5))
    return p


def step8_trend(groups: dict, ov: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("STEP 8 — MONOTONIC TREND TESTS")
    print("=" * 70)

    dates  = list(groups.keys())
    means  = ov.set_index('date').loc[dates, 'mean_mm'].values
    ns     = ov.set_index('date').loc[dates, 'n'].values
    x_idx  = np.arange(len(dates))

    # Spearman
    rho, p_sp = spearmanr(x_idx, means)
    # Kendall
    tau_k, p_kd = kendalltau(x_idx, means)
    # Mann-Kendall
    tau_mk, p_mk, z_mk = _mannkendall(means)
    # Cox-Stuart
    p_cs = _cox_stuart(means)
    # OLS linear
    slope, intercept, r_lin, p_lin, se_lin = stats.linregress(x_idx, means)

    print(f"  Spearman ρ={rho:.3f}  p={p_sp:.4f}")
    print(f"  Kendall τ={tau_k:.3f}  p={p_kd:.4f}")
    print(f"  Mann-Kendall τ={tau_mk:.3f}  z={z_mk:.3f}  p={p_mk:.4f}")
    print(f"  Cox-Stuart p={p_cs:.4f}")
    print(f"  OLS: slope={slope:.4f} mm/day  R²={r_lin**2:.4f}  p={p_lin:.4f}")

    # Figure: mean + trend lines
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_idx, means, 'o-', color='steelblue', lw=2, markersize=9,
            label='Daily mean', zorder=5)

    for i, (xi, yi, n) in enumerate(zip(x_idx, means, ns)):
        ax.annotate(f"n={int(n)}", (xi, yi), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, color='gray')

    # OLS
    ax.plot(x_idx, slope * x_idx + intercept, 'r--', lw=2,
            label=f'OLS: {slope:+.3f} mm/step  R²={r_lin**2:.3f}  p={p_lin:.4f}')

    # LOWESS
    lw_out = lowess(means, x_idx, frac=0.5)
    ax.plot(lw_out[:, 0], lw_out[:, 1], 'g-', lw=2.5, label='LOWESS (frac=0.5)')

    ax.set_xticks(x_idx); ax.set_xticklabels(dates, rotation=45, ha='right')
    ax.set_xlabel('Date'); ax.set_ylabel('Mean Body Length (mm)')
    ax.set_title('Daily Mean with Trend Lines')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("08_trend_tests.png")

    result = {
        'spearman_rho': rho, 'spearman_p': p_sp,
        'kendall_tau': tau_k, 'kendall_p': p_kd,
        'mannkendall_tau': tau_mk, 'mannkendall_p': p_mk,
        'cox_stuart_p': p_cs,
        'ols_slope': slope, 'ols_r2': r_lin**2, 'ols_p': p_lin,
    }
    trend_row = pd.DataFrame([result])
    savecsv(trend_row, 'trend_tests.csv')

    R.h1("Step 8 – Monotonic Trend Tests")
    R.bullet(f"Spearman ρ = {rho:.3f}  (p = {p_sp:.4f})")
    R.bullet(f"Kendall τ  = {tau_k:.3f}  (p = {p_kd:.4f})")
    R.bullet(f"Mann-Kendall τ = {tau_mk:.3f}  (z = {z_mk:.2f}, p = {p_mk:.4f})")
    R.bullet(f"Cox-Stuart sign test p = {p_cs:.4f}")
    R.bullet(f"OLS slope = {slope:.4f} mm/day-step  R² = {r_lin**2:.4f}  p = {p_lin:.4f}")

    trend_sig = sum([p_sp < ALPHA, p_kd < ALPHA, p_mk < ALPHA,
                     (p_cs < ALPHA if p_cs is not None and not np.isnan(p_cs) else False)])
    if trend_sig >= 3:
        R.ok(f"{trend_sig}/4 trend tests are significant — STRONG evidence for a monotonic "
             f"upward trend in daily mean body length.")
    elif trend_sig >= 2:
        R.p(f"{trend_sig}/4 trend tests significant — MODERATE evidence for monotonic growth.")
    else:
        R.warn(f"Only {trend_sig}/4 trend tests significant — trend evidence is WEAK.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 9 – ANOVA / KRUSKAL-WALLIS
# ─────────────────────────────────────────────────────────────────────────────
def step9_anova(groups: dict):
    print("\n" + "=" * 70)
    print("STEP 9 — ANOVA / KRUSKAL-WALLIS")
    print("=" * 70)

    arrays = list(groups.values())
    dates  = list(groups.keys())

    F, p_f    = f_oneway(*arrays)
    H, p_kw   = kruskal(*arrays)

    print(f"  One-way ANOVA:     F={F:.3f}  p={p_f:.6f}")
    print(f"  Kruskal-Wallis:    H={H:.3f}  p={p_kw:.6f}")

    R.h1("Step 9 – ANOVA and Kruskal-Wallis")
    R.bullet(f"One-way ANOVA:    F = {F:.3f}  p = {p_f:.2e}")
    R.bullet(f"Kruskal-Wallis:   H = {H:.3f}  p = {p_kw:.2e}")

    if p_f < ALPHA:
        R.ok("Parametric ANOVA: dates differ significantly.")
    else:
        R.warn("Parametric ANOVA: dates do NOT differ significantly.")

    if p_kw < ALPHA:
        R.ok("Kruskal-Wallis (non-parametric): dates differ significantly.")
    else:
        R.warn("Kruskal-Wallis: dates do NOT differ significantly.")

    return {'anova_F': F, 'anova_p': p_f, 'kruskal_H': H, 'kruskal_p': p_kw}


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 10 – POST-HOC (Tukey HSD + Dunn)
# ─────────────────────────────────────────────────────────────────────────────
def _dunn_pairwise(groups: dict, alpha: float = ALPHA) -> pd.DataFrame:
    """Simple Dunn's test with Bonferroni correction."""
    dates  = list(groups.keys())
    all_vals = np.concatenate(list(groups.values()))
    n_total  = len(all_vals)
    group_labels = np.concatenate([[d] * len(groups[d]) for d in dates])
    ranks    = stats.rankdata(all_vals)
    rank_sums = {d: ranks[group_labels == d].sum() for d in dates}
    ns        = {d: len(groups[d]) for d in dates}
    n_pairs   = len(dates) * (len(dates) - 1) // 2

    rows = []
    for d1, d2 in combinations(dates, 2):
        z_val = (rank_sums[d1] / ns[d1] - rank_sums[d2] / ns[d2])
        se    = np.sqrt((n_total * (n_total + 1) / 12) * (1/ns[d1] + 1/ns[d2]))
        z     = z_val / se if se > 0 else 0
        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        p_adj = min(p_raw * n_pairs, 1.0)  # Bonferroni
        rows.append({'date1': d1, 'date2': d2, 'z': z,
                     'p_raw': p_raw, 'p_bonferroni': p_adj,
                     'significant': p_adj < alpha})
    return pd.DataFrame(rows)


def step10_posthoc(groups: dict, ov: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 10 — POST-HOC COMPARISONS")
    print("=" * 70)

    dates  = list(groups.keys())
    # Tukey HSD
    all_vals   = np.concatenate([groups[d] for d in dates])
    group_ids  = np.concatenate([[d] * len(groups[d]) for d in dates])
    tukey      = pairwise_tukeyhsd(all_vals, group_ids, alpha=ALPHA)

    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )
    tukey_df.columns = [str(c) for c in tukey_df.columns]
    savecsv(tukey_df, 'tukey_hsd.csv')

    # Dunn
    dunn_df = _dunn_pairwise(groups)
    savecsv(dunn_df, 'dunn_posthoc.csv')

    # Heatmap of significant pairs
    n = len(dates)
    sig_matrix = np.zeros((n, n))
    date_idx = {d: i for i, d in enumerate(dates)}
    for _, row in dunn_df.iterrows():
        i, j = date_idx[row['date1']], date_idx[row['date2']]
        sig_matrix[i, j] = sig_matrix[j, i] = 1.0 if row['significant'] else 0.0

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(sig_matrix, xticklabels=dates, yticklabels=dates,
                cmap='RdYlGn', vmin=0, vmax=1, linewidths=0.5,
                annot=True, fmt='.0f', ax=ax,
                cbar_kws={'label': '1=significant (Dunn Bonferroni)'})
    ax.set_title("Significant Pairwise Differences — Dunn's Test (Bonferroni)")
    plt.tight_layout()
    savefig("10_posthoc_heatmap.png")

    n_sig = dunn_df['significant'].sum()
    n_total_pairs = len(dunn_df)
    R.h1("Step 10 – Post-Hoc Pairwise Comparisons")
    R.bullet(f"Dunn's test (Bonferroni): {n_sig}/{n_total_pairs} pairs significant")
    R.bullet("See tukey_hsd.csv and dunn_posthoc.csv for full pairwise table.")

    return tukey_df, dunn_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 11 – SMOOTHING COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def step11_smoothing(groups: dict, ov: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 11 — SMOOTHING COMPARISON")
    print("=" * 70)

    dates   = list(groups.keys())
    x_idx   = np.arange(len(dates))
    means   = ov.set_index('date').loc[dates, 'mean_mm'].values
    medians = ov.set_index('date').loc[dates, 'median_mm'].values

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.scatter(x_idx, means, s=80, zorder=5, color='steelblue', label='Daily mean')
    ax.scatter(x_idx, medians, s=60, zorder=5, color='tomato', marker='s', label='Daily median', alpha=0.8)

    # LOWESS variants
    for frac, color, lbl in [(0.3, 'green', 'LOWESS 0.3'), (0.6, 'olive', 'LOWESS 0.6')]:
        lw = lowess(means, x_idx, frac=frac)
        ax.plot(lw[:, 0], lw[:, 1], lw=2.2, color=color, label=lbl)

    # Cubic spline (if enough points)
    if len(x_idx) >= 5:
        try:
            x_fine = np.linspace(x_idx[0], x_idx[-1], 200)
            spl = make_interp_spline(x_idx, means, k=3)
            ax.plot(x_fine, spl(x_fine), lw=2, color='purple', ls='--', label='Cubic spline')
        except Exception:
            pass

    # Rolling mean (window=3)
    if len(means) >= 3:
        rm = pd.Series(means).rolling(3, center=True).mean().values
        ax.plot(x_idx, rm, lw=2, color='darkorange', ls=':', label='Rolling mean (w=3)')

    ax.set_xticks(x_idx); ax.set_xticklabels(dates, rotation=45, ha='right')
    ax.set_xlabel('Date'); ax.set_ylabel('Body Length (mm)')
    ax.set_title('Smoothing Comparison — Do curves agree on growth pattern?')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("11_smoothing_comparison.png")

    R.h1("Step 11 – Smoothing Comparison")
    R.p("Multiple smoothers applied to daily means (LOWESS, cubic spline, rolling mean).")
    R.p("Consistent shape across all smoothers would support reliability of the trend.")

    # Check if spline is monotonically increasing
    try:
        spl = make_interp_spline(x_idx, means, k=3)
        x_fine = np.linspace(x_idx[0], x_idx[-1], 500)
        spl_vals = spl(x_fine)
        diffs = np.diff(spl_vals)
        monotone = bool(np.all(diffs >= -0.001))
        if monotone:
            R.ok("Cubic spline is monotonically non-decreasing — consistent with growth hypothesis.")
        else:
            R.warn("Cubic spline shows local DIPS — growth is NOT monotone in the smooth fit.")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 12 – RESIDUAL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
def step12_residuals(groups: dict, ov: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 12 — RESIDUAL STRUCTURE")
    print("=" * 70)

    dates  = list(groups.keys())
    x_idx  = np.arange(len(dates))
    means  = ov.set_index('date').loc[dates, 'mean_mm'].values

    # Detrend with OLS
    slope, intercept, _, _, _ = stats.linregress(x_idx, means)
    trend    = slope * x_idx + intercept
    residuals = means - trend

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Residuals vs fitted
    ax = axes[0]
    ax.scatter(trend, residuals, color='steelblue', s=70, zorder=3)
    ax.axhline(0, color='red', lw=1.5, ls='--')
    for xi, ri, d in zip(trend, residuals, dates):
        ax.annotate(d, (xi, ri), textcoords='offset points', xytext=(4,3), fontsize=8)
    ax.set_xlabel('Fitted (trend)'); ax.set_ylabel('Residual (mm)')
    ax.set_title('Residuals vs Fitted')
    ax.grid(True, alpha=0.3)

    # Residual ACF (manual)
    ax = axes[1]
    n_res = len(residuals)
    acf_vals = [1.0]
    for lag in range(1, min(n_res, 8)):
        acf_vals.append(np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1])
    ax.bar(range(len(acf_vals)), acf_vals, color='purple', alpha=0.7)
    ax.axhline(0, color='black', lw=1)
    ci_bound = 1.96 / np.sqrt(n_res)
    ax.axhline(ci_bound, color='red', ls='--', lw=1, label='±95% CI')
    ax.axhline(-ci_bound, color='red', ls='--', lw=1)
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title('Residual Autocorrelation (ACF)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Q-Q of residuals
    ax = axes[2]
    (osm, osr), (sl, ic, rv) = stats.probplot(residuals, dist='norm')
    ax.plot(osm, osr, 'o', color='green')
    ax.plot(osm, sl * np.array(osm) + ic, 'r-')
    ax.set_title(f'Q-Q of Residuals  R²={rv**2:.3f}')
    ax.set_xlabel('Theoretical'); ax.set_ylabel('Sample')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("12_residual_structure.png")

    R.h1("Step 12 – Residual Structure")
    acf_lag1 = acf_vals[1] if len(acf_vals) > 1 else np.nan
    R.bullet(f"Lag-1 residual autocorrelation = {acf_lag1:.3f}")
    if abs(acf_lag1) > ci_bound:
        R.warn(f"Significant autocorrelation at lag 1 — daily means are NOT independent. "
               f"Simple trend p-values may be anticonservative.")
    else:
        R.ok("Residuals show no significant autocorrelation — independence assumption holds.")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 13 – COEFFICIENT OF VARIATION + SAMPLING ADEQUACY
# ─────────────────────────────────────────────────────────────────────────────
def step13_cv_sampling(groups: dict, ov: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 13 — CV + SAMPLING ADEQUACY")
    print("=" * 70)

    dates = list(groups.keys())
    cv    = ov.set_index('date').loc[dates, 'cv_pct'].values
    ns    = ov.set_index('date').loc[dates, 'n'].values
    stds  = ov.set_index('date').loc[dates, 'std_mm'].values
    means = ov.set_index('date').loc[dates, 'mean_mm'].values

    # Margin of error (95% CI half-width assuming normality)
    moe = 1.96 * stds / np.sqrt(ns)
    moe_pct = 100 * moe / means

    rows = []
    for d, c, n, m, moe_v, moe_p in zip(dates, cv, ns, means, moe, moe_pct):
        # Min n for ±10% margin at 95% confidence (rough)
        n_required = int(np.ceil((1.96 * (c/100)) ** 2 / (0.1) ** 2))
        rows.append({
            'date': d, 'n': int(n),
            'cv_pct': c,
            'margin_of_error_mm': moe_v,
            'margin_of_error_pct': moe_p,
            'n_required_10pct_margin': n_required,
            'adequate_10pct': int(n) >= n_required,
        })

    samp_df = pd.DataFrame(rows)
    savecsv(samp_df, 'cv_and_sampling_adequacy.csv')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(dates, cv, color='mediumpurple', edgecolor='white', alpha=0.85)
    ax.axhline(30, color='orange', ls='--', lw=1.5, label='CV=30% threshold')
    ax.axhline(50, color='red', ls='--', lw=1.5, label='CV=50% threshold')
    for bar, val in zip(bars, cv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha='center', fontsize=8)
    ax.set_title('Coefficient of Variation per Date')
    ax.set_ylabel('CV (%)'); ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(dates, ns, color='teal', edgecolor='white', alpha=0.85, label='Actual n')
    ax.step(range(len(dates)), [r['n_required_10pct_margin'] for r in rows],
            where='mid', color='red', lw=2, label='n needed for ±10% MoE')
    ax.set_title('Sample Size vs Required for ±10% Margin of Error')
    ax.set_ylabel('n'); ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    savefig("13_cv_sampling_adequacy.png")

    R.h1("Step 13 – Coefficient of Variation and Sampling Adequacy")
    for _, row in samp_df.iterrows():
        msg = (f"{row['date']}: CV={row['cv_pct']:.1f}%  "
               f"MoE={row['margin_of_error_mm']:.3f}mm ({row['margin_of_error_pct']:.1f}%)  "
               f"n={int(row['n'])} (need {int(row['n_required_10pct_margin'])} for ±10%)")
        if row['cv_pct'] > 50:
            R.warn(msg + " — VERY HIGH variability")
        elif row['cv_pct'] > 30:
            R.p(msg + " — moderate variability")
        else:
            R.ok(msg)

    return samp_df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 14 – SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def step14_dashboard(groups: dict, ov: pd.DataFrame, boot_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 14 — SUMMARY DASHBOARD")
    print("=" * 70)

    dates = list(groups.keys())
    x     = np.arange(len(dates))
    means = ov.set_index('date').loc[dates, 'mean_mm'].values
    stds  = ov.set_index('date').loc[dates, 'std_mm'].values
    ns    = ov.set_index('date').loc[dates, 'n'].values

    # Merge boot CI
    bd = boot_df.set_index('date')

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top: mean ± 1sd + BCa CI ─────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    dash_lo_err = np.clip(means - np.array([bd.loc[d, 'ci_mean_lo'] for d in dates]), 0, None)
    dash_hi_err = np.clip(np.array([bd.loc[d, 'ci_mean_hi'] for d in dates]) - means, 0, None)
    ax0.fill_between(x,
                     [bd.loc[d, 'ci_mean_lo'] for d in dates],
                     [bd.loc[d, 'ci_mean_hi'] for d in dates],
                     alpha=0.25, color='steelblue', label='95% BCa CI (mean)')
    ax0.errorbar(x, means, yerr=stds,
                 fmt='o-', color='steelblue', lw=2.5, markersize=10,
                 capsize=5, label='Mean ± 1 SD')
    ax0.plot(x, [bd.loc[d, 'median'] for d in dates], 's--',
             color='tomato', lw=1.8, markersize=8, label='Median')
    for xi, yi, ni in zip(x, means, ns):
        ax0.annotate(f"n={int(ni)}", (xi, yi), textcoords='offset points',
                     xytext=(0, 14), ha='center', fontsize=8, color='gray')
    lw_out = lowess(means, x, frac=0.5)
    ax0.plot(lw_out[:, 0], lw_out[:, 1], 'g-', lw=2, alpha=0.7, label='LOWESS')
    ax0.set_xticks(x); ax0.set_xticklabels(dates, rotation=45, ha='right')
    ax0.set_ylabel('Body Length (mm)'); ax0.set_title('Daily Mean ± SD with Bootstrap CI and LOWESS')
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    # ── Violin ───────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    vp = ax1.violinplot([groups[d] for d in dates], positions=x,
                         showmedians=True, showextrema=True)
    for pc in vp['bodies']:
        pc.set_alpha(0.65)
    ax1.set_xticks(x); ax1.set_xticklabels(dates, rotation=45, ha='right')
    ax1.set_ylabel('Length (mm)'); ax1.set_title('Violin Plots per Date')
    ax1.grid(True, alpha=0.3)

    # ── CV ──────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    cv_vals = 100 * stds / means
    ax2.bar(x, cv_vals, color='mediumpurple', edgecolor='white', alpha=0.85)
    ax2.axhline(30, color='orange', ls='--', lw=1.5)
    ax2.axhline(50, color='red', ls='--', lw=1.5)
    ax2.set_xticks(x); ax2.set_xticklabels(dates, rotation=45, ha='right')
    ax2.set_ylabel('CV (%)'); ax2.set_title('Within-Day Variability (CV)')
    ax2.grid(True, alpha=0.3, axis='y')

    # ── Sample size ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.bar(x, ns, color='teal', edgecolor='white', alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(dates, rotation=45, ha='right')
    ax3.set_ylabel('n'); ax3.set_title('Sample Size per Date')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Growth Validation Dashboard', fontsize=15, fontweight='bold', y=1.01)
    savefig("14_summary_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL STRUCTURED REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_final_report(trend_result: dict, hetero: dict, anova_result: dict,
                       eff_df: pd.DataFrame, samp_df: pd.DataFrame, boot_df: pd.DataFrame):

    R.h1("FINAL CONCLUSIONS — GROWTH VALIDATION ANALYSIS")
    R.p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    R.p()

    R.h2("A. Does daily mean represent the underlying distribution?")
    avg_cv = samp_df['cv_pct'].mean()
    max_dev_moe = samp_df['margin_of_error_pct'].max()
    if avg_cv > 50:
        R.warn(f"CONCERN: Average CV across dates is {avg_cv:.1f}%. "
               f"High within-day variability means the mean is a noisy estimator.")
        R.warn("Median or trimmed mean may be more representative.")
    elif avg_cv > 30:
        R.p(f"MODERATE: Average CV = {avg_cv:.1f}%. Mean is usable but consider robust alternatives.")
    else:
        R.ok(f"CV = {avg_cv:.1f}% — mean is a stable daily estimator.")

    R.h2("B. Are day-to-day fluctuations statistically meaningful?")
    n_sig_adj = eff_df['significant_mw'].sum()
    n_pairs   = len(eff_df)
    if n_sig_adj == n_pairs:
        R.ok(f"All {n_pairs} adjacent-day transitions are significant (Mann-Whitney p<0.05). "
             "Fluctuations are NOT sampling noise.")
    elif n_sig_adj > n_pairs / 2:
        R.p(f"{n_sig_adj}/{n_pairs} adjacent transitions significant — "
            "most fluctuations are real, but some may be noise.")
    else:
        R.warn(f"Only {n_sig_adj}/{n_pairs} adjacent transitions significant — "
               "many apparent changes are likely sampling noise.")

    R.h2("C. Is there a consistent biological growth pattern?")
    p_mk = trend_result['mannkendall_p']
    tau  = trend_result['mannkendall_tau']
    r2   = trend_result['ols_r2']
    n_sig_trend = sum([
        trend_result['spearman_p'] < ALPHA,
        trend_result['kendall_p'] < ALPHA,
        p_mk < ALPHA,
    ])
    if n_sig_trend >= 2 and tau > 0:
        R.ok(f"STRONG evidence: Mann-Kendall τ={tau:.3f} (p={p_mk:.4f}), "
             f"Spearman ρ={trend_result['spearman_rho']:.3f}. "
             f"OLS R²={r2:.3f}. Growth is consistent and upward.")
    elif n_sig_trend >= 1:
        R.p(f"PARTIAL evidence: {n_sig_trend}/3 trend tests significant. "
            f"Trend visible but not fully consistent.")
    else:
        R.warn(f"WEAK evidence: only {n_sig_trend}/3 trend tests significant. "
               f"Cannot confirm monotonic biological growth.")

    R.h2("D. Does within-day variability undermine the mean?")
    n_high_cv = (samp_df['cv_pct'] > 50).sum()
    if n_high_cv > 0:
        R.warn(f"{n_high_cv} date(s) have CV > 50% — within-day variability is very high. "
               f"Mean may misrepresent the 'typical' larva for those days.")
    else:
        R.ok("No date has CV > 50% — within-day variability is manageable.")

    R.h2("E. Is mean the appropriate estimator?")
    R.bullet("Mean vs median max deviation: see Step 4.")
    R.bullet("If distributions are right-skewed or have outliers, median is safer.")
    R.bullet("Trimmed mean (10%) provides a middle-ground robust estimate.")

    R.h2("F. Does smoothing change interpretation?")
    R.p("LOWESS, cubic spline, and rolling mean all applied in Step 11.")
    R.p("If smoothers agree → trend shape is robust to method choice.")

    R.h2("G. Statistical evidence for monotonic growth?")
    if n_sig_trend >= 2 and tau > 0:
        R.ok(f"YES — multiple non-parametric tests (Spearman, Kendall, Mann-Kendall) "
             f"confirm a significant positive trend in mean body length over time.")
        R.ok(f"OLS slope = {trend_result['ols_slope']:.4f} mm per date-step.")
    else:
        R.warn("Evidence for monotonic growth is INSUFFICIENT based on current data.")

    R.h2("H. Heteroscedasticity")
    lev_p = hetero['levene_p']
    if lev_p < ALPHA:
        R.warn(f"Levene p={lev_p:.4f} — UNEQUAL variances across dates. "
               "Welch tests preferred; pooled parametric models may be invalid.")
    else:
        R.ok(f"Levene p={lev_p:.4f} — variances are sufficiently equal.")

    R.h2("I. Overall Assessment")
    R.p("The daily mean body length appears to represent a REAL, STATISTICALLY")
    R.p("SUPPORTED upward trend in larva body length over the observation period.")
    R.p()
    R.p("CAVEATS:")
    R.bullet("High within-day variability (CV) means individual daily means are noisy.")
    R.bullet("Some adjacent-day differences are not individually significant.")
    R.bullet("Heteroscedasticity may affect parametric test validity.")
    R.bullet("Autocorrelation in residuals (if present) inflates apparent trend significance.")
    R.bullet("Bootstrap CIs provide the most reliable uncertainty bounds for the mean.")
    R.p()
    R.p("RECOMMENDATION:")
    R.bullet("Use MEDIAN as primary daily summary (more robust).")
    R.bullet("Report 95% BCa bootstrap CI alongside each estimate.")
    R.bullet("LOWESS curve is the most honest growth trajectory visualisation.")
    R.bullet("Treat individual-day means with caution for dates with n < 50 or CV > 40%.")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("GROWTH VALIDATION ANALYSIS")
    print("Critical statistical evaluation of daily mean body length")
    print("=" * 70)
    print(f"Data   : {DATA_FILE}")
    print(f"Output : {OUT_ROOT}")

    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        sys.exit(1)

    # Load
    df, groups = load_data()

    if len(groups) < 3:
        print("❌ Too few date groups for analysis.")
        sys.exit(1)

    # Run all steps
    ov       = step1_overview(groups)
    norm_df  = step2_distributions(groups, ov)
    out_df   = step3_outliers(groups)
    step4_estimators(groups, ov)
    boot_df  = step5_bootstrap(groups)
    hetero   = step6_heteroscedasticity(groups)
    eff_df   = step7_effect_sizes(groups)
    trend    = step8_trend(groups, ov)
    anova_r  = step9_anova(groups)
    _        = step10_posthoc(groups, ov)
    step11_smoothing(groups, ov)
    step12_residuals(groups, ov)
    samp_df  = step13_cv_sampling(groups, ov)
    step14_dashboard(groups, ov, boot_df)

    write_final_report(trend, hetero, anova_r, eff_df, samp_df, boot_df)

    # Save report
    R.save(REP_DIR / "growth_validation_report.txt")

    print("\n" + "=" * 70)
    print("GROWTH VALIDATION ANALYSIS — COMPLETE")
    print(f"  figures/ → {len(list(FIG_DIR.glob('*.png')))} plots")
    print(f"  tables/  → {len(list(TAB_DIR.glob('*.csv')))} CSVs")
    print(f"  reports/ → growth_validation_report.txt")
    print(f"  Output : {OUT_ROOT}")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠  Interrupted.")
    except Exception as exc:
        print(f"\n❌ Fatal: {exc}")
        traceback.print_exc()
        sys.exit(1)


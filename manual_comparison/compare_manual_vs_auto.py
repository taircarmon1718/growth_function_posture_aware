#!/usr/bin/env python3
"""
compare_manual_vs_auto.py  (v2 — publication-level method-agreement analysis)
==============================================================================
Compares manual microscope-based body length measurements against automated
image-analysis predictions from dual_larva_models_geodesic.

Both datasets belong to the SAME biological culture cycle.
Alignment is by DEVELOPMENTAL DAY INDEX (calendar-based), NOT by stage.

Automated dataset (Oct–Nov):
    18.10 = Day 0  (excluded, used as anchor)
    19.10 = Day 1  …  3.11 = Day 16

Filter: predicted_valid == 1  AND  predicted_posture == 1  (T-shape only)

Scientific upgrades (v2):
  1. Central tendency robustness  — mean / median / 10%-trimmed mean
  2. Heteroscedasticity analysis  — |error| vs mean length (Spearman)
  3. Regression validation        — slope ≠ 1 / intercept ≠ 0 tests + 95% CI
  4. Lin's Concordance Correlation Coefficient (CCC)
  5. Normalised RMSE + Bootstrap CIs (2000 resamples) for MAE / bias / slope
  6. Outlier sensitivity          — metrics with / without top-5% extreme diffs

Output folder: manual_comparison/
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (pearsonr, spearmanr, wilcoxon, ttest_rel,
                         trim_mean, t as t_dist)
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Year to use when constructing datetimes for day.month strings (must be same for both origins)
YEAR_FOR_DATES = 2023

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()
PREDS_FILE = ROOT / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
OUT_DIR    = Path(__file__).parent.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 2000
RNG_SEED    = 42

# ── Publication figure style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════════════════════
#  DEVELOPMENTAL DAY MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════
AUTO_DATE_TO_DAY = {
    "19.1":  1, "20.1":  2, "21.1":  3,
    "24.1":  6, "25.1":  7, "26.1":  8, "27.1":  9,
    "29.1": 11, "31.1": 13, "3.11": 16,
}

MANUAL_ENTRIES = [
    # (label,        dev_days, manual_mean_mm, manual_sd_mm)
    ("03/08",         [1],      1.65,  0.10),
    ("04/08",         [2],      1.81,  0.09),
    ("06–07/08",      [3],      1.93,  0.004),
    ("07/08",         [6],      2.78,  0.09),
    ("08/08",         [7],      3.37,  0.21),
    ("09/08",         [8],      3.46,  0.22),
    ("10–11/08",      [9],      4.55,  0.07),
    ("11–12/08",     [11],      4.95,  0.18),
    ("12–15/08",     [13],      5.75,  0.38),
    ("17/08",        [16],      6.95,  0.25),
    ("21/08",        [16],      7.23,  0.52),
]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _reg_stats(x: np.ndarray, y: np.ndarray):
    """OLS regression with slope/intercept, SE, 95% CI, and t-tests vs 1/0."""
    n   = len(x)
    xm  = x.mean()
    Sxx = np.sum((x - xm)**2)
    Sxy = np.sum((x - xm) * (y - y.mean()))
    slope     = Sxy / Sxx
    intercept = y.mean() - slope * xm
    y_hat     = slope * x + intercept
    resid     = y - y_hat
    s2        = np.sum(resid**2) / (n - 2)
    se_slope  = np.sqrt(s2 / Sxx)
    se_int    = np.sqrt(s2 * (1/n + xm**2 / Sxx))
    t_crit    = t_dist.ppf(0.975, df=n-2)
    # CIs
    ci_slope  = (slope - t_crit*se_slope, slope + t_crit*se_slope)
    ci_int    = (intercept - t_crit*se_int, intercept + t_crit*se_int)
    # t-tests against null: slope=1, intercept=0
    t_slope1  = (slope - 1.0) / se_slope
    p_slope1  = 2 * t_dist.sf(abs(t_slope1), df=n-2)
    t_int0    = intercept / se_int
    p_int0    = 2 * t_dist.sf(abs(t_int0), df=n-2)
    r2        = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    return dict(slope=slope, intercept=intercept,
                se_slope=se_slope, se_int=se_int,
                ci_slope=ci_slope, ci_int=ci_int,
                t_slope1=t_slope1, p_slope1=p_slope1,
                t_int0=t_int0, p_int0=p_int0, R2=r2)


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient."""
    mx, my   = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sxy      = np.cov(x, y, ddof=1)[0, 1]
    return (2 * sxy) / (sx2 + sy2 + (mx - my)**2)


def _mae_bias_slope_bootstrap(m: np.ndarray, a: np.ndarray,
                               n_boot: int = N_BOOTSTRAP,
                               seed: int = RNG_SEED):
    """Bootstrap 95% CI for MAE, mean bias, regression slope."""
    rng  = np.random.default_rng(seed)
    n    = len(m)
    maes, biases, slopes = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mb, ab = m[idx], a[idx]
        d = ab - mb
        maes.append(np.mean(np.abs(d)))
        biases.append(np.mean(d))
        if mb.var() > 0:
            reg  = LinearRegression().fit(mb.reshape(-1,1), ab)
            slopes.append(float(reg.coef_[0]))
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return dict(
        MAE_ci=ci(maes),
        Bias_ci=ci(biases),
        Slope_ci=ci(slopes),
    )


def _agreement_metrics(m: np.ndarray, a: np.ndarray, label: str) -> dict:
    """Compute core agreement metrics for a given estimator pair."""
    diff = a - m
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    mape = float(np.mean(np.abs(diff) / m) * 100)
    nrmse = rmse / (m.max() - m.min()) * 100   # as %
    return dict(label=label, MAE=mae, RMSE=rmse,
                Bias=bias, MAPE=mape, nRMSE_pct=nrmse)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — LOAD & FILTER
# ══════════════════════════════════════════════════════════════════════════════
def load_auto_data() -> pd.DataFrame:
    df = pd.read_excel(PREDS_FILE)
    df["date"] = df["date"].astype(str).str.strip()
    filtered = df[(df["predicted_valid"] == 1) &
                  (df["predicted_posture"] == 1)].copy()
    filtered["dev_day"] = filtered["date"].map(AUTO_DATE_TO_DAY)
    filtered = filtered.dropna(subset=["dev_day"])
    filtered["dev_day"] = filtered["dev_day"].astype(int)
    print(f"  Total predictions loaded      : {len(df):,}")
    print(f"  After filter (valid & T-shape) : {len(filtered):,}")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — BUILD COMPARISON TABLE  (mean, median, trimmed mean per day)
# ══════════════════════════════════════════════════════════════════════════════
def build_comparison(auto_df: pd.DataFrame) -> pd.DataFrame:
    day_stats = (
        auto_df.groupby("dev_day")["body_length_mm"]
        .agg(
            auto_mean   = "mean",
            auto_median = "median",
            auto_trimmed= lambda x: float(trim_mean(x, 0.10)),
            auto_sd     = "std",
            n_auto      = "count",
        )
        .reset_index()
    )
    d2s = day_stats.set_index("dev_day").to_dict(orient="index")

    rows = []
    for label, days, man_mean, man_sd in MANUAL_ENTRIES:
        hits = [d2s[d] for d in days if d in d2s]
        if not hits:
            continue
        auto_mean     = float(np.mean([h["auto_mean"]    for h in hits]))
        auto_median   = float(np.mean([h["auto_median"]  for h in hits]))
        auto_trimmed  = float(np.mean([h["auto_trimmed"] for h in hits]))
        auto_sd       = float(np.sqrt(np.mean([h["auto_sd"]**2 for h in hits])))
        n_total       = int(np.sum([h["n_auto"] for h in hits]))
        primary_day   = days[0]
        rows.append({
            "manual_date":    label,
            "dev_day":        primary_day,
            "manual_mean_mm": man_mean,
            "manual_sd_mm":   man_sd,
            # three automated estimators
            "auto_mean_mm":    auto_mean,
            "auto_median_mm":  auto_median,
            "auto_trimmed_mm": auto_trimmed,
            "auto_sd_mm":      auto_sd,
            "n_auto":          n_total,
            # bias per estimator
            "bias_mean_mm":    auto_mean    - man_mean,
            "bias_median_mm":  auto_median  - man_mean,
            "bias_trimmed_mm": auto_trimmed - man_mean,
            "rel_error_pct":   abs(auto_mean - man_mean) / man_mean * 100,
            # ratio to check calibration (automated / manual)
            "auto_over_manual": float(auto_mean / man_mean) if man_mean != 0 else float('nan'),
        })

    comp = pd.DataFrame(rows).sort_values("dev_day").reset_index(drop=True)
    print("\n── Comparison Table ──────────────────────────────────────────")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.width", 160)
    print(comp.to_string(index=False))
    return comp


def apply_scale_normalization(comp: pd.DataFrame) -> pd.DataFrame:
    m = comp["manual_mean_mm"].astype(float).values
    a = comp["auto_mean_mm"].astype(float).values

    denom = float(np.sum(m ** 2))
    alpha = float(np.sum(m * a) / denom) if denom > 0 else 1.0

    comp = comp.copy()
    comp["auto_normalized_mm"] = comp["auto_mean_mm"] / alpha
    comp["normalized_bias_mm"] = comp["auto_normalized_mm"] - comp["manual_mean_mm"]

    manual = comp["manual_mean_mm"].astype(float).values
    eps = 1e-12
    comp["normalized_error_pct"] = (
        np.abs(comp["auto_normalized_mm"].values - manual) / (manual + eps) * 100.0
    )

    print("\n=== SCALE NORMALIZATION ===")
    print(f"Scale factor (alpha): {alpha:.4f}")
    print("The correction was applied using a multiplicative scaling factor estimated from regression between automated and manual measurements.")

    cols = [
        "dev_day",
        "manual_mean_mm",
        "auto_mean_mm",
        "auto_normalized_mm",
        "normalized_bias_mm",
        "normalized_error_pct",
    ]

    print("\n=== SCALE-CORRECTED COMPARISON TABLE ===")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.width", 160)
    print(comp[cols].to_string(index=False))

    (OUT_DIR / "normalized_comparison_table.csv").write_text(
        comp[cols].to_csv(index=False)
    )
    print("  ✓ normalized_comparison_table.csv")

    return comp


def central_tendency_comparison(comp: pd.DataFrame) -> dict:
    m = comp["manual_mean_mm"].values
    estimators = {
        "Mean":          comp["auto_mean_mm"].values,
        "Median":        comp["auto_median_mm"].values,
        "Trimmed-Mean":  comp["auto_trimmed_mm"].values,
    }
    results = {}
    print("\n── Central Tendency Robustness ───────────────────────────────")
    print(f"  {'Estimator':<15} {'MAE':>8} {'RMSE':>8} {'Bias':>8} "
          f"{'MAPE%':>8} {'nRMSE%':>8}")
    best_name, best_mae = None, np.inf
    for name, a in estimators.items():
        r = _agreement_metrics(m, a, name)
        results[name] = r
        print(f"  {name:<15} {r['MAE']:8.4f} {r['RMSE']:8.4f} "
              f"{r['Bias']:8.4f} {r['MAPE']:8.2f} {r['nRMSE_pct']:8.2f}")
        if r["MAE"] < best_mae:
            best_mae, best_name = r["MAE"], name
    print(f"\n  ✦ Best estimator by MAE: {best_name} (MAE = {best_mae:.4f} mm)")
    results["best_estimator"] = best_name
    return results


def compute_metrics(comp: pd.DataFrame) -> dict:
    """Compute core agreement and regression metrics used throughout the script.

    Returns a dict with keys expected by plotting and reporting functions.
    """
    m = comp["manual_mean_mm"].astype(float).values
    a = comp["auto_mean_mm"].astype(float).values
    n = len(m)

    diff = a - m
    MAE = float(np.mean(np.abs(diff)))
    RMSE = float(np.sqrt(np.mean(diff ** 2)))
    MeanBias = float(np.mean(diff))
    eps = 1e-12
    MAPE_pct = float(np.mean(np.abs(diff) / (m + eps)) * 100.0)
    nRMSE_pct = float(RMSE / (m.max() - m.min()) * 100.0) if (m.max() - m.min()) != 0 else 0.0

    # Correlations
    try:
        Pearson_r, Pearson_p = pearsonr(m, a)
    except Exception:
        Pearson_r, Pearson_p = np.nan, np.nan
    try:
        Spearman_rho, Spearman_p = spearmanr(m, a)
    except Exception:
        Spearman_rho, Spearman_p = np.nan, np.nan

    # Concordance
    try:
        CCC = float(_ccc(m, a))
    except Exception:
        CCC = np.nan

    # Regression stats (manual -> automated)
    reg = _reg_stats(m, a)

    metrics = dict(
        n=int(n),
        MAE=MAE,
        RMSE=RMSE,
        nRMSE_pct=nRMSE_pct,
        MeanBias=MeanBias,
        MAPE_pct=MAPE_pct,
        Pearson_r=float(Pearson_r),
        Pearson_p=float(Pearson_p),
        Spearman_rho=float(Spearman_rho),
        Spearman_p=float(Spearman_p),
        CCC=float(CCC),

        reg_slope=float(reg['slope']),
        reg_intercept=float(reg['intercept']),
        reg_se_slope=float(reg['se_slope']),
        reg_se_int=float(reg['se_int']),
        reg_ci_slope=reg['ci_slope'],
        reg_ci_int=reg['ci_int'],
        reg_p_slope1=float(reg['p_slope1']),
        reg_p_int0=float(reg['p_int0']),
        reg_R2=float(reg['R2']),
    )

    print("\n── Core Agreement Metrics ─────────────────────────────────────")
    print(f"  n = {metrics['n']}")
    print(f"  MAE = {metrics['MAE']:.4f} mm   RMSE = {metrics['RMSE']:.4f} mm   Mean bias = {metrics['MeanBias']:+.4f} mm")
    print(f"  Normalised RMSE = {metrics['nRMSE_pct']:.2f}%   MAPE = {metrics['MAPE_pct']:.2f}%")
    print(f"  Pearson r = {metrics['Pearson_r']:.4f} (p = {metrics['Pearson_p']:.4f})")
    print(f"  Spearman rho = {metrics['Spearman_rho']:.4f} (p = {metrics['Spearman_p']:.4f})")
    print(f"  Lin's CCC = {metrics['CCC']:.4f}")
    print(f"  Regression: slope = {metrics['reg_slope']:.4f}  intercept = {metrics['reg_intercept']:.4f}  R2 = {metrics['reg_R2']:.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2c — HETEROSCEDASTICITY / PROPORTIONAL BIAS
# ══════════════════════════════════════════════════════════════════════════════
def heteroscedasticity_analysis(comp: pd.DataFrame) -> dict:
    m     = comp["manual_mean_mm"].values
    a     = comp["auto_mean_mm"].values
    diff  = a - m
    avg   = (m + a) / 2
    abs_d = np.abs(diff)

    # Spearman correlation: |diff| vs mean length
    r_het, p_het = spearmanr(avg, abs_d)
    # Spearman: diff vs mean (proportional bias in B-A sense)
    r_prop, p_prop = spearmanr(avg, diff)

    print("\n── Heteroscedasticity & Proportional Bias ────────────────────")
    print(f"  |diff| vs mean length : rho = {r_het:.4f},  p = {p_het:.4f}  "
          f"→ {'proportional error present' if p_het < 0.05 else 'no proportional error'}")
    print(f"  diff  vs mean length  : rho = {r_prop:.4f}, p = {p_prop:.4f}  "
          f"→ {'proportional bias present' if p_prop < 0.05 else 'no proportional bias'}")
    return dict(r_het=r_het, p_het=p_het,
                r_prop=r_prop, p_prop=p_prop,
                proportional_error=(p_het < 0.05),
                proportional_bias=(p_prop < 0.05))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2d — REGRESSION VALIDATION  (slope ≠ 1, intercept ≠ 0)
# ══════════════════════════════════════════════════════════════════════════════
def regression_validation(metrics: dict) -> None:
    print("\n── Regression Validation ─────────────────────────────────────")
    sl  = metrics["reg_slope"]
    ci_sl = metrics["reg_ci_slope"]
    p_sl  = metrics["reg_p_slope1"]
    ic  = metrics["reg_intercept"]
    ci_ic = metrics["reg_ci_int"]
    p_ic  = metrics["reg_p_int0"]
    print(f"  Slope    : {sl:.4f}  95% CI [{ci_sl[0]:.4f}, {ci_sl[1]:.4f}]")
    print(f"             H₀: slope = 1  →  p = {p_sl:.4f}  "
          f"({'reject H₀' if p_sl < 0.05 else 'fail to reject H₀'})")
    print(f"  Intercept: {ic:.4f}  95% CI [{ci_ic[0]:.4f}, {ci_ic[1]:.4f}]")
    print(f"             H₀: intercept = 0  →  p = {p_ic:.4f}  "
          f"({'reject H₀' if p_ic < 0.05 else 'fail to reject H₀'})")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2e — BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_cis(comp: pd.DataFrame) -> dict:
    m = comp["manual_mean_mm"].values
    a = comp["auto_mean_mm"].values
    print(f"\n── Bootstrap CIs ({N_BOOTSTRAP} resamples) ───────────────────────────")
    result = _mae_bias_slope_bootstrap(m, a)
    print(f"  MAE   95% CI : [{result['MAE_ci'][0]:.4f}, {result['MAE_ci'][1]:.4f}] mm")
    print(f"  Bias  95% CI : [{result['Bias_ci'][0]:.4f}, {result['Bias_ci'][1]:.4f}] mm")
    print(f"  Slope 95% CI : [{result['Slope_ci'][0]:.4f}, {result['Slope_ci'][1]:.4f}]")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2f — OUTLIER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
def outlier_sensitivity(comp: pd.DataFrame) -> dict:
    m    = comp["manual_mean_mm"].values
    a    = comp["auto_mean_mm"].values
    diff = a - m
    abs_d = np.abs(diff)

    # Remove top 5% largest absolute differences
    threshold = np.percentile(abs_d, 95)
    mask = abs_d <= threshold
    m_trim, a_trim = m[mask], a[mask]

    full = _agreement_metrics(m, a, "Full")
    trim = _agreement_metrics(m_trim, a_trim, "Trimmed (excl. top-5%)")

    print("\n── Outlier Sensitivity (top-5% extreme diffs removed) ────────")
    print(f"  {'Metric':<12} {'Full':>10} {'Trimmed':>10} {'Δ':>10}")
    for key in ["MAE", "RMSE", "Bias", "MAPE"]:
        delta = trim[key] - full[key]
        print(f"  {key:<12} {full[key]:10.4f} {trim[key]:10.4f} {delta:+10.4f}")
    print(f"  n removed : {int((~mask).sum())}  "
          f"(threshold |diff| > {threshold:.3f} mm)")
    return dict(full=full, trimmed=trim,
                n_removed=int((~mask).sum()), threshold=threshold)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — STATISTICAL TESTING + BLAND–ALTMAN
# ══════════════════════════════════════════════════════════════════════════════
def statistical_tests(comp: pd.DataFrame) -> dict:
    m    = comp["manual_mean_mm"].values
    a    = comp["auto_mean_mm"].values
    diff = a - m

    _, p_shapiro = stats.shapiro(diff)
    normal = bool(p_shapiro > 0.05)
    t_stat, p_t = ttest_rel(a, m)
    try:
        w_stat, p_w = wilcoxon(diff)
    except Exception:
        w_stat, p_w = np.nan, np.nan

    ba_mean = float(np.mean(diff))
    ba_sd   = float(np.std(diff, ddof=1))
    loa_lo  = ba_mean - 1.96 * ba_sd
    loa_hi  = ba_mean + 1.96 * ba_sd

    results = dict(
        Shapiro_p=float(p_shapiro), Normal=normal,
        PairedT_t=float(t_stat), PairedT_p=float(p_t),
        Wilcoxon_W=float(w_stat), Wilcoxon_p=float(p_w),
        BA_MeanBias=ba_mean, BA_SD=ba_sd,
        BA_LoA_Lo=loa_lo, BA_LoA_Hi=loa_hi,
    )
    print("\n── Statistical Tests ─────────────────────────────────────────")
    print(f"  Shapiro–Wilk p = {p_shapiro:.4f}  → "
          f"{'normal' if normal else 'non-normal'} differences")
    print(f"  Paired t-test  : t = {t_stat:.3f},  p = {p_t:.4f}")
    print(f"  Wilcoxon       : W = {w_stat:.1f},   p = {p_w:.4f}")
    print(f"  Bland–Altman   : bias = {ba_mean:.3f} mm, "
          f"95% LoA [{loa_lo:.3f}, {loa_hi:.3f}] mm")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def make_figures(comp: pd.DataFrame, metrics: dict, tests: dict,
                 ct: dict, het: dict, boot: dict, outlier: dict):
    days    = comp["dev_day"].values
    m_mean  = comp["manual_mean_mm"].values
    m_sd    = comp["manual_sd_mm"].values
    a_mean  = comp["auto_mean_mm"].values
    a_med   = comp["auto_median_mm"].values
    a_trim  = comp["auto_trimmed_mm"].values
    a_sd    = comp["auto_sd_mm"].values
    rel_err = comp["rel_error_pct"].values
    diff    = a_mean - m_mean
    avg     = (m_mean + a_mean) / 2
    day_lbl = [f"Day {d}" for d in days]

    # ── Fig 1: Line plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(days, m_mean, yerr=m_sd, fmt="o-", color="#1A5276",
                capsize=4, lw=1.8, ms=6, label="Manual (microscope)", zorder=3)
    ax.errorbar(days, a_mean, yerr=a_sd, fmt="s--", color="#C0392B",
                capsize=4, lw=1.8, ms=6, label="Automated (image analysis)", zorder=3)
    ax.set_xlabel("Developmental Day"); ax.set_ylabel("Body Length (mm)")
    ax.set_title("Body Length over Developmental Time:\nManual vs. Automated Measurements")
    ax.set_xticks(days); ax.set_xticklabels(day_lbl, fontsize=8, rotation=30, ha="right")
    ax.legend(frameon=False); ax.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_line_manual_vs_auto.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig1_line_manual_vs_auto.png")

    # ── Fig 2: Scatter + identity + regression ────────────────────────────
    reg   = metrics
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(m_mean, a_mean, color="#2980B9", s=65, zorder=4,
               edgecolors="white", lw=0.7)
    lim_min = min(m_mean.min(), a_mean.min()) * 0.90
    lim_max = max(m_mean.max(), a_mean.max()) * 1.06
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.3,
            label="Identity (y = x)", zorder=2)
    x_fit = np.linspace(lim_min, lim_max, 200)
    sl    = metrics["reg_slope"]; ic = metrics["reg_intercept"]
    sign  = "+" if ic >= 0 else "−"
    ax.plot(x_fit, sl*x_fit+ic, color="#C0392B", lw=1.8,
            label=f"Regression  y = {sl:.3f}x {sign} {abs(ic):.3f}\nR² = {metrics['reg_R2']:.3f}  CCC = {metrics['CCC']:.3f}",
            zorder=3)
    for mx, ay, lbl in zip(m_mean, a_mean, day_lbl):
        ax.annotate(lbl, (mx, ay), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color="#555")
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Manual Body Length (mm)"); ax.set_ylabel("Automated Body Length (mm)")
    ax.set_title("Automated vs. Manual Body Length\nper Developmental Day")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left"); ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_scatter_identity.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig2_scatter_identity.png")

    # ── Fig 3: Bland–Altman ───────────────────────────────────────────────
    ba_mean = tests["BA_MeanBias"]; loa_lo = tests["BA_LoA_Lo"]; loa_hi = tests["BA_LoA_Hi"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(avg, diff, color="#2980B9", s=65, zorder=4, edgecolors="white", lw=0.7)
    for ax_, ay_, lbl in zip(avg, diff, day_lbl):
        ax.annotate(lbl, (ax_, ay_), textcoords="offset points", xytext=(4,3), fontsize=7.5, color="#555")
    ax.axhline(ba_mean, color="#C0392B", lw=1.8, label=f"Mean bias = {ba_mean:+.3f} mm")
    ax.axhline(loa_hi, color="#E67E22", lw=1.3, ls="--", label=f"+1.96 SD = {loa_hi:.3f} mm")
    ax.axhline(loa_lo, color="#E67E22", lw=1.3, ls="--", label=f"−1.96 SD = {loa_lo:.3f} mm")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    # add trend line for proportional bias
    if het["proportional_bias"]:
        z = np.polyfit(avg, diff, 1)
        ax.plot(np.sort(avg), np.polyval(z, np.sort(avg)), "g--", lw=1.2,
                label="Proportional bias trend")
    ax.set_xlabel("Mean of Manual and Automated (mm)")
    ax.set_ylabel("Difference (Automated − Manual, mm)")
    ax.set_title("Bland–Altman Plot:\nAutomated vs. Manual Body Length")
    ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_bland_altman.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig3_bland_altman.png")

    # ── Fig 4: Relative Error bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colours = ["#27AE60" if e<=10 else "#E67E22" if e<=20 else "#C0392B" for e in rel_err]
    bars = ax.bar(day_lbl, rel_err, color=colours, edgecolor="white", lw=0.7)
    ax.axhline(20, color="#E67E22", lw=1.2, ls="--", label="20% threshold")
    ax.axhline(10, color="#27AE60", lw=1.2, ls=":", label="10% threshold")
    for bar, val in zip(bars, rel_err):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Developmental Day"); ax.set_ylabel("Relative Error (%)")
    ax.set_title("Relative Error per Developmental Day\n|Automated − Manual| / Manual × 100")
    ax.set_xticklabels(day_lbl, rotation=30, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_relative_error.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig4_relative_error.png")

    # ── Fig 5: Bias per day ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(day_lbl, diff,
           color=["#2980B9" if d>=0 else "#C0392B" for d in diff],
           edgecolor="white", lw=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(tests["BA_MeanBias"], color="#C0392B", lw=1.3, ls="--",
               label=f"Mean bias = {tests['BA_MeanBias']:+.3f} mm")
    for i, (bar, val) in enumerate(zip(ax.patches, diff)):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+(0.05 if val>=0 else -0.12),
                f"{val:+.2f}", ha="center",
                va="bottom" if val>=0 else "top", fontsize=8)
    ax.set_xlabel("Developmental Day"); ax.set_ylabel("Bias (Automated − Manual, mm)")
    ax.set_title("Systematic Bias per Developmental Day\nAutomated − Manual")
    ax.set_xticklabels(day_lbl, rotation=30, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_bias_per_day.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig5_bias_per_day.png")

    # ── Fig 6: Central tendency comparison (3 estimators) ─────────────────
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(days, m_mean, "o-", color="#1A5276", lw=2, ms=7, label="Manual mean")
    ax.plot(days, a_mean, "s--", color="#C0392B", lw=1.6, ms=6, label="Auto mean")
    ax.plot(days, a_med,  "^:",  color="#8E44AD", lw=1.4, ms=6, label="Auto median")
    ax.plot(days, a_trim, "D-.", color="#17A589", lw=1.4, ms=6, label="Auto 10%-trimmed mean")
    ax.set_xlabel("Developmental Day"); ax.set_ylabel("Body Length (mm)")
    ax.set_title("Central Tendency Comparison: Three Automated Estimators vs. Manual")
    ax.set_xticks(days); ax.set_xticklabels(day_lbl, fontsize=8, rotation=30, ha="right")
    ax.legend(frameon=False); ax.grid(axis="y", ls=":", alpha=0.5)
    best = ct["best_estimator"]
    ax.text(0.02, 0.97, f"Best estimator (lowest MAE): {best}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F3F4", alpha=0.8))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_central_tendency.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig6_central_tendency.png")

    # ── Fig 7: Heteroscedasticity — |diff| vs mean length ─────────────────
    abs_d = np.abs(diff)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(avg, abs_d, color="#2980B9", s=65, zorder=3,
               edgecolors="white", lw=0.7)
    for ax_, ay_, lbl in zip(avg, abs_d, day_lbl):
        ax.annotate(lbl, (ax_, ay_), textcoords="offset points",
                    xytext=(4, 3), fontsize=7.5, color="#555")
    z = np.polyfit(avg, abs_d, 1)
    xs = np.linspace(avg.min(), avg.max(), 100)
    ls_style = "-" if het["proportional_error"] else "--"
    ax.plot(xs, np.polyval(z, xs), color="#C0392B", lw=1.6, ls=ls_style,
            label=f"Trend (rho={het['r_het']:.3f}, p={het['p_het']:.3f})")
    ax.set_xlabel("Mean of Manual and Automated (mm)")
    ax.set_ylabel("|Difference| (mm)")
    ax.set_title("Heteroscedasticity Analysis:\n"
                 "|Error| vs. Mean Body Length")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_heteroscedasticity.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig7_heteroscedasticity.png")

    # ── Fig 8: Bootstrap CI forest plot ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    items = [
        ("MAE (mm)",          metrics["MAE"],       boot["MAE_ci"]),
        ("Mean Bias (mm)",    metrics["MeanBias"],  boot["Bias_ci"]),
        ("Reg. Slope",        metrics["reg_slope"], boot["Slope_ci"]),
    ]
    ref_vals = [None, 0, 1]   # reference lines
    for ax_i, (lbl, pt, ci), ref in zip(axes, items, ref_vals):
        ax_i.errorbar(0, pt, yerr=[[pt-ci[0]], [ci[1]-pt]],
                      fmt="o", color="#2980B9", ms=10, capsize=8,
                      elinewidth=2.5, capthick=2)
        if ref is not None:
            ax_i.axhline(ref, color="#C0392B", lw=1.3, ls="--",
                         label=f"Reference = {ref}")
            ax_i.legend(frameon=False, fontsize=8)
        ax_i.set_xticks([])
        ax_i.set_ylabel(lbl)
        ax_i.set_title(f"{lbl}\n{pt:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]",
                       fontsize=9)
        ax_i.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle(f"Bootstrap 95% Confidence Intervals  (n = {N_BOOTSTRAP} resamples)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_bootstrap_ci.png", dpi=200, bbox_inches="tight")
    plt.close(); print("  ✓ fig8_bootstrap_ci.png")

    # ── Fig 9: Line plot comparing Manual, Automated, and Calibrated Automated ──
    try:
        sl = float(metrics.get('reg_slope', np.nan))
        ic = float(metrics.get('reg_intercept', np.nan))
    except Exception:
        sl = np.nan
        ic = np.nan

    # Original automated mean and sd (already available)
    auto_mean = a_mean
    auto_sd = a_sd

    # Calibrated automated measurements: invert regression (automated -> manual)
    if np.isfinite(sl) and abs(sl) > 1e-8:
        auto_cal_mean = (auto_mean - ic) / sl
        # propagate SD by dividing by slope (intercept does not affect SD)
        # ensure arrays
        auto_cal_mean = np.array(auto_cal_mean, dtype=float)
        auto_cal_sd = np.array(auto_sd, dtype=float) / float(sl)
    else:
        # fallback to original if slope invalid
        auto_cal_mean = np.array(auto_mean, dtype=float)
        auto_cal_sd = np.array(auto_sd, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    # Manual: solid blue with circular markers
    ax.errorbar(days, m_mean, yerr=m_sd, fmt='o-', color='#1A5276',
                capsize=4, lw=1.8, ms=6, label='Manual (microscope)', zorder=4)
    # Automated original: dashed red with square markers
    ax.errorbar(days, auto_mean, yerr=auto_sd, fmt='s--', color='#C0392B',
                capsize=4, lw=1.8, ms=6, label='Automated (image analysis)', zorder=3)
    # Automated calibrated: dotted green with triangle markers
    ax.errorbar(days, auto_cal_mean, yerr=auto_cal_sd, fmt='^:', color='#17A589',
                capsize=4, lw=1.8, ms=6, label='Automated (calibrated)', zorder=5)

    ax.set_xlabel('Developmental Day')
    ax.set_ylabel('Body Length (mm)')
    ax.set_title('Body Length over Developmental Time:\nEffect of Calibration on Automated Measurements')
    ax.set_xticks(days)
    ax.set_xticklabels(day_lbl, fontsize=8, rotation=30, ha='right')
    ax.legend(frameon=False)
    # grid only on y-axis, light dashed
    ax.grid(axis='y', ls=':', alpha=0.5)
    # remove top/right spines (consistent with rcParams)
    try:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_line_manual_vs_auto_with_calibration.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_line_manual_vs_auto_with_calibration.png")

    # end of make_figures


def check_calibration(comp: pd.DataFrame) -> dict:
    """Check calibration by computing automated/manual ratios per row.

    Saves ratios to OUT_DIR/calibration_ratios.csv and prints a short summary.
    Returns a dict with summary stats.
    """
    comp = comp.copy()
    if "auto_over_manual" not in comp.columns:
        comp["auto_over_manual"] = comp["auto_mean_mm"] / comp["manual_mean_mm"].replace({0: np.nan})
    ratios = comp["auto_over_manual"].astype(float).values
    # Basic stats
    mean_r = float(np.nanmean(ratios))
    med_r = float(np.nanmedian(ratios))
    std_r = float(np.nanstd(ratios, ddof=1)) if len(ratios) > 1 else 0.0
    min_r = float(np.nanmin(ratios))
    max_r = float(np.nanmax(ratios))

    # flag days with >5% and >10% calibration deviation
    dev_pct = np.abs(ratios - 1.0) * 100.0
    n_gt5 = int(np.sum(dev_pct > 5.0))
    n_gt10 = int(np.sum(dev_pct > 10.0))

    summary = {
        "mean_ratio": mean_r,
        "median_ratio": med_r,
        "std_ratio": std_r,
        "min_ratio": min_r,
        "max_ratio": max_r,
        "n_points": int(len(ratios)),
        "n_gt5pct": n_gt5,
        "n_gt10pct": n_gt10,
    }

    print("\n── Calibration check (automated / manual ratios) ───────────────")
    print(f"  n points         : {summary['n_points']}")
    print(f"  mean ratio       : {mean_r:.4f}  median: {med_r:.4f}  std: {std_r:.4f}")
    print(f"  range            : [{min_r:.4f}, {max_r:.4f}]")
    print(f"  n days >5% dev   : {n_gt5}   n days >10% dev: {n_gt10}")

    # Save detailed table with ratio and percent deviation
    out = comp.copy()
    out['auto_over_manual'] = ratios
    out['calib_dev_pct'] = np.abs(out['auto_over_manual'] - 1.0) * 100.0
    out.to_csv(OUT_DIR / 'calibration_ratios.csv', index=False)
    print(f"  ✓ calibration_ratios.csv saved ({OUT_DIR / 'calibration_ratios.csv'})")

    return summary


def save_outputs(comp: pd.DataFrame, metrics: dict, tests: dict,
                 ct: dict, het: dict, boot: dict, outlier: dict):
    """Save comparison table, metrics summary, and LaTeX section.

    Keeps filenames unchanged from previous pipeline.
    """
    comp.round(4).to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print("  ✓ comparison_table.csv")

    # metrics_summary.txt
    sl = metrics.get("reg_slope", np.nan)
    ic = metrics.get("reg_intercept", np.nan)
    lines = [
        "=" * 70,
        "AGREEMENT METRICS  v2  (developmental-day alignment)",
        "=" * 70,
        f"  n data points              : {metrics.get('n', 'NA')}",
        f"  MAE                        : {metrics.get('MAE', float('nan')):.4f} mm",
        f"  RMSE                       : {metrics.get('RMSE', float('nan')):.4f} mm",
        f"  Normalised RMSE            : {metrics.get('nRMSE_pct', float('nan')):.2f} %",
        f"  Mean Bias (Auto−Manual)    : {metrics.get('MeanBias', float('nan')):+.4f} mm",
        f"  MAPE                       : {metrics.get('MAPE_pct', float('nan')):.2f} %",
        f"  Pearson r                  : {metrics.get('Pearson_r', float('nan')):.4f}  (p = {metrics.get('Pearson_p', float('nan')):.4f})",
        f"  Spearman rho               : {metrics.get('Spearman_rho', float('nan')):.4f}  (p = {metrics.get('Spearman_p', float('nan')):.4f})",
        f"  Lin's CCC                  : {metrics.get('CCC', float('nan')):.4f}",
        f"  Regression slope           : {sl:.4f}  95% CI [{metrics.get('reg_ci_slope', (float('nan'), float('nan')))[0]:.4f}, {metrics.get('reg_ci_slope', (float('nan'), float('nan')))[1]:.4f}]",
        f"    H₀: slope = 1            : p = {metrics.get('reg_p_slope1', float('nan')):.4f}",
        f"  Regression intercept       : {ic:.4f}  95% CI [{metrics.get('reg_ci_int', (float('nan'), float('nan')))[0]:.4f}, {metrics.get('reg_ci_int', (float('nan'), float('nan')))[1]:.4f}]",
        f"    H₀: intercept = 0        : p = {metrics.get('reg_p_int0', float('nan')):.4f}",
        f"  R²                         : {metrics.get('reg_R2', float('nan')):.4f}",
        "",
        "─" * 70,
        "CENTRAL TENDENCY ROBUSTNESS",
        "─" * 70,
    ]
    for name, r in ct.items():
        if name == 'best_estimator':
            continue
        lines.append(f"  {name:<16} MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  Bias={r['Bias']:+.4f}  MAPE={r['MAPE']:.2f}%  nRMSE={r['nRMSE_pct']:.2f}%")
    lines.append(f"  → Best estimator by MAE: {ct.get('best_estimator', 'NA')}")

    lines += [
        "",
        "─" * 70,
        "HETEROSCEDASTICITY & PROPORTIONAL BIAS",
        "─" * 70,
        f"  |diff| vs mean length      : rho = {het.get('r_het', float('nan')):.4f},  p = {het.get('p_het', float('nan')):.4f}",
        f"  diff  vs mean length       : rho = {het.get('r_prop', float('nan')):.4f}, p = {het.get('p_prop', float('nan')):.4f}",
        "",
        "─" * 70,
        f"BOOTSTRAP CIs  (n = {N_BOOTSTRAP} resamples)",
        "─" * 70,
        f"  MAE   95% CI               : [{boot.get('MAE_ci', (float('nan'), float('nan')))[0]:.4f}, {boot.get('MAE_ci', (float('nan'), float('nan')))[1]:.4f}] mm",
        f"  Bias  95% CI               : [{boot.get('Bias_ci', (float('nan'), float('nan')))[0]:.4f}, {boot.get('Bias_ci', (float('nan'), float('nan')))[1]:.4f}] mm",
        f"  Slope 95% CI               : [{boot.get('Slope_ci', (float('nan'), float('nan')))[0]:.4f}, {boot.get('Slope_ci', (float('nan'), float('nan')))[1]:.4f}]",
        "",
        "─" * 70,
        "OUTLIER SENSITIVITY  (top-5% extreme diffs removed)",
        "─" * 70,
        f"  n removed                  : {outlier.get('n_removed', 0)}  (|diff| > {outlier.get('threshold', float('nan')):.3f} mm)",
        f"  MAE   full / trimmed       : {outlier.get('full', {}).get('MAE', float('nan')):.4f} / {outlier.get('trimmed', {}).get('MAE', float('nan')):.4f} mm",
        f"  RMSE  full / trimmed       : {outlier.get('full', {}).get('RMSE', float('nan')):.4f} / {outlier.get('trimmed', {}).get('RMSE', float('nan')):.4f} mm",
        f"  Bias  full / trimmed       : {outlier.get('full', {}).get('Bias', float('nan')):+.4f} / {outlier.get('trimmed', {}).get('Bias', float('nan')):+.4f} mm",
        "",
        "─" * 70,
        "STATISTICAL TESTS",
        "─" * 70,
        f"  Shapiro–Wilk p-value       : {tests.get('Shapiro_p', float('nan')):.4f}  → {'normal' if tests.get('Normal', False) else 'non-normal'} differences",
        f"  Paired t-test              : t = {tests.get('PairedT_t', float('nan')):.3f},  p = {tests.get('PairedT_p', float('nan')):.4f}",
        f"  Wilcoxon signed-rank       : W = {tests.get('Wilcoxon_W', float('nan')):.1f},  p = {tests.get('Wilcoxon_p', float('nan')):.4f}",
        "",
        "BLAND–ALTMAN",
        f"  Mean bias                  : {tests.get('BA_MeanBias', float('nan')):+.4f} mm",
        f"  SD of differences          : {tests.get('BA_SD', float('nan')):.4f} mm",
        f"  95% LoA lower              : {tests.get('BA_LoA_Lo', float('nan')):.4f} mm",
        f"  95% LoA upper              : {tests.get('BA_LoA_Hi', float('nan')):.4f} mm",
        "=" * 70,
    ]

    (OUT_DIR / "metrics_summary.txt").write_text("\n".join(lines))
    print("  ✓ metrics_summary.txt")

    # Minimal LaTeX section
    sl = metrics.get('reg_slope', float('nan'))
    ic = metrics.get('reg_intercept', float('nan'))
    latex = rf"""
\subsection{{Comparison with Manual Microscope-Based Measurements}}

Regression slope = {sl:.3f}, intercept = {ic:.3f}.

"""
    (OUT_DIR / "latex_section.tex").write_text(latex.strip())
    print("  ✓ latex_section.tex")

    return None


def _gompertz_model(t, A, k, t0):
    """Gompertz model in the form requested:
    L(t) = A * exp(-exp(-k * (t - t0)))
    """
    t = np.asarray(t, dtype=float)
    return A * np.exp(-np.exp(-k * (t - t0)))


def _weighted_auto_by_day(auto_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted mean per developmental day using pipeline weighting.
       Uses dev_day present in auto_df (already set by load_auto_data).
       weight = 0.7 * valid_confidence + 0.3 * posture_confidence
       If posture_confidence is NaN, use valid_confidence only.
       Returns DataFrame with columns: dev_day, t (dev_day), mean_mm, sd_mm, n
    """
    df = auto_df.copy()
    def row_weight(r):
        v = float(r.get('valid_confidence', 0.0))
        p = r.get('posture_confidence', np.nan)
        if pd.notna(p):
            return 0.7 * v + 0.3 * float(p)
        return v
    df['__w__'] = df.apply(row_weight, axis=1).astype(float)

    rows = []
    for d, g in df.groupby('dev_day'):
        if pd.isna(d):
            continue
        w = g['__w__'].values.astype(float)
        L = g['body_length_mm'].astype(float).values
        sum_w = float(w.sum())
        if sum_w > 0:
            mean_w = float((w * L).sum() / sum_w)
            if len(L) > 1:
                var_w = float(((w * (L - mean_w) ** 2).sum()) / sum_w)
                sd_w = float(np.sqrt(max(var_w, 0.0)))
            else:
                sd_w = 0.0
        else:
            mean_w = float(L.mean())
            sd_w = float(L.std(ddof=0)) if len(L) > 1 else 0.0
        rows.append({'dev_day': int(d), 't': int(d), 'mean_mm': mean_w, 'sd_mm': sd_w, 'n': len(L)})
    out = pd.DataFrame(rows).sort_values('dev_day').reset_index(drop=True)
    return out


def _fit_gompertz(t: np.ndarray, L: np.ndarray):
    """Fit gompertz model and return popt and diagnostics (R2, RMSE).
    Uses robust initial guesses and bounds.
    """
    t = np.asarray(t, dtype=float)
    L = np.asarray(L, dtype=float)
    if len(t) < 3 or np.all(np.isnan(L)):
        return None

    # initial guesses
    A0 = float(np.nanmax(L) * 1.05) if np.nanmax(L) > 0 else 1.0
    k0 = 0.3
    t0_0 = float(np.median(t))
    p0 = [A0, k0, t0_0]
    bounds = ([0.0, 1e-6, t.min() - 5.0], [A0 * 10.0 if A0>0 else 100.0, 5.0, t.max() + 5.0])
    try:
        popt, pcov = curve_fit(_gompertz_model, t, L, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        # try without bounds
        try:
            popt, pcov = curve_fit(_gompertz_model, t, L, p0=p0, maxfev=20000)
        except Exception:
            return None
    # predictions and metrics
    L_hat = _gompertz_model(t, *popt)
    ss_res = np.sum((L - L_hat) ** 2)
    ss_tot = np.sum((L - np.nanmean(L)) ** 2)
    R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    RMSE = float(np.sqrt(np.mean((L - L_hat) ** 2)))
    return dict(popt=popt, pcov=pcov, R2=R2, RMSE=RMSE, L_hat=L_hat)


def run_gompertz_comparison(auto_df: pd.DataFrame, comp: pd.DataFrame):
    """Perform the full Gompertz comparison analysis and save outputs.
    Saves figure to outputs/figures/gompertz_comparison_publication.png and parameters to
    outputs/tables/gompertz_parameters.csv and a short Results text.
    """
    # prepare output dirs
    out_fig_dir = OUT_DIR / 'outputs' / 'figures'
    out_tab_dir = OUT_DIR / 'outputs' / 'tables'
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_tab_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: prepare time and means
    auto_w = _weighted_auto_by_day(auto_df)

    # Manual: use dev_day from comp directly as the time axis (simplified)
    manual_df = comp[['dev_day', 'manual_mean_mm', 'manual_sd_mm']].copy()
    manual_df = manual_df.rename(columns={'manual_mean_mm': 'mean_mm', 'manual_sd_mm': 'sd_mm'})
    manual_df['dev_day'] = manual_df['dev_day'].astype(int)
    manual_df['t'] = manual_df['dev_day'].astype(int)

    # Step 2: align by dev_day and fit Gompertz to each
    # Ensure both datasets use the same developmental days (intersection)
    days_auto = set(auto_w['dev_day'].astype(int).tolist()) if not auto_w.empty else set()
    days_man = set(manual_df['dev_day'].astype(int).tolist()) if not manual_df.empty else set()
    common_days = sorted(list(days_auto.intersection(days_man)))
    if len(common_days) == 0:
        print('  ⚠ No overlapping developmental days between automated and manual datasets — skipping Gompertz fit.')
        return dict(params=pd.DataFrame(rows), mad=float('nan'), fig='')

    auto_w = auto_w[auto_w['dev_day'].isin(common_days)].sort_values('dev_day').reset_index(drop=True)
    manual_df = manual_df[manual_df['dev_day'].isin(common_days)].sort_values('dev_day').reset_index(drop=True)

    t_auto = auto_w['dev_day'].values.astype(float)
    y_auto = auto_w['mean_mm'].values.astype(float)
    t_man = manual_df['dev_day'].values.astype(float)
    y_man = manual_df['mean_mm'].values.astype(float)

    # use specified p0 and bounds per instructions
    def _fit_gompertz_fixed(t, L):
        if len(t) < 3:
            return None
        p0 = [10.0, 0.5, 5.0]
        bounds = (0, [20.0, 5.0, 20.0])
        try:
            popt, pcov = curve_fit(_gompertz_model, t, L, p0=p0, bounds=bounds, maxfev=20000)
        except Exception:
            try:
                popt, pcov = curve_fit(_gompertz_model, t, L, p0=p0, maxfev=20000)
            except Exception:
                return None
        L_hat = _gompertz_model(t, *popt)
        ss_res = np.sum((L - L_hat) ** 2)
        ss_tot = np.sum((L - np.nanmean(L)) ** 2)
        R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
        RMSE = float(np.sqrt(np.mean((L - L_hat) ** 2)))
        return dict(popt=popt, pcov=pcov, R2=R2, RMSE=RMSE, L_hat=L_hat)

    res_auto = _fit_gompertz_fixed(t_auto, y_auto) if len(t_auto)>0 else None
    res_man  = _fit_gompertz_fixed(t_man, y_man) if len(t_man)>0 else None

    # Prepare parameters table
    rows = []
    def _row_from_res(source, res):
        if res is None:
            return dict(source=source, A=float('nan'), k=float('nan'), t0=float('nan'), R2=float('nan'), RMSE=float('nan'))
        popt = res['popt']
        return dict(source=source, A=float(popt[0]), k=float(popt[1]), t0=float(popt[2]), R2=float(res['R2']), RMSE=float(res['RMSE']))

    rows.append(_row_from_res('automated', res_auto))
    rows.append(_row_from_res('manual', res_man))
    params_df = pd.DataFrame(rows)
    params_df.to_csv(out_tab_dir / 'gompertz_parameters.csv', index=False)

    # Step 3: curve similarity and metrics
    t_min = min(np.nanmin(t_auto) if len(t_auto)>0 else np.nan, np.nanmin(t_man) if len(t_man)>0 else np.nan)
    t_max = max(np.nanmax(t_auto) if len(t_auto)>0 else np.nan, np.nanmax(t_man) if len(t_man)>0 else np.nan)
    if np.isnan(t_min) or np.isnan(t_max):
        t_fine = np.linspace(1, 16, 200)
    else:
        # keep fine grid within integer dev_day bounds
        t_fine = np.linspace(max(1, t_min), t_max, 200)

    if res_auto is not None:
        y_auto_f = _gompertz_model(t_fine, *res_auto['popt'])
    else:
        y_auto_f = np.full_like(t_fine, np.nan)
    if res_man is not None:
        y_man_f = _gompertz_model(t_fine, *res_man['popt'])
    else:
        y_man_f = np.full_like(t_fine, np.nan)

    mad = float(np.nanmean(np.abs(y_auto_f - y_man_f)))

    # Step 4: Publication-quality visualization
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # scatter points
    if len(t_auto)>0:
        ax.scatter(t_auto, y_auto, color='#1A5276', marker='o', s=40, label='Automated measurements')
    if len(t_man)>0:
        ax.scatter(t_man, y_man, color='#C0392B', marker='s', s=40, label='Manual microscope measurements')

    # fitted curves (solid)
    if res_auto is not None:
        ax.plot(t_fine, y_auto_f, color='#1A5276', lw=2.5, label='Gompertz fit (automated)')
    if res_man is not None:
        ax.plot(t_fine, y_man_f, color='#C0392B', lw=2.5, label='Gompertz fit (manual)')

    ax.set_xlabel('Developmental Day', fontsize=12)
    ax.set_ylabel('Body length (mm)', fontsize=12)
    ax.set_title('Gompertz model comparison — Automated vs Manual', fontsize=13)
    ax.grid(color='#E5E5E5', linestyle='-', linewidth=0.8)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    fig_path = out_fig_dir / 'gompertz_comparison_publication.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Step 5: Save textual results for manuscript
    txt_lines = []
    txt_lines.append('Gompertz model comparison — automated vs manual')
    txt_lines.append('')
    if res_auto is not None:
        pa = res_auto['popt']
        txt_lines.append(f'Automated fit: A={pa[0]:.3f} mm, k={pa[1]:.4f} day^-1, t0={pa[2]:.3f} days; R2={res_auto["R2"]:.3f}, RMSE={res_auto["RMSE"]:.3f} mm')
    else:
        txt_lines.append('Automated fit: failed')
    if res_man is not None:
        pm = res_man['popt']
        txt_lines.append(f'Manual fit   : A={pm[0]:.3f} mm, k={pm[1]:.4f} day^-1, t0={pm[2]:.3f} days; R2={res_man["R2"]:.3f}, RMSE={res_man["RMSE"]:.3f} mm')
    else:
        txt_lines.append('Manual fit: failed')

    txt_lines.append('')
    txt_lines.append(f'Curve similarity (mean absolute diff) = {mad:.3f} mm')
    txt_lines.append('')
    # Scientific interpretation
    if res_auto is not None and res_man is not None:
        txt_lines.append('Results:')
        txt_lines.append(f' The Gompertz model fit quality: automated R² = {res_auto["R2"]:.3f}; manual R² = {res_man["R2"]:.3f}.')
        txt_lines.append(f' Asymptotic size (A): automated = {pa[0]:.2f} mm; manual = {pm[0]:.2f} mm.')
        txt_lines.append(f' Growth rate (k): automated = {pa[1]:.4f} day⁻¹; manual = {pm[1]:.4f} day⁻¹.')
        txt_lines.append(f' Inflection time (t0): automated = {pa[2]:.2f} days; manual = {pm[2]:.2f} days.')
        txt_lines.append(' Interpretation: compare k and t0 to assess whether the automated pipeline preserves the temporal dynamics of growth; MAD gives curve-level deviation independent of scale.')
    else:
        txt_lines.append('Insufficient fits to provide full comparison.')

    (out_tab_dir / 'gompertz_results.txt').write_text('\n'.join(txt_lines))

    print(f'  ✓ {fig_path.name}')
    print(f'  ✓ gompertz_parameters.csv')
    print(f'  ✓ gompertz_results.txt')

    return dict(params=params_df, mad=mad, fig=str(fig_path))


# Insert call to run_gompertz_comparison inside main (after save_outputs)
def main():
    print("\n" + "=" * 70)
    print("MANUAL vs. AUTOMATED — Publication-Level Method Agreement Analysis v2")
    print("=" * 70)
    print(f"Predictions : {PREDS_FILE}")
    print(f"Output      : {OUT_DIR}")
    print("=" * 70)

    print("\nSTEP 0 — Loading and filtering automated predictions...")
    auto_df = load_auto_data()

    print("\nSTEP 1 — Building comparison table (mean + median + trimmed mean)...")
    comp = build_comparison(auto_df)

    print("\nSTEP 1b — Calibration check (automated / manual)…")
    calib_summary = check_calibration(comp)

    print("\nSTEP 2 — Core agreement metrics (MAE, RMSE, nRMSE, CCC, regression)...")
    metrics = compute_metrics(comp)

    print("\nSTEP 2b — Central tendency robustness...")
    ct = central_tendency_comparison(comp)

    print("\nSTEP 2c — Heteroscedasticity & proportional bias...")
    het = heteroscedasticity_analysis(comp)

    print("\nSTEP 2d — Regression validation (slope ≠ 1, intercept ≠ 0)...")
    regression_validation(metrics)

    print("\nSTEP 2e — Bootstrap confidence intervals...")
    boot = bootstrap_cis(comp)

    print("\nSTEP 2f — Outlier sensitivity analysis...")
    outlier = outlier_sensitivity(comp)

    print("\nSTEP 3 — Statistical testing + Bland–Altman...")
    tests = statistical_tests(comp)

    print("\nSTEP 4 — Generating publication figures (8 figures)...")
    make_figures(comp, metrics, tests, ct, het, boot, outlier)

    print("\nSTEP 5 — Saving all outputs...")
    save_outputs(comp, metrics, tests, ct, het, boot, outlier)

    print("\nSTEP 6 — Gompertz model comparison (auto vs manual)...")
    try:
        _ = run_gompertz_comparison(auto_df, comp)
    except Exception as e:
        print(f"  ⚠ Gompertz comparison failed: {e}")

    print("\n" + "=" * 70)
    print("DONE — all outputs saved to:")
    print(f"  {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

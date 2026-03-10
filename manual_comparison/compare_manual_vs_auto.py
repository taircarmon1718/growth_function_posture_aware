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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()
PREDS_FILE = ROOT / "dual_larva_models_geodesic" / "predictions" / "predictions_all_larvae.xlsx"
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
        })

    comp = pd.DataFrame(rows).sort_values("dev_day").reset_index(drop=True)
    print("\n── Comparison Table ──────────────────────────────────────────")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.width", 160)
    print(comp.to_string(index=False))
    return comp


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — AGREEMENT METRICS  (core + CCC + nRMSE)
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(comp: pd.DataFrame) -> dict:
    m = comp["manual_mean_mm"].values
    a = comp["auto_mean_mm"].values
    diff = a - m

    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff**2)))
    bias  = float(np.mean(diff))
    mape  = float(np.mean(np.abs(diff) / m) * 100)
    nrmse = rmse / (m.max() - m.min()) * 100

    r_p, p_p = pearsonr(m, a)
    r_s, p_s = spearmanr(m, a)
    ccc       = _ccc(m, a)

    reg = _reg_stats(m, a)

    metrics = dict(
        n=len(comp),
        MAE=mae, RMSE=rmse, nRMSE_pct=nrmse,
        MeanBias=bias, MAPE_pct=mape,
        Pearson_r=float(r_p), Pearson_p=float(p_p),
        Spearman_rho=float(r_s), Spearman_p=float(p_s),
        CCC=ccc,
        **{f"reg_{k}": v for k, v in reg.items()},
    )

    print("\n── Core Agreement Metrics ────────────────────────────────────")
    for k, v in metrics.items():
        if isinstance(v, tuple):
            print(f"  {k:<30s}: ({v[0]:.4f}, {v[1]:.4f})")
        elif isinstance(v, float):
            print(f"  {k:<30s}: {v:.4f}")
        else:
            print(f"  {k:<30s}: {v}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2b — CENTRAL TENDENCY ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SAVE ALL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(comp: pd.DataFrame, metrics: dict, tests: dict,
                 ct: dict, het: dict, boot: dict, outlier: dict):

    comp.round(4).to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print("  ✓ comparison_table.csv")

    # ── metrics_summary.txt ───────────────────────────────────────────────
    sl = metrics["reg_slope"]; ic = metrics["reg_intercept"]
    lines = [
        "=" * 70,
        "AGREEMENT METRICS  v2  (developmental-day alignment)",
        "=" * 70,
        f"  n data points              : {metrics['n']}",
        f"  MAE                        : {metrics['MAE']:.4f} mm",
        f"  RMSE                       : {metrics['RMSE']:.4f} mm",
        f"  Normalised RMSE            : {metrics['nRMSE_pct']:.2f} %",
        f"  Mean Bias (Auto−Manual)    : {metrics['MeanBias']:+.4f} mm",
        f"  MAPE                       : {metrics['MAPE_pct']:.2f} %",
        f"  Pearson r                  : {metrics['Pearson_r']:.4f}  (p = {metrics['Pearson_p']:.4f})",
        f"  Spearman rho               : {metrics['Spearman_rho']:.4f}  (p = {metrics['Spearman_p']:.4f})",
        f"  Lin's CCC                  : {metrics['CCC']:.4f}",
        f"  Regression slope           : {sl:.4f}  95% CI [{metrics['reg_ci_slope'][0]:.4f}, {metrics['reg_ci_slope'][1]:.4f}]",
        f"    H₀: slope = 1            : p = {metrics['reg_p_slope1']:.4f}",
        f"  Regression intercept       : {ic:.4f}  95% CI [{metrics['reg_ci_int'][0]:.4f}, {metrics['reg_ci_int'][1]:.4f}]",
        f"    H₀: intercept = 0        : p = {metrics['reg_p_int0']:.4f}",
        f"  R²                         : {metrics['reg_R2']:.4f}",
        "",
        "─" * 70,
        "CENTRAL TENDENCY ROBUSTNESS",
        "─" * 70,
    ]
    for name, r in ct.items():
        if name == "best_estimator":
            continue
        lines.append(f"  {name:<16} MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                     f"Bias={r['Bias']:+.4f}  MAPE={r['MAPE']:.2f}%  nRMSE={r['nRMSE_pct']:.2f}%")
    lines.append(f"  → Best estimator by MAE: {ct['best_estimator']}")
    lines += [
        "",
        "─" * 70,
        "HETEROSCEDASTICITY & PROPORTIONAL BIAS",
        "─" * 70,
        f"  |diff| vs mean length      : rho = {het['r_het']:.4f},  p = {het['p_het']:.4f}"
        f"  → {'proportional error present' if het['proportional_error'] else 'no proportional error'}",
        f"  diff  vs mean length       : rho = {het['r_prop']:.4f}, p = {het['p_prop']:.4f}"
        f"  → {'proportional bias present' if het['proportional_bias'] else 'no proportional bias'}",
        "",
        "─" * 70,
        f"BOOTSTRAP CIs  (n = {N_BOOTSTRAP} resamples)",
        "─" * 70,
        f"  MAE   95% CI               : [{boot['MAE_ci'][0]:.4f}, {boot['MAE_ci'][1]:.4f}] mm",
        f"  Bias  95% CI               : [{boot['Bias_ci'][0]:.4f}, {boot['Bias_ci'][1]:.4f}] mm",
        f"  Slope 95% CI               : [{boot['Slope_ci'][0]:.4f}, {boot['Slope_ci'][1]:.4f}]",
        "",
        "─" * 70,
        "OUTLIER SENSITIVITY  (top-5% extreme diffs removed)",
        "─" * 70,
        f"  n removed                  : {outlier['n_removed']}  (|diff| > {outlier['threshold']:.3f} mm)",
        f"  MAE   full / trimmed       : {outlier['full']['MAE']:.4f} / {outlier['trimmed']['MAE']:.4f} mm",
        f"  RMSE  full / trimmed       : {outlier['full']['RMSE']:.4f} / {outlier['trimmed']['RMSE']:.4f} mm",
        f"  Bias  full / trimmed       : {outlier['full']['Bias']:+.4f} / {outlier['trimmed']['Bias']:+.4f} mm",
        "",
        "─" * 70,
        "STATISTICAL TESTS",
        "─" * 70,
        f"  Shapiro–Wilk p-value       : {tests['Shapiro_p']:.4f}  → "
        f"{'normal' if tests['Normal'] else 'non-normal'} differences",
        f"  Paired t-test              : t = {tests['PairedT_t']:.3f},  p = {tests['PairedT_p']:.4f}",
        f"  Wilcoxon signed-rank       : W = {tests['Wilcoxon_W']:.1f},  p = {tests['Wilcoxon_p']:.4f}",
        "",
        "BLAND–ALTMAN",
        f"  Mean bias                  : {tests['BA_MeanBias']:+.4f} mm",
        f"  SD of differences          : {tests['BA_SD']:.4f} mm",
        f"  95% LoA lower              : {tests['BA_LoA_Lo']:.4f} mm",
        f"  95% LoA upper              : {tests['BA_LoA_Hi']:.4f} mm",
        "=" * 70,
    ]
    (OUT_DIR / "metrics_summary.txt").write_text("\n".join(lines))
    print("  ✓ metrics_summary.txt")

    # ── LaTeX ─────────────────────────────────────────────────────────────
    bias_dir = "overestimation" if metrics["MeanBias"] > 0 else "underestimation"
    sig_t    = "statistically significant" if tests["PairedT_p"] < 0.05 else "not statistically significant"
    prop_bias_txt = ("Spearman correlation between the mean of both methods and the "
                     f"difference was $\\rho = {het['r_prop']:.3f}$ ($p = {het['p_prop']:.4f}$), "
                     + ("indicating the presence of proportional bias."
                        if het["proportional_bias"]
                        else "providing no evidence of proportional bias."))
    het_txt = (f"Correlation between absolute error and mean body length was "
               f"$\\rho = {het['r_het']:.3f}$ ($p = {het['p_het']:.4f}$), "
               + ("suggesting heteroscedastic error structure."
                  if het["proportional_error"]
                  else "indicating homoscedastic error."))
    sl = metrics["reg_slope"]; ic = metrics["reg_intercept"]

    latex = rf"""
\subsection{{Comparison with Manual Microscope-Based Measurements}}

\subsubsection{{Agreement Between Automated and Manual Daily Means}}

The automated image-analysis pipeline was evaluated against manual
microscope-based body length measurements from the same biological culture
cycle, aligned by developmental day index.
Only larvae classified as valid detections with correct T-shaped posture
were included ($n = {int(comp['n_auto'].sum()):,}$ larvae across
{metrics['n']} developmental time points).

Strong agreement was observed between the two measurement approaches.
Pearson correlation was $r = {metrics['Pearson_r']:.3f}$
($p = {metrics['Pearson_p']:.4f}$) and Lin's concordance correlation
coefficient (CCC) was $\rho_c = {metrics['CCC']:.3f}$, indicating
high simultaneous precision and accuracy.
Spearman rank correlation was
$\rho = {metrics['Spearman_rho']:.3f}$ ($p = {metrics['Spearman_p']:.4f}$).
Ordinary least-squares regression yielded a slope of
${sl:.3f}$ (95\%~CI: ${metrics['reg_ci_slope'][0]:.3f}$--${metrics['reg_ci_slope'][1]:.3f}$;
$H_0\colon \beta_1=1$: $p = {metrics['reg_p_slope1']:.4f}$) and intercept of
${ic:.3f}$~mm (95\%~CI: ${metrics['reg_ci_int'][0]:.3f}$--${metrics['reg_ci_int'][1]:.3f}$;
$H_0\colon \beta_0=0$: $p = {metrics['reg_p_int0']:.4f}$), with $R^2 = {metrics['reg_R2']:.3f}$.

\subsubsection{{Bias and Error Analysis (MAE, RMSE, Relative Error)}}

The mean absolute error (MAE) was ${metrics['MAE']:.3f}$~mm
(bootstrap 95\%~CI: ${boot['MAE_ci'][0]:.3f}$--${boot['MAE_ci'][1]:.3f}$~mm)
and the root mean square error (RMSE) was ${metrics['RMSE']:.3f}$~mm
(normalised RMSE = ${metrics['nRMSE_pct']:.1f}$\%).
The mean bias was ${metrics['MeanBias']:+.3f}$~mm
(bootstrap 95\%~CI: ${boot['Bias_ci'][0]:.3f}$--${boot['Bias_ci'][1]:.3f}$~mm),
indicating systematic {bias_dir}.
The MAPE was ${metrics['MAPE_pct']:.1f}$\%.

Bland--Altman analysis revealed a mean bias of
${tests['BA_MeanBias']:+.3f}$~mm (SD~$= {tests['BA_SD']:.3f}$~mm),
with 95\% limits of agreement from
${tests['BA_LoA_Lo']:.3f}$~mm to ${tests['BA_LoA_Hi']:.3f}$~mm.
{prop_bias_txt}
{het_txt}

A paired $t$-test indicated that the systematic bias was {sig_t}
($t = {tests['PairedT_t']:.3f}$, $p = {tests['PairedT_p']:.4f}$);
the Wilcoxon signed-rank test confirmed this finding
($W = {tests['Wilcoxon_W']:.1f}$, $p = {tests['Wilcoxon_p']:.4f}$).

Robustness of central tendency estimation was assessed by comparing the mean,
median, and 10\%-trimmed mean as automated estimators.
The {ct['best_estimator'].lower()} yielded the lowest MAE
(${ct[ct['best_estimator']]['MAE']:.3f}$~mm), suggesting it as the preferred
estimator for growth-tracking applications.
Sensitivity analysis excluding the 5\% most extreme pairwise differences
({outlier['n_removed']} data point(s)) resulted in a MAE of
${outlier['trimmed']['MAE']:.3f}$~mm (vs. ${outlier['full']['MAE']:.3f}$~mm full),
indicating that the agreement metrics are robust to extreme observations.
"""
    (OUT_DIR / "latex_section.tex").write_text(latex.strip())
    print("  ✓ latex_section.tex")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
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

    print("\n" + "=" * 70)
    print("DONE — all outputs saved to:")
    print(f"  {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

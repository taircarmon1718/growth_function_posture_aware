#!/usr/bin/env python3
"""
growth_trajectory_dynamics.py
==============================
GROWTH TRAJECTORY DYNAMICS ANALYSIS

Evaluates whether the automated system reproduces the SAME GROWTH DYNAMICS
over developmental time — independent of any absolute scale bias.

Goal: NOT absolute agreement.
Goal: Do the two systems track the same biological growth pattern?

Analyses:
  1. Offset-normalised trajectory (bias removed)
  2. Day-to-day growth increments (first differences)
  3. Relative Growth Rate (RGR)
  4. Gompertz growth model fit comparison

All outputs saved to: manual_comparison/

Data source:
  dual_larva_models/predictions/predictions_all_larvae.xlsx
  Filter: predicted_valid == 1  AND  predicted_posture == 1
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, trim_mean
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.resolve()
PREDS_FILE = ROOT / "dual_larva_models" / "predictions" / "predictions_all_larvae.xlsx"
OUT_DIR    = Path(__file__).parent.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
RNG_SEED    = 42

# ── Publication style ──────────────────────────────────────────────────────────
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

COL_MAN  = "#1A5276"   # dark blue  – manual
COL_AUTO = "#C0392B"   # dark red   – automated


def _safe_pearsonr(x, y):
    """Return (r, p) or (nan, nan) if either array is constant."""
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    return pearsonr(x, y)

# ══════════════════════════════════════════════════════════════════════════════
#  DEVELOPMENTAL DAY MAPPINGS  (identical to compare_manual_vs_auto.py)
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
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_comparison_table() -> pd.DataFrame:
    """Rebuild the per-day comparison table (same logic as main script).

    Note: duplicate dev_day rows (two manual entries mapping to the same
    automated day, e.g. Day 16) are averaged so that the series used for
    first-difference analysis is strictly monotone in dev_day.
    """
    df = pd.read_excel(PREDS_FILE)
    df["date"] = df["date"].astype(str).str.strip()
    filtered = df[(df["predicted_valid"] == 1) &
                  (df["predicted_posture"] == 1)].copy()
    filtered["dev_day"] = filtered["date"].map(AUTO_DATE_TO_DAY)
    filtered = filtered.dropna(subset=["dev_day"])
    filtered["dev_day"] = filtered["dev_day"].astype(int)

    day_stats = (
        filtered.groupby("dev_day")["body_length_mm"]
        .agg(auto_mean="mean", auto_sd="std", n_auto="count")
        .reset_index()
    )
    d2s = day_stats.set_index("dev_day").to_dict(orient="index")

    rows = []
    for label, days, man_mean, man_sd in MANUAL_ENTRIES:
        hits = [d2s[d] for d in days if d in d2s]
        if not hits:
            continue
        auto_mean = float(np.mean([h["auto_mean"] for h in hits]))
        auto_sd   = float(np.sqrt(np.mean([h["auto_sd"]**2 for h in hits])))
        n_total   = int(np.sum([h["n_auto"] for h in hits]))
        rows.append({
            "manual_date":    label,
            "dev_day":        days[0],
            "manual_mean_mm": man_mean,
            "manual_sd_mm":   man_sd,
            "auto_mean_mm":   auto_mean,
            "auto_sd_mm":     auto_sd,
            "n_auto":         n_total,
        })

    comp = pd.DataFrame(rows).sort_values("dev_day").reset_index(drop=True)

    # Average duplicate dev_day rows so diff-based analyses work correctly
    comp = (comp.groupby("dev_day", as_index=False)
            .agg({
                "manual_date":    "first",
                "manual_mean_mm": "mean",
                "manual_sd_mm":   "mean",
                "auto_mean_mm":   "mean",
                "auto_sd_mm":     "mean",
                "n_auto":         "sum",
            })
            .sort_values("dev_day").reset_index(drop=True))

    print(f"  Loaded {len(comp)} unique developmental time points.")
    return comp


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — OFFSET-NORMALISED TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
def normalised_trajectory(comp: pd.DataFrame) -> dict:
    days  = comp["dev_day"].values.astype(float)
    m     = comp["manual_mean_mm"].values
    a     = comp["auto_mean_mm"].values
    m_sd  = comp["manual_sd_mm"].values
    a_sd  = comp["auto_sd_mm"].values

    # Subtract Day-1 value from each series
    m_norm = m - m[0]
    a_norm = a - a[0]

    diff    = a_norm - m_norm
    mae     = float(np.mean(np.abs(diff)))
    rmse    = float(np.sqrt(np.mean(diff**2)))
    r, p    = _safe_pearsonr(m_norm, a_norm)
    day_lbl = [f"Day {int(d)}" for d in days]

    print(f"\n  [1] Normalised trajectory: r={r:.4f}, MAE={mae:.4f} mm, RMSE={rmse:.4f} mm")

    # ── Figure 9 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(days, m_norm - m_sd, m_norm + m_sd,
                    alpha=0.15, color=COL_MAN)
    ax.fill_between(days, a_norm - a_sd, a_norm + a_sd,
                    alpha=0.12, color=COL_AUTO)
    ax.plot(days, m_norm, "o-", color=COL_MAN, lw=2, ms=7,
            label="Manual (normalised)")
    ax.plot(days, a_norm, "s--", color=COL_AUTO, lw=2, ms=6,
            label="Automated (normalised)")
    ax.set_xlabel("Developmental Day")
    ax.set_ylabel("Body Length Increment from Day 1 (mm)")
    ax.set_title("Offset-Normalised Growth Trajectory\n"
                 "(Day-1 value subtracted from each series)")
    ax.set_xticks(days)
    ax.set_xticklabels(day_lbl, fontsize=8, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.text(0.02, 0.97,
            f"Pearson r = {r:.3f}   MAE = {mae:.3f} mm   RMSE = {rmse:.3f} mm",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F3F4", alpha=0.85))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig9_normalized_growth.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig9_normalized_growth.png")

    return dict(r=r, p=p, MAE=mae, RMSE=rmse,
                m_norm=m_norm, a_norm=a_norm, days=days)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DAY-TO-DAY GROWTH INCREMENTS
# ══════════════════════════════════════════════════════════════════════════════
def growth_increments(comp: pd.DataFrame) -> dict:
    days = comp["dev_day"].values.astype(float)
    m    = comp["manual_mean_mm"].values
    a    = comp["auto_mean_mm"].values

    # First differences — weighted by actual day gap
    day_gaps = np.diff(days)
    dm = np.diff(m)
    da = np.diff(a)

    # Per-day increments (mm/day)
    dm_per_day = dm / day_gaps
    da_per_day = da / day_gaps

    diff  = da_per_day - dm_per_day
    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff**2)))
    r, p  = _safe_pearsonr(dm_per_day, da_per_day)

    intervals = [f"D{int(days[i])}→D{int(days[i+1])}" for i in range(len(days)-1)]
    print(f"\n  [2] Growth increments: r={r:.4f}, MAE={mae:.4f} mm/day, RMSE={rmse:.4f} mm/day")

    # ── Figure 10 ─────────────────────────────────────────────────────────
    x     = np.arange(len(intervals))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars_m = ax.bar(x - width/2, dm_per_day, width, color=COL_MAN,
                    label="Manual", edgecolor="white", lw=0.6, alpha=0.85)
    bars_a = ax.bar(x + width/2, da_per_day, width, color=COL_AUTO,
                    label="Automated", edgecolor="white", lw=0.6, alpha=0.85)
    for bar, val in zip(bars_m, dm_per_day):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.03 if val >= 0 else -0.08),
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=7.5, color=COL_MAN)
    for bar, val in zip(bars_a, da_per_day):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.03 if val >= 0 else -0.08),
                f"{val:.2f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=7.5, color=COL_AUTO)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(intervals, rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("Day Interval")
    ax.set_ylabel("Growth Increment (mm / day)")
    ax.set_title("Day-to-Day Growth Increments: Manual vs. Automated\n"
                 "(mm per developmental day, accounting for unequal intervals)")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.text(0.02, 0.97,
            f"Pearson r = {r:.3f}   MAE = {mae:.3f} mm/day",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F3F4", alpha=0.85))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig10_growth_increments.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig10_growth_increments.png")

    return dict(r=r, p=p, MAE=mae, RMSE=rmse,
                dm=dm_per_day, da=da_per_day,
                intervals=intervals, days=days)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — RELATIVE GROWTH RATE (RGR)
# ══════════════════════════════════════════════════════════════════════════════
def relative_growth_rate(comp: pd.DataFrame) -> dict:
    days = comp["dev_day"].values.astype(float)
    m    = comp["manual_mean_mm"].values
    a    = comp["auto_mean_mm"].values

    day_gaps    = np.diff(days)
    # RGR = (L(t) - L(t-1)) / (L(t-1) * Δt)   [fractional growth per day]
    rgr_m = np.diff(m)   / (m[:-1] * day_gaps)
    rgr_a = np.diff(a)   / (a[:-1] * day_gaps)

    diff  = rgr_a - rgr_m
    mae   = float(np.mean(np.abs(diff)))
    r, p  = _safe_pearsonr(rgr_m, rgr_a)

    mid_days  = [(days[i] + days[i+1]) / 2 for i in range(len(days)-1)]
    intervals = [f"D{int(days[i])}→D{int(days[i+1])}" for i in range(len(days)-1)]
    print(f"\n  [3] Relative Growth Rate: r={r:.4f}, MAE={mae:.4f} day⁻¹")

    # ── Figure 11 ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(mid_days, rgr_m, "o-", color=COL_MAN, lw=2, ms=7,
            label="RGR Manual")
    ax.plot(mid_days, rgr_a, "s--", color=COL_AUTO, lw=2, ms=6,
            label="RGR Automated")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Developmental Day (midpoint of interval)")
    ax.set_ylabel("Relative Growth Rate (day⁻¹)")
    ax.set_title("Relative Growth Rate: Manual vs. Automated\n"
                 "RGR(t) = ΔL(t) / [L(t−1) · Δt]")
    ax.set_xticks(mid_days)
    ax.set_xticklabels(intervals, rotation=40, ha="right", fontsize=8)
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.text(0.02, 0.97,
            f"Pearson r = {r:.3f}   MAE = {mae:.4f} day⁻¹",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F3F4", alpha=0.85))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig11_relative_growth_rate.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig11_relative_growth_rate.png")

    return dict(r=r, p=p, MAE=mae,
                rgr_m=rgr_m, rgr_a=rgr_a, intervals=intervals)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GOMPERTZ GROWTH MODEL FIT
# ══════════════════════════════════════════════════════════════════════════════
def _gompertz(t, A, B, k):
    """L(t) = A * exp(-B * exp(-k * t))"""
    return A * np.exp(-B * np.exp(-k * t))


def _bootstrap_gompertz(t, L, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap 95% CI for Gompertz parameters via residual resampling."""
    rng   = np.random.default_rng(seed)
    try:
        popt, _ = curve_fit(_gompertz, t, L, p0=[max(L)*1.2, 3.0, 0.1],
                             maxfev=10000, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    except Exception:
        return None, None, None

    resid = L - _gompertz(t, *popt)
    boots = []
    for _ in range(n_boot):
        L_boot = _gompertz(t, *popt) + rng.choice(resid, size=len(resid), replace=True)
        try:
            p_b, _ = curve_fit(_gompertz, t, L_boot, p0=popt,
                                maxfev=5000,
                                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            boots.append(p_b)
        except Exception:
            continue

    if len(boots) < 50:
        return popt, None, None

    boots  = np.array(boots)
    ci_lo  = np.percentile(boots, 2.5, axis=0)
    ci_hi  = np.percentile(boots, 97.5, axis=0)
    return popt, ci_lo, ci_hi


def gompertz_fit(comp: pd.DataFrame) -> dict:
    days = comp["dev_day"].values.astype(float)
    m    = comp["manual_mean_mm"].values
    a    = comp["auto_mean_mm"].values

    print("\n  [4] Fitting Gompertz model ...")
    p_m, ci_lo_m, ci_hi_m = _bootstrap_gompertz(days, m)
    p_a, ci_lo_a, ci_hi_a = _bootstrap_gompertz(days, a)

    if p_m is None or p_a is None:
        print("  ⚠  Gompertz fit failed — skipping figure 12.")
        return {}

    t_fine = np.linspace(days.min(), days.max() * 1.1, 300)

    for name, popt, ci_lo, ci_hi in [
        ("Manual",    p_m, ci_lo_m, ci_hi_m),
        ("Automated", p_a, ci_lo_a, ci_hi_a),
    ]:
        ci_str = ""
        if ci_lo is not None:
            ci_str = (f"   95% CI:  A=[{ci_lo[0]:.2f},{ci_hi[0]:.2f}]  "
                      f"k=[{ci_lo[2]:.4f},{ci_hi[2]:.4f}]  "
                      f"B=[{ci_lo[1]:.3f},{ci_hi[1]:.3f}]")
        print(f"  {name:10s}: A={popt[0]:.3f}  k={popt[2]:.4f}  B={popt[1]:.3f}{ci_str}")

    # ── Figure 12 ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overlaid fitted curves
    ax = axes[0]
    ax.scatter(days, m, color=COL_MAN,  s=55, zorder=4, label="Manual data")
    ax.scatter(days, a, color=COL_AUTO, s=55, zorder=4, marker="s", label="Auto data")
    y_m = _gompertz(t_fine, *p_m)
    y_a = _gompertz(t_fine, *p_a)
    ax.plot(t_fine, y_m, color=COL_MAN,  lw=2,
            label=f"Gompertz Manual\nA={p_m[0]:.2f}, k={p_m[2]:.4f}")
    ax.plot(t_fine, y_a, color=COL_AUTO, lw=2, ls="--",
            label=f"Gompertz Auto\nA={p_a[0]:.2f}, k={p_a[2]:.4f}")
    if ci_lo_m is not None:
        ax.fill_between(t_fine,
                        _gompertz(t_fine, ci_lo_m[0], ci_lo_m[1], ci_lo_m[2]),
                        _gompertz(t_fine, ci_hi_m[0], ci_hi_m[1], ci_hi_m[2]),
                        alpha=0.12, color=COL_MAN)
    if ci_lo_a is not None:
        ax.fill_between(t_fine,
                        _gompertz(t_fine, ci_lo_a[0], ci_lo_a[1], ci_lo_a[2]),
                        _gompertz(t_fine, ci_hi_a[0], ci_hi_a[1], ci_hi_a[2]),
                        alpha=0.10, color=COL_AUTO)
    ax.set_xlabel("Developmental Day")
    ax.set_ylabel("Body Length (mm)")
    ax.set_title("Gompertz Growth Model Fit\nManual vs. Automated")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)

    # Right: parameter comparison bar chart
    ax2 = axes[1]
    params   = ["A (asymptote, mm)", "k (growth rate)", "B (displacement)"]
    vals_m   = [p_m[0], p_m[2], p_m[1]]
    vals_a   = [p_a[0], p_a[2], p_a[1]]
    x        = np.arange(len(params))
    w        = 0.35
    bars_m2  = ax2.bar(x - w/2, vals_m, w, color=COL_MAN,  label="Manual",    alpha=0.85, edgecolor="white")
    bars_a2  = ax2.bar(x + w/2, vals_a, w, color=COL_AUTO, label="Automated", alpha=0.85, edgecolor="white")
    # CI error bars — order: [A, k, B] to match params list
    if ci_lo_m is not None:
        vm   = np.array([p_m[0], p_m[2], p_m[1]])
        lo_m = np.array([ci_lo_m[0], ci_lo_m[2], ci_lo_m[1]])
        hi_m = np.array([ci_hi_m[0], ci_hi_m[2], ci_hi_m[1]])
        err_m = np.vstack([np.clip(vm - lo_m, 0, None),
                           np.clip(hi_m - vm, 0, None)])
        ax2.errorbar(x - w/2, vm, yerr=err_m, fmt="none",
                     color="black", capsize=5, lw=1.5)
    if ci_lo_a is not None:
        va   = np.array([p_a[0], p_a[2], p_a[1]])
        lo_a = np.array([ci_lo_a[0], ci_lo_a[2], ci_lo_a[1]])
        hi_a = np.array([ci_hi_a[0], ci_hi_a[2], ci_hi_a[1]])
        err_a = np.vstack([np.clip(va - lo_a, 0, None),
                           np.clip(hi_a - va, 0, None)])
        ax2.errorbar(x + w/2, va, yerr=err_a, fmt="none",
                     color="black", capsize=5, lw=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(params, fontsize=8.5)
    ax2.set_ylabel("Parameter Value")
    ax2.set_title("Gompertz Parameter Comparison\n(error bars = 95% bootstrap CI)")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig12_gompertz_fit.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig12_gompertz_fit.png")

    return dict(
        manual=dict(A=p_m[0], k=p_m[2], B=p_m[1],
                    ci_lo=ci_lo_m, ci_hi=ci_hi_m),
        auto=dict(A=p_a[0], k=p_a[2], B=p_a[1],
                  ci_lo=ci_lo_a, ci_hi=ci_hi_a),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def save_dynamics_summary(norm: dict, inc: dict, rgr: dict, gomp: dict):
    dyn_preserved = (norm["r"] > 0.90 and inc["r"] > 0.80)

    lines = [
        "=" * 70,
        "GROWTH TRAJECTORY DYNAMICS ANALYSIS",
        "=" * 70,
        "",
        "1. OFFSET-NORMALISED TRAJECTORY (bias removed)",
        "─" * 70,
        f"   Pearson r          : {norm['r']:.4f}  (p = {norm['p']:.4f})",
        f"   MAE                : {norm['MAE']:.4f} mm",
        f"   RMSE               : {norm['RMSE']:.4f} mm",
        "",
        "2. DAY-TO-DAY GROWTH INCREMENTS",
        "─" * 70,
        f"   Pearson r          : {inc['r']:.4f}  (p = {inc['p']:.4f})",
        f"   MAE                : {inc['MAE']:.4f} mm/day",
        f"   RMSE               : {inc['RMSE']:.4f} mm/day",
        "",
        "   Increment table (mm/day):",
        f"   {'Interval':<18} {'ΔManual':>10} {'ΔAuto':>10} {'Δ Error':>10}",
    ]
    for iv, dm, da in zip(inc["intervals"], inc["dm"], inc["da"]):
        lines.append(f"   {iv:<18} {dm:10.4f} {da:10.4f} {da-dm:+10.4f}")

    lines += [
        "",
        "3. RELATIVE GROWTH RATE",
        "─" * 70,
        f"   Pearson r          : {rgr['r']:.4f}  (p = {rgr['p']:.4f})",
        f"   MAE                : {rgr['MAE']:.6f} day⁻¹",
        "",
        "   RGR table (day⁻¹):",
        f"   {'Interval':<18} {'RGR Manual':>12} {'RGR Auto':>12} {'Δ':>12}",
    ]
    for iv, rm, ra in zip(rgr["intervals"], rgr["rgr_m"], rgr["rgr_a"]):
        lines.append(f"   {iv:<18} {rm:12.6f} {ra:12.6f} {ra-rm:+12.6f}")

    if gomp:
        gm = gomp["manual"]; ga = gomp["auto"]
        ci_m = ""
        ci_a = ""
        if gm["ci_lo"] is not None:
            ci_m = (f"   A 95% CI: [{gm['ci_lo'][0]:.3f}, {gm['ci_hi'][0]:.3f}]  "
                    f"k 95% CI: [{gm['ci_lo'][2]:.4f}, {gm['ci_hi'][2]:.4f}]")
            ci_a = (f"   A 95% CI: [{ga['ci_lo'][0]:.3f}, {ga['ci_hi'][0]:.3f}]  "
                    f"k 95% CI: [{ga['ci_lo'][2]:.4f}, {ga['ci_hi'][2]:.4f}]")
        lines += [
            "",
            "4. GOMPERTZ GROWTH MODEL",
            "─" * 70,
            f"   Manual  : A={gm['A']:.3f} mm,  k={gm['k']:.4f} day⁻¹,  B={gm['B']:.3f}",
            ci_m,
            f"   Automated: A={ga['A']:.3f} mm,  k={ga['k']:.4f} day⁻¹,  B={ga['B']:.3f}",
            ci_a,
            f"   ΔA (Auto−Manual) : {ga['A']-gm['A']:+.3f} mm",
            f"   Δk (Auto−Manual) : {ga['k']-gm['k']:+.4f} day⁻¹",
        ]

    lines += [
        "",
        "=" * 70,
        "INTERPRETATION",
        "=" * 70,
        "",
    ]

    # Auto-generate interpretation
    interp = []
    if norm["r"] > 0.90:
        interp.append(
            f"The normalised growth trajectories show very high correlation "
            f"(r = {norm['r']:.3f}), indicating that after removing the "
            f"constant scale offset the two systems track the same biological "
            f"growth pattern across developmental time."
        )
    else:
        interp.append(
            f"The normalised trajectories show moderate correlation "
            f"(r = {norm['r']:.3f}), suggesting partial divergence in "
            f"growth dynamics beyond the mean bias."
        )

    if inc["r"] > 0.80:
        interp.append(
            f"Day-to-day growth increments are strongly correlated "
            f"(r = {inc['r']:.3f}, MAE = {inc['MAE']:.3f} mm/day), "
            f"confirming that the automated system captures similar growth "
            f"velocity patterns to manual microscopy."
        )
    else:
        interp.append(
            f"Day-to-day growth increments show weaker correlation "
            f"(r = {inc['r']:.3f}), indicating that growth velocity "
            f"estimates differ between methods."
        )

    if dyn_preserved:
        interp.append(
            "CONCLUSION: Both the normalised trajectory correlation (> 0.90) "
            "and the increment correlation (> 0.80) exceed the preset thresholds. "
            "Growth dynamics are considered PRESERVED by the automated system. "
            "The systematic offset observed in absolute measurements reflects a "
            "scale difference between methods, not a difference in growth pattern."
        )
    else:
        interp.append(
            "CONCLUSION: One or more thresholds were not met "
            f"(norm r = {norm['r']:.3f}, incr r = {inc['r']:.3f}). "
            "Growth dynamics may not be fully preserved."
        )

    for para in interp:
        # word-wrap at ~70 chars
        words  = para.split()
        line   = "   "
        for w in words:
            if len(line) + len(w) + 1 > 73:
                lines.append(line)
                line = "   " + w + " "
            else:
                line += w + " "
        lines.append(line)
        lines.append("")

    lines.append("=" * 70)

    out_path = OUT_DIR / "growth_dynamics_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"  ✓ growth_dynamics_summary.txt")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("GROWTH TRAJECTORY DYNAMICS ANALYSIS")
    print("=" * 70)
    print(f"Predictions : {PREDS_FILE}")
    print(f"Output      : {OUT_DIR}")
    print("=" * 70)

    print("\nLoading comparison data...")
    comp = load_comparison_table()

    print("\nSECTION 1 — Offset-normalised trajectory...")
    norm = normalised_trajectory(comp)

    print("\nSECTION 2 — Day-to-day growth increments...")
    inc = growth_increments(comp)

    print("\nSECTION 3 — Relative Growth Rate (RGR)...")
    rgr = relative_growth_rate(comp)

    print("\nSECTION 4 — Gompertz growth model fit...")
    gomp = gompertz_fit(comp)

    print("\nSaving summary...")
    save_dynamics_summary(norm, inc, rgr, gomp)

    print("\n" + "=" * 70)
    print("GROWTH TRAJECTORY DYNAMICS ANALYSIS COMPLETE")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()


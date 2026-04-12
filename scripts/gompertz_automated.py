#!/usr/bin/env python3
"""
Fit and plot a Gompertz curve for automated measurements only.
Saves:
 - outputs/figures/gompertz_automated.png
 - outputs/tables/gompertz_automated_params.csv

Logic:
 - Load predictions from dual_larva_models_geodesic2/predictions/predictions_all_larvae.xlsx
 - Filter: predicted_valid == 1 AND predicted_posture == 1 (robust to column name variants)
 - Compute weights per row: w = 0.7 * valid_confidence + 0.3 * posture_confidence
   if posture_confidence is NaN, use valid_confidence
 - Group by date, sort by date using date_sort_key (day.month)
 - For each date compute weighted mean and weighted std (population style)
 - Map dates to dev_day index 1..N and fit Gompertz: L(t)=A*exp(-exp(-k*(t-t0))) using curve_fit
 - Save plot and params CSV
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import warnings

ROOT = Path(__file__).resolve().parents[1]
PRED_XLSX = ROOT / 'dual_larva_models_geodesic2' / 'predictions' / 'predictions_all_larvae.xlsx'
OUT_FIG = ROOT / 'outputs' / 'figures' / 'gompertz_automated.png'
OUT_CSV = ROOT / 'outputs' / 'tables' / 'gompertz_automated_params.csv'
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Helper: date sort key for strings like '19.10' -> (month, day)
def date_sort_key(name):
    try:
        day, mon = str(name).split('.')
        return (int(mon), int(day))
    except Exception:
        return (999, 999)

# Weighted stats
def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    if np.sum(weights[mask]) == 0:
        return np.nan
    return float(np.sum(weights[mask] * values[mask]) / np.sum(weights[mask]))

def weighted_std(values, weights, mean=None):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    if mean is None:
        mean = weighted_mean(values, weights)
    denom = np.sum(weights[mask])
    if denom == 0:
        return np.nan
    var = float(np.sum(weights[mask] * (values[mask] - mean) ** 2) / denom)
    return float(np.sqrt(var))

# Gompertz model
def gompertz(t, A, k, t0):
    return A * np.exp(-np.exp(-k * (t - t0)))


def main():
    if not PRED_XLSX.exists():
        raise FileNotFoundError(f"Predictions file not found: {PRED_XLSX}")
    df = pd.read_excel(PRED_XLSX, engine='openpyxl')

    # detect relevant columns robustly
    cols_lower = {c.lower(): c for c in df.columns}
    col_date = cols_lower.get('date') or cols_lower.get('image_date') or cols_lower.get('img_date')
    col_body_mm = cols_lower.get('body_length_mm') or cols_lower.get('body_length') or cols_lower.get('length_mm')
    col_pred_valid = cols_lower.get('predicted_valid') or cols_lower.get('valid_pred') or cols_lower.get('pred_valid')
    col_pred_posture = cols_lower.get('predicted_posture') or cols_lower.get('posture_pred') or cols_lower.get('pred_posture')
    col_valid_conf = cols_lower.get('valid_confidence') or cols_lower.get('v_conf') or cols_lower.get('valid_conf')
    col_posture_conf = cols_lower.get('posture_confidence') or cols_lower.get('posture_conf') or cols_lower.get('p_conf')

    if col_date is None or col_body_mm is None:
        raise RuntimeError('Required columns not found in predictions file (date, body_length_mm)')

    # Ensure presence of predicted flags; if missing, try to infer from columns named 'predicted' variants
    if col_pred_valid is None or col_pred_posture is None:
        # attempt to find boolean-like columns
        # If not present, raise
        raise RuntimeError('Predicted validity/posture columns not found in predictions file')

    # Filter automated valid & posture
    mask_valid = df[col_pred_valid].astype(float) == 1
    mask_posture = df[col_pred_posture].astype(float) == 1
    df_auto = df[mask_valid & mask_posture].copy()

    if df_auto.empty:
        raise RuntimeError('No automated rows after filtering predicted_valid==1 AND predicted_posture==1')

    # compute weights
    def compute_weight(row):
        vconf = row[col_valid_conf] if col_valid_conf in df.columns else np.nan
        pconf = row[col_posture_conf] if col_posture_conf in df.columns else np.nan
        try:
            v = float(vconf)
        except Exception:
            v = np.nan
        try:
            p = float(pconf)
        except Exception:
            p = np.nan
        if np.isnan(p):
            if np.isnan(v):
                return 1.0
            return float(v)
        if np.isnan(v):
            return float(p)
        return float(0.7 * v + 0.3 * p)

    df_auto['weight'] = df_auto.apply(compute_weight, axis=1)

    # group by date
    df_auto['date_fixed'] = df_auto[col_date].apply(lambda x: str(x).strip())
    grouped = df_auto.groupby('date_fixed')

    dates = []
    dev_days = []
    means = []
    stds = []
    ns = []

    # sort date keys
    date_keys = sorted(grouped.groups.keys(), key=date_sort_key)
    for idx, d in enumerate(date_keys, start=1):
        g = grouped.get_group(d)
        vals = g[col_body_mm].astype(float).values
        w = g['weight'].astype(float).values
        mean = weighted_mean(vals, w)
        std = weighted_std(vals, w, mean=mean)
        dates.append(d)
        dev_days.append(idx)
        means.append(mean)
        stds.append(std)
        ns.append(len(g))

    t = np.array(dev_days, dtype=float)
    y = np.array(means, dtype=float)
    yerr = np.array(stds, dtype=float)

    # Fit Gompertz
    try:
        from scipy.optimize import curve_fit
    except Exception:
        raise RuntimeError('scipy is required to fit the Gompertz model (scipy.optimize.curve_fit)')

    # initial guesses
    A0 = np.nanmax(y) if not np.all(np.isnan(y)) else 1.0
    k0 = 0.5
    t00 = np.median(t)
    p0 = [A0, k0, t00]
    # bounds
    lower = [0.0, 0.0, -50.0]
    upper = [A0 * 5 if not np.isnan(A0) else 20.0, 5.0, 50.0]

    # remove NaNs
    mask_fit = ~np.isnan(y)
    if mask_fit.sum() < 3:
        raise RuntimeError('Not enough data points to fit Gompertz (need >=3 dates with values)')

    try:
        popt, pcov = curve_fit(gompertz, t[mask_fit], y[mask_fit], p0=p0, bounds=(lower, upper), maxfev=10000)
    except Exception as e:
        warnings.warn(f'Gompertz fit failed: {e}')
        popt = np.array([np.nan, np.nan, np.nan])
        pcov = np.full((3, 3), np.nan)

    # compute predictions and metrics
    A_est, k_est, t0_est = popt
    y_pred = gompertz(t, A_est, k_est, t0_est)
    # R2 and RMSE
    ss_res = np.sum((y[mask_fit] - gompertz(t[mask_fit], *popt)) ** 2)
    ss_tot = np.sum((y[mask_fit] - np.mean(y[mask_fit])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    rmse = float(np.sqrt(np.mean((y[mask_fit] - gompertz(t[mask_fit], *popt)) ** 2)))

    # Save parameters CSV
    df_out = pd.DataFrame({
        'source': ['automated'],
        'A': [A_est],
        'k': [k_est],
        't0': [t0_est],
        'R2': [r2],
        'RMSE': [rmse]
    })
    df_out.to_csv(OUT_CSV, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    ax.errorbar(t, y, yerr=yerr, fmt='o', color='tab:blue', ecolor='lightgray', capsize=3, markersize=6, label='Automated (mean ± SD)')
    t_smooth = np.linspace(t.min(), t.max(), 200)
    if not np.any(np.isnan(popt)):
        ax.plot(t_smooth, gompertz(t_smooth, *popt), '-', color='tab:orange', lw=2.2, label='Gompertz fit')

    ax.set_xlabel('Developmental day (index)', fontsize=12)
    ax.set_ylabel('Body length (mm)', fontsize=12)
    ax.set_xticks(t)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=600)
    plt.close(fig)

    print('Saved figure to', OUT_FIG)
    print('Saved parameters to', OUT_CSV)


if __name__ == '__main__':
    main()


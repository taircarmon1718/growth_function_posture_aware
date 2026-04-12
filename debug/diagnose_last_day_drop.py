#!/usr/bin/env python3
"""
diagnose_last_day_drop.py
==========================
Diagnostic script to analyze why the last date shows a drop in average larva body length.

This script performs comprehensive statistical and visual analysis to understand
whether the drop represents:
  - True biological decline
  - Sampling artifact
  - Outlier effect
  - Cohort composition shift

Usage:
    python diagnose_last_day_drop.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr

# ============================================================================
#  CONFIGURATION
# ============================================================================
ROOT_DIR = Path(__file__).parent.resolve()
DATA_FILE = ROOT_DIR / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
OUTPUT_DIR = ROOT_DIR / "dual_larva_models_geodesic2" / "diagnostics_last_day_analysis"

# Date order (chronological Oct-Nov)
DATE_ORDER = ['19.10', '20.10', '21.10', '24.10', '25.10', 
              '26.10', '27.10', '29.10', '31.10', '3.11']

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================
def save_fig(filename):
    """Save figure to output directory"""
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def date_sort_key(date_str):
    """Sort dates chronologically"""
    try:
        day, mon = str(date_str).split('.')
        return (int(mon), int(day))
    except:
        return (999, 999)


# ============================================================================
#  STEP 0: LOAD AND FILTER DATA
# ============================================================================
def load_data():
    """Load and filter predictions data"""
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        sys.exit(1)
    
    # Load data
    df_raw = pd.read_excel(DATA_FILE)
    print(f"  Raw data: {len(df_raw)} rows")
    
    # Filter: valid larvae with positive body length
    df = df_raw[
        (df_raw['predicted_valid'] == 1) & 
        (df_raw['body_length_mm'] > 0)
    ].copy()
    
    print(f"  Filtered (valid + length>0): {len(df)} rows")
    
    # Normalize date format
    df['date'] = df['date'].astype(str).apply(
        lambda s: f"{s.split('.')[0]}.10" if s.split('.')[-1] == '1' else s
    )
    
    # Keep only dates in DATE_ORDER
    df = df[df['date'].isin(DATE_ORDER)].copy()
    
    # Add date index for sorting
    df['date_idx'] = df['date'].map({d: i for i, d in enumerate(DATE_ORDER)})
    df = df.sort_values('date_idx')
    
    print(f"  Final dataset: {len(df)} rows")
    print(f"  Dates: {sorted(df['date'].unique(), key=date_sort_key)}")
    
    # Count per date
    print("\n  Larvae per date:")
    for date in DATE_ORDER:
        n = len(df[df['date'] == date])
        print(f"    {date}: {n}")
    
    return df


# ============================================================================
#  STEP 1: PER-DATE STATISTICS
# ============================================================================
def compute_per_date_statistics(df):
    """Compute comprehensive statistics per date"""
    print("\n" + "=" * 70)
    print("STEP 1: COMPUTING PER-DATE STATISTICS")
    print("=" * 70)
    
    stats_rows = []
    
    for date in DATE_ORDER:
        date_data = df[df['date'] == date]['body_length_mm'].values
        
        if len(date_data) == 0:
            continue
        
        stats_rows.append({
            'date': date,
            'n': len(date_data),
            'mean': np.mean(date_data),
            'median': np.median(date_data),
            'std': np.std(date_data, ddof=1),
            'min': np.min(date_data),
            'max': np.max(date_data),
            'p10': np.percentile(date_data, 10),
            'p25': np.percentile(date_data, 25),
            'p75': np.percentile(date_data, 75),
            'p90': np.percentile(date_data, 90),
            'iqr': np.percentile(date_data, 75) - np.percentile(date_data, 25),
            'cv_percent': 100 * np.std(date_data, ddof=1) / np.mean(date_data) if np.mean(date_data) > 0 else np.nan
        })
    
    stats_df = pd.DataFrame(stats_rows)
    
    # Save to CSV
    output_path = OUTPUT_DIR / "per_date_statistics.csv"
    stats_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved: per_date_statistics.csv")
    
    # Print summary
    print("\n  Summary:")
    print(stats_df[['date', 'n', 'mean', 'median', 'std']].to_string(index=False))
    
    return stats_df


# ============================================================================
#  STEP 2: DISTRIBUTION VISUALIZATION PER DATE
# ============================================================================
def plot_histograms_per_date(df):
    """Create histogram grid with KDE curves"""
    print("\n" + "=" * 70)
    print("STEP 2: CREATING HISTOGRAM GRID")
    print("=" * 70)
    
    n_dates = len(DATE_ORDER)
    n_cols = 3
    n_rows = (n_dates + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, date in enumerate(DATE_ORDER):
        ax = axes[i]
        date_data = df[df['date'] == date]['body_length_mm'].values
        
        if len(date_data) == 0:
            ax.text(0.5, 0.5, f'{date}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Histogram
        ax.hist(date_data, bins=25, density=True, alpha=0.6, 
               color='steelblue', edgecolor='black')
        
        # KDE curve
        try:
            kde = stats.gaussian_kde(date_data, bw_method='scott')
            x_range = np.linspace(date_data.min() - 0.5, date_data.max() + 0.5, 200)
            ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        except:
            pass
        
        # Mean and median lines
        ax.axvline(np.mean(date_data), color='green', ls='--', lw=2, label=f'Mean: {np.mean(date_data):.2f}')
        ax.axvline(np.median(date_data), color='orange', ls=':', lw=2, label=f'Median: {np.median(date_data):.2f}')
        
        ax.set_title(f'{date} (n={len(date_data)})', fontweight='bold')
        ax.set_xlabel('Body Length (mm)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Body Length Distributions per Date', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig('histograms_per_date.png')


# ============================================================================
#  STEP 3: BOXPLOT COMPARISON
# ============================================================================
def plot_boxplot_comparison(df):
    """Create boxplot comparing all dates"""
    print("\n" + "=" * 70)
    print("STEP 3: CREATING BOXPLOT COMPARISON")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data for boxplot
    data_by_date = [df[df['date'] == date]['body_length_mm'].values 
                    for date in DATE_ORDER]
    
    bp = ax.boxplot(data_by_date, labels=DATE_ORDER, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='green', linewidth=2, linestyle='--'))
    
    # Highlight last date
    bp['boxes'][-1].set_facecolor('salmon')
    bp['boxes'][-1].set_alpha(0.8)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Body Length Distribution Comparison Across Dates\n(Last date highlighted in red)', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Median'),
        Line2D([0], [0], color='green', lw=2, ls='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    save_fig('boxplot_lengths_per_date.png')


# ============================================================================
#  STEP 4: QUANTILE GROWTH CURVES
# ============================================================================
def plot_quantile_growth_curves(df, stats_df):
    """Plot mean, median, and percentile curves"""
    print("\n" + "=" * 70)
    print("STEP 4: CREATING QUANTILE GROWTH CURVES")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(DATE_ORDER))
    
    # Plot curves
    ax.plot(x_pos, stats_df['mean'], 'o-', lw=3, ms=10, 
           color='black', label='Mean', zorder=5)
    ax.plot(x_pos, stats_df['median'], 's-', lw=2.5, ms=8, 
           color='red', label='Median (P50)', zorder=4)
    ax.plot(x_pos, stats_df['p10'], '^--', lw=2, ms=7, 
           color='blue', alpha=0.7, label='P10', zorder=3)
    ax.plot(x_pos, stats_df['p90'], 'v--', lw=2, ms=7, 
           color='green', alpha=0.7, label='P90', zorder=3)
    ax.plot(x_pos, stats_df['p25'], 'd--', lw=1.5, ms=6, 
           color='cyan', alpha=0.6, label='P25', zorder=2)
    ax.plot(x_pos, stats_df['p75'], 'd--', lw=1.5, ms=6, 
           color='orange', alpha=0.6, label='P75', zorder=2)
    
    # Highlight last point
    ax.scatter([x_pos[-1]], [stats_df['mean'].iloc[-1]], 
              s=200, color='red', marker='o', zorder=10, 
              edgecolor='black', linewidth=2, label='Last date')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DATE_ORDER, rotation=45, ha='right')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Quantile Growth Curves\n(Does only mean drop, or all quantiles?)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig('quantile_growth_curves.png')


# ============================================================================
#  STEP 5: TOP-COHORT ANALYSIS
# ============================================================================
def plot_leading_cohort_growth(df):
    """Analyze top 20% larvae (leading cohort)"""
    print("\n" + "=" * 70)
    print("STEP 5: ANALYZING LEADING COHORT (TOP 20%)")
    print("=" * 70)
    
    cohort_stats = []
    
    for date in DATE_ORDER:
        date_data = df[df['date'] == date]['body_length_mm'].values
        
        if len(date_data) == 0:
            cohort_stats.append({
                'date': date,
                'mean_top20': np.nan,
                'median_top20': np.nan,
                'n_top20': 0
            })
            continue
        
        # Top 20%
        threshold = np.percentile(date_data, 80)
        top20 = date_data[date_data >= threshold]
        
        cohort_stats.append({
            'date': date,
            'mean_top20': np.mean(top20),
            'median_top20': np.median(top20),
            'n_top20': len(top20)
        })
    
    cohort_df = pd.DataFrame(cohort_stats)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(DATE_ORDER))
    
    # Overall mean (from earlier)
    overall_means = [df[df['date'] == d]['body_length_mm'].mean() for d in DATE_ORDER]
    
    ax.plot(x_pos, overall_means, 'o-', lw=3, ms=10, 
           color='gray', label='Overall Mean', alpha=0.7, zorder=3)
    ax.plot(x_pos, cohort_df['mean_top20'], 's-', lw=3, ms=10, 
           color='darkgreen', label='Top 20% Mean', zorder=4)
    ax.plot(x_pos, cohort_df['median_top20'], '^--', lw=2, ms=8, 
           color='lime', label='Top 20% Median', alpha=0.8, zorder=4)
    
    # Highlight last point
    ax.scatter([x_pos[-1]], [cohort_df['mean_top20'].iloc[-1]], 
              s=200, color='red', marker='s', zorder=10, 
              edgecolor='black', linewidth=2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DATE_ORDER, rotation=45, ha='right')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Leading Cohort Growth (Top 20% Largest Larvae)\n(Does the largest cohort still grow?)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig('leading_cohort_growth.png')
    
    return cohort_df


# ============================================================================
#  STEP 6: SAMPLE SIZE VISUALIZATION
# ============================================================================
def plot_sample_size_per_date(stats_df):
    """Plot number of larvae per date"""
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZING SAMPLE SIZES")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(DATE_ORDER))
    colors = ['steelblue'] * (len(DATE_ORDER) - 1) + ['red']
    
    bars = ax.bar(x_pos, stats_df['n'], color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (x, y) in enumerate(zip(x_pos, stats_df['n'])):
        ax.text(x, y + 1, str(int(y)), ha='center', va='bottom', 
               fontweight='bold', fontsize=10)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DATE_ORDER, rotation=45, ha='right')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Larvae', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size per Date\n(Last date highlighted in red)', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_fig('sample_size_per_date.png')


# ============================================================================
#  STEP 7: OUTLIER DETECTION
# ============================================================================
def detect_outliers_last_day(df):
    """Detect outliers on last date using IQR method"""
    print("\n" + "=" * 70)
    print("STEP 7: DETECTING OUTLIERS ON LAST DATE")
    print("=" * 70)
    
    last_date = DATE_ORDER[-1]
    last_data = df[df['date'] == last_date].copy()
    
    if len(last_data) == 0:
        print("  ⚠️  No data for last date")
        return None
    
    lengths = last_data['body_length_mm'].values
    
    # IQR method
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Identify outliers
    outliers = last_data[
        (last_data['body_length_mm'] < lower_bound) | 
        (last_data['body_length_mm'] > upper_bound)
    ].copy()
    
    print(f"  Last date: {last_date}")
    print(f"  Total larvae: {len(last_data)}")
    print(f"  Q1 = {q1:.3f}, Q3 = {q3:.3f}, IQR = {iqr:.3f}")
    print(f"  Outlier bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
    print(f"  Outliers detected: {len(outliers)}")
    
    if len(outliers) > 0:
        print(f"  Lower outliers: {len(outliers[outliers['body_length_mm'] < lower_bound])}")
        print(f"  Upper outliers: {len(outliers[outliers['body_length_mm'] > upper_bound])}")
        
        # Save outliers
        output_path = OUTPUT_DIR / "outliers_last_day.csv"
        outliers.to_csv(output_path, index=False)
        print(f"  ✓ Saved: outliers_last_day.csv")
    else:
        print("  ✓ No outliers detected")
        # Save empty file
        pd.DataFrame().to_csv(OUTPUT_DIR / "outliers_last_day.csv", index=False)
    
    return outliers


# ============================================================================
#  STEP 8: LAST-DAY VS PREVIOUS-DAY COMPARISON
# ============================================================================
def plot_last_vs_previous_distribution(df):
    """Compare last date vs previous date distributions"""
    print("\n" + "=" * 70)
    print("STEP 8: COMPARING LAST DATE VS PREVIOUS DATE")
    print("=" * 70)
    
    if len(DATE_ORDER) < 2:
        print("  ⚠️  Not enough dates for comparison")
        return
    
    last_date = DATE_ORDER[-1]
    prev_date = DATE_ORDER[-2]
    
    last_data = df[df['date'] == last_date]['body_length_mm'].values
    prev_data = df[df['date'] == prev_date]['body_length_mm'].values
    
    if len(last_data) == 0 or len(prev_data) == 0:
        print("  ⚠️  Insufficient data for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # KDE curves
    try:
        kde_prev = stats.gaussian_kde(prev_data, bw_method='scott')
        kde_last = stats.gaussian_kde(last_data, bw_method='scott')
        
        x_min = min(prev_data.min(), last_data.min()) - 0.5
        x_max = max(prev_data.max(), last_data.max()) + 0.5
        x_range = np.linspace(x_min, x_max, 300)
        
        ax.plot(x_range, kde_prev(x_range), '-', lw=3, 
               color='blue', label=f'{prev_date} (n={len(prev_data)})', alpha=0.8)
        ax.plot(x_range, kde_last(x_range), '-', lw=3, 
               color='red', label=f'{last_date} (n={len(last_data)})', alpha=0.8)
        
        # Fill under curves
        ax.fill_between(x_range, 0, kde_prev(x_range), alpha=0.2, color='blue')
        ax.fill_between(x_range, 0, kde_last(x_range), alpha=0.2, color='red')
        
    except Exception as e:
        print(f"  Warning: KDE failed: {e}")
    
    # Mean lines
    ax.axvline(np.mean(prev_data), color='blue', ls='--', lw=2, 
              label=f'{prev_date} mean: {np.mean(prev_data):.2f}')
    ax.axvline(np.mean(last_data), color='red', ls='--', lw=2, 
              label=f'{last_date} mean: {np.mean(last_data):.2f}')
    
    # Statistical test
    stat, pval = stats.mannwhitneyu(prev_data, last_data, alternative='two-sided')
    
    ax.set_xlabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution Comparison: {prev_date} vs {last_date}\n' + 
                f'Mann-Whitney U test: p={pval:.4f}' + 
                (' (SIGNIFICANT difference)' if pval < 0.05 else ' (not significant)'),
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig('last_day_vs_previous_distribution.png')
    
    print(f"  {prev_date}: n={len(prev_data)}, mean={np.mean(prev_data):.3f}, median={np.median(prev_data):.3f}")
    print(f"  {last_date}: n={len(last_data)}, mean={np.mean(last_data):.3f}, median={np.median(last_data):.3f}")
    print(f"  Mann-Whitney U test: p={pval:.4f}")


# ============================================================================
#  STEP 9: TREND ANALYSIS
# ============================================================================
def compute_trend_statistics(stats_df):
    """Compute Spearman correlation for overall trend"""
    print("\n" + "=" * 70)
    print("STEP 9: COMPUTING TREND STATISTICS")
    print("=" * 70)
    
    date_indices = np.arange(len(stats_df))
    means = stats_df['mean'].values
    medians = stats_df['median'].values
    
    # Spearman correlation
    rho_mean, pval_mean = spearmanr(date_indices, means)
    rho_median, pval_median = spearmanr(date_indices, medians)
    
    # Linear regression
    slope_mean, intercept_mean, r_value_mean, _, _ = stats.linregress(date_indices, means)
    slope_median, intercept_median, r_value_median, _, _ = stats.linregress(date_indices, medians)
    
    # Change from first to last
    mean_change = means[-1] - means[0]
    median_change = medians[-1] - medians[0]
    mean_pct_change = 100 * mean_change / means[0] if means[0] > 0 else np.nan
    median_pct_change = 100 * median_change / medians[0] if medians[0] > 0 else np.nan
    
    # Change from penultimate to last
    if len(means) >= 2:
        last_step_mean = means[-1] - means[-2]
        last_step_median = medians[-1] - medians[-2]
    else:
        last_step_mean = np.nan
        last_step_median = np.nan
    
    # Save to text file
    output_path = OUTPUT_DIR / "trend_statistics.txt"
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TREND STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL TREND (all dates)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Spearman correlation (mean):   ρ = {rho_mean:.4f}, p = {pval_mean:.4f}\n")
        f.write(f"Spearman correlation (median): ρ = {rho_median:.4f}, p = {pval_median:.4f}\n")
        f.write(f"Linear regression (mean):      slope = {slope_mean:.4f}, R² = {r_value_mean**2:.4f}\n")
        f.write(f"Linear regression (median):    slope = {slope_median:.4f}, R² = {r_value_median**2:.4f}\n")
        
        if pval_mean < 0.05:
            f.write(f"✓ Significant positive trend detected (p < 0.05)\n")
        else:
            f.write(f"⚠ No significant trend (p >= 0.05)\n")
        
        f.write("\n")
        f.write("OVERALL CHANGE (first to last date)\n")
        f.write("-" * 40 + "\n")
        f.write(f"First date ({stats_df['date'].iloc[0]}): mean = {means[0]:.3f}, median = {medians[0]:.3f}\n")
        f.write(f"Last date ({stats_df['date'].iloc[-1]}):  mean = {means[-1]:.3f}, median = {medians[-1]:.3f}\n")
        f.write(f"Mean change:   {mean_change:+.3f} mm ({mean_pct_change:+.1f}%)\n")
        f.write(f"Median change: {median_change:+.3f} mm ({median_pct_change:+.1f}%)\n")
        
        f.write("\n")
        f.write("LAST STEP CHANGE (penultimate to last date)\n")
        f.write("-" * 40 + "\n")
        if len(means) >= 2:
            f.write(f"Penultimate date ({stats_df['date'].iloc[-2]}): mean = {means[-2]:.3f}, median = {medians[-2]:.3f}\n")
            f.write(f"Last date ({stats_df['date'].iloc[-1]}):        mean = {means[-1]:.3f}, median = {medians[-1]:.3f}\n")
            f.write(f"Mean change:   {last_step_mean:+.3f} mm\n")
            f.write(f"Median change: {last_step_median:+.3f} mm\n")
            
            if last_step_mean < 0:
                f.write(f"⚠ MEAN DROPPED on last date!\n")
            if last_step_median < 0:
                f.write(f"⚠ MEDIAN DROPPED on last date!\n")
        
        f.write("\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        if pval_mean < 0.05 and rho_mean > 0 and last_step_mean < 0:
            f.write("Despite overall positive growth trend, the last date shows a DROP.\n")
            f.write("This suggests:\n")
            f.write("  • Sampling artifact (small sample size?)\n")
            f.write("  • Cohort composition shift (more small larvae sampled?)\n")
            f.write("  • True biological decline (unlikely without biological explanation)\n")
            f.write("  • Outlier effect (presence of unusually small larvae?)\n")
        elif pval_mean >= 0.05:
            f.write("No significant overall growth trend detected.\n")
            f.write("Data may be too noisy or sample sizes too small.\n")
        elif last_step_mean >= 0:
            f.write("Last date shows continued growth (no drop).\n")
        
        f.write("\n")
    
    print(f"  ✓ Saved: trend_statistics.txt")
    
    print(f"\n  Spearman ρ (mean): {rho_mean:.4f} (p={pval_mean:.4f})")
    print(f"  Overall change: {mean_change:+.3f} mm ({mean_pct_change:+.1f}%)")
    if len(means) >= 2:
        print(f"  Last step: {last_step_mean:+.3f} mm")
        if last_step_mean < 0:
            print("  ⚠️  MEAN DROPPED on last date!")


# ============================================================================
#  MAIN PIPELINE
# ============================================================================
def main():
    """Main analysis pipeline"""
    print("\n" + "=" * 70)
    print("LAST DAY DROP DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    print(f"Data file: {DATA_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    df = load_data()
    
    # Step 1: Per-date statistics
    stats_df = compute_per_date_statistics(df)
    
    # Step 2: Distribution visualizations
    plot_histograms_per_date(df)
    
    # Step 3: Boxplot comparison
    plot_boxplot_comparison(df)
    
    # Step 4: Quantile growth curves
    plot_quantile_growth_curves(df, stats_df)
    
    # Step 5: Leading cohort analysis
    cohort_df = plot_leading_cohort_growth(df)
    
    # Step 6: Sample size visualization
    plot_sample_size_per_date(stats_df)
    
    # Step 7: Outlier detection
    outliers = detect_outliers_last_day(df)
    
    # Step 8: Last vs previous distribution
    plot_last_vs_previous_distribution(df)
    
    # Step 9: Trend statistics
    compute_trend_statistics(stats_df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"  • {file.name}")
    
    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    last_date = DATE_ORDER[-1]
    prev_date = DATE_ORDER[-2] if len(DATE_ORDER) >= 2 else None
    
    last_mean = stats_df[stats_df['date'] == last_date]['mean'].iloc[0]
    last_n = stats_df[stats_df['date'] == last_date]['n'].iloc[0]
    
    if prev_date:
        prev_mean = stats_df[stats_df['date'] == prev_date]['mean'].iloc[0]
        change = last_mean - prev_mean
        pct_change = 100 * change / prev_mean if prev_mean > 0 else 0
        
        print(f"\n1. MEAN BODY LENGTH:")
        print(f"   {prev_date}: {prev_mean:.3f} mm")
        print(f"   {last_date}: {last_mean:.3f} mm")
        print(f"   Change: {change:+.3f} mm ({pct_change:+.1f}%)")
        
        if change < 0:
            print(f"   ⚠️  DROPPED on last date!")
        else:
            print(f"   ✓  Continued growth")
    
    print(f"\n2. SAMPLE SIZE:")
    print(f"   {last_date}: n = {last_n}")
    if last_n < 30:
        print(f"   ⚠️  Small sample size may cause unstable estimates")
    
    if outliers is not None and len(outliers) > 0:
        print(f"\n3. OUTLIERS:")
        print(f"   {len(outliers)} outliers detected on {last_date}")
        n_lower = len(outliers[outliers['body_length_mm'] < stats_df[stats_df['date'] == last_date]['p25'].iloc[0]])
        if n_lower > 0:
            print(f"   ⚠️  {n_lower} unusually SMALL larvae may be pulling mean down")
    
    print("\n" + "=" * 70)
    print("Review the generated figures and statistics for detailed analysis.")
    print("=" * 70)


if __name__ == "__main__":
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


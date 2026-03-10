#!/usr/bin/env python3
"""
central_tendency_analysis.py
============================
Standalone exploratory analysis script for comparing multiple central tendency
estimators and dispersion measures on larvae body length data from the
dual_larva_models pipeline.

This script is purely for visualization and exploration - it does not modify
any existing models, pipelines, or outputs.

Input: dual_larva_models/predictions/predictions_all_larvae.xlsx
Filter: All larvae (configurable in load_data method)
Output: analysis_central_tendency_exploration/

Author: Automated Analysis System
Date: March 4, 2026
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import trim_mean
import traceback

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================
ROOT_DIR = Path(__file__).parent.resolve()
# Use regular dual_larva_models which should have data for all dates
INPUT_FILE = ROOT_DIR / "dual_larva_models_geodesic" / "predictions" / "predictions_all_larvae.xlsx"
OUTPUT_DIR = ROOT_DIR / "analysis_central_tendency_exploration"

# Date ordering (chronological)
DATE_ORDER = ["19.10", "20.10", "21.10", "24.10", "25.10", "26.10", "27.10", "29.10", "31.10", "3.11"]

# Visualization settings
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# ============================================================
#  HELPERS
# ============================================================
def _normalize_date_value(val: object) -> str:
    """Normalize date representations from predictions file to 'DD.MM' strings.

    The predictions Excel stores dates as floats like 19.1, 20.1, 3.11, etc.
    We want canonical strings that match DATE_ORDER:
        19.1  -> '19.10'
        20.1  -> '20.10'
        3.11  -> '3.11' (already correct)

    Logic:
      - If value is float or int, format with one decimal place and then
        replace any trailing '.1' with '.10'.
      - If value is a string, return as-is.
    """
    # Already a string
    if isinstance(val, str):
        return val

    # Numeric representation (float/int from Excel)
    if isinstance(val, (int, float)):
        # Represent with one decimal place, e.g. 19.1, 20.1, 3.1, 3.11
        s = f"{val:g}"
        # If it ends with '.1' and not '.11', interpret as '.10'
        if s.endswith('.1') and not s.endswith('.11'):
            return s[:-2] + '.10'
        return s

    # Fallback: string conversion
    return str(val)


# ============================================================
#  ROBUST ESTIMATORS
# ============================================================
def winsorized_mean(data, limits=(0.1, 0.1)):
    """Compute winsorized mean with specified limits."""
    from scipy.stats.mstats import winsorize
    return np.mean(winsorize(data, limits=limits))

def hodges_lehmann_estimator(data):
    """Compute Hodges-Lehmann estimator (median of pairwise means)."""
    n = len(data)
    if n == 0:
        return np.nan
    if n == 1:
        return data[0]

    # For large datasets, use a sample to avoid memory issues
    if n > 1000:
        data = np.random.choice(data, size=1000, replace=False)
        n = len(data)

    pairwise_means = []
    for i in range(n):
        for j in range(i, n):
            pairwise_means.append((data[i] + data[j]) / 2)

    return np.median(pairwise_means)

def median_absolute_deviation(data):
    """Compute median absolute deviation (MAD)."""
    median_val = np.median(data)
    return np.median(np.abs(data - median_val))

# ============================================================
#  DATA LOADING AND PROCESSING
# ============================================================
class CentralTendencyAnalyzer:

    def __init__(self):
        self.data = None
        self.stats_df = None
        self.filtered_data = {}

    def load_data(self):
        """Load and filter the predictions data."""
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        print("Loading predictions data...")
        df = pd.read_excel(INPUT_FILE)

        # Use all larvae to ensure we get data for all dates
        # Filter options (choose one):
        # Option 1: All larvae (most permissive)
        self.data = df[
            (df['predicted_valid'] == 1) &
            (df['predicted_posture'] == 1)
            ].copy()
        filter_description = "all larvae"

        # Option 2: Valid larvae only (uncomment to use)
        # self.data = df[df['predicted_valid'] == 1].copy()
        # filter_description = "valid larvae only"

        # Option 3: Posture-filtered larvae (strictest, uncomment to use)
        # self.data = df[(df['predicted_valid'] == 1) & (df['predicted_posture'] == 1)].copy()
        # filter_description = "posture-filtered larvae"

        # Normalize date column to canonical 'DD.MM' strings
        self.data['date'] = self.data['date'].apply(_normalize_date_value)

        # Filter to known dates in chronological order
        self.data = self.data[self.data['date'].isin(DATE_ORDER)]

        print(f"Total larvae loaded: {len(df)}")
        print(f"Filtered larvae ({filter_description}): {len(self.data)}")

        # Check how many larvae per date
        print("Larvae per date:")
        date_counts = self.data['date'].value_counts().sort_index()
        for date in DATE_ORDER:
            count = date_counts.get(date, 0)
            print(f"  {date}: {count} larvae")

        # Group data by date (only dates with valid body_length_mm data)
        self.filtered_data.clear()
        for date in DATE_ORDER:
            subset = self.data[self.data['date'] == date]['body_length_mm'].dropna()
            if len(subset) > 0:
                self.filtered_data[date] = subset.values

        print(f"Valid dates with data: {len(self.filtered_data)}")
        print(f"Dates with data: {sorted(self.filtered_data.keys())}")

        # Check if we're missing most dates
        if len(self.filtered_data) <= 2:
            print()
            print("⚠️  WARNING: Only found data for very few dates!")
            print("   This suggests the pipeline has not been run or exported correctly for all date folders,")
            print("   or that date formatting in the predictions file is inconsistent.")
            print("   Available date folders in project: 18.10, 19.10, 20.10, 21.10, 24.10, 25.10, 26.10, 27.10, 29.10, 31.10, 3.11")
            print("   Current analysis will proceed with available data only.")
            print()

    def compute_statistics(self):
        """Compute all central tendency and dispersion statistics."""

        stats_list = []

        for date in DATE_ORDER:
            if date not in self.filtered_data:
                continue

            data = self.filtered_data[date]
            n = len(data)

            if n < 2:
                continue

            print(f"Computing statistics for {date} (n={n})...")

            # Central tendency estimators
            mean_val = np.mean(data)
            median_val = np.median(data)
            trimmed_mean_val = trim_mean(data, proportiontocut=0.1)  # 10% trimmed mean

            try:
                winsorized_mean_val = winsorized_mean(data, limits=(0.1, 0.1))
            except:
                winsorized_mean_val = np.nan

            try:
                hodges_lehmann_val = hodges_lehmann_estimator(data)
            except:
                hodges_lehmann_val = np.nan

            # Dispersion measures
            sd = np.std(data, ddof=1)
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            mad = median_absolute_deviation(data)

            stats_dict = {
                'date': date,
                'n': n,
                'mean': mean_val,
                'median': median_val,
                'trimmed_mean': trimmed_mean_val,
                'winsorized_mean': winsorized_mean_val,
                'hodges_lehmann': hodges_lehmann_val,
                'sd': sd,
                'iqr': iqr,
                'mad': mad
            }

            stats_list.append(stats_dict)

        self.stats_df = pd.DataFrame(stats_list)
        print(f"Statistics computed for {len(self.stats_df)} dates")

    def save_statistics(self):
        """Save statistics to CSV file."""
        output_file = OUTPUT_DIR / "central_tendency_statistics.csv"
        self.stats_df.to_csv(output_file, index=False)
        print(f"✓ Statistics saved to: {output_file}")

# ============================================================
#  VISUALIZATION FUNCTIONS
# ============================================================
def create_figure1_estimators_comparison(analyzer):
    """Figure 1: Line plot comparing all central tendency estimators."""

    fig, ax = plt.subplots(figsize=(12, 8))

    stats_df = analyzer.stats_df

    # Plot each estimator
    estimators = [
        ('mean', 'Mean', '-', 'blue'),
        ('median', 'Median', '-', 'red'),
        ('trimmed_mean', 'Trimmed Mean (10%)', '-', 'green'),
        ('winsorized_mean', 'Winsorized Mean (10%)', '-', 'orange'),
        ('hodges_lehmann', 'Hodges-Lehmann', '-', 'purple')
    ]

    x_positions = range(len(stats_df))

    for col, label, linestyle, color in estimators:
        values = stats_df[col].values
        valid_mask = ~np.isnan(values)

        if np.any(valid_mask):
            ax.plot(np.array(x_positions)[valid_mask], values[valid_mask],
                   linestyle=linestyle, marker='o', color=color,
                   label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Central Tendency Estimators Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stats_df['date'], rotation=45)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_estimators_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Created fig1_estimators_comparison.png")

def create_figure2_mean_sd(analyzer):
    """Figure 2: Mean ± SD error bars."""

    fig, ax = plt.subplots(figsize=(10, 6))

    stats_df = analyzer.stats_df
    x_positions = range(len(stats_df))

    ax.errorbar(x_positions, stats_df['mean'], yerr=stats_df['sd'],
               marker='o', capsize=5, capthick=2, linewidth=2,
               markersize=8, color='blue', label='Mean ± SD')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Mean with Standard Deviation', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stats_df['date'], rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_mean_sd.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Created fig2_mean_sd.png")

def create_figure3_median_iqr(analyzer):
    """Figure 3: Median with IQR error bars."""

    fig, ax = plt.subplots(figsize=(10, 6))

    stats_df = analyzer.stats_df
    x_positions = range(len(stats_df))

    ax.errorbar(x_positions, stats_df['median'], yerr=stats_df['iqr']/2,
               marker='s', capsize=5, capthick=2, linewidth=2,
               markersize=8, color='red', label='Median ± IQR/2')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Median with Interquartile Range', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stats_df['date'], rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_median_iqr.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Created fig3_median_iqr.png")

def create_figure4_trimmed_mean_sd(analyzer):
    """Figure 4: Trimmed mean ± SD."""

    fig, ax = plt.subplots(figsize=(10, 6))

    stats_df = analyzer.stats_df
    x_positions = range(len(stats_df))

    ax.errorbar(x_positions, stats_df['trimmed_mean'], yerr=stats_df['sd'],
               marker='^', capsize=5, capthick=2, linewidth=2,
               markersize=8, color='green', label='Trimmed Mean (10%) ± SD')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Trimmed Mean with Standard Deviation', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stats_df['date'], rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_trimmed_mean_sd.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Created fig4_trimmed_mean_sd.png")

def create_figure5_dispersion_comparison(analyzer):
    """Figure 5: Comparison of dispersion measures."""

    fig, ax = plt.subplots(figsize=(12, 8))

    stats_df = analyzer.stats_df
    x_positions = range(len(stats_df))
    width = 0.25

    # Plot each dispersion measure
    ax.bar([x - width for x in x_positions], stats_df['sd'], width,
           label='Standard Deviation', alpha=0.7, color='blue')
    ax.bar(x_positions, stats_df['iqr'], width,
           label='Interquartile Range', alpha=0.7, color='red')
    ax.bar([x + width for x in x_positions], stats_df['mad'], width,
           label='Median Absolute Deviation', alpha=0.7, color='green')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dispersion (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Dispersion Measures Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stats_df['date'], rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_dispersion_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Created fig5_dispersion_comparison.png")

def create_figure6_boxplot_estimators(analyzer):
    """Figure 6: Boxplot with central tendency estimators overlaid."""

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for boxplot
    data_for_boxplot = []
    labels = []

    for date in analyzer.stats_df['date']:
        if date in analyzer.filtered_data:
            data_for_boxplot.append(analyzer.filtered_data[date])
            labels.append(date)

    # Create boxplot
    bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Overlay central tendency estimators
    stats_df = analyzer.stats_df
    x_positions = range(1, len(stats_df) + 1)  # boxplot uses 1-based indexing

    ax.plot(x_positions, stats_df['mean'], 'o', color='blue',
            markersize=8, label='Mean', markeredgecolor='white', markeredgewidth=1)
    ax.plot(x_positions, stats_df['median'], 's', color='red',
            markersize=8, label='Median', markeredgecolor='white', markeredgewidth=1)
    ax.plot(x_positions, stats_df['trimmed_mean'], '^', color='green',
            markersize=8, label='Trimmed Mean', markeredgecolor='white', markeredgewidth=1)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Body Length (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution per Date with Central Tendency Estimators', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_boxplot_estimators.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Created fig6_boxplot_estimators.png")

# ============================================================
#  STABILITY ANALYSIS
# ============================================================
def analyze_estimator_stability(analyzer):
    """Analyze and rank estimators by stability (lowest CV) and distribution characteristics."""

    stats_df = analyzer.stats_df

    estimators = ['mean', 'median', 'trimmed_mean', 'winsorized_mean', 'hodges_lehmann']
    stability_results = []

    print("\n" + "="*60)
    print("ESTIMATOR STABILITY ANALYSIS")
    print("="*60)

    if len(stats_df) == 1:
        # Single date case - analyze distribution characteristics
        print("⚠️  LIMITATION: Only one date with sufficient data found.")
        print("   Temporal stability analysis requires multiple time points.")
        print("   Performing distribution characteristics analysis instead.")
        print("   For meaningful central tendency comparison across dates,")
        print("   please run the pipeline on all date folders.")
        print()

        date = stats_df.iloc[0]['date']
        n = int(stats_df.iloc[0]['n'])
        data = analyzer.filtered_data[date]

        print(f"Date: {date} (n = {n})")
        print("-" * 30)

        for estimator in estimators:
            value = stats_df.iloc[0][estimator]

            if not np.isnan(value):
                # Calculate how close the estimator is to the median (robust reference)
                median_val = stats_df.iloc[0]['median']
                deviation_from_median = abs(value - median_val)
                relative_deviation = (deviation_from_median / median_val) * 100 if median_val != 0 else 0

                stability_results.append({
                    'estimator': estimator,
                    'value': value,
                    'deviation_from_median': deviation_from_median,
                    'relative_deviation': relative_deviation
                })

                print(f"{estimator.replace('_', ' ').title()}:")
                print(f"  Value: {value:.3f} mm")
                print(f"  Deviation from median: {deviation_from_median:.3f} mm ({relative_deviation:.1f}%)")
                print()

        # Sort by deviation from median (ascending = more robust)
        stability_results.sort(key=lambda x: x['relative_deviation'])

        print("ROBUSTNESS RANKING (most robust first):")
        print("-" * 40)
        for i, result in enumerate(stability_results, 1):
            estimator_name = result['estimator'].replace('_', ' ').title()
            print(f"{i}. {estimator_name} (deviation = {result['relative_deviation']:.1f}%)")

        # Additional distribution analysis
        print()
        print("DISTRIBUTION CHARACTERISTICS:")
        print("-" * 30)
        print(f"Standard Deviation: {stats_df.iloc[0]['sd']:.3f} mm")
        print(f"IQR: {stats_df.iloc[0]['iqr']:.3f} mm")
        print(f"MAD: {stats_df.iloc[0]['mad']:.3f} mm")
        print(f"Coefficient of Variation: {(stats_df.iloc[0]['sd'] / stats_df.iloc[0]['mean']) * 100:.1f}%")

        # Outlier analysis
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr
        n_outliers = np.sum((data < outlier_threshold_low) | (data > outlier_threshold_high))
        outlier_percentage = (n_outliers / len(data)) * 100

        print(f"Outliers (IQR method): {n_outliers} ({outlier_percentage:.1f}%)")

    else:
        # Multiple dates case - original temporal stability analysis
        for estimator in estimators:
            values = stats_df[estimator].dropna()

            if len(values) < 2:
                continue

            global_mean = np.mean(values)
            global_sd = np.std(values, ddof=1)
            cv = (global_sd / global_mean) * 100 if global_mean != 0 else np.inf

            stability_results.append({
                'estimator': estimator,
                'global_mean': global_mean,
                'global_sd': global_sd,
                'cv': cv
            })

            print(f"{estimator.replace('_', ' ').title()}:")
            print(f"  Global Mean: {global_mean:.3f} mm")
            print(f"  Global SD:   {global_sd:.3f} mm")
            print(f"  CV:          {cv:.2f}%")
            print()

        # Sort by coefficient of variation (ascending = more stable)
        stability_results.sort(key=lambda x: x['cv'])

        print("STABILITY RANKING (most stable first):")
        print("-" * 40)
        for i, result in enumerate(stability_results, 1):
            estimator_name = result['estimator'].replace('_', ' ').title()
            print(f"{i}. {estimator_name} (CV = {result['cv']:.2f}%)")

    print("="*60)

    return stability_results

# ============================================================
#  MAIN EXECUTION
# ============================================================
def main():
    """Main execution function."""

    print("="*80)
    print("CENTRAL TENDENCY EXPLORATORY ANALYSIS")
    print("="*80)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create output directory
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {OUTPUT_DIR}")
    except Exception as e:
        print(f"❌ Error creating output directory: {e}")
        sys.exit(1)

    try:
        # Initialize analyzer
        analyzer = CentralTendencyAnalyzer()

        # Load and process data
        analyzer.load_data()
        analyzer.compute_statistics()

        # Save statistics
        analyzer.save_statistics()

        # Create visualizations
        print("\nGenerating visualizations...")
        create_figure1_estimators_comparison(analyzer)
        create_figure2_mean_sd(analyzer)
        create_figure3_median_iqr(analyzer)
        create_figure4_trimmed_mean_sd(analyzer)
        create_figure5_dispersion_comparison(analyzer)
        create_figure6_boxplot_estimators(analyzer)

        # Analyze stability
        stability_results = analyze_estimator_stability(analyzer)

        print(f"\n✓ Analysis complete! All outputs saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        for file_path in sorted(OUTPUT_DIR.iterdir()):
            if file_path.is_file():
                print(f"  {file_path.name}")

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Please ensure the dual_larva_models_geodesic pipeline has been run first.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

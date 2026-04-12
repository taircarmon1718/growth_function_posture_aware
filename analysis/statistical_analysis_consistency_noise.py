#!/usr/bin/env python3
"""
statistical_analysis_consistency_noise.py
==========================================
Standalone statistical analysis module that evaluates the internal consistency
and variability of posture-filtered geodesic body-length measurements.

Analyzes:
- Measurement variability per day
- Temporal stability of the growth signal
- Noise characteristics and signal-to-noise behavior
- Uncertainty of daily mean length values
- Comparison before/after posture filtering
- Statistical significance testing
- Temporal smoothness and trend analysis

Outputs:
- CSV files with all numerical results
- Publication-ready figures
- Structured summary report
"""

import sys
from pathlib import Path
import warnings
import traceback
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    pass
from datetime import datetime
import scipy.stats as stats
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================
ROOT_DIR = Path(__file__).parent.resolve()
GEODESIC_MODELS_DIR = ROOT_DIR / "dual_larva_models_geodesic2"
PREDICTIONS_FILE = GEODESIC_MODELS_DIR / "predictions" / "predictions_all_larvae.xlsx"
OUTPUT_DIR = ROOT_DIR / "statistical_analysis" / "consistency_noise"

# Date ordering (chronological)
DATE_ORDER = ["19.10", "20.10", "21.10", "24.10", "25.10", "26.10", "27.10", "29.10", "31.10", "3.11"]

# Visualization settings
try:
    plt.style.use('default')
    sns.set_palette("husl")
except:
    pass  # Use default if style loading fails

# ============================================================
#  DATA LOADING AND PREPROCESSING
# ============================================================
class ConsistencyAnalyzer:

    def __init__(self):
        self.df_raw = None
        self.df_filtered = None
        self.daily_stats_raw = None
        self.daily_stats_filtered = None
        self.results = {}

    def load_data(self):
        """Load predictions data and prepare datasets."""
        if not PREDICTIONS_FILE.exists():
            raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_FILE}")

        print("Loading predictions data...")
        self.df_raw = pd.read_excel(PREDICTIONS_FILE)

        # Clean and standardize dates
        self.df_raw['date'] = self.df_raw['date'].astype(str)
        self.df_raw = self.df_raw[self.df_raw['date'].isin(DATE_ORDER)]

        # Create posture-filtered dataset
        self.df_filtered = self.df_raw[
            (self.df_raw['predicted_valid'] == 1) &
            (self.df_raw['predicted_posture'] == 1)
        ].copy()

        print(f"Total larvae: {len(self.df_raw)}")
        print(f"Posture-filtered larvae: {len(self.df_filtered)}")

        # Compute daily statistics
        self.compute_daily_statistics()

    def compute_daily_statistics(self):
        """Compute comprehensive daily statistics for both datasets."""

        def daily_stats(df, suffix=""):
            stats_list = []
            for date in DATE_ORDER:
                subset = df[df['date'] == date]
                lengths = subset['body_length_mm'].dropna()

                if len(lengths) > 0:
                    stats_dict = {
                        'date': date,
                        f'count{suffix}': len(lengths),
                        f'mean{suffix}': lengths.mean(),
                        f'median{suffix}': lengths.median(),
                        f'std{suffix}': lengths.std(),
                        f'sem{suffix}': lengths.std() / np.sqrt(len(lengths)),  # Standard error of mean
                        f'cv{suffix}': (lengths.std() / lengths.mean()) * 100,  # Coefficient of variation
                        f'q25{suffix}': lengths.quantile(0.25),
                        f'q75{suffix}': lengths.quantile(0.75),
                        f'iqr{suffix}': lengths.quantile(0.75) - lengths.quantile(0.25),
                        f'min{suffix}': lengths.min(),
                        f'max{suffix}': lengths.max(),
                        f'range{suffix}': lengths.max() - lengths.min(),
                        f'skewness{suffix}': stats.skew(lengths),
                        f'kurtosis{suffix}': stats.kurtosis(lengths),
                        f'outliers{suffix}': self._count_outliers(lengths)
                    }
                    stats_list.append(stats_dict)

            return pd.DataFrame(stats_list)

        # Raw data statistics
        self.daily_stats_raw = daily_stats(self.df_raw, "_raw")

        # Filtered data statistics
        self.daily_stats_filtered = daily_stats(self.df_filtered, "_filtered")

        # Merge statistics
        self.daily_stats = pd.merge(self.daily_stats_raw, self.daily_stats_filtered,
                                   on='date', how='outer').fillna(0)

    def _count_outliers(self, data, method='iqr'):
        """Count outliers using IQR method."""
        if len(data) < 4:
            return 0
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return len(data[(data < lower) | (data > upper)])

# ============================================================
#  VARIABILITY ANALYSIS
# ============================================================
def analyze_measurement_variability(analyzer: ConsistencyAnalyzer):
    """Analyze within-day and between-day variability."""

    results = {}

    # Within-day variability analysis
    within_day_stats = []
    for date in DATE_ORDER:
        raw_subset = analyzer.df_raw[analyzer.df_raw['date'] == date]['body_length_mm'].dropna()
        filt_subset = analyzer.df_filtered[analyzer.df_filtered['date'] == date]['body_length_mm'].dropna()

        if len(raw_subset) > 1 and len(filt_subset) > 1:
            within_day_stats.append({
                'date': date,
                'raw_cv': (raw_subset.std() / raw_subset.mean()) * 100,
                'filtered_cv': (filt_subset.std() / filt_subset.mean()) * 100,
                'raw_range': raw_subset.max() - raw_subset.min(),
                'filtered_range': filt_subset.max() - filt_subset.min(),
                'raw_iqr': raw_subset.quantile(0.75) - raw_subset.quantile(0.25),
                'filtered_iqr': filt_subset.quantile(0.75) - filt_subset.quantile(0.25)
            })

    within_day_df = pd.DataFrame(within_day_stats)

    # Overall variability metrics
    results['overall_variability'] = {
        'raw_mean_cv': within_day_df['raw_cv'].mean(),
        'filtered_mean_cv': within_day_df['filtered_cv'].mean(),
        'cv_reduction_percent': ((within_day_df['raw_cv'].mean() - within_day_df['filtered_cv'].mean()) /
                                within_day_df['raw_cv'].mean()) * 100,
        'raw_mean_range': within_day_df['raw_range'].mean(),
        'filtered_mean_range': within_day_df['filtered_range'].mean(),
        'range_reduction_percent': ((within_day_df['raw_range'].mean() - within_day_df['filtered_range'].mean()) /
                                   within_day_df['raw_range'].mean()) * 100
    }

    # Between-day variability (growth signal stability)
    raw_means = analyzer.daily_stats['mean_raw'].dropna()
    filtered_means = analyzer.daily_stats['mean_filtered'].dropna()

    results['between_day_variability'] = {
        'raw_growth_cv': (raw_means.std() / raw_means.mean()) * 100,
        'filtered_growth_cv': (filtered_means.std() / filtered_means.mean()) * 100,
        'growth_signal_improvement': ((raw_means.std() / raw_means.mean()) -
                                     (filtered_means.std() / filtered_means.mean())) * 100
    }

    return results, within_day_df

# ============================================================
#  TEMPORAL STABILITY ANALYSIS
# ============================================================
def analyze_temporal_stability(analyzer: ConsistencyAnalyzer):
    """Analyze temporal stability and trend characteristics."""

    results = {}

    # Prepare data for regression analysis
    valid_stats = analyzer.daily_stats.dropna(subset=['mean_filtered'])
    if len(valid_stats) < 3:
        return {"error": "Insufficient data for temporal analysis"}

    # Convert dates to numeric for regression
    day_index = np.arange(len(valid_stats))
    mean_lengths = valid_stats['mean_filtered'].values
    std_lengths = valid_stats['std_filtered'].values

    # Linear trend analysis
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(day_index, mean_lengths)

        results['linear_trend'] = {
            'slope_mm_per_day': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'trend_significant': p_value < 0.05
        }
    except Exception as e:
        results['linear_trend'] = {
            'error': f"Could not compute linear trend: {str(e)}",
            'slope_mm_per_day': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_error': np.nan,
            'trend_significant': False
        }

    # Smoothness analysis using Savitzky-Golay filter
    if len(mean_lengths) >= 5:
        try:
            # Ensure window length is odd and appropriate for data size
            window_length = min(5, len(mean_lengths))
            if window_length % 2 == 0:
                window_length -= 1

            if window_length >= 3:  # Minimum window for polyorder=2
                polyorder = min(2, window_length - 1)
                smoothed = savgol_filter(mean_lengths, window_length=window_length, polyorder=polyorder)
                residuals = mean_lengths - smoothed

                results['smoothness'] = {
                    'residual_std': np.std(residuals),
                    'residual_mean_abs': np.mean(np.abs(residuals)),
                    'smoothness_index': np.std(residuals) / np.std(mean_lengths)  # Lower = smoother
                }
        except Exception as e:
            results['smoothness'] = {"error": f"Could not compute smoothness metrics: {str(e)}"}

    # Autocorrelation analysis
    if len(mean_lengths) >= 4:
        # First-order autocorrelation
        autocorr_1 = np.corrcoef(mean_lengths[:-1], mean_lengths[1:])[0, 1]
        results['autocorrelation'] = {
            'first_order': autocorr_1,
            'temporal_persistence': autocorr_1 > 0.5
        }

    # Volatility analysis (day-to-day changes)
    if len(mean_lengths) >= 2:
        daily_changes = np.diff(mean_lengths)
        results['volatility'] = {
            'mean_daily_change': np.mean(daily_changes),
            'std_daily_change': np.std(daily_changes),
            'max_daily_change': np.max(np.abs(daily_changes)),
            'change_coefficient': np.std(daily_changes) / np.mean(mean_lengths)
        }

    return results

# ============================================================
#  NOISE ANALYSIS
# ============================================================
def analyze_noise_characteristics(analyzer: ConsistencyAnalyzer):
    """Analyze noise characteristics and signal-to-noise ratio."""

    results = {}

    # Signal-to-noise analysis per day
    snr_stats = []
    for date in DATE_ORDER:
        subset = analyzer.df_filtered[analyzer.df_filtered['date'] == date]['body_length_mm'].dropna()

        if len(subset) > 1:
            signal = subset.mean()  # Mean as signal
            noise = subset.std()    # Standard deviation as noise
            snr = signal / noise if noise > 0 else np.inf

            snr_stats.append({
                'date': date,
                'signal': signal,
                'noise': noise,
                'snr': snr,
                'snr_db': 20 * np.log10(snr) if snr > 0 else np.inf
            })

    snr_df = pd.DataFrame(snr_stats)

    if len(snr_df) > 0:
        results['signal_to_noise'] = {
            'mean_snr': snr_df['snr'].mean(),
            'mean_snr_db': snr_df['snr_db'].replace([np.inf, -np.inf], np.nan).mean(),
            'min_snr': snr_df['snr'].min(),
            'max_snr': snr_df['snr'].max(),
            'snr_consistency': 1 / snr_df['snr'].std()  # Higher = more consistent SNR
        }

    # Noise distribution analysis
    all_residuals = []
    daily_means = analyzer.daily_stats_filtered.set_index('date')['mean_filtered']

    for date in DATE_ORDER:
        subset = analyzer.df_filtered[analyzer.df_filtered['date'] == date]['body_length_mm'].dropna()
        if len(subset) > 0 and date in daily_means.index:
            residuals = subset - daily_means[date]
            all_residuals.extend(residuals.tolist())

    if len(all_residuals) > 10:
        residuals_array = np.array(all_residuals)

        # Test for normality
        _, normality_p = stats.shapiro(residuals_array[:5000] if len(residuals_array) > 5000 else residuals_array)

        results['noise_distribution'] = {
            'residual_mean': np.mean(residuals_array),
            'residual_std': np.std(residuals_array),
            'residual_skewness': stats.skew(residuals_array),
            'residual_kurtosis': stats.kurtosis(residuals_array),
            'normality_p_value': normality_p,
            'is_normal_noise': normality_p > 0.05
        }

    return results, snr_df

# ============================================================
#  UNCERTAINTY ESTIMATION
# ============================================================
def estimate_daily_uncertainties(analyzer: ConsistencyAnalyzer):
    """Estimate uncertainty of daily mean length values."""

    uncertainty_stats = []

    for date in DATE_ORDER:
        subset = analyzer.df_filtered[analyzer.df_filtered['date'] == date]['body_length_mm'].dropna()

        if len(subset) > 1:
            n = len(subset)
            mean_val = subset.mean()
            std_val = subset.std()
            sem = std_val / np.sqrt(n)

            # Bootstrap confidence interval
            bootstrap_means = []
            for _ in range(1000):
                boot_sample = np.random.choice(subset, size=n, replace=True)
                bootstrap_means.append(np.mean(boot_sample))

            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)

            uncertainty_stats.append({
                'date': date,
                'sample_size': n,
                'mean': mean_val,
                'std': std_val,
                'sem': sem,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'relative_uncertainty': (ci_upper - ci_lower) / (2 * mean_val) * 100
            })

    uncertainty_df = pd.DataFrame(uncertainty_stats)

    return uncertainty_df

# ============================================================
#  STATISTICAL TESTING
# ============================================================
def perform_statistical_tests(analyzer: ConsistencyAnalyzer):
    """Perform statistical significance tests."""

    results = {}

    # Test for significant differences in variability before/after filtering
    cv_before = []
    cv_after = []

    for date in DATE_ORDER:
        raw_subset = analyzer.df_raw[analyzer.df_raw['date'] == date]['body_length_mm'].dropna()
        filt_subset = analyzer.df_filtered[analyzer.df_filtered['date'] == date]['body_length_mm'].dropna()

        if len(raw_subset) > 1 and len(filt_subset) > 1:
            cv_before.append((raw_subset.std() / raw_subset.mean()) * 100)
            cv_after.append((filt_subset.std() / filt_subset.mean()) * 100)

    if len(cv_before) > 1 and len(cv_after) > 1:
        # Paired t-test for CV reduction
        t_stat, p_val = stats.ttest_rel(cv_before, cv_after)
        results['cv_reduction_test'] = {
            'mean_cv_before': np.mean(cv_before),
            'mean_cv_after': np.mean(cv_after),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant_reduction': p_val < 0.05 and t_stat > 0
        }

    # Test for temporal trend significance
    valid_stats = analyzer.daily_stats.dropna(subset=['mean_filtered'])
    if len(valid_stats) >= 3:
        day_index = np.arange(len(valid_stats))
        mean_lengths = valid_stats['mean_filtered'].values

        # Mann-Kendall trend test
        def mann_kendall_test(data):
            n = len(data)
            s = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    s += np.sign(data[j] - data[i])

            var_s = n * (n - 1) * (2*n + 5) / 18

            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0

            p = 2 * (1 - stats.norm.cdf(abs(z)))
            return z, p

        mk_z, mk_p = mann_kendall_test(mean_lengths)
        results['trend_test'] = {
            'mann_kendall_z': mk_z,
            'mann_kendall_p': mk_p,
            'significant_trend': mk_p < 0.05
        }

    return results

# ============================================================
#  VISUALIZATION
# ============================================================
def create_visualizations(analyzer: ConsistencyAnalyzer, within_day_df, snr_df, uncertainty_df):
    """Create comprehensive visualizations."""

    fig_dir = OUTPUT_DIR / "figures"

    # 1. Variability comparison plot
    plt.figure(figsize=(12, 8))

    # Subplot 1: CV comparison
    plt.subplot(2, 2, 1)
    x = np.arange(len(within_day_df))
    width = 0.35
    plt.bar(x - width/2, within_day_df['raw_cv'], width, label='Raw', alpha=0.7)
    plt.bar(x + width/2, within_day_df['filtered_cv'], width, label='Posture-filtered', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('Within-day Variability: CV Comparison')
    plt.xticks(x, within_day_df['date'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Daily means with error bars
    plt.subplot(2, 2, 2)
    valid_stats = analyzer.daily_stats.dropna(subset=['mean_filtered'])
    x = np.arange(len(valid_stats))
    plt.errorbar(x, valid_stats['mean_filtered'], yerr=valid_stats['sem_filtered'],
                marker='o', capsize=5, label='Posture-filtered')
    plt.errorbar(x, valid_stats['mean_raw'], yerr=valid_stats['sem_raw'],
                marker='s', capsize=5, alpha=0.7, label='Raw')
    plt.xlabel('Date')
    plt.ylabel('Body Length (mm)')
    plt.title('Daily Means with Standard Error')
    plt.xticks(x, valid_stats['date'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Signal-to-noise ratio
    plt.subplot(2, 2, 3)
    if len(snr_df) > 0:
        plt.plot(snr_df['date'], snr_df['snr'], marker='o', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Signal-to-Noise Ratio')
        plt.title('Signal-to-Noise Ratio Over Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    # Subplot 4: Uncertainty estimates
    plt.subplot(2, 2, 4)
    if len(uncertainty_df) > 0:
        plt.fill_between(uncertainty_df['date'], uncertainty_df['ci_lower'],
                        uncertainty_df['ci_upper'], alpha=0.3, label='95% CI')
        plt.plot(uncertainty_df['date'], uncertainty_df['mean'],
                marker='o', linewidth=2, label='Mean')
        plt.xlabel('Date')
        plt.ylabel('Body Length (mm)')
        plt.title('Daily Means with 95% Confidence Intervals')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "variability_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Distribution analysis plot
    plt.figure(figsize=(15, 10))

    # Subplot 1: Box plots by date
    plt.subplot(2, 3, 1)
    filtered_data = []
    dates = []
    for date in DATE_ORDER:
        subset = analyzer.df_filtered[analyzer.df_filtered['date'] == date]['body_length_mm'].dropna()
        if len(subset) > 0:
            filtered_data.append(subset.tolist())
            dates.append(date)

    plt.boxplot(filtered_data, labels=dates)
    plt.xlabel('Date')
    plt.ylabel('Body Length (mm)')
    plt.title('Distribution by Date (Posture-filtered)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Overall distribution comparison
    plt.subplot(2, 3, 2)
    raw_all = analyzer.df_raw['body_length_mm'].dropna()
    filt_all = analyzer.df_filtered['body_length_mm'].dropna()

    plt.hist(raw_all, bins=30, alpha=0.5, label='Raw', density=True)
    plt.hist(filt_all, bins=30, alpha=0.5, label='Posture-filtered', density=True)
    plt.xlabel('Body Length (mm)')
    plt.ylabel('Density')
    plt.title('Overall Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Q-Q plot for normality check
    plt.subplot(2, 3, 3)
    stats.probplot(filt_all, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Posture-filtered Data vs Normal')
    plt.grid(True, alpha=0.3)

    # Subplot 4: Residuals analysis
    plt.subplot(2, 3, 4)
    all_residuals = []
    daily_means = analyzer.daily_stats_filtered.set_index('date')['mean_filtered']

    for date in DATE_ORDER:
        subset = analyzer.df_filtered[analyzer.df_filtered['date'] == date]['body_length_mm'].dropna()
        if len(subset) > 0 and date in daily_means.index:
            residuals = subset - daily_means[date]
            all_residuals.extend(residuals.tolist())

    if len(all_residuals) > 0:
        plt.hist(all_residuals, bins=30, density=True, alpha=0.7)
        plt.xlabel('Residuals from Daily Mean (mm)')
        plt.ylabel('Density')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)

    # Subplot 5: Temporal trend
    plt.subplot(2, 3, 5)
    if len(valid_stats) > 0:
        day_index = np.arange(len(valid_stats))
        plt.scatter(day_index, valid_stats['mean_filtered'], alpha=0.7)

        # Fit trend line with error handling
        try:
            if len(valid_stats) > 1:  # Need at least 2 points for trend
                z = np.polyfit(day_index, valid_stats['mean_filtered'], 1)
                p = np.poly1d(z)
                plt.plot(day_index, p(day_index), "r--", alpha=0.8, label=f'Trend: {z[0]:.3f} mm/day')
                plt.legend()
        except (np.linalg.LinAlgError, np.RankWarning):
            # Skip trend line if SVD doesn't converge
            print("Warning: Could not fit trend line due to numerical issues")

        plt.xlabel('Day Index')
        plt.ylabel('Mean Body Length (mm)')
        plt.title('Temporal Trend Analysis')
        plt.grid(True, alpha=0.3)

    # Subplot 6: Sample size per date
    plt.subplot(2, 3, 6)
    x = np.arange(len(valid_stats))
    width = 0.35
    plt.bar(x - width/2, valid_stats['count_raw'], width, label='Raw', alpha=0.7)
    plt.bar(x + width/2, valid_stats['count_filtered'], width, label='Posture-filtered', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Sample Size')
    plt.title('Sample Size Comparison')
    plt.xticks(x, valid_stats['date'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualizations to: {fig_dir}")

# ============================================================
#  REPORT GENERATION
# ============================================================
def generate_summary_report(analyzer: ConsistencyAnalyzer, all_results: Dict):
    """Generate a comprehensive summary report."""

    report_file = OUTPUT_DIR / "consistency_analysis_report.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS: CONSISTENCY AND NOISE CHARACTERISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: {PREDICTIONS_FILE}\n")
        f.write("\n")

        # Data Summary
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total larvae analyzed: {len(analyzer.df_raw)}\n")
        f.write(f"Posture-filtered larvae: {len(analyzer.df_filtered)}\n")
        f.write(f"Filtering efficiency: {len(analyzer.df_filtered)/len(analyzer.df_raw)*100:.1f}%\n")
        f.write(f"Date range: {DATE_ORDER[0]} to {DATE_ORDER[-1]}\n")
        f.write(f"Number of time points: {len(DATE_ORDER)}\n")
        f.write("\n")

        # Variability Analysis Results
        if 'variability' in all_results:
            var_results = all_results['variability']
            f.write("VARIABILITY ANALYSIS\n")
            f.write("-" * 40 + "\n")

            if 'overall_variability' in var_results:
                ov = var_results['overall_variability']
                f.write("Within-day variability:\n")
                f.write(f"  Raw data CV (mean): {ov['raw_mean_cv']:.2f}%\n")
                f.write(f"  Filtered data CV (mean): {ov['filtered_mean_cv']:.2f}%\n")
                f.write(f"  CV reduction: {ov['cv_reduction_percent']:.1f}%\n")
                f.write(f"  Range reduction: {ov['range_reduction_percent']:.1f}%\n")

            if 'between_day_variability' in var_results:
                bv = var_results['between_day_variability']
                f.write("\nBetween-day variability:\n")
                f.write(f"  Raw growth signal CV: {bv['raw_growth_cv']:.2f}%\n")
                f.write(f"  Filtered growth signal CV: {bv['filtered_growth_cv']:.2f}%\n")
                f.write(f"  Growth signal improvement: {bv['growth_signal_improvement']:.2f}%\n")
            f.write("\n")

        # Temporal Stability Results
        if 'temporal' in all_results:
            temp_results = all_results['temporal']
            f.write("TEMPORAL STABILITY ANALYSIS\n")
            f.write("-" * 40 + "\n")

            if 'linear_trend' in temp_results:
                lt = temp_results['linear_trend']
                f.write("Linear trend analysis:\n")
                f.write(f"  Slope: {lt['slope_mm_per_day']:.4f} mm/day\n")
                f.write(f"  R²: {lt['r_squared']:.3f}\n")
                f.write(f"  P-value: {lt['p_value']:.6f}\n")
                f.write(f"  Significant trend: {'Yes' if lt['trend_significant'] else 'No'}\n")

            if 'smoothness' in temp_results and 'residual_std' in temp_results['smoothness']:
                sm = temp_results['smoothness']
                f.write(f"\nSmoothness metrics:\n")
                f.write(f"  Residual standard deviation: {sm['residual_std']:.4f}\n")
                f.write(f"  Smoothness index: {sm['smoothness_index']:.4f}\n")

            if 'volatility' in temp_results:
                vol = temp_results['volatility']
                f.write(f"\nVolatility analysis:\n")
                f.write(f"  Mean daily change: {vol['mean_daily_change']:.4f} mm\n")
                f.write(f"  Standard deviation of changes: {vol['std_daily_change']:.4f} mm\n")
                f.write(f"  Maximum daily change: {vol['max_daily_change']:.4f} mm\n")
            f.write("\n")

        # Noise Analysis Results
        if 'noise' in all_results:
            noise_results = all_results['noise']
            f.write("NOISE CHARACTERISTICS\n")
            f.write("-" * 40 + "\n")

            if 'signal_to_noise' in noise_results:
                snr = noise_results['signal_to_noise']
                f.write("Signal-to-noise analysis:\n")
                f.write(f"  Mean SNR: {snr['mean_snr']:.2f}\n")
                f.write(f"  Mean SNR (dB): {snr['mean_snr_db']:.1f}\n")
                f.write(f"  SNR range: {snr['min_snr']:.2f} - {snr['max_snr']:.2f}\n")

            if 'noise_distribution' in noise_results:
                nd = noise_results['noise_distribution']
                f.write(f"\nNoise distribution:\n")
                f.write(f"  Residual mean: {nd['residual_mean']:.4f}\n")
                f.write(f"  Residual std: {nd['residual_std']:.4f}\n")
                f.write(f"  Skewness: {nd['residual_skewness']:.3f}\n")
                f.write(f"  Kurtosis: {nd['residual_kurtosis']:.3f}\n")
                f.write(f"  Normal distribution: {'Yes' if nd['is_normal_noise'] else 'No'} (p={nd['normality_p_value']:.4f})\n")
            f.write("\n")

        # Statistical Tests Results
        if 'tests' in all_results:
            test_results = all_results['tests']
            f.write("STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-" * 40 + "\n")

            if 'cv_reduction_test' in test_results:
                cv_test = test_results['cv_reduction_test']
                f.write("Variability reduction test:\n")
                f.write(f"  Mean CV before filtering: {cv_test['mean_cv_before']:.2f}%\n")
                f.write(f"  Mean CV after filtering: {cv_test['mean_cv_after']:.2f}%\n")
                f.write(f"  T-statistic: {cv_test['t_statistic']:.3f}\n")
                f.write(f"  P-value: {cv_test['p_value']:.6f}\n")
                f.write(f"  Significant reduction: {'Yes' if cv_test['significant_reduction'] else 'No'}\n")

            if 'trend_test' in test_results:
                trend_test = test_results['trend_test']
                f.write(f"\nMann-Kendall trend test:\n")
                f.write(f"  Z-statistic: {trend_test['mann_kendall_z']:.3f}\n")
                f.write(f"  P-value: {trend_test['mann_kendall_p']:.6f}\n")
                f.write(f"  Significant trend: {'Yes' if trend_test['significant_trend'] else 'No'}\n")
            f.write("\n")

        # Conclusions
        f.write("SUMMARY AND CONCLUSIONS\n")
        f.write("-" * 40 + "\n")

        conclusions = []

        if 'variability' in all_results and 'overall_variability' in all_results['variability']:
            cv_reduction = all_results['variability']['overall_variability']['cv_reduction_percent']
            if cv_reduction > 10:
                conclusions.append(f"• Posture filtering substantially reduces measurement variability ({cv_reduction:.1f}% CV reduction)")
            elif cv_reduction > 0:
                conclusions.append(f"• Posture filtering modestly reduces measurement variability ({cv_reduction:.1f}% CV reduction)")
            else:
                conclusions.append("• Posture filtering shows minimal effect on measurement variability")

        if 'temporal' in all_results and 'linear_trend' in all_results['temporal']:
            if all_results['temporal']['linear_trend']['trend_significant']:
                slope = all_results['temporal']['linear_trend']['slope_mm_per_day']
                r2 = all_results['temporal']['linear_trend']['r_squared']
                conclusions.append(f"• Significant temporal trend detected ({slope:.4f} mm/day, R²={r2:.3f})")
            else:
                conclusions.append("• No significant temporal trend detected in the growth signal")

        if 'noise' in all_results and 'signal_to_noise' in all_results['noise']:
            snr = all_results['noise']['signal_to_noise']['mean_snr']
            if snr > 10:
                conclusions.append(f"• High signal-to-noise ratio achieved (SNR = {snr:.1f})")
            elif snr > 5:
                conclusions.append(f"• Moderate signal-to-noise ratio (SNR = {snr:.1f})")
            else:
                conclusions.append(f"• Low signal-to-noise ratio may affect measurement precision (SNR = {snr:.1f})")

        if 'tests' in all_results and 'cv_reduction_test' in all_results['tests']:
            if all_results['tests']['cv_reduction_test']['significant_reduction']:
                conclusions.append("• Statistical testing confirms significant improvement in measurement consistency")

        for conclusion in conclusions:
            f.write(conclusion + "\n")

        if not conclusions:
            f.write("• Analysis completed. Detailed results available in CSV files and figures.\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")

    print(f"✓ Saved summary report to: {report_file}")

# ============================================================
#  NEW: PER-DATASET DAILY COMPARISON
# ============================================================

def compute_daily_comparison_summary(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Compute per-date statistics for RAW, VALID, and POSTURE-filtered datasets.

    RAW      : all detections
    VALID    : predicted_valid == 1
    POSTURE  : predicted_valid == 1 AND predicted_posture == 1

    For each date and dataset, compute:
      - n (sample size)
      - mean body_length_mm
      - standard deviation
      - coefficient of variation (CV)
      - min, max, range (max - min)

    Returns
    -------
    daily_df : pd.DataFrame
        Table with per-date stats for all three datasets.
    global_metrics : dict
        Aggregated metrics across all dates for the clean summary block.
    """
    required_cols = {"date", "body_length_mm", "predicted_valid", "predicted_posture"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Predictions DataFrame missing required columns: {required_cols - set(df.columns)}")

    df = df.copy()
    df["date"] = df["date"].astype(str)

    # Predefine containers
    records: list[dict] = []

    # Group by date and compute stats per subset
    for date, sub in df.groupby("date"):
        # RAW: all detections
        raw = sub["body_length_mm"].dropna().values
        valid_mask = sub["predicted_valid"] == 1
        posture_mask = valid_mask & (sub["predicted_posture"] == 1)
        valid = sub.loc[valid_mask, "body_length_mm"].dropna().values
        posture = sub.loc[posture_mask, "body_length_mm"].dropna().values

        def _stats(x: np.ndarray) -> tuple[int, float, float, float, float, float]:
            if x.size == 0:
                return 0, np.nan, np.nan, np.nan, np.nan, np.nan
            n = int(x.size)
            mean = float(np.mean(x))
            std = float(np.std(x, ddof=1)) if n > 1 else 0.0
            cv = float(std / mean * 100) if mean != 0 else np.nan
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            rng = x_max - x_min
            return n, mean, std, cv, x_min, rng

        n_raw, mean_raw, std_raw, cv_raw, _, range_raw = _stats(raw)
        n_valid, mean_valid, std_valid, cv_valid, _, range_valid = _stats(valid)
        n_posture, mean_posture, std_posture, cv_posture, _, range_posture = _stats(posture)

        records.append({
            "date": date,
            "n_raw": n_raw,
            "n_valid": n_valid,
            "n_posture": n_posture,
            "mean_raw": mean_raw,
            "mean_valid": mean_valid,
            "mean_posture": mean_posture,
            "std_raw": std_raw,
            "std_valid": std_valid,
            "std_posture": std_posture,
            "cv_raw": cv_raw,
            "cv_valid": cv_valid,
            "cv_posture": cv_posture,
            "range_raw": range_raw,
            "range_valid": range_valid,
            "range_posture": range_posture,
        })

    daily_df = pd.DataFrame.from_records(records).sort_values("date")

    # Save to CSV
    out_path = out_dir / "daily_comparison_summary.csv"
    daily_df.to_csv(out_path, index=False)

    # Compute global metrics across all dates (ignoring NaNs)
    total_raw = int(daily_df["n_raw"].sum())
    total_valid = int(daily_df["n_valid"].sum())
    total_posture = int(daily_df["n_posture"].sum())

    retention_rate = (total_posture / total_raw * 100.0) if total_raw > 0 else np.nan

    mean_cv_raw = float(daily_df["cv_raw"].mean(skipna=True))
    mean_cv_valid = float(daily_df["cv_valid"].mean(skipna=True))
    mean_cv_posture = float(daily_df["cv_posture"].mean(skipna=True))

    cv_reduction = (mean_cv_raw - mean_cv_posture) / mean_cv_raw * 100.0 if mean_cv_raw not in (0, np.nan) else np.nan

    mean_range_raw = float(daily_df["range_raw"].mean(skipna=True))
    mean_range_posture = float(daily_df["range_posture"].mean(skipna=True))
    range_reduction = (mean_range_raw - mean_range_posture) / mean_range_raw * 100.0 if mean_range_raw not in (0, np.nan) else np.nan

    global_metrics = {
        "total_raw": total_raw,
        "total_valid": total_valid,
        "total_posture": total_posture,
        "retention_rate": retention_rate,
        "mean_cv_raw": mean_cv_raw,
        "mean_cv_valid": mean_cv_valid,
        "mean_cv_posture": mean_cv_posture,
        "cv_reduction_raw_to_posture": cv_reduction,
        "mean_range_raw": mean_range_raw,
        "mean_range_posture": mean_range_posture,
        "range_reduction_raw_to_posture": range_reduction,
    }

    return daily_df, global_metrics


def print_measurement_consistency_summary(global_metrics: dict):
    """Print a clean, publication-ready summary block for Results section."""
    print("""
--------------------------------------------------
MEASUREMENT CONSISTENCY SUMMARY
--------------------------------------------------
Total detections: {total_raw:d}
Valid larvae: {total_valid:d}
Posture-filtered larvae: {total_posture:d}
Retention rate: {retention_rate:.1f} %

Mean CV (raw): {mean_cv_raw:.1f} %
Mean CV (valid): {mean_cv_valid:.1f} %
Mean CV (posture): {mean_cv_posture:.1f} %

CV reduction raw → posture: {cv_reduction_raw_to_posture:.1f} %

Mean range (raw): {mean_range_raw:.2f} mm
Mean range (posture): {mean_range_posture:.2f} mm
Range reduction: {range_reduction_raw_to_posture:.1f} %
--------------------------------------------------
""".format(**global_metrics))


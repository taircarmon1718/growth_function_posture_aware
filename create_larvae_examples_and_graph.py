#!/usr/bin/env python3
"""
Larva Examples and Length Analysis Script
==========================================
1. Extracts one example larva image from each date folder
2. Creates a summary figure showing all examples
3. Generates a graph of average body length per date

Author: Research Pipeline
Date: 2026-02-28
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ======================== CONFIGURATION ========================
ROOT_DIR = Path(__file__).parent
ANALYSIS_DIR = ROOT_DIR / "analysis_full"
OUTPUT_DIR = ROOT_DIR / "larvae_examples_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Output files
EXAMPLES_FIGURE = OUTPUT_DIR / "larvae_examples_per_date.png"
LENGTH_GRAPH = OUTPUT_DIR / "average_length_per_date.png"
SUMMARY_CSV = OUTPUT_DIR / "length_statistics_per_date.csv"

# ======================== FUNCTIONS ========================

def collect_larvae_per_date():
    """
    Collect all larvae from each date folder.

    Returns:
        dict: {date: [(image_name, larva_filename, larva_path, morphometrics_path), ...]}
    """
    print("="*70)
    print("SCANNING LARVAE ACROSS ALL DATES")
    print("="*70)

    date_folders = sorted([d for d in ANALYSIS_DIR.iterdir()
                          if d.is_dir() and d.name[0].isdigit()])

    larvae_by_date = {}

    print(f"\nFound {len(date_folders)} date folders")

    for date_folder in date_folders:
        date_name = date_folder.name
        larvae_list = []

        # Find all image folders
        image_folders = sorted([d for d in date_folder.iterdir() if d.is_dir()])

        for image_folder in image_folders:
            image_name = image_folder.name
            larvae_reports_dir = image_folder / "larvae_reports"
            morphometrics_csv = image_folder / "morphometrics.csv"

            if not larvae_reports_dir.exists():
                continue

            # Find all larva images
            larva_files = sorted([f for f in larvae_reports_dir.iterdir()
                                 if f.name.startswith("larva_") and f.name.endswith(".png")])

            for larva_file in larva_files:
                larvae_list.append((image_name, larva_file.name, larva_file, morphometrics_csv))

        if larvae_list:
            larvae_by_date[date_name] = larvae_list
            print(f"  {date_name}: {len(larvae_list)} larvae")

    return larvae_by_date


def extract_example_larvae(larvae_by_date, num_examples=1):
    """
    Extract example larvae from each date.

    Args:
        larvae_by_date: Dictionary of larvae per date
        num_examples: Number of examples per date

    Returns:
        dict: {date: [(larva_path, image_name, larva_filename), ...]}
    """
    print("\n" + "="*70)
    print("EXTRACTING EXAMPLE LARVAE")
    print("="*70)

    examples = {}

    for date, larvae_list in sorted(larvae_by_date.items()):
        # Take the first N larvae as examples
        selected = larvae_list[:num_examples]
        examples[date] = [(path, img_name, larva_name)
                         for img_name, larva_name, path, _ in selected]

        print(f"  {date}: Selected {len(selected)} example(s)")

    return examples


def create_examples_figure(examples):
    """
    Create a figure showing one example larva from each date.

    Args:
        examples: Dictionary of example larvae per date
    """
    print("\n" + "="*70)
    print("CREATING EXAMPLES FIGURE")
    print("="*70)

    dates = sorted(examples.keys())
    n_dates = len(dates)

    # Calculate grid size
    n_cols = 4
    n_rows = (n_dates + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, date in enumerate(dates):
        ax = axes[idx]

        # Get first example for this date
        larva_path, img_name, larva_name = examples[date][0]

        # Load image
        img = cv2.imread(str(larva_path))
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract visual portion only (remove text panel if present)
            h, w = img_rgb.shape[:2]
            gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            visual_width = w
            for x in range(w - 1, w // 2, -1):
                if np.mean(gray_check[:, x]) > 20:
                    visual_width = x + 1
                    break
            if visual_width == w:
                visual_width = int(w * 0.75)

            img_visual = img_rgb[:, :visual_width]

            ax.imshow(img_visual)
            ax.set_title(f"{date}\n{img_name}/{larva_name}", fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"{date}\nImage not found",
                   ha='center', va='center', fontsize=10)

        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_dates, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Example Larva from Each Date', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(EXAMPLES_FIGURE, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Examples figure saved to: {EXAMPLES_FIGURE}")


def calculate_length_statistics(larvae_by_date):
    """
    Calculate average body length per date from morphometrics.

    Args:
        larvae_by_date: Dictionary of larvae per date

    Returns:
        pd.DataFrame: Statistics per date
    """
    print("\n" + "="*70)
    print("CALCULATING LENGTH STATISTICS")
    print("="*70)

    stats_list = []

    for date, larvae_list in sorted(larvae_by_date.items()):
        lengths = []

        for image_name, larva_filename, larva_path, morphometrics_csv in larvae_list:
            # Try to extract length from morphometrics CSV
            if morphometrics_csv.exists():
                try:
                    df = pd.read_csv(morphometrics_csv)

                    # Extract larva ID from filename (e.g., larva_001.png -> 1)
                    larva_id = int(larva_filename.replace('larva_', '').replace('.png', ''))

                    # Find the row for this larva
                    larva_row = df[df['larva_id'] == larva_id]

                    if not larva_row.empty and 'body_length' in df.columns:
                        body_length = larva_row['body_length'].values[0]
                        if body_length > 0:  # Only include valid measurements
                            lengths.append(body_length)
                except Exception as e:
                    pass

        if lengths:
            stats_list.append({
                'date': date,
                'num_larvae': len(lengths),
                'avg_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'min_length': np.min(lengths),
                'max_length': np.max(lengths),
                'median_length': np.median(lengths)
            })

            print(f"  {date}: {len(lengths)} larvae, avg length = {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        else:
            print(f"  {date}: No valid length data")

    stats_df = pd.DataFrame(stats_list)

    # Save to CSV
    stats_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n✓ Statistics saved to: {SUMMARY_CSV}")

    return stats_df


def create_length_graph(stats_df):
    """
    Create a graph showing average body length per date.

    Args:
        stats_df: DataFrame with length statistics per date
    """
    print("\n" + "="*70)
    print("CREATING LENGTH GRAPH")
    print("="*70)

    if stats_df.empty:
        print("⚠️  No data to plot")
        return

    # Set style
    sns.set_style("whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot line with error bars
    x_pos = range(len(stats_df))
    ax.errorbar(x_pos, stats_df['avg_length'], yerr=stats_df['std_length'],
               fmt='o-', capsize=5, linewidth=2.5, markersize=10,
               color='#2E86AB', ecolor='#A23B72', capthick=2,
               label='Mean ± Std Dev')

    # Add points for min/max range
    ax.scatter(x_pos, stats_df['min_length'], marker='v', s=80,
              color='#F18F01', alpha=0.6, label='Min', zorder=3)
    ax.scatter(x_pos, stats_df['max_length'], marker='^', s=80,
              color='#C73E1D', alpha=0.6, label='Max', zorder=3)

    # Customize
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df['date'], rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Body Length (pixels)', fontsize=14, fontweight='bold')
    ax.set_title('Average Larva Body Length per Date',
                fontsize=16, fontweight='bold', pad=20)

    # Add sample size annotations
    for i, row in stats_df.iterrows():
        ax.text(i, row['avg_length'] + row['std_length'] + 10,
               f"n={row['num_larvae']}",
               ha='center', va='bottom', fontsize=9, color='gray')

    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(LENGTH_GRAPH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Length graph saved to: {LENGTH_GRAPH}")


def create_summary_report(stats_df):
    """
    Print a summary report to console and save to file.

    Args:
        stats_df: DataFrame with statistics
    """
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    report_lines = []
    report_lines.append("="*70)
    report_lines.append("LARVAE LENGTH ANALYSIS - SUMMARY REPORT")
    report_lines.append("="*70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append(f"Total dates analyzed: {len(stats_df)}")
    report_lines.append(f"Total larvae measured: {stats_df['num_larvae'].sum()}")
    report_lines.append("")
    report_lines.append("Per-Date Statistics:")
    report_lines.append("-"*70)

    for _, row in stats_df.iterrows():
        report_lines.append(f"\n{row['date']}:")
        report_lines.append(f"  Number of larvae: {row['num_larvae']}")
        report_lines.append(f"  Average length: {row['avg_length']:.2f} ± {row['std_length']:.2f} pixels")
        report_lines.append(f"  Range: {row['min_length']:.2f} - {row['max_length']:.2f} pixels")
        report_lines.append(f"  Median: {row['median_length']:.2f} pixels")

    report_lines.append("\n" + "="*70)
    report_lines.append("Overall Statistics:")
    report_lines.append("-"*70)
    report_lines.append(f"  Grand mean length: {stats_df['avg_length'].mean():.2f} pixels")
    report_lines.append(f"  Std dev of means: {stats_df['avg_length'].std():.2f} pixels")
    report_lines.append(f"  Range of means: {stats_df['avg_length'].min():.2f} - {stats_df['avg_length'].max():.2f} pixels")
    report_lines.append("="*70)

    report_text = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save to file
    report_file = OUTPUT_DIR / "summary_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to: {report_file}")


# ======================== MAIN ========================

def main():
    """Main execution."""
    print("="*70)
    print("LARVAE EXAMPLES AND LENGTH ANALYSIS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # Step 1: Collect all larvae
    larvae_by_date = collect_larvae_per_date()

    if not larvae_by_date:
        print("\n❌ No larvae found in analysis_full/")
        return

    # Step 2: Extract examples
    examples = extract_example_larvae(larvae_by_date, num_examples=1)

    # Step 3: Create examples figure
    create_examples_figure(examples)

    # Step 4: Calculate length statistics
    stats_df = calculate_length_statistics(larvae_by_date)

    # Step 5: Create length graph
    if not stats_df.empty:
        create_length_graph(stats_df)

    # Step 6: Create summary report
    if not stats_df.empty:
        create_summary_report(stats_df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print(f"  1. Examples figure: {EXAMPLES_FIGURE.name}")
    print(f"  2. Length graph: {LENGTH_GRAPH.name}")
    print(f"  3. Statistics CSV: {SUMMARY_CSV.name}")
    print(f"  4. Summary report: summary_report.txt")
    print(f"\nAll saved in: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()


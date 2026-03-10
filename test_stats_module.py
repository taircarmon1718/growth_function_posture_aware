#!/usr/bin/env python3
"""
Minimal test version of the statistical analysis module.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Configuration
ROOT_DIR = Path(__file__).parent.resolve()
GEODESIC_MODELS_DIR = ROOT_DIR / "dual_larva_models_geodesic"
PREDICTIONS_FILE = GEODESIC_MODELS_DIR / "predictions" / "predictions_all_larvae.xlsx"

def main():
    print("Testing basic functionality...")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Predictions file: {PREDICTIONS_FILE}")
    print(f"File exists: {PREDICTIONS_FILE.exists()}")

    if PREDICTIONS_FILE.exists():
        try:
            df = pd.read_excel(PREDICTIONS_FILE)
            print(f"Data loaded: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:\n{df.head(2)}")
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print("Predictions file not found. Please run the dual_larva_classification_pipeline_geodesic.py first.")

if __name__ == "__main__":
    main()

# Posture-Aware Image-Based Growth Monitoring of *Macrobrachium rosenbergii* Larvae

---

## Overview

Accurate monitoring of larval growth is essential for effective prawn hatchery management. This repository implements a posture-aware framework for extracting growth trajectories from image-based observations of *Macrobrachium rosenbergii* larvae.

The approach combines feature-based larval characterization with a hierarchical classification framework to filter invalid detections and isolate posture-suitable individuals prior to growth estimation. Posture-aware filtering reduced measurement variability by approximately **82%** and eliminated biologically inconsistent trends present in raw data.

Growth curves derived from the filtered measurements exhibited a clear sigmoidal pattern well described by the **Gompertz model** (R² = 0.954, RMSE = 0.356 mm), closely matching reference microscope-based measurements (R² = 0.991, RMSE = 0.186 mm). Temporal growth patterns show strong agreement (Spearman ρ = 0.98, DTW similarity = 0.91).

---

## Method

The pipeline operates in four stages:

### 1. Image Preprocessing and Segmentation

Input images (3024 × 4032 px, iPhone 11, top-down at 18 cm) are converted to grayscale. A two-stage saliency enhancement is applied — morphological dilation for background estimation followed by a black-hat transform — producing a saliency map:

```
S(x, y) = 0.7 · D(x, y) + 0.3 · B(x, y)
```

Binary segmentation via Otsu thresholding, followed by connected-component analysis with area filtering (80 ≤ A ≤ 0.05·H·W), yields larval candidates.

### 2. Morphometric Feature Extraction

For each candidate component, an 8-dimensional feature vector is extracted:

| Feature | Description |
|---------|-------------|
| `area` | Foreground pixel count |
| `aspect_ratio` | Bounding box height / width |
| `solidity` | Area / convex hull area |
| `perimeter` | Component perimeter |
| `mean_width` | Mean local body width (via distance transform) |
| `max_width` | Maximum local body width |
| `pca_variance_ratio` | λ₂/λ₁ from PCA on pixel coordinates |
| `body_length` | Principal-axis projection extent |

Body length is estimated by projecting foreground pixels onto the PCA dominant axis and converting to mm (1 pixel = 0.232255814 mm).

### 3. Hierarchical Classification (Posture-Aware Filtering)

Two sequential binary classifiers filter detections before growth analysis:

**Stage 1 — Valid-Larva Classifier** (Random Forest, threshold τ = 0.85)
- Distinguishes true larvae from debris and segmentation artifacts
- F₁ = 0.92, Precision = 0.87, Recall = 0.97

**Stage 2 — Posture-Suitability Classifier** (XGBoost)
- Applied only to valid larvae; identifies posture-suitable individuals (shape_score ≥ 1)
- F₁ = 0.93 for correct posture class, overall accuracy = 0.89

Only larvae passing both stages enter downstream growth analysis.

### 4. Gompertz Growth Curve Fitting

Daily median body lengths from posture-filtered larvae are fitted to the Gompertz growth model. The resulting growth trajectory is validated against a reference dataset from manual microscope-based measurements.

---

## Dataset

Images were collected across 11 sampling dates during larval development (October–November 2021): 18/10, 19/10, 20/10, 21/10, 24/10, 25/10, 26/10, 27/10, 29/10, 31/10, and 03/11. The first date (18/10) was excluded from morphometric analysis as larvae were below the effective imaging resolution.

- **330 images** total (30 images per sampling day)
- **11,011 larvae** detected in the raw dataset
- **1,200 larvae** manually annotated for classifier training

The date folders (e.g. `18.10/`, `19.10/`) contain the raw images. Processed outputs are written to `analysis_full/` and `analysis_primary/`.

---

## Repository Structure

```
growth_function_posture_aware/
│
├── run_pipeline.py                        # Main analysis pipeline (all dates)
├── run_pipeline_binary_masks.py           # Binary mask variant of the pipeline
├── batch_analysis_pipeline.py            # Batch processing across dates
├── larva_feature_extraction.py           # Single source of truth for feature extraction
│
├── dual_larva_classification_pipeline.py          # Hierarchical classifier (main)
├── dual_larva_classification_pipeline_improved.py # Improved variant
├── dual_larva_classification_pipeline_geodesic.py # Geodesic length variant
├── dual_larva_pipeline_full_labels.py             # Full-label training pipeline
├── dual_larva_pipeline_geodesic_fixed.py
├── dual_larva_pipeline_geodesic_original.py
│
├── interactive_labeling_tool.py          # Keyboard-driven manual annotation tool
├── filter_larvae_by_confidence.py        # Batch inference and confidence filtering
├── label_efficiency_ground_truth.py      # Annotation efficiency experiment
│
├── larva_analysis.py                     # Per-date larval analysis
├── larval_quality_scientific_analysis.py # Statistical quality analysis
├── raw_dataset_analysis.py               # Raw dataset statistics
├── central_tendency_analysis.py          # Central tendency exploration
├── statistical_analysis_consistency_noise.py
├── generate_summary.py                   # Summary table generation
│
├── scripts/                              # Figure generation and publication scripts
│   ├── gompertz_automated.py             # Automated Gompertz curve fitting
│   ├── generate_paper_figures.py
│   ├── plot_pipeline_figures.py
│   ├── create_graphical_abstract.py
│   └── ...
│
├── 18.10/ 19.10/ ... 3.11/              # Raw image folders by date
├── analysis_full/                        # Full pipeline outputs
├── analysis_primary/                     # Primary analysis outputs
├── figures/                              # Generated figures
├── outputs/                              # Evaluation outputs
├── feature_cache/                        # Cached extracted features
└── dual_larva_models/                    # Trained classifier models
```

---

## Key Results

| Stage | n | Mean body length |
|-------|---|-----------------|
| Raw detections | 11,011 | 8.17 mm (SD = 12.95) |
| After valid-larva filtering | 562 | 6.11 mm |
| After posture filtering | 476 | 5.93 mm |

Posture filtering reduced body length standard deviation by **~82%**, removing biologically inconsistent trends and enabling clean Gompertz curve fitting.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `opencv-python`, `scikit-image`, `scikit-learn`, `xgboost`, `scipy`, `pandas`, `numpy`, `matplotlib`, `joblib`

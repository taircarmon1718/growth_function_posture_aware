#!/usr/bin/env python3
"""
Deep debug: compare feature construction and filtering vs training predict_all.

Run from project root:
  python scripts/debug_filter_pipeline_deep.py

Requires: joblib, pandas, sklearn, openpyxl (for optional xlsx sample columns).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from larva_feature_extraction import (  # noqa: E402
    FEATURE_KEYS,
    assert_model_input_matrix,
    extract_features,
    features_dict_to_vector,
    inference_features_from_cache_row,
)

ANALYSIS_DIR = ROOT / "analysis_full_binary_masks_only"
CACHE_FILE = ROOT / "dual_larva_models_geodesic2" / "cached_all_features.pkl"
VALID_MODEL = ROOT / "dual_larva_models_geodesic2" / "valid_model" / "model.pkl"
POSTURE_MODEL = ROOT / "dual_larva_models_geodesic2" / "posture_model" / "model.pkl"
STATS_JSON = ROOT / "dual_larva_models_geodesic2" / "training_feature_stats.json"
XLSX = ROOT / "dual_larva_models_geodesic2" / "predictions" / "predictions_all_larvae.xlsx"
VALID_THRESHOLD = 0.85


def fix_date(d) -> str:
    s = str(d).strip()
    if s.endswith(".0"):
        s = s[:-2]
    parts = s.split(".")
    if len(parts) == 2 and parts[1].strip() == "1":
        return f"{parts[0].strip()}.10"
    return s


def resolve_img(row: pd.Series) -> Path:
    p = row.get("_img_path")
    if p is not None and str(p).strip():
        return Path(str(p))
    d = fix_date(row["date"])
    return ANALYSIS_DIR / d / str(row["image_name"]) / "larvae_reports" / str(row["larva_filename"])


def build_X_pipeline(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        path = resolve_img(row)
        feats = inference_features_from_cache_row(row, img_path=path, full_extract=False)
        if feats is None:
            continue
        rows.append(features_dict_to_vector(feats))
    if not rows:
        raise RuntimeError("No rows with readable larva PNGs for pipeline-aligned X.")
    X = np.vstack(rows).astype(np.float64)
    assert_model_input_matrix(X)
    return X


def build_X_full(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        path = resolve_img(row)
        f = extract_features(path)
        if f is None:
            continue
        rows.append(features_dict_to_vector(f))
    if not rows:
        raise RuntimeError("No rows with readable larva PNGs for full-extract X.")
    X = np.vstack(rows).astype(np.float64)
    assert_model_input_matrix(X)
    return X


def df_with_readable_pngs(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows as feature extraction: PNG exists and extract_features succeeds."""
    keep = []
    for i, row in df.iterrows():
        path = resolve_img(row)
        if extract_features(path) is not None:
            keep.append(i)
    return df.loc[keep].reset_index(drop=True)


def predict_training_style(valid_m, posture_m, X: np.ndarray):
    v_proba = valid_m.predict_proba(X)
    n = len(X)
    v_prob1 = v_proba[:, 1]
    v_pred = (v_prob1 >= VALID_THRESHOLD).astype(int)
    v_conf = v_proba[np.arange(n), v_pred]
    p_proba = posture_m.predict_proba(X)
    p_pred = np.argmax(p_proba, axis=1)
    p_conf = p_proba[np.arange(n), p_pred]
    keep = (v_pred == 1) & (v_conf >= VALID_THRESHOLD) & (p_pred == 1)
    return v_pred, v_conf, p_pred, p_conf, v_prob1, p_proba[:, 1], keep


def feat_row_str(X: np.ndarray, i: int) -> str:
    parts = []
    for j, k in enumerate(FEATURE_KEYS):
        parts.append(f"{k}={X[i, j]:.4g}")
    return "  " + " | ".join(parts)


def main():
    print("=" * 72)
    print("DEEP FILTER / TRAINING PIPELINE DEBUG")
    print("=" * 72)

    cache = joblib.load(CACHE_FILE)
    df = cache[cache["_valid"] == True].copy()
    print(f"\nLoaded cache: {len(cache)} rows, _valid True: {len(df)}")
    df = df_with_readable_pngs(df)
    print(f"After dropping rows without readable larva PNG / extract_features: {len(df)}")

    vm = joblib.load(VALID_MODEL)
    pm = joblib.load(POSTURE_MODEL)

    print("\n--- (1) Feature construction ---")
    print("Training predict_all uses: 7 features FROM PKL + body_length from extract_features(PNG).")
    print("extract_features() reads larvae_reports/*.png as BGR → gray → Otsu → invert if mean>0.5.")
    print("There is NO separate 'skeleton length' feature; body_length = PCA axis extent on bbox crop.")
    print("If segmentation merges larvae or leaves debris, those morphometrics still look 'larva-like' to RF.\n")

    print("Building X_pipeline (matches predict_all) …")
    Xp = build_X_pipeline(df)
    print("Building X_full (all 8 from extract_features) …")
    Xf = build_X_full(df)

    vp1, vcp, pp1, pcp, vp1s, pp1s, kp = predict_training_style(vm, pm, Xp)
    _, _, _, _, _, _, kf = predict_training_style(vm, pm, Xf)
    print(f"\nKeep count (pipeline X): {int(kp.sum())}")
    print(f"Keep count (full X):     {int(kf.sum())}")
    print(f"Rows where keep differs:   {int(np.sum(kp != kf))}  ← main cause of 'wrong' filter vs xlsx if you used full extract.")

    # Training stats
    if STATS_JSON.exists():
        with open(STATS_JSON, encoding="utf-8") as fh:
            ref = json.load(fh)
        print("\n--- (5) Inference (pipeline X) vs training_feature_stats.json (mean) ---")
        for j, name in enumerate(FEATURE_KEYS):
            cm = float(Xp[:, j].mean())
            rm = ref.get(name, {}).get("mean")
            if rm is None:
                continue
            rm = float(rm)
            rel = abs(cm - rm) / max(abs(rm), 1e-9)
            flag = " ⚠️" if rel > 0.35 else ""
            print(f"  {name:22s}  infer_mean={cm:.5g}  train_mean={rm:.5g}  |rel diff|={rel:.3f}{flag}")
    else:
        print(f"\n(no {STATS_JSON.name} — run training pipeline once to generate)")

    # Top 20 pass (pipeline): sort by valid_confidence * posture_confidence among kept
    idx_kept = np.where(kp)[0]
    if len(idx_kept):
        score = vcp[idx_kept] * pcp[idx_kept]
        order = idx_kept[np.argsort(-score)[:20]]
    else:
        order = np.array([], dtype=int)

    print("\n--- (3) Top 20 PASS filter (pipeline X), by valid_conf*posture_conf ---")
    for rank, i in enumerate(order, 1):
        path = resolve_img(df.iloc[i])
        print(f"\n#{rank}  {path}")
        print(f"  predicted_valid=1 valid_conf={vcp[i]:.4f}  P(valid=1)={vp1s[i]:.4f}")
        print(f"  predicted_posture={pp1[i]} posture_conf={pcp[i]:.4f}  P(posture=1)={pp1s[i]:.4f}")
        print(feat_row_str(Xp, i))

    # Fail but high confidence: valid passes threshold but posture 0, OR posture 1 but valid fail
    mask_fail = ~kp
    # "high conf" failures: max(v_prob1, p_prob1) high among failures
    hi = np.maximum(vp1s, pp1s)
    fail_idx = np.where(mask_fail)[0]
    if len(fail_idx):
        sub = fail_idx[np.argsort(-hi[fail_idx])[:20]]
    else:
        sub = np.array([], dtype=int)

    print("\n--- (3) Top 20 FAIL filter but HIGH max(P(valid=1), P(posture=1)) ---")
    for rank, i in enumerate(sub, 1):
        path = resolve_img(df.iloc[i])
        print(f"\n#{rank}  {path}")
        print(
            f"  v_pred={vp1[i]} valid_conf={vcp[i]:.4f} P(valid=1)={vp1s[i]:.4f} | "
            f"p_posture={pp1[i]} posture_conf={pcp[i]:.4f} P(posture=1)={pp1s[i]:.4f}"
        )
        print(feat_row_str(Xp, i))

    print("\n--- (4)(6)(7) Why junk can still PASS ---")
    print(
        "  (A) Segmentation: one PNG = one connected component; merged larvae / shards still yield finite area, solidity, aspect_ratio.\n"
        "  (B) Features: only 8 morphometrics — no texture / CNN; debris can mimic elongation (high aspect_ratio, body_length).\n"
        "  (C) Model: RF was fit on human labels on similar features; it can be overconfident on OOD shapes.\n"
        "  (D) Threshold 0.85 on P(valid=1) is sharp but does not guarantee human-visible 'larva' if training noise exists.\n"
        "  (E) Using FILTER_FULL_EXTRACT_FEATURES=1 in filter_larvae_by_confidence.py changed X vs predict_all → different passes."
    )

    if XLSX.exists():
        try:
            xdf = pd.read_excel(XLSX)
            n_x = int(
                (
                    (pd.to_numeric(xdf["predicted_valid"], errors="coerce") == 1)
                    & (pd.to_numeric(xdf["predicted_posture"], errors="coerce") == 1)
                ).sum()
            )
            print(f"\n--- xlsx rows (predicted_valid==1 & predicted_posture==1): {n_x}")
            print(f"    pipeline keep from this script's logic on cache rows: {int(kp.sum())}")
        except Exception as e:
            print(f"\n(could not read xlsx: {e})")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()

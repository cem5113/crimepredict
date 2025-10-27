#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_train_mid.py
MID motoru (3–30 gün / 72–720 saat) için OOF Stacking + Kalibrasyon eğitim scripti.

Girdi:
  data/features/mid/features_h{H}.parquet
    Zorunlu asgari kolonlar:
      geoid, category, datetime, Y_label_h{H}
    Tipik feature'lar:
      hour, dow, month, is_weekend, is_holiday, season,
      ev_sum_*h, ev_mean_*h, y_sum_*h, y_mean_*h, ev_ewm_24h, y_ewm_24h,
      (varsa) neighbor_crime_*, poi_*, distance_*, population,
      (ops.) 911/311 lag/proxy/forecast sütunları (gelecek sızıntısı yoksa!)

Çıktılar:
  models/mid/
    base_xgb_h{H}_C{cat}.pkl
    base_lgb_h{H}_C{cat}.pkl
    (ops.) base_rf_h{H}_C{cat}.pkl
    meta_stack_h{H}_C{cat}.pkl
    calibrator_h{H}_C{cat}.pkl
    feature_order_h{H}_C{cat}.json
  metrics/mid/
    h{H}_C{cat}_aucpr.json
    h{H}_C{cat}_precision_at_k.json
    h{H}_C{cat}_calibration.json
"""

import os
import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib


# -----------------------
# Argümanlar
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, default="data/features/mid",
                    help="Horizon bazlı feature dosyalarının klasörü (features_h{H}.parquet)")
    ap.add_argument("--out_models", type=str, default="models/mid",
                    help="Model artefaktlarının yazılacağı klasör")
    ap.add_argument("--out_metrics", type=str, default="metrics/mid",
                    help="Metrik dosyalarının yazılacağı klasör")
    ap.add_argument("--horizons", type=str, default="96,168,336,504,720",
                    help="Saat cinsinden ufuk listesi (virgülle ayrılmış)")
    ap.add_argument("--n_folds", type=int, default=3,
                    help="TimeSeriesSplit kat sayısı")
    ap.add_argument("--precision_k", type=str, default="20,50",
                    help="Precision@K için K değerleri (virgülle)")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--use_rf", type=str, default="False",
                    help="RF base modelini de dahil et (True/False)")
    return ap.parse_args()


# -----------------------
# Yardımcılar
# -----------------------
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def list_categories(df: pd.DataFrame) -> List[str]:
    cats = df["category"].dropna().astype(str).unique().tolist()
    return sorted(cats)


def pick_features(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    MID için özellik seçimi:
    - Kimlik/zaman/label dışındaki sayısal sütunlar
    - Objeler (string) otomatik dışlanır
    - Gelecek sızıntısı yaratabilecek isim kalıpları elenir
    """
    leak_like = ("timestamp", "target", "label_h", "y_label_h",
                 "future_", "lead_", "ahead_", "horizon_", "y_label", "Y_label_h")
    drop_cols = {"geoid", "category", "datetime", label_col}

    Xcols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if df[c].dtype == "O":
            continue
        low = c.lower()
        if any(tok in low for tok in leak_like):
            continue
        Xcols.append(c)

    if not Xcols:
        raise ValueError("Feature listesi boş kaldı! (filtre çok agresif olabilir)")

    X = df[Xcols].astype("float32")
    y = df[label_col].astype(int)
    return X, y, Xcols


def time_series_splits(df: pd.DataFrame, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Zaman sıralı fold'lar (leakage yok).
    """
    order = df["datetime"].values.argsort(kind="mergesort")
    tscv = TimeSeriesSplit(n_splits=n_folds)
    splits = []
    for tr, va in tscv.split(order):
        splits.append((order[tr], order[va]))
    return splits


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos <= 0:
        return 1.0
    return float(neg / max(pos, 1))


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def precision_at_k_per_timestamp(df: pd.DataFrame, prob_col: str, k: int, label_col: str) -> float:
    """
    Her timestamp için olasılığa göre sıralayıp Top-K içindeki etiket oranının ortalaması.
    """
    precs = []
    for _, g in df.groupby("datetime", sort=False):
        top = g.sort_values(prob_col, ascending=False).head(k)
        if len(top) == 0:
            continue
        precs.append(float(top[label_col].mean()))
    return float(np.mean(precs)) if precs else 0.0


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "prob_mean": None, "label_rate": None})
        else:
            out.append({
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": n,
                "prob_mean": float(y_prob[mask].mean()),
                "label_rate": float(y_true[mask].mean()),
            })
    return {"bins": out}


# -----------------------
# Eğitim / Stacking
# -----------------------
def train_one_horizon_category(
    df: pd.DataFrame,
    H: int,
    cat: str,
    n_folds: int,
    out_models_dir: str,
    out_metrics_dir: str,
    rs: int,
    k_list: List[int],
    use_rf: bool = False
):
    label_col = f"Y_label_h{H}"
    dfx = df[df["category"].astype(str) == str(cat)].copy()
    if dfx.empty:
        print(f"[MID][H{H}][{cat}] veri yok, atlandı.")
        return

    # datetime güvenliği
    dfx["datetime"] = pd.to_datetime(dfx["datetime"], errors="coerce", utc=False)
    dfx = dfx.dropna(subset=["datetime"])
    if dfx.empty:
        print(f"[MID][H{H}][{cat}] 'datetime' sonrası veri yok, atlandı.")
        return

    # Özellik/etiket
    X, y, Xcols = pick_features(dfx, label_col)
    spw = compute_scale_pos_weight(y)

    # Zaman bazlı fold'lar
    splits = time_series_splits(dfx, n_folds)
    if not splits:
        print(f"[MID][H{H}][{cat}] split üretilemedi, atlanıyor.")
        return

    # Base modeller
    xgb = XGBClassifier(
        max_depth=6, n_estimators=1000, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        reg_lambda=1.0, random_state=rs, n_jobs=-1, eval_metric="aucpr",
        scale_pos_weight=spw
    )
    lgb = LGBMClassifier(
        num_leaves=63, n_estimators=1200, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8, objective="binary",
        random_state=rs, n_jobs=-1
    )
    rf = None
    if use_rf:
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=rs
        )

    # OOF matris
    oof = pd.DataFrame({"idx": np.arange(len(dfx)), "y": y.values})
    oof["p_xgb"] = np.nan
    oof["p_lgb"] = np.nan
    if use_rf:
        oof["p_rf"] = np.nan

    fold_metrics = []
    for fi, (tr_idx, va_idx) in enumerate(splits, start=1):
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]

        # Fit base modeller
        xgb.fit(Xtr, ytr)
        lgb.fit(Xtr, ytr)
        if use_rf:
            rf.fit(Xtr, ytr)

        # Valid tahminleri (OOF)
        oof.loc[va_idx, "p_xgb"] = xgb.predict_proba(Xva)[:, 1]
        oof.loc[va_idx, "p_lgb"] = lgb.predict_proba(Xva)[:, 1]
        if use_rf:
            oof.loc[va_idx, "p_rf"] = rf.predict_proba(Xva)[:, 1]

        # Fold metrikleri
        ap_xgb = pr_auc(yva.values, oof.loc[va_idx, "p_xgb"].values)
        ap_lgb = pr_auc(yva.values, oof.loc[va_idx, "p_lgb"].values)
        rec = {"fold": fi, "ap_xgb": ap_xgb, "ap_lgb": ap_lgb}
        if use_rf:
            rec["ap_rf"] = pr_auc(yva.values, oof.loc[va_idx, "p_rf"].values)
        fold_metrics.append(rec)

    # Meta model (OOF ile)
    meta_inputs = ["p_xgb", "p_lgb"] + (["p_rf"] if use_rf else [])
    meta = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
    meta.fit(oof[meta_inputs].fillna(0.0), y.values)

    # Kalibrasyon
    cal = CalibratedClassifierCV(base_estimator=meta, method="isotonic", cv="prefit")
    cal.fit(oof[meta_inputs].fillna(0.0), y.values)

    # Full fit: base modelleri tüm veride yeniden eğit
    xgb.fit(X, y)
    lgb.fit(X, y)
    if use_rf:
        rf.fit(X, y)

    # OOF meta tahmini + kalibre
    meta_cal = cal.predict_proba(oof[meta_inputs].fillna(0.0))[:, 1]

    # Metrikler (OOF)
    auc_roc = float(roc_auc_score(y, meta_cal))
    ap_pr   = float(average_precision_score(y, meta_cal))

    # Precision@K
    eval_df = pd.DataFrame({
        "datetime": dfx["datetime"].values,
        "geoid": dfx["geoid"].astype(str).values,
        "y": y.values,
        "p_stack": meta_cal
    })
    prec_k = {}
    for k in k_list:
        prec_k[f"P@{k}"] = precision_at_k_per_timestamp(eval_df, "p_stack", k, "y")

    # Calibration bins
    cal_bins = reliability_bins(y.values, meta_cal, n_bins=10)

    # Kaydet
    os.makedirs(out_models_dir, exist_ok=True)
    os.makedirs(out_metrics_dir, exist_ok=True)
    tag = f"h{H}_C{cat}"

    joblib.dump(xgb, os.path.join(out_models_dir, f"base_xgb_{tag}.pkl"))
    joblib.dump(lgb, os.path.join(out_models_dir, f"base_lgb_{tag}.pkl"))
    if use_rf:
        joblib.dump(rf, os.path.join(out_models_dir, f"base_rf_{tag}.pkl"))
    joblib.dump(meta, os.path.join(out_models_dir, f"meta_stack_{tag}.pkl"))
    joblib.dump(cal,  os.path.join(out_models_dir, f"calibrator_{tag}.pkl"))

    with open(os.path.join(out_models_dir, f"feature_order_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"features": Xcols}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_metrics_dir, f"{tag}_aucpr.json"), "w", encoding="utf-8") as f:
        json.dump({
            "horizon_h": H,
            "category": cat,
            "oof_roc_auc": auc_roc,
            "oof_pr_auc": ap_pr,
            "fold_base_ap": fold_metrics
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_metrics_dir, f"{tag}_precision_at_k.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_h": H, "category": cat, **prec_k}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_metrics_dir, f"{tag}_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_h": H, "category": cat, **cal_bins}, f, ensure_ascii=False, indent=2)

    print(f"[OK][MID] H={H}, C={cat} | OOF PR-AUC={ap_pr:.4f} ROC-AUC={auc_roc:.4f} | {prec_k}")


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    ensure_dirs(args.out_models, args.out_metrics)

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    k_list   = [int(k.strip()) for k in args.precision_k.split(",") if k.strip()]
    use_rf   = str(args.use_rf).lower() in ("1", "true", "yes", "y")

    for H in horizons:
        path = os.path.join(args.features_dir, f"features_h{H}.parquet")
        if not os.path.exists(path):
            print(f"[Uyarı][MID] {path} bulunamadı, atlanıyor.")
            continue

        df = pd.read_parquet(path)

        # Minimum kolon kontrolleri
        min_cols = {"geoid", "category", "datetime"}
        missing = [c for c in min_cols if c not in df.columns]
        if missing:
            print(f"[Uyarı][MID] {path} dosyasında eksik kolon(lar): {missing} — atlanıyor.")
            continue

        # datetime güvenliği
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)

        label_col = f"Y_label_h{H}"
        if label_col not in df.columns:
            print(f"[Uyarı][MID] {label_col} yok ({path}), atlanıyor.")
            continue

        # Temel temizlik
        df = df.dropna(subset=[label_col, "datetime", "geoid", "category"]).copy()
        if df.empty:
            print(f"[Uyarı][MID] {path} temizleme sonrası boş — atlanıyor.")
            continue

        cats = list_categories(df)

        for cat in cats:
            try:
                train_one_horizon_category(
                    df=df,
                    H=H,
                    cat=cat,
                    n_folds=args.n_folds,
                    out_models_dir=args.out_models,
                    out_metrics_dir=args.out_metrics,
                    rs=args.random_state,
                    k_list=k_list,
                    use_rf=use_rf
                )
            except Exception as e:
                print(f"[H{H}][{cat}] HATA: {e}")

    print("✅ 03_train_mid.py tamamlandı.")


if __name__ == "__main__":
    main()

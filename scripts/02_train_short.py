#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_short.py
SHORT motor (0–72 saat) için OOF Stacking + Kalibrasyon eğitim scripti.

Girdi:
  data/features/short/features_h{H}.parquet   (H ∈ {1,2,3,6,12,24,48,72} önerilen)
    Kolonlar (en az): geoid, category, datetime, hour, dow, month, is_weekend,
                      ev_sum_*h, ev_mean_*h, y_sum_*h, y_mean_*h, ev_ewm_24h, y_ewm_24h,
                      Y_label_h{H}

Çıktılar:
  models/short/
    base_xgb_h{H}_C{cat}.pkl
    base_lgb_h{H}_C{cat}.pkl
    base_rf_h{H}_C{cat}.pkl
    meta_stack_h{H}_C{cat}.pkl
    calibrator_h{H}_C{cat}.pkl
    feature_order_h{H}_C{cat}.json
  metrics/short/
    h{H}_C{cat}_aucpr.json
    h{H}_C{cat}_precision_at_k.json
    h{H}_C{cat}_calibration.json

Kullanım:
  python scripts/02_train_short.py \
    --features_dir data/features/short \
    --out_models models/short \
    --out_metrics metrics/short \
    --horizons 1,2,3,6,12,24,48,72 \
    --n_folds 4 \
    --precision_k 20,50
"""

import os
import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
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
    ap.add_argument("--features_dir", type=str, default="data/features/short")
    ap.add_argument("--out_models", type=str, default="models/short")
    ap.add_argument("--out_metrics", type=str, default="metrics/short")
    ap.add_argument("--horizons", type=str, default="1,2,3,6,12,24,48,72")
    ap.add_argument("--n_folds", type=int, default=4)
    ap.add_argument("--precision_k", type=str, default="20,50")
    ap.add_argument("--random_state", type=int, default=42)
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
    # Hedef dışındaki sayısal sütunları al (id/zaman hariç).
    drop_cols = {"geoid", "category", "datetime", label_col}
    Xcols = [c for c in df.columns if c not in drop_cols and df[c].dtype != "O"]
    X = df[Xcols].astype("float32")
    y = df[label_col].astype(int)
    return X, y, Xcols


def time_series_splits(df: pd.DataFrame, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Zaman sırasını bozmadan fold’lar üretir.
    df 'datetime' içeriyor olmalı.
    """
    order = df["datetime"].values.argsort(kind="mergesort")
    tscv = TimeSeriesSplit(n_splits=n_folds)
    splits = []
    for tr_idx, va_idx in tscv.split(order):
        splits.append((order[tr_idx], order[va_idx]))
    return splits


def compute_class_weight(y: pd.Series) -> float:
    # scale_pos_weight ≈ neg/pos (XGB/LGB için)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0:
        return 1.0
    return float(neg / max(pos, 1))


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def precision_at_k_per_timestamp(df: pd.DataFrame, prob_col: str, k: int, label_col: str) -> float:
    """
    Her timestamp için prob'a göre sıralayıp Top-K içinde kaç pozitif yakalanmış? -> ortalama precision.
    df: en azından ['datetime','geoid',prob_col,label_col] içerir
    """
    precs = []
    for ts, g in df.groupby("datetime", sort=False):
        g = g.sort_values(prob_col, ascending=False)
        top = g.head(k)
        if len(top) == 0:
            continue
        prec = top[label_col].mean()  # (TP / K)
        precs.append(prec)
    return float(np.mean(precs)) if precs else 0.0


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if m.sum() == 0:
            out.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "prob_mean": None, "label_rate": None})
        else:
            out.append({
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": int(m.sum()),
                "prob_mean": float(y_prob[m].mean()),
                "label_rate": float(y_true[m].mean())
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
    k_list: List[int]
):
    label_col = f"Y_label_h{H}"
    dfc = df[df["category"].astype(str) == str(cat)].copy()
    if dfc.empty:
        print(f"[H{H}][{cat}] VERİ YOK, atlanıyor.")
        return

    # Özellik/etiket
    X, y, Xcols = pick_features(dfc, label_col)
    spw = compute_class_weight(y)

    # Fold’ları oluştur
    splits = time_series_splits(dfc, n_folds)
    oof = pd.DataFrame({"idx": np.arange(len(dfc)), "y": y.values})

    # Base modeller
    xgb = XGBClassifier(
        max_depth=6, n_estimators=800, learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        reg_lambda=1.0, random_state=rs, n_jobs=-1, eval_metric="aucpr",
        scale_pos_weight=spw
    )
    lgb = LGBMClassifier(
        num_leaves=63, n_estimators=1000, learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.8, objective="binary",
        random_state=rs, n_jobs=-1, class_weight=None  # pos_weight, is_unbalance alternativa
    )
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=rs
    )

    # OOF üretimi
    oof["p_xgb"] = np.nan
    oof["p_lgb"] = np.nan
    oof["p_rf"]  = np.nan

    fold_metrics = []
    for fi, (tr_idx, va_idx) in enumerate(splits, start=1):
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]

        # Fit base’ler
        xgb.fit(Xtr, ytr)
        lgb.fit(Xtr, ytr)
        rf.fit(Xtr, ytr)

        # Valid tahminleri (OOF)
        oof.loc[va_idx, "p_xgb"] = xgb.predict_proba(Xva)[:, 1]
        oof.loc[va_idx, "p_lgb"] = lgb.predict_proba(Xva)[:, 1]
        oof.loc[va_idx, "p_rf"]  = rf.predict_proba(Xva)[:, 1]

        # Fold metrikleri (PR-AUC ağırlıklı)
        ap_xgb = pr_auc(yva.values, oof.loc[va_idx, "p_xgb"].values)
        ap_lgb = pr_auc(yva.values, oof.loc[va_idx, "p_lgb"].values)
        ap_rf  = pr_auc(yva.values, oof.loc[va_idx, "p_rf"].values)
        fold_metrics.append({"fold": fi, "ap_xgb": ap_xgb, "ap_lgb": ap_lgb, "ap_rf": ap_rf})

    # Meta model (OOF ile)
    meta = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
    meta.fit(oof[["p_xgb", "p_lgb", "p_rf"]].fillna(0.0), y.values)

    # Kalibrasyon (isotonic; gerekirse platt)
    # Kalibrasyon için OOF’u kullanmak iyi pratik: gerçek hatayı yansıtır.
    cal = CalibratedClassifierCV(base_estimator=meta, method="isotonic", cv="prefit")
    cal.fit(oof[["p_xgb", "p_lgb", "p_rf"]].fillna(0.0), y.values)

    # Full fit: base modelleri tüm veride yeniden eğit (inference’ta kullanılacak)
    xgb.fit(X, y)
    lgb.fit(X, y)
    rf.fit(X, y)

    # OOF meta tahmini + kalibre
    meta_raw = meta.predict_proba(oof[["p_xgb", "p_lgb", "p_rf"]].fillna(0.0))[:, 1]
    meta_cal = cal.predict_proba(oof[["p_xgb", "p_lgb", "p_rf"]].fillna(0.0))[:, 1]

    # Metrikler (OOF üzerinde rapor)
    auc_roc = float(roc_auc_score(y, meta_cal))
    ap_pr   = float(average_precision_score(y, meta_cal))

    # Precision@K (OOF valid tahminlerini datetime bazında değerlendir)
    eval_df = pd.DataFrame({
        "datetime": dfc["datetime"].values,
        "geoid": dfc["geoid"].astype(str).values,
        "y": y.values,
        "p_stack": meta_cal
    })
    prec_k = {}
    for k in k_list:
        prec_k[f"P@{k}"] = precision_at_k_per_timestamp(eval_df, "p_stack", k, "y")

    # Calibration bins
    cal_bins = reliability_bins(y.values, meta_cal, n_bins=10)

    # Kaydetme
    ensure_dirs(out_models_dir, out_metrics_dir)
    tag = f"h{H}_C{cat}"

    joblib.dump(xgb, os.path.join(out_models_dir, f"base_xgb_{tag}.pkl"))
    joblib.dump(lgb, os.path.join(out_models_dir, f"base_lgb_{tag}.pkl"))
    joblib.dump(rf,  os.path.join(out_models_dir, f"base_rf_{tag}.pkl"))
    joblib.dump(meta,os.path.join(out_models_dir, f"meta_stack_{tag}.pkl"))
    joblib.dump(cal, os.path.join(out_models_dir, f"calibrator_{tag}.pkl"))

    with open(os.path.join(out_models_dir, f"feature_order_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"features": Xcols}, f, ensure_ascii=False, indent=2)

    # Metrikleri yaz
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

    print(f"[OK] SHORT train tamam: H={H}, C={cat} | OOF PR-AUC={ap_pr:.4f} ROC-AUC={auc_roc:.4f} | {prec_k}")


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    ensure_dirs(args.out_models, args.out_metrics)

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    k_list   = [int(k.strip()) for k in args.precision_k.split(",") if k.strip()]

    for H in horizons:
        path = os.path.join(args.features_dir, f"features_h{H}.parquet")
        if not os.path.exists(path):
            print(f"[Uyarı] {path} bulunamadı, atlanıyor.")
            continue

        df = pd.read_parquet(path)
        # Kategori listesi
        cats = list_categories(df)

        # Temizlik: eksik label’ları at
        label_col = f"Y_label_h{H}"
        if label_col not in df.columns:
            print(f"[Uyarı] {label_col} yok ({path}), atlanıyor.")
            continue
        df = df.dropna(subset=[label_col, "datetime", "geoid", "category"]).copy()

        # Eğitim
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
                    k_list=k_list
                )
            except Exception as e:
                print(f"[H{H}][{cat}] HATA: {e}")

    print("✅ 02_train_short.py tamamlandı.")


if __name__ == "__main__":
    main()

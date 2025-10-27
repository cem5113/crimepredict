#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_long.py
LONG motor (1–6 ay; ör. H ∈ {960, 1440, 2160}) için eğitim.
Yaklaşım: Mevsimsel baseline (geoid×category×hour×dow) + deterministik/statik özellikler
üzerinde Logistic Regression; ardından isotonic kalibrasyon.

Girdi:
  data/features/long/features_h{H}.parquet
    Zorunlu min. kolonlar: geoid, category, datetime, hour, dow, month, is_weekend,
                           (ops.: is_holiday, season, statikler: poi_*, distance_*, population),
                           Y_label_h{H}

Çıktılar:
  models/long/
    lr_h{H}_C{cat}.pkl
    calibrator_h{H}_C{cat}.pkl
    feature_order_h{H}_C{cat}.json
    baseline_map_h{H}_C{cat}.parquet
  metrics/long/
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
from sklearn.isotonic import IsotonicRegression
import joblib


# -----------------------
# Argümanlar
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, default="data/features/long")
    ap.add_argument("--out_models", type=str, default="models/long")
    ap.add_argument("--out_metrics", type=str, default="metrics/long")
    ap.add_argument("--horizons", type=str, default="960,1440,2160")
    ap.add_argument("--n_folds", type=int, default=3)
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
    return sorted(df["category"].dropna().astype(str).unique().tolist())

def time_series_splits(df: pd.DataFrame, n_folds: int):
    order = df["datetime"].values.argsort(kind="mergesort")
    tss = TimeSeriesSplit(n_splits=n_folds)
    return [(order[tr], order[va]) for tr, va in tss.split(order)]

def pr_auc(y_true, y_prob) -> float:
    return float(average_precision_score(y_true, y_prob))

def precision_at_k_per_ts(df: pd.DataFrame, prob_col: str, k: int, label_col: str) -> float:
    vals = []
    for _, g in df.groupby("datetime", sort=False):
        top = g.sort_values(prob_col, ascending=False).head(k)
        if len(top) == 0:
            continue
        vals.append(float(top[label_col].mean()))
    return float(np.mean(vals)) if vals else 0.0

def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "prob_mean": None, "label_rate": None})
        else:
            out.append({
                "bin": f"[{lo:.1f},{hi:.1f})", "n": n,
                "prob_mean": float(y_prob[mask].mean()),
                "label_rate": float(y_true[mask].mean())
            })
    return {"bins": out}

def build_baseline_map(train_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    (geoid,category,hour,dow) → rate; ayrıca category+hour+dow ve hour+dow fallback seviyeleri.
    """
    keys_full = ["geoid", "category", "hour", "dow"]
    keys_cat  = ["category", "hour", "dow"]
    keys_hd   = ["hour", "dow"]

    def agg_rate(df, keys):
        g = df.groupby(keys, observed=True)[label_col].mean().reset_index()
        g = g.rename(columns={label_col: "rate"})
        g["key_level"] = "_".join(keys)
        return g

    m_full = agg_rate(train_df, keys_full)
    m_cat  = agg_rate(train_df, keys_cat)
    m_hd   = agg_rate(train_df, keys_hd)

    for m in (m_full, m_cat, m_hd):
        m["rate"] = m["rate"].clip(0.0, 1.0)

    return pd.concat([m_full, m_cat, m_hd], ignore_index=True)

def attach_baseline(df: pd.DataFrame, baseline_map: pd.DataFrame, global_mean: float) -> pd.Series:
    """
    Baseline oranını bağlar; bulunamazsa category+hour+dow; yine yoksa hour+dow; en sonda global_mean.
    """
    out = pd.merge(
        df[["geoid", "category", "hour", "dow"]].copy(),
        baseline_map[baseline_map["key_level"] == "geoid_category_hour_dow"],
        on=["geoid","category","hour","dow"], how="left"
    )["rate"]

    miss = out.isna()
    if miss.any():
        tmp = pd.merge(
            df.loc[miss, ["category","hour","dow"]],
            baseline_map[baseline_map["key_level"] == "category_hour_dow"],
            on=["category","hour","dow"], how="left"
        )["rate"]
        out.loc[miss] = tmp.values

    miss = out.isna()
    if miss.any():
        tmp = pd.merge(
            df.loc[miss, ["hour","dow"]],
            baseline_map[baseline_map["key_level"] == "hour_dow"],
            on=["hour","dow"], how="left"
        )["rate"]
        out.loc[miss] = tmp.values

    if out.isna().any():
        out = out.fillna(float(global_mean))

    return out.clip(0.0, 1.0)

def pick_features(df: pd.DataFrame, label_col: str, extra_num_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    LONG için yalnız deterministik/statik + baseline kullanılacak.
    - Kimlik/zaman/label hariç numeric kolonlar
    - Gelecek sızıntısı yaratabilecek isim kalıpları dışlanır
    """
    drop_cols = {"geoid","category","datetime",label_col}
    forbidden_tokens = [
        "future_", "lead_", "ahead_", "horizon_", "911", "311", "wx_", "weather",
        "lag_", "prev_", "y_sum_", "y_mean_", "ev_sum_", "ev_mean_", "ewm"
    ]
    Xcols = []
    for c in df.columns:
        if c in drop_cols or df[c].dtype == "O":
            continue
        low = c.lower()
        if any(tok in low for tok in forbidden_tokens):
            continue
        Xcols.append(c)

    for c in extra_num_cols:
        if c in df.columns and c not in Xcols and pd.api.types.is_numeric_dtype(df[c]):
            Xcols.append(c)

    if not Xcols:
        raise ValueError("Feature listesi boş! (filtre çok agresif olabilir)")

    X = df[Xcols].astype("float32")
    y = df[label_col].astype(int)
    return X, y, Xcols


# -----------------------
# Eğitim
# -----------------------
def train_one_h_cat(df: pd.DataFrame, H: int, cat: str,
                    n_folds: int, out_models_dir: str, out_metrics_dir: str, rs: int,
                    k_list: List[int]):

    label_col = f"Y_label_h{H}"
    dfx = df[df["category"].astype(str) == str(cat)].copy()
    if dfx.empty:
        print(f"[LONG][H{H}][{cat}] veri yok, atlanıyor.")
        return

    dfx["datetime"] = pd.to_datetime(dfx["datetime"], errors="coerce", utc=False)
    dfx = dfx.dropna(subset=["datetime"])
    if dfx.empty:
        print(f"[LONG][H{H}][{cat}] 'datetime' sonrası veri yok, atlanıyor.")
        return

    # Zaman bazlı fold'lar
    splits = time_series_splits(dfx, n_folds)
    if not splits:
        print(f"[LONG][H{H}][{cat}] split üretilemedi, atlanıyor.")
        return

    # OOF konteyner
    oof_prob_raw = np.zeros(len(dfx), dtype="float32")

    fold_recs = []
    for fi, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr = dfx.iloc[tr_idx].copy()
        va = dfx.iloc[va_idx].copy()

        # Baseline haritası (train döneminden)
        baseline_map = build_baseline_map(tr, label_col)
        global_mean = float(tr[label_col].mean()) if label_col in tr.columns else 0.0

        # Baseline oranını ekle
        tr["baseline_rate"] = attach_baseline(tr, baseline_map, global_mean).astype("float32")
        va["baseline_rate"] = attach_baseline(va, baseline_map, global_mean).astype("float32")

        # Deterministik/statik + baseline
        extra = ["baseline_rate", "hour","dow","month","is_weekend"]
        if "is_holiday" in tr.columns: extra.append("is_holiday")

        Xtr, ytr, cols = pick_features(tr, label_col, extra_num_cols=extra)
        Xva, yva, _    = pick_features(va, label_col, extra_num_cols=extra)

        # LR
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
        lr.fit(Xtr, ytr)

        # Valid tahminleri (OOF raw)
        p_va = lr.predict_proba(Xva)[:, 1].astype("float32")
        oof_prob_raw[va_idx] = p_va

        # Fold metrikleri (raw)
        ap  = pr_auc(yva.values, p_va)
        roc = float(roc_auc_score(yva.values, p_va))
        fold_recs.append({"fold": fi, "pr_auc_raw": ap, "roc_auc_raw": roc})

    # Isotonic kalibrasyon (OOF üzerinde)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_prob_raw, dfx[label_col].values.astype(int))
    oof_prob_cal = iso.predict(oof_prob_raw).astype("float32")

    # OOF metrikleri (kalibre)
    auc_roc = float(roc_auc_score(dfx[label_col].values, oof_prob_cal))
    ap_pr   = float(average_precision_score(dfx[label_col].values, oof_prob_cal))

    # Precision@K (kalibre)
    eval_oof = pd.DataFrame({
        "datetime": dfx["datetime"].values,
        "geoid": dfx["geoid"].astype(str).values,
        "y": dfx[label_col].values.astype(int),
        "p_cal": oof_prob_cal
    })
    prec_k = {}
    tmp = eval_oof.rename(columns={"p_cal": "p_stack"})
    for k in k_list:
        prec_k[f"P@{k}"] = precision_at_k_per_ts(tmp, "p_stack", k, "y")

    cal_bins = reliability_bins(eval_oof["y"].values, oof_prob_cal, n_bins=10)

    # Full-train: final LR ve baseline_map
    baseline_map_global = build_baseline_map(dfx, label_col)
    global_mean_full = float(dfx[label_col].mean())
    dfx["baseline_rate"] = attach_baseline(dfx, baseline_map_global, global_mean_full).astype("float32")
    extra = ["baseline_rate", "hour","dow","month","is_weekend"]
    if "is_holiday" in dfx.columns: extra.append("is_holiday")
    Xfull, yfull, cols_full = pick_features(dfx, label_col, extra_num_cols=extra)
    lr_final = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
    lr_final.fit(Xfull, yfull)

    # Kaydet
    ensure_dirs(out_models_dir, out_metrics_dir)
    tag = f"h{H}_C{cat}"

    joblib.dump(lr_final, os.path.join(out_models_dir, f"lr_{tag}.pkl"))
    joblib.dump(iso,      os.path.join(out_models_dir, f"calibrator_{tag}.pkl"))

    with open(os.path.join(out_models_dir, f"feature_order_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"features": cols_full}, f, ensure_ascii=False, indent=2)

    baseline_path = os.path.join(out_models_dir, f"baseline_map_{tag}.parquet")
    baseline_map_global.to_parquet(baseline_path, index=False)

    with open(os.path.join(out_metrics_dir, f"{tag}_aucpr.json"), "w", encoding="utf-8") as f:
        json.dump({
            "horizon_h": H,
            "category": cat,
            "oof_pr_auc": ap_pr,
            "oof_roc_auc": auc_roc,
            "folds": fold_recs
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_metrics_dir, f"{tag}_precision_at_k.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_h": H, "category": cat, **prec_k}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_metrics_dir, f"{tag}_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_h": H, "category": cat, **cal_bins}, f, ensure_ascii=False, indent=2)

    print(f"[OK][LONG] H={H}, C={cat} | OOF (cal) PR-AUC={ap_pr:.4f} ROC-AUC={auc_roc:.4f} | baseline_map rows={len(baseline_map_global):,}")


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
            print(f"[Uyarı][LONG] {path} bulunamadı, atlanıyor.")
            continue

        df = pd.read_parquet(path)

        # Minimum kolonlar
        required = {"geoid", "category", "datetime", "hour", "dow", "month"}
        miss = [c for c in required if c not in df.columns]
        if miss:
            print(f"[Uyarı][LONG] {path} eksik kolon(lar): {miss} — atlanıyor.")
            continue

        # datetime güvenliği
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)

        label_col = f"Y_label_h{H}"
        if label_col not in df.columns:
            print(f"[Uyarı][LONG] {label_col} yok ({path}), atlanıyor.")
            continue

        # Temizlik
        df = df.dropna(subset=[label_col, "datetime", "geoid", "category", "hour", "dow", "month"]).copy()
        if df.empty:
            print(f"[Uyarı][LONG] {path} temizleme sonrası boş — atlanıyor.")
            continue

        cats = list_categories(df)
        for cat in cats:
            try:
                train_one_h_cat(
                    df=df, H=H, cat=cat,
                    n_folds=args.n_folds,
                    out_models_dir=args.out_models,
                    out_metrics_dir=args.out_metrics,
                    rs=args.random_state,
                    k_list=k_list
                )
            except Exception as e:
                print(f"[H{H}][{cat}] HATA: {e}")

    print("✅ 04_train_long.py tamamlandı.")


if __name__ == "__main__":
    main()

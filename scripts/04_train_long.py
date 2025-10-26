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
                           (opsiyonel: is_holiday, season, statikler: poi_*, distance_*, population),
                           Y_label_h{H}

Çıktılar:
  models/long/
    lr_h{H}_C{cat}.pkl                     # lojistik regresyon
    calibrator_h{H}_C{cat}.pkl             # isotonic kalibratör
    feature_order_h{H}_C{cat}.json         # inference'ta sütun sırası
    baseline_map_h{H}_C{cat}.parquet       # (geoid,category,hour,dow) → baseline_rate
  metrics/long/
    h{H}_C{cat}_aucpr.json
    h{H}_C{cat}_precision_at_k.json
    h{H}_C{cat}_calibration.json

Kullanım:
  python scripts/04_train_long.py \
    --features_dir data/features/long \
    --out_models models/long \
    --out_metrics metrics/long \
    --horizons 960,1440,2160 \
    --n_folds 3 \
    --precision_k 20,50
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
    from sklearn.metrics import average_precision_score
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
    Train döneminde Y_label üzerinden baseline oran tablosu üretir:
    (geoid,category,hour,dow) → rate. Fallback sütunları da üretir.
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

    # Oranları 0..1 aralığında güvenli tut
    for m in (m_full, m_cat, m_hd):
        m["rate"] = m["rate"].clip(0.0, 1.0)

    # Birleştirirken inference tarafında sırayla bakacağız (full → cat → hd)
    # Burada hepsini tek Parquet’te saklayalım, level kolonu ile.
    baseline_map = pd.concat([m_full, m_cat, m_hd], ignore_index=True)
    return baseline_map

def attach_baseline(df: pd.DataFrame, baseline_map: pd.DataFrame) -> pd.Series:
    """
    Train/valid sırasında baseline oranını (geoid,category,hour,dow) → rate olarak bağlar;
    bulunamazsa category+hour+dow; yine yoksa hour+dow fallback.
    """
    # Önce full anahtar
    out = pd.merge(
        df[["geoid", "category", "hour", "dow"]].copy(),
        baseline_map[baseline_map["key_level"] == "geoid_category_hour_dow"],
        on=["geoid","category","hour","dow"], how="left"
    )["rate"]

    # Fallback 1: category+hour+dow
    missing = out.isna()
    if missing.any():
        tmp = pd.merge(
            df.loc[missing, ["category","hour","dow"]],
            baseline_map[baseline_map["key_level"] == "category_hour_dow"],
            on=["category","hour","dow"], how="left"
        )["rate"]
        out.loc[missing] = tmp.values

    # Fallback 2: hour+dow
    missing = out.isna()
    if missing.any():
        tmp = pd.merge(
            df.loc[missing, ["hour","dow"]],
            baseline_map[baseline_map["key_level"] == "hour_dow"],
            on=["hour","dow"], how="left"
        )["rate"]
        out.loc[missing] = tmp.values

    # Son çare: global ortalama
    if out.isna().any():
        out = out.fillna(df[label_col].mean())

    return out.clip(0.0, 1.0)

def pick_features(df: pd.DataFrame, label_col: str, extra_num_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    LONG için yalnız deterministik/statik + baseline kullanılacak.
    - Kimlik/zaman/label hariç numeric kolonlar içinden güvenli olanları al.
    - İsim bazlı filtre: future/lead/911/311/weather gibi kolonları dışla.
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

    # Eklemek istediğimiz zorunlu deterministik/statik kolonlar
    for c in extra_num_cols:
        if c in df.columns and c not in Xcols:
            Xcols.append(c)

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

    # Fold'lar (zaman)
    splits = time_series_splits(dfx, n_folds)

    # OOF konteyner
    oof_idx = np.arange(len(dfx))
    oof_prob = np.zeros(len(dfx), dtype="float32")

    # Fold bazlı eğitim
    fold_recs = []
    feat_used = None
    baseline_map_global = None  # en son full-train'den kaydetmek için

    for fi, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr = dfx.iloc[tr_idx].copy()
        va = dfx.iloc[va_idx].copy()

        # 1) Train kısmında baseline haritası oluştur
        baseline_map = build_baseline_map(tr, label_col)

        # 2) Train/valid'e baseline_rate'i ekle
        tr["baseline_rate"] = attach_baseline(tr, baseline_map).astype("float32")
        va["baseline_rate"] = attach_baseline(va, baseline_map).astype("float32")

        # 3) Deterministik/statik kolonlar (güvenli) + baseline_rate
        extra = ["baseline_rate", "hour","dow","month","is_weekend"]
        if "is_holiday" in tr.columns: extra.append("is_holiday")
        if "season" in tr.columns:     extra.append("season")  # kategorik ise kullanılmayacak
        # season text ise atlanır; numeric değilse pick_features dışlar.

        Xtr, ytr, cols = pick_features(tr, label_col, extra_num_cols=extra)
        Xva, yva, _    = pick_features(va, label_col, extra_num_cols=extra)
        feat_used = cols  # son fold'un kolonlarını referans alacağız (aynı kalmalı)

        # 4) Lojistik regresyon (basit, açıklanabilir)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
        lr.fit(Xtr, ytr)

        # 5) Valid tahmini (OOF)
        p_va = lr.predict_proba(Xva)[:, 1].astype("float32")
        oof_prob[va_idx] = p_va

        # 6) Fold metrikleri
        ap  = pr_auc(yva.values, p_va)
        roc = float(roc_auc_score(yva.values, p_va))
        fold_recs.append({"fold": fi, "pr_auc": ap, "roc_auc": roc})

    # Kalibrasyon (isotonic) — OOF üzerinde
    cal = CalibratedClassifierCV(base_estimator=LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs),
                                 method="isotonic", cv="prefit")
    # Not: isotonic için bir base model lazım; pratikte kalibratörü, OOF prob'ları ile fit edebilmek için,
    # logistic'i prefit saymak yerine, isotonic'i doğrudan 'y=oof_prob' üzerinden uyarlamak isterdik.
    # sklearn kısıtı nedeniyle şu yöntemi uygulayacağız:
    #   - meta gibi davranıp 'cal'ı probs üstünde fit etmek için küçük bir hile: 1 özellikli LR fit + isotonic.
    # Daha temiz bir yol: net isotonic uygulaması (sklearn.isotonic.IsotonicRegression).
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_prob, dfx[label_col].values.astype(int))
    # iso nesnesini kaydedeceğiz; inference'ta lr probs → iso(probs)

    # Precision@K ve kalibrasyon raporları (OOF)
    eval_oof = pd.DataFrame({
        "datetime": dfx["datetime"].values,
        "geoid": dfx["geoid"].astype(str).values,
        "y": dfx[label_col].values.astype(int),
        "p_raw": oof_prob
    })
    prec_k = {}
    for k in k_list:
        prec_k[f"P@{k}"] = precision_at_k_per_ts(eval_oof.rename(columns={"p_raw":"p_stack"}), "p_stack", k, "y")
    cal_bins = reliability_bins(eval_oof["y"].values, oof_prob, n_bins=10)

    # Full-train ile final LR ve baseline_map üret
    baseline_map_global = build_baseline_map(dfx, label_col)
    dfx["baseline_rate"] = attach_baseline(dfx, baseline_map_global).astype("float32")
    extra = ["baseline_rate", "hour","dow","month","is_weekend"]
    if "is_holiday" in dfx.columns: extra.append("is_holiday")
    Xfull, yfull, cols_full = pick_features(dfx, label_col, extra_num_cols=extra)
    lr_final = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=rs)
    lr_final.fit(Xfull, yfull)

    # Kaydet
    ensure_dirs(out_models_dir, out_metrics_dir)
    tag = f"h{H}_C{cat}"

    joblib.dump(lr_final, os.path.join(out_models_dir, f"lr_{tag}.pkl"))
    # IsotonicRegression objesini ayrı kaydediyoruz (kalibratör)
    joblib.dump(iso, os.path.join(out_models_dir, f"calibrator_{tag}.pkl"))

    # Feature order
    with open(os.path.join(out_models_dir, f"feature_order_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"features": cols_full}, f, ensure_ascii=False, indent=2)

    # Baseline haritasını da kaydet (inference'ta join edeceğiz)
    baseline_path = os.path.join(out_models_dir, f"baseline_map_{tag}.parquet")
    baseline_map_global.to_parquet(baseline_path, index=False)

    # Metrikler
    with open(os.path.join(out_metrics_dir, f"{tag}_aucpr.json"), "w", encoding="utf-8") as f:
        json.dump({
            "horizon_h": H,
            "category": cat,
            "oof_pr_auc": float(pr_auc(dfx[label_col].values, oof_prob)),
            "oof_roc_auc": float(roc_auc_score(dfx[label_col].values, oof_prob)),
            "folds": fold_recs
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_metrics_dir, f"{tag}_precision_at_k.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_h": H, "category": cat, **prec_k}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_metrics_dir, f"{tag}_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_h": H, "category": cat, **cal_bins}, f, ensure_ascii=False, indent=2)

    print(f"[OK][LONG] H={H}, C={cat} | OOF PR-AUC={pr_auc(dfx[label_col].values, oof_prob):.4f} | baseline_map rows={len(baseline_map_global):,}")


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
        label_col = f"Y_label_h{H}"
        if label_col not in df.columns:
            print(f"[Uyarı][LONG] {label_col} yok ({path}), atlanıyor.")
            continue

        df = df.dropna(subset=[label_col, "datetime", "geoid", "category","hour","dow","month"]).copy()
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

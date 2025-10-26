#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_infer_long.py
LONG motor (1–6 ay) için mevsimsel baseline + LR + isotonic ile tahmin üretir.
Çıktı: data/outputs/risk_long.parquet

Kullanım:
  python scripts/07_infer_long.py \
    --features_dir data/features/long \
    --models_dir models/long \
    --out_path data/outputs/risk_long.parquet \
    --horizons 960,1440,2160 \
    --confidence_lambda 0.15

Not:
- Eğitim sırasında kaydedilen baseline_map_h{H}_C{cat}.parquet dosyaları JOIN edilir.
- Feature sırası feature_order_h{H}_C{cat}.json ile garanti edilir.
"""

import os
import json
import argparse
from typing import List
import numpy as np
import pandas as pd
import joblib
from math import exp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, default="data/features/long")
    ap.add_argument("--models_dir", type=str, default="models/long")
    ap.add_argument("--out_path", type=str, default="data/outputs/risk_long.parquet")
    ap.add_argument("--horizons", type=str, default="960,1440,2160")
    ap.add_argument("--confidence_lambda", type=float, default=0.15)
    return ap.parse_args()


def read_latest_features(features_dir: str, H: int) -> pd.DataFrame:
    path = os.path.join(features_dir, f"features_h{H}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature dosyası yok: {path}")
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        raise ValueError("'datetime' yok")
    # Her (geoid,category) için en güncel t0
    df = df.sort_values(["geoid","category","datetime"])
    idx = df.groupby(["geoid","category"], as_index=False)["datetime"].idxmax()
    return df.loc[idx.values].reset_index(drop=True)


def list_categories_from_models(models_dir: str, H: int) -> List[str]:
    cats = set()
    pref = f"lr_h{H}_C"
    for fn in os.listdir(models_dir):
        if fn.startswith("lr_") and f"_h{H}_C" in fn and fn.endswith(".pkl"):
            # lr_h{H}_C{cat}.pkl
            cat = fn.replace("lr_", "").replace(".pkl", "")
            try:
                cat = cat.split("_C", 1)[1]
                cats.add(cat)
            except Exception:
                pass
    return sorted(cats)


def load_artifacts(models_dir: str, H: int, cat: str):
    tag = f"h{H}_C{cat}"
    paths = {
        "model": os.path.join(models_dir, f"lr_{tag}.pkl"),
        "cal":   os.path.join(models_dir, f"calibrator_{tag}.pkl"),
        "order": os.path.join(models_dir, f"feature_order_{tag}.json"),
        "base":  os.path.join(models_dir, f"baseline_map_{tag}.parquet"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Artefakt yok: {p}")
    lr  = joblib.load(paths["model"])
    cal = joblib.load(paths["cal"])     # IsotonicRegression
    with open(paths["order"], "r", encoding="utf-8") as f:
        order = json.load(f)["features"]
    base_map = pd.read_parquet(paths["base"])
    return lr, cal, order, base_map


def attach_baseline(df: pd.DataFrame, baseline_map: pd.DataFrame) -> pd.Series:
    # sırayla (geoid,category,hour,dow) → (category,hour,dow) → (hour,dow)
    out = pd.merge(
        df[["geoid","category","hour","dow"]],
        baseline_map[baseline_map["key_level"]=="geoid_category_hour_dow"],
        on=["geoid","category","hour","dow"], how="left"
    )["rate"]

    miss = out.isna()
    if miss.any():
        tmp = pd.merge(
            df.loc[miss, ["category","hour","dow"]],
            baseline_map[baseline_map["key_level"]=="category_hour_dow"],
            on=["category","hour","dow"], how="left"
        )["rate"]
        out.loc[miss] = tmp.values

    miss = out.isna()
    if miss.any():
        tmp = pd.merge(
            df.loc[miss, ["hour","dow"]],
            baseline_map[baseline_map["key_level"]=="hour_dow"],
            on=["hour","dow"], how="left"
        )["rate"]
        out.loc[miss] = tmp.values

    if out.isna().any():
        out = out.fillna( df.get("Y_label_h_dummy_global_mean", pd.Series([0.01]*len(df))).iloc[0] )
    return out.clip(0.0, 1.0)


def ensure_X(df: pd.DataFrame, order: list) -> pd.DataFrame:
    X = df.copy()
    for c in order:
        if c not in X.columns:
            X[c] = 0.0
    return X[order].astype("float32")


def conf_from_h(h: int, lam: float) -> float:
    return float(max(0.0, min(1.0, exp(-lam * (h / 24.0)))))


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    outs = []

    for H in horizons:
        try:
            feats = read_latest_features(args.features_dir, H)
        except Exception as e:
            print(f"[LONG][H{H}] Feature okunamadı: {e}")
            continue

        cats = list_categories_from_models(args.models_dir, H)
        if not cats:
            print(f"[LONG][H{H}] Uyarı: model bulunamadı; atlanıyor.")
            continue

        for cat in cats:
            try:
                lr, cal, order, base_map = load_artifacts(args.models_dir, H, cat)
            except Exception as e:
                print(f"[LONG][H{H}][{cat}] Artefakt yüklenemedi: {e}")
                continue

            dff = feats[feats["category"].astype(str) == str(cat)].copy()
            if dff.empty:
                print(f"[LONG][H{H}][{cat}] İnference için satır yok.")
                continue

            # baseline_rate ekle
            dff["baseline_rate"] = attach_baseline(dff, base_map).astype("float32")

            # Feature sırası
            X = ensure_X(dff, order)

            # LR olasılık + isotonic kalibrasyon
            p_raw = lr.predict_proba(X)[:, 1]
            p_cal = cal.predict(p_raw)  # IsotonicRegression: predict(x) → kalibre olasılık

            ts = pd.to_datetime(dff["datetime"]) + pd.to_timedelta(H, unit="h")
            conf = conf_from_h(H, args.confidence_lambda)

            outs.append(pd.DataFrame({
                "timestamp": ts.values,
                "horizon_h": H,
                "geoid": dff["geoid"].astype(str).values,
                "category": dff["category"].astype(str).values,
                "p_stack": p_cal.astype("float32"),
                "engine": "long",
                "confidence": conf
            }))

        print(f"[LONG][H{H}] tamamlandı.")

    if not outs:
        print("Uyarı: LONG inference çıktı üretmedi.")
        return

    final = pd.concat(outs, ignore_index=True)
    final = (final
             .groupby(["timestamp","horizon_h","geoid","category","engine"], as_index=False)
             .agg(p_stack=("p_stack","mean"), confidence=("confidence","mean")))
    final = final.sort_values(["timestamp","geoid","category"]).reset_index(drop=True)
    final.to_parquet(args.out_path, index=False)
    print(f"✅ LONG inference tamam: {args.out_path} | satır: {len(final):,}")


if __name__ == "__main__":
    main()

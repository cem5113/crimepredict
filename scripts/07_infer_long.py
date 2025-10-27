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
"""

import os
import json
import argparse
from typing import List, Dict
import numpy as np
import pandas as pd
import joblib
from math import exp


# -----------------------
# Argümanlar
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, default="data/features/long")
    ap.add_argument("--models_dir", type=str, default="models/long")
    ap.add_argument("--out_path", type=str, default="data/outputs/risk_long.parquet")
    ap.add_argument("--horizons", type=str, default="960,1440,2160")
    ap.add_argument("--confidence_lambda", type=float, default=0.15)
    return ap.parse_args()


# -----------------------
# Yardımcılar
# -----------------------
def _safe_to_datetime(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=False)
    return s

def read_latest_features(features_dir: str, H: int) -> pd.DataFrame:
    """
    features_h{H}.parquet içinden her (geoid, category) için en güncel (max datetime) kaydı döner.
    hour/dow eksikse güvenli şekilde üretir.
    """
    path = os.path.join(features_dir, f"features_h{H}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature dosyası yok: {path}")
    df = pd.read_parquet(path)

    need = {"geoid", "category", "datetime"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path} içinde eksik kolon(lar): {missing}")

    df["datetime"] = _safe_to_datetime(df["datetime"])
    df = df.dropna(subset=["datetime", "geoid", "category"])
    if df.empty:
        raise ValueError(f"{path} temizleme sonrası boş")

    # hour/dow yoksa türet
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour.astype("int16")
    if "dow" not in df.columns:
        df["dow"] = df["datetime"].dt.dayofweek.astype("int16")

    df = df.sort_values(["geoid", "category", "datetime"])
    idx = df.groupby(["geoid", "category"])["datetime"].idxmax()
    latest = df.loc[idx].reset_index(drop=True)
    return latest


def list_categories_from_models(models_dir: str, H: int) -> List[str]:
    """
    Kategori listesini, mevcut model artefakt isimlerinden çıkarır.
    """
    if not os.path.isdir(models_dir):
        return []
    cats = set()
    # lr_h{H}_C{cat}.pkl veya feature_order_h{H}_C{cat}.json üzerinden yakala
    tag = f"_h{H}_C"
    for fn in os.listdir(models_dir):
        if tag in fn:
            if fn.startswith("lr_") and fn.endswith(".pkl"):
                try:
                    cat = fn.replace("lr_", "").replace(".pkl", "").split("_C", 1)[1]
                    cats.add(cat)
                except Exception:
                    pass
            elif fn.startswith("feature_order_") and fn.endswith(".json"):
                try:
                    cat = fn.replace("feature_order_", "").replace(".json", "").split("_C", 1)[1]
                    cats.add(cat)
                except Exception:
                    pass
    return sorted(cats)


def paths_for(models_dir: str, H: int, cat: str) -> Dict[str, str]:
    tag = f"h{H}_C{cat}"
    return {
        "model": os.path.join(models_dir, f"lr_{tag}.pkl"),
        "cal":   os.path.join(models_dir, f"calibrator_{tag}.pkl"),       # IsotonicRegression
        "order": os.path.join(models_dir, f"feature_order_{tag}.json"),
        "base":  os.path.join(models_dir, f"baseline_map_{tag}.parquet"),
    }


def load_artifacts(models_dir: str, H: int, cat: str):
    """
    LR, Isotonic kalibratör, feature_order ve baseline_map'i yükler.
    """
    p = paths_for(models_dir, H, cat)
    required = ["model", "cal", "order", "base"]
    missing = [k for k in required if not os.path.exists(p[k])]
    if missing:
        raise FileNotFoundError(f"[LONG][H{H}][{cat}] Eksik artefakt(lar): {missing}")

    lr  = joblib.load(p["model"])
    cal = joblib.load(p["cal"])  # sklearn.isotonic.IsotonicRegression
    with open(p["order"], "r", encoding="utf-8") as f:
        order = json.load(f).get("features", [])
    if not isinstance(order, list) or not order:
        raise ValueError(f"[LONG][H{H}][{cat}] feature_order boş/bozuk: {p['order']}")
    base_map = pd.read_parquet(p["base"])
    return lr, cal, order, base_map


def attach_baseline(df: pd.DataFrame, baseline_map: pd.DataFrame) -> pd.Series:
    """
    (geoid,category,hour,dow) → rate
    fallback: (category,hour,dow) → rate
    fallback: (hour,dow) → rate
    son çare: baseline ortalaması
    """
    gm = float(baseline_map["rate"].mean()) if "rate" in baseline_map.columns and len(baseline_map) else 0.01
    out = pd.merge(
        df[["geoid", "category", "hour", "dow"]],
        baseline_map[baseline_map["key_level"] == "geoid_category_hour_dow"],
        on=["geoid", "category", "hour", "dow"],
        how="left"
    )["rate"]

    miss = out.isna()
    if miss.any():
        tmp = pd.merge(
            df.loc[miss, ["category", "hour", "dow"]],
            baseline_map[baseline_map["key_level"] == "category_hour_dow"],
            on=["category", "hour", "dow"], how="left"
        )["rate"]
        out.loc[miss] = tmp.values

    miss = out.isna()
    if miss.any():
        tmp = pd.merge(
            df.loc[miss, ["hour", "dow"]],
            baseline_map[baseline_map["key_level"] == "hour_dow"],
            on=["hour", "dow"], how="left"
        )["rate"]
        out.loc[miss] = tmp.values

    if out.isna().any():
        out = out.fillna(gm)
    return out.clip(0.0, 1.0)


def ensure_X(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in order:
        if c not in X.columns:
            X[c] = 0.0
    return X[order].astype("float32")


def conf_from_horizon(h: int, lam: float) -> float:
    return float(max(0.0, min(1.0, exp(-lam * (h / 24.0)))))


# -----------------------
# Inference
# -----------------------
def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    outputs: List[pd.DataFrame] = []

    for H in horizons:
        # 1) Feature’ları oku
        try:
            feats = read_latest_features(args.features_dir, H)
        except Exception as e:
            print(f"[LONG][H{H}] Feature okunamadı: {e}")
            continue

        # 2) Kategori listesi
        cats = list_categories_from_models(args.models_dir, H)
        if not cats:
            print(f"[LONG][H{H}] Uyarı: '{args.models_dir}' altında kategoriye ait model bulunamadı; atlanıyor.")
            continue

        # 3) Kategori bazında tahmin
        for cat in cats:
            try:
                lr, cal, order, base_map = load_artifacts(args.models_dir, H, cat)
            except Exception as e:
                print(f"[LONG][H{H}][{cat}] Artefakt yüklenemedi: {e}")
                continue

            dff = feats[feats["category"].astype(str) == str(cat)].copy()
            if dff.empty:
                print(f"[LONG][H{H}][{cat}] İnference için satır yok; atlandı.")
                continue

            # baseline_rate ekle
            try:
                dff["baseline_rate"] = attach_baseline(dff, base_map).astype("float32")
            except Exception as e:
                print(f"[LONG][H{H}][{cat}] baseline attach hatası: {e}")
                continue

            # Feature sırası
            try:
                X = ensure_X(dff, order)
            except Exception as e:
                print(f"[LONG][H{H}][{cat}] Feature hizalama hatası: {e}")
                continue

            # LR olasılık + isotonic kalibrasyon
            try:
                p_raw = lr.predict_proba(X)[:, 1]
                # IsotonicRegression: predict(prob_raw) → kalibre probability
                p_cal = cal.predict(p_raw).astype("float32")
            except Exception as e:
                print(f"[LONG][H{H}][{cat}] Tahmin/kalibrasyon hatası: {e}")
                continue

            ts = _safe_to_datetime(dff["datetime"]) + pd.to_timedelta(H, unit="h")
            conf = conf_from_horizon(H, args.confidence_lambda)

            outputs.append(pd.DataFrame({
                "timestamp": ts.values,
                "horizon_h": H,
                "geoid": dff["geoid"].astype(str).values,
                "category": dff["category"].astype(str).values,
                "p_stack": p_cal,
                "engine": "long",
                "confidence": conf
            }))

        print(f"[LONG][H{H}] tamamlandı.")

    if not outputs:
        print("Uyarı: LONG inference çıktı üretmedi.")
        return

    final = pd.concat(outputs, ignore_index=True)
    final = (
        final.groupby(["timestamp", "horizon_h", "geoid", "category", "engine"], as_index=False)
             .agg(p_stack=("p_stack", "mean"), confidence=("confidence", "mean"))
             .sort_values(["timestamp", "geoid", "category"])
             .reset_index(drop=True)
    )
    final.to_parquet(args.out_path, index=False)
    print(f"✅ LONG inference tamam: {args.out_path} | satır: {len(final):,}")


if __name__ == "__main__":
    main()

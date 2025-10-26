#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_infer_mid.py
- MID motor (3–30 gün / 72–720 saat) için stacking modelleriyle tahmin üretir.
- Çıktı: data/outputs/risk_mid.parquet

Kullanım:
  python scripts/06_infer_mid.py \
    --features_dir data/features/mid \
    --models_dir models/mid \
    --out_path data/outputs/risk_mid.parquet \
    --horizons 96,168,336,504,720 \
    --confidence_lambda 0.09

Not:
- 03_train_mid.py ile üretilmiş artefaktlar (base_xgb/lgb/(rf), meta_stack, calibrator, feature_order) beklenir.
- Feature dosyalarından her (geoid, category) için en güncel (t0) satır seçilir,
  ve hedef zaman damgası t0 + H saat olarak hesaplanır.
"""

import os
import json
import argparse
from typing import List
import numpy as np
import pandas as pd
import joblib
from math import exp


# -----------------------
# Argümanlar
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=str, default="data/features/mid",
                    help="Horizon bazlı feature dosyalarının klasörü (features_h{H}.parquet)")
    ap.add_argument("--models_dir", type=str, default="models/mid",
                    help="MID modellerinin kaydedildiği klasör")
    ap.add_argument("--out_path", type=str, default="data/outputs/risk_mid.parquet",
                    help="Çıktı dosyası (Parquet)")
    ap.add_argument("--horizons", type=str, default="96,168,336,504,720",
                    help="Saat cinsinden ufuk listesi (virgülle)")
    ap.add_argument("--confidence_lambda", type=float, default=0.09,
                    help="Confidence için exponential decay katsayısı (gün bazında)")
    return ap.parse_args()


# -----------------------
# Yardımcılar
# -----------------------
def read_latest_features(features_dir: str, H: int) -> pd.DataFrame:
    """
    features_h{H}.parquet içinden her (geoid, category) için en güncel (max datetime) kaydı döner.
    """
    path = os.path.join(features_dir, f"features_h{H}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature dosyası yok: {path}")
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        raise ValueError(f"'datetime' kolonu bulunamadı: {path}")

    df = df.sort_values(["geoid", "category", "datetime"])
    idx = df.groupby(["geoid", "category"], as_index=False)["datetime"].idxmax()
    latest = df.loc[idx.values].reset_index(drop=True)
    return latest


def list_categories_from_models(models_dir: str, H: int) -> List[str]:
    """
    Mevcut meta_stack dosyalarına bakarak kategori listesini çıkarır.
    """
    cats = set()
    pref = f"meta_stack_h{H}_C"
    for fn in os.listdir(models_dir):
        if fn.startswith(pref) and fn.endswith(".pkl"):
            # meta_stack_h{H}_C{cat}.pkl
            cat = fn.replace("meta_stack_", "").replace(".pkl", "")
            # kalan: h{H}_C{cat}
            try:
                cat = cat.split("_C", 1)[1]
                cats.add(cat)
            except Exception:
                pass
    return sorted(cats)


def load_artifacts(models_dir: str, H: int, cat: str):
    """
    İlgili horizon + kategori için base, meta, kalibratör ve feature sırasını yükler.
    RF opsiyonel olduğu için olmayabilir; varsa kullanılır.
    """
    tag = f"h{H}_C{cat}"
    paths = {
        "xgb": os.path.join(models_dir, f"base_xgb_{tag}.pkl"),
        "lgb": os.path.join(models_dir, f"base_lgb_{tag}.pkl"),
        "rf":  os.path.join(models_dir, f"base_rf_{tag}.pkl"),  # opsiyonel
        "meta":os.path.join(models_dir, f"meta_stack_{tag}.pkl"),
        "cal": os.path.join(models_dir, f"calibrator_{tag}.pkl"),
        "ord": os.path.join(models_dir, f"feature_order_{tag}.json"),
    }

    # Zorunlular
    for k in ["xgb", "lgb", "meta", "cal", "ord"]:
        if not os.path.exists(paths[k]):
            raise FileNotFoundError(f"Artefakt yok: {paths[k]}")

    xgb = joblib.load(paths["xgb"])
    lgb = joblib.load(paths["lgb"])
    rf  = joblib.load(paths["rf"]) if os.path.exists(paths["rf"]) else None
    meta= joblib.load(paths["meta"])
    cal = joblib.load(paths["cal"])

    with open(paths["ord"], "r", encoding="utf-8") as f:
        order = json.load(f)["features"]

    return xgb, lgb, rf, meta, cal, order


def ensure_X(df: pd.DataFrame, order: list) -> pd.DataFrame:
    """
    Feature sırasını garanti eder; eksik kolonları 0.0 ile doldurur.
    """
    X = df.copy()
    for c in order:
        if c not in X.columns:
            X[c] = 0.0
    X = X[order].astype("float32")
    return X


def conf_from_horizon(h: int, lam: float) -> float:
    """
    Confidence = exp(-λ * horizon_days) ; MID için λ genelde SHORT'tan büyük.
    """
    return float(max(0.0, min(1.0, exp(-lam * (h / 24.0)))))


# -----------------------
# Inference
# -----------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    outputs = []

    for H in horizons:
        try:
            feats = read_latest_features(args.features_dir, H)
        except Exception as e:
            print(f"[H{H}] Feature okunamadı: {e}")
            continue

        cats = list_categories_from_models(args.models_dir, H)
        if not cats:
            print(f"[H{H}] Uyarı: {args.models_dir} altında meta_stack bulunamadı (kategori yok). Atlanıyor.")
            continue

        for cat in cats:
            try:
                xgb, lgb, rf, meta, cal, order = load_artifacts(args.models_dir, H, cat)
            except Exception as e:
                print(f"[H{H}][{cat}] Artefakt yüklenemedi: {e}")
                continue

            dff = feats[feats["category"].astype(str) == str(cat)].copy()
            if dff.empty:
                print(f"[H{H}][{cat}] İnference için satır yok, atlandı.")
                continue

            X = ensure_X(dff, order)
            p1 = xgb.predict_proba(X)[:, 1]
            p2 = lgb.predict_proba(X)[:, 1]
            if rf is not None:
                p3 = rf.predict_proba(X)[:, 1]
                stack = np.vstack([p1, p2, p3]).T
            else:
                stack = np.vstack([p1, p2]).T

            p_meta  = meta.predict_proba(stack)[:, 1]
            p_stack = cal.predict_proba(stack)[:, 1]

            # Hedef zaman: t0 + H saat
            ts = pd.to_datetime(dff["datetime"]) + pd.to_timedelta(H, unit="h")

            conf = conf_from_horizon(H, args.confidence_lambda)

            out = pd.DataFrame({
                "timestamp": ts.values,
                "horizon_h": H,
                "geoid": dff["geoid"].astype(str).values,
                "category": dff["category"].astype(str).values,
                "p_stack": p_stack.astype("float32"),
                "engine": "mid",
                "confidence": conf
            })
            outputs.append(out)

        print(f"[MID][H{H}] tamamlandı.")

    if not outputs:
        print("Uyarı: MID inference için üretilecek satır bulunamadı.")
        return

    final = pd.concat(outputs, ignore_index=True)
    final = (final
             .groupby(["timestamp", "horizon_h", "geoid", "category", "engine"], as_index=False)
             .agg(p_stack=("p_stack", "mean"), confidence=("confidence", "mean")))
    final = final.sort_values(["timestamp", "geoid", "category"]).reset_index(drop=True)

    final.to_parquet(args.out_path, index=False)
    print(f"✅ MID inference tamam: {args.out_path} | satır: {len(final):,}")


if __name__ == "__main__":
    main()

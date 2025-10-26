#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_infer_short.py
- SHORT motor (0–72 saat) için stacking modelleriyle tahmin üretir.
- Çıktı: data/outputs/risk_short.parquet

Kullanım (varsayılan yollarla):
  python scripts/05_infer_short.py \
    --features_dir data/features/short \
    --models_dir models/short \
    --out_path data/outputs/risk_short.parquet \
    --horizons 1,2,3,6,12,24,48,72

Girdiler:
- data/features/short/features_h{H}.parquet
  (içinden, her geoid×category için **en güncel** feature satırını (t0) alır ve t0+H için tahmin yazar)
- models/short/
  base_xgb_h{H}_C{cat}.pkl, base_lgb_h{H}_C{cat}.pkl, base_rf_h{H}_C{cat}.pkl,
  meta_stack_h{H}_C{cat}.pkl, calibrator_h{H}_C{cat}.pkl,
  feature_order_h{H}_C{cat}.json

Çıktı şeması:
- timestamp (t0 + H saat)
- horizon_h
- geoid
- category
- p_stack  (kalibre edilmiş)
- engine   = 'short'
- confidence ∈ [0,1]
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
    ap.add_argument("--features_dir", type=str, default="data/features/short",
                    help="Horizon bazlı feature dosyalarının klasörü")
    ap.add_argument("--models_dir", type=str, default="models/short",
                    help="SHORT modellerinin kaydedildiği klasör")
    ap.add_argument("--out_path", type=str, default="data/outputs/risk_short.parquet",
                    help="Çıkış dosyası")
    ap.add_argument("--horizons", type=str, default="1,2,3,6,12,24,48,72",
                    help="Saat cinsinden ufuk listesi (virgülle)")
    ap.add_argument("--confidence_lambda", type=float, default=0.05,
                    help="Confidence için exponential decay katsayısı (gün bazında)")
    return ap.parse_args()


# -----------------------
# Yardımcılar
# -----------------------
def read_features_for_horizon(features_dir: str, H: int) -> pd.DataFrame:
    """
    Eğitimde kullandığımız feature dosyasını okur.
    Inference'ta **her geoid×category** için en **güncel** (max datetime) satırı alırız.
    """
    path = os.path.join(features_dir, f"features_h{H}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature dosyası yok: {path}")
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        raise ValueError(f"features_h{H}.parquet içinde 'datetime' yok")

    # Her (geoid, category) için en güncel t0
    df = df.sort_values(["geoid", "category", "datetime"])
    idx = df.groupby(["geoid", "category"], as_index=False)["datetime"].idxmax()
    latest = df.loc[idx.values].reset_index(drop=True)

    # Modelin istemediği kolonları burada dert etmiyoruz; seçim ve sırayı feature_order ile yapacağız.
    return latest


def list_categories_from_models(models_dir: str, H: int) -> List[str]:
    """
    Modellerden kategori listesi çıkar (meta veya feature_order dosyalarına bakarak).
    """
    cats = []
    prefix = f"_h{H}_C"
    for fname in os.listdir(models_dir):
        if fname.startswith("meta_stack") and prefix in fname and fname.endswith(".pkl"):
            # meta_stack_h{H}_C{cat}.pkl
            # kategoriyi sondan çek
            tag = fname.replace("meta_stack_", "").replace(".pkl", "")
            # tag: h{H}_C{cat}
            try:
                cat = tag.split("_C", 1)[1]
                cats.append(cat)
            except Exception:
                pass
    return sorted(list(set(cats)))


def load_artifacts_for(H: int, cat: str, models_dir: str):
    """
    İlgili horizon+kategori için base, meta, kalibratör ve feature sırasını yükler.
    """
    tag = f"h{H}_C{cat}"
    paths = {
        "xgb": os.path.join(models_dir, f"base_xgb_{tag}.pkl"),
        "lgb": os.path.join(models_dir, f"base_lgb_{tag}.pkl"),
        "rf":  os.path.join(models_dir, f"base_rf_{tag}.pkl"),
        "meta":os.path.join(models_dir, f"meta_stack_{tag}.pkl"),
        "cal": os.path.join(models_dir, f"calibrator_{tag}.pkl"),
        "order": os.path.join(models_dir, f"feature_order_{tag}.json"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Artefakt yok: {p}")

    xgb = joblib.load(paths["xgb"])
    lgb = joblib.load(paths["lgb"])
    rf  = joblib.load(paths["rf"])
    meta= joblib.load(paths["meta"])
    cal = joblib.load(paths["cal"])
    with open(paths["order"], "r", encoding="utf-8") as f:
        order = json.load(f)["features"]
    return xgb, lgb, rf, meta, cal, order


def ensure_feature_matrix(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """
    Feature sırasını garanti eder; eksik kolonları 0 ile doldurur, fazla kolonları düşürür.
    """
    X = df.copy()
    for c in order:
        if c not in X.columns:
            X[c] = 0.0
    X = X[order].astype("float32")
    return X


def calc_confidence(h: int, lam: float) -> float:
    """
    Ufuk büyüdükçe azalan güven: confidence = exp(-λ * horizon_days)
    """
    horizon_days = h / 24.0
    c = exp(-lam * horizon_days)
    return max(0.0, min(1.0, float(c)))


# -----------------------
# Inference
# -----------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    all_rows = []

    for H in horizons:
        # 1) En güncel feature satırları (t0) — her geoid×category için
        feat_latest = read_features_for_horizon(args.features_dir, H)

        # 2) Kategori listesini modellerden oku (hangi kategorilere model var?)
        cats = list_categories_from_models(args.models_dir, H)
        if not cats:
            print(f"[H{H}] Uyarı: models/short altında kategoriye ait meta dosyası bulunamadı; atlanıyor.")
            continue

        # 3) Her kategori için tahmin
        for cat in cats:
            # İlgili artefaktları yükle
            try:
                xgb, lgb, rf, meta, cal, order = load_artifacts_for(H, cat, args.models_dir)
            except Exception as e:
                print(f"[H{H}][{cat}] Artefakt yüklenemedi: {e}")
                continue

            dff = feat_latest[feat_latest["category"].astype(str) == str(cat)].copy()
            if dff.empty:
                print(f"[H{H}][{cat}] İnference için veri yok; atlandı.")
                continue

            # Feature matrisi
            X = ensure_feature_matrix(dff, order)

            # Base model olasılıkları
            p1 = xgb.predict_proba(X)[:, 1]
            p2 = lgb.predict_proba(X)[:, 1]
            p3 = rf.predict_proba(X)[:, 1]
            stack_mat = np.vstack([p1, p2, p3]).T

            # Meta + kalibrasyon
            p_meta = meta.predict_proba(stack_mat)[:, 1]
            p_stack = cal.predict_proba(stack_mat)[:, 1]

            # Hedef zaman damgası: t0 + H saat
            ts = pd.to_datetime(dff["datetime"]) + pd.to_timedelta(H, unit="h")

            # Confidence
            conf = calc_confidence(H, args.confidence_lambda)

            out = pd.DataFrame({
                "timestamp": ts.values,
                "horizon_h": H,
                "geoid": dff["geoid"].astype(str).values,
                "category": dff["category"].astype(str).values,
                "p_stack": p_stack.astype("float32"),
                "engine": "short",
                "confidence": conf
            })
            all_rows.append(out)

        print(f"[H{H}] tamamlandı.")

    if not all_rows:
        print("Uyarı: Üretilecek satır bulunamadı. Model veya feature kontrol edin.")
        return

    final = pd.concat(all_rows, ignore_index=True)
    # Aynı (timestamp, geoid, category, horizon_h) için birden fazla satır varsa ortala (teorik olarak olmamalı)
    final = (final
             .groupby(["timestamp", "horizon_h", "geoid", "category", "engine"], as_index=False)
             .agg(p_stack=("p_stack", "mean"), confidence=("confidence", "mean")))

    final = final.sort_values(["timestamp", "geoid", "category"]).reset_index(drop=True)
    final.to_parquet(args.out_path, index=False)
    print(f"✅ SHORT inference tamam: {args.out_path} | satır: {len(final):,}")


if __name__ == "__main__":
    main()

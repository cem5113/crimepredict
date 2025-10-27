#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_infer_short.py
- SHORT motor (0–72 saat) için stacking modelleriyle tahmin üretir.
- Çıktı: data/outputs/risk_short.parquet

Kullanım:
  python scripts/05_infer_short.py \
    --features_dir data/features/short \
    --models_dir models/short \
    --out_path data/outputs/risk_short.parquet \
    --horizons 1,2,3,6,12,24,48,72
"""

import os
import json
import argparse
from typing import List, Dict, Tuple
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
                    help="confidence = exp(-λ * horizon_days)")
    return ap.parse_args()


# -----------------------
# Yardımcılar
# -----------------------
def read_features_for_horizon(features_dir: str, H: int) -> pd.DataFrame:
    """
    features_h{H}.parquet -> her (geoid, category) için en güncel satır (t0).
    """
    path = os.path.join(features_dir, f"features_h{H}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature dosyası yok: {path}")

    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        raise ValueError(f"features_h{H}.parquet içinde 'datetime' yok")

    # (geoid,category) için max datetime satırı
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    df = df.dropna(subset=["datetime", "geoid", "category"])
    if df.empty:
        raise ValueError(f"features_h{H}.parquet temizleme sonrası boş")

    df = df.sort_values(["geoid", "category", "datetime"])
    # pandas >=1.4: idxmax çalışır; eski sürümler için agg ile de çözülebilir
    idx = df.groupby(["geoid", "category"])["datetime"].idxmax()
    latest = df.loc[idx].reset_index(drop=True)
    return latest


def list_categories_from_models(models_dir: str, H: int) -> List[str]:
    """
    Modellerden kategori listesi çıkar (meta_stack veya feature_order dosyalarına bakarak).
    """
    if not os.path.isdir(models_dir):
        return []
    cats = set()
    prefix = f"_h{H}_C"
    for fname in os.listdir(models_dir):
        if prefix in fname:
            if fname.startswith("meta_stack_") and fname.endswith(".pkl"):
                tag = fname.replace("meta_stack_", "").replace(".pkl", "")  # h{H}_C{cat}
                try:
                    cat = tag.split("_C", 1)[1]
                    cats.add(cat)
                except Exception:
                    pass
            elif fname.startswith("feature_order_") and fname.endswith(".json"):
                tag = fname.replace("feature_order_", "").replace(".json", "")
                try:
                    cat = tag.split("_C", 1)[1]
                    cats.add(cat)
                except Exception:
                    pass
    return sorted(cats)


def paths_for(H: int, cat: str, models_dir: str) -> Dict[str, str]:
    tag = f"h{H}_C{cat}"
    return {
        "xgb":   os.path.join(models_dir, f"base_xgb_{tag}.pkl"),
        "lgb":   os.path.join(models_dir, f"base_lgb_{tag}.pkl"),
        "rf":    os.path.join(models_dir, f"base_rf_{tag}.pkl"),
        "meta":  os.path.join(models_dir, f"meta_stack_{tag}.pkl"),
        "cal":   os.path.join(models_dir, f"calibrator_{tag}.pkl"),
        "order": os.path.join(models_dir, f"feature_order_{tag}.json"),
    }


def load_artifacts_for(H: int, cat: str, models_dir: str):
    """
    İlgili horizon+kategori için base, meta, kalibratör ve feature sırasını yükler.
    """
    p = paths_for(H, cat, models_dir)
    missing = [k for k, v in p.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"[H{H}][{cat}] Eksik artefakt(lar): {missing}")

    xgb  = joblib.load(p["xgb"])
    lgb  = joblib.load(p["lgb"])
    rf   = joblib.load(p["rf"])
    meta = joblib.load(p["meta"])
    cal  = joblib.load(p["cal"])
    with open(p["order"], "r", encoding="utf-8") as f:
        order = json.load(f)["features"]
    if not isinstance(order, list) or not order:
        raise ValueError(f"[H{H}][{cat}] feature_order boş/bozuk: {p['order']}")
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
    Ufuk büyüdükçe azalan güven: confidence = exp(-λ * (h/24))
    """
    return float(max(0.0, min(1.0, exp(-lam * (h / 24.0)))))


# -----------------------
# Inference
# -----------------------
def infer_one_horizon(H: int,
                      features_dir: str,
                      models_dir: str,
                      confidence_lambda: float) -> pd.DataFrame:
    """
    Tek bir horizon için tüm kategorilerde tahmin üret.
    """
    feat_latest = read_features_for_horizon(features_dir, H)
    cats = list_categories_from_models(models_dir, H)
    if not cats:
        print(f"[H{H}] Uyarı: '{models_dir}' altında kategoriye ait model bulunamadı; atlanıyor.")
        return pd.DataFrame()

    out_rows = []

    for cat in cats:
        try:
            xgb, lgb, rf, meta, cal, order = load_artifacts_for(H, cat, models_dir)
        except Exception as e:
            print(f"[H{H}][{cat}] Artefakt yüklenemedi: {e}")
            continue

        dff = feat_latest[feat_latest["category"].astype(str) == str(cat)].copy()
        if dff.empty:
            print(f"[H{H}][{cat}] İnference için veri yok; atlandı.")
            continue

        X = ensure_feature_matrix(dff, order)

        # Base olasılıklar
        try:
            p1 = xgb.predict_proba(X)[:, 1]
            p2 = lgb.predict_proba(X)[:, 1]
            p3 = rf.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"[H{H}][{cat}] Base predict_proba hatası: {e}")
            continue

        stack_mat = np.vstack([p1, p2, p3]).T

        # Meta + kalibrasyon
        try:
            p_meta  = meta.predict_proba(stack_mat)[:, 1]
            p_stack = cal.predict_proba(stack_mat)[:, 1]
        except Exception as e:
            print(f"[H{H}][{cat}] Meta/Cal predict_proba hatası: {e}")
            continue

        ts = pd.to_datetime(dff["datetime"]) + pd.to_timedelta(H, unit="h")
        conf = calc_confidence(H, confidence_lambda)

        out = pd.DataFrame({
            "timestamp": ts.values,
            "horizon_h": H,
            "geoid": dff["geoid"].astype(str).values,
            "category": dff["category"].astype(str).values,
            "p_stack": p_stack.astype("float32"),
            "engine": "short",
            "confidence": conf
        })
        out_rows.append(out)

    if not out_rows:
        return pd.DataFrame()

    res = pd.concat(out_rows, ignore_index=True)
    # Güvenli birleştirme (teorik olarak dupe olmamalı)
    res = (res.groupby(["timestamp", "horizon_h", "geoid", "category", "engine"], as_index=False)
              .agg(p_stack=("p_stack", "mean"), confidence=("confidence", "mean")))
    return res.sort_values(["timestamp", "geoid", "category"]).reset_index(drop=True)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    all_parts: List[pd.DataFrame] = []

    for H in horizons:
        try:
            part = infer_one_horizon(
                H=H,
                features_dir=args.features_dir,
                models_dir=args.models_dir,
                confidence_lambda=args.confidence_lambda,
            )
        except Exception as e:
            print(f"[H{H}] HATA: {e}")
            part = pd.DataFrame()

        if not part.empty:
            all_parts.append(part)
            print(f"[H{H}] tamamlandı: {len(part):,} satır.")
        else:
            print(f"[H{H}] çıktı boş.")

    if not all_parts:
        print("Uyarı: Üretilecek satır bulunamadı. Model/feature dizinlerini kontrol edin.")
        return

    final = pd.concat(all_parts, ignore_index=True)
    final.to_parquet(args.out_path, index=False)
    print(f"✅ SHORT inference tamam: {args.out_path} | satır: {len(final):,}")


if __name__ == "__main__":
    main()

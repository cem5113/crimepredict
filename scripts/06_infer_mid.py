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
    ap.add_argument("--features_dir", type=str, default="data/features/mid",
                    help="Horizon bazlı feature dosyalarının klasörü (features_h{H}.parquet)")
    ap.add_argument("--models_dir", type=str, default="models/mid",
                    help="MID modellerinin kaydedildiği klasör")
    ap.add_argument("--out_path", type=str, default="data/outputs/risk_mid.parquet",
                    help="Çıktı Parquet dosyası")
    ap.add_argument("--horizons", type=str, default="96,168,336,504,720",
                    help="Saat cinsinden ufuk listesi (virgülle)")
    ap.add_argument("--confidence_lambda", type=float, default=0.09,
                    help="confidence = exp(-λ * (h/24)) için λ")
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
    """
    path = os.path.join(features_dir, f"features_h{H}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature dosyası yok: {path}")
    df = pd.read_parquet(path)

    if "datetime" not in df.columns:
        raise ValueError(f"'datetime' kolonu bulunamadı: {path}")
    if "geoid" not in df.columns or "category" not in df.columns:
        raise ValueError(f"'geoid' / 'category' kolonu yok: {path}")

    df["datetime"] = _safe_to_datetime(df["datetime"])
    df = df.dropna(subset=["datetime", "geoid", "category"])
    if df.empty:
        raise ValueError(f"{path} temizleme sonrası boş")

    # (geoid,category) için max datetime satırı
    df = df.sort_values(["geoid", "category", "datetime"])
    idx = df.groupby(["geoid", "category"])["datetime"].idxmax()
    latest = df.loc[idx].reset_index(drop=True)
    return latest


def list_categories_from_models(models_dir: str, H: int) -> List[str]:
    """
    Mevcut artefakt isimlerinden kategori listesi çıkar (meta_stack ve/veya feature_order).
    """
    if not os.path.isdir(models_dir):
        return []
    cats = set()
    tag = f"_h{H}_C"
    for fn in os.listdir(models_dir):
        if tag in fn:
            if fn.startswith("meta_stack_") and fn.endswith(".pkl"):
                # meta_stack_h{H}_C{cat}.pkl
                try:
                    cat = fn.replace("meta_stack_", "").replace(".pkl", "").split("_C", 1)[1]
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
    base = f"h{H}_C{cat}"
    return {
        "xgb":   os.path.join(models_dir, f"base_xgb_{base}.pkl"),
        "lgb":   os.path.join(models_dir, f"base_lgb_{base}.pkl"),
        "rf":    os.path.join(models_dir, f"base_rf_{base}.pkl"),   # opsiyonel
        "meta":  os.path.join(models_dir, f"meta_stack_{base}.pkl"),
        "cal":   os.path.join(models_dir, f"calibrator_{base}.pkl"),
        "order": os.path.join(models_dir, f"feature_order_{base}.json"),
    }


def load_artifacts(models_dir: str, H: int, cat: str):
    """
    İlgili horizon + kategori için base, meta, kalibratör ve feature sırasını yükler.
    RF opsiyonel olduğu için olmayabilir; varsa kullanılır.
    """
    p = paths_for(models_dir, H, cat)
    # zorunlu set
    required = ["xgb", "lgb", "meta", "cal", "order"]
    missing = [k for k in required if not os.path.exists(p[k])]
    if missing:
        raise FileNotFoundError(f"[H{H}][{cat}] Eksik artefakt(lar): {missing}")

    xgb  = joblib.load(p["xgb"])
    lgb  = joblib.load(p["lgb"])
    rf   = joblib.load(p["rf"]) if os.path.exists(p["rf"]) else None
    meta = joblib.load(p["meta"])
    cal  = joblib.load(p["cal"])
    with open(p["order"], "r", encoding="utf-8") as f:
        order = json.load(f).get("features", [])
    if not isinstance(order, list) or not order:
        raise ValueError(f"[H{H}][{cat}] feature_order boş/bozuk: {p['order']}")
    return xgb, lgb, rf, meta, cal, order


def ensure_X(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
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
    Confidence = exp(-λ * horizon_days)
    """
    return float(max(0.0, min(1.0, exp(-lam * (h / 24.0)))))


def infer_one_horizon(H: int,
                      features_dir: str,
                      models_dir: str,
                      confidence_lambda: float) -> pd.DataFrame:
    """
    Tek bir horizon için tüm kategorilerde tahmin üret.
    """
    feats = read_latest_features(features_dir, H)
    cats = list_categories_from_models(models_dir, H)
    if not cats:
        print(f"[MID][H{H}] Uyarı: '{models_dir}' altında kategoriye ait model bulunamadı; atlanıyor.")
        return pd.DataFrame()

    outs = []

    for cat in cats:
        try:
            xgb, lgb, rf, meta, cal, order = load_artifacts(models_dir, H, cat)
        except Exception as e:
            print(f"[MID][H{H}][{cat}] Artefakt yüklenemedi: {e}")
            continue

        dff = feats[feats["category"].astype(str) == str(cat)].copy()
        if dff.empty:
            print(f"[MID][H{H}][{cat}] İnference için satır yok, atlandı.")
            continue

        X = ensure_X(dff, order)

        try:
            p1 = xgb.predict_proba(X)[:, 1]
            p2 = lgb.predict_proba(X)[:, 1]
            if rf is not None:
                p3 = rf.predict_proba(X)[:, 1]
                stack = np.vstack([p1, p2, p3]).T
            else:
                stack = np.vstack([p1, p2]).T
        except Exception as e:
            print(f"[MID][H{H}][{cat}] Base predict_proba hatası: {e}")
            continue

        try:
            p_meta  = meta.predict_proba(stack)[:, 1]
            p_stack = cal.predict_proba(stack)[:, 1]
        except Exception as e:
            print(f"[MID][H{H}][{cat}] Meta/Cal predict_proba hatası: {e}")
            continue

        ts = _safe_to_datetime(dff["datetime"]) + pd.to_timedelta(H, unit="h")
        conf = conf_from_horizon(H, confidence_lambda)

        outs.append(pd.DataFrame({
            "timestamp": ts.values,
            "horizon_h": H,
            "geoid": dff["geoid"].astype(str).values,
            "category": dff["category"].astype(str).values,
            "p_stack": p_stack.astype("float32"),
            "engine": "mid",
            "confidence": conf
        }))

    if not outs:
        return pd.DataFrame()

    res = pd.concat(outs, ignore_index=True)
    # Teorik olarak dupe olmamalı; yine de emniyetli toplama
    res = (res.groupby(["timestamp", "horizon_h", "geoid", "category", "engine"], as_index=False)
             .agg(p_stack=("p_stack", "mean"), confidence=("confidence", "mean")))
    return res.sort_values(["timestamp", "geoid", "category"]).reset_index(drop=True)


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    parts: List[pd.DataFrame] = []

    for H in horizons:
        try:
            part = infer_one_horizon(
                H=H,
                features_dir=args.features_dir,
                models_dir=args.models_dir,
                confidence_lambda=args.confidence_lambda,
            )
        except Exception as e:
            print(f"[MID][H{H}] HATA: {e}")
            part = pd.DataFrame()

        if not part.empty:
            parts.append(part)
            print(f"[MID][H{H}] tamamlandı: {len(part):,} satır.")
        else:
            print(f"[MID][H{H}] çıktı boş.")

    if not parts:
        print("Uyarı: MID inference için üretilecek satır bulunamadı.")
        return

    final = pd.concat(parts, ignore_index=True)
    final.to_parquet(args.out_path, index=False)
    print(f"✅ MID inference tamam: {args.out_path} | satır: {len(final):,}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_merge_outputs.py
- risk_short.parquet, risk_mid.parquet, risk_long.parquet dosyalarını
  TEK şemada birleştirir: risk_hourly_by_category.parquet
- Öncelik: short > mid > long (aynı (timestamp, geoid, category) çakışırsa)
- Confidence değerlerini 0..1 aralığında kırpar
- Opsiyonel: günlük agregasyon (mean veya max) üretip ayrı dosyaya yazar

Kullanım:
  python scripts/08_merge_outputs.py \
    --short data/outputs/risk_short.parquet \
    --mid   data/outputs/risk_mid.parquet \
    --long  data/outputs/risk_long.parquet \
    --out   data/outputs/risk_hourly_by_category.parquet \
    --out_daily data/outputs/risk_daily_by_category.parquet \
    --daily_how mean
"""

import os
import argparse
import pandas as pd
import numpy as np

ENGINE_PRIORITY = {"short": 3, "mid": 2, "long": 1}
REQUIRED_COLS = ["timestamp","horizon_h","geoid","category","p_stack","engine","confidence"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", type=str, default="data/outputs/risk_short.parquet")
    ap.add_argument("--mid",   type=str, default="data/outputs/risk_mid.parquet")
    ap.add_argument("--long",  type=str, default="data/outputs/risk_long.parquet")
    ap.add_argument("--out",   type=str, default="data/outputs/risk_hourly_by_category.parquet")
    ap.add_argument("--out_daily", type=str, default="", help="(opsiyonel) günlük dosya yolu")
    ap.add_argument("--daily_how", type=str, default="mean", choices=["mean","max"],
                    help="Günlük agregasyon yöntemi (opsiyonel çıktıda)")
    return ap.parse_args()

def _empty_df():
    return pd.DataFrame(columns=REQUIRED_COLS)

def _read_any(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return _empty_df()
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[Uyarı] '{path}' okunamadı: {e}; boş geçiliyor.")
        return _empty_df()

def _normalize_schema(df: pd.DataFrame, src: str) -> pd.DataFrame:
    if df.empty:
        return _empty_df()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{src} içinde eksik kolon(lar): {missing}")

    out = df[REQUIRED_COLS].copy()

    # timestamp → UTC aware
    # (Naive ise UTC varsayılır; tz-aware ise UTC’ye çevrilir)
    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["timestamp"] = ts

    # tip güvenliği
    out["horizon_h"] = pd.to_numeric(out["horizon_h"], errors="coerce").fillna(0).astype("int32")
    out["geoid"]     = out["geoid"].astype(str)
    out["category"]  = out["category"].astype(str)
    out["engine"]    = out["engine"].astype(str).str.lower().str.strip()

    out["p_stack"]   = pd.to_numeric(out["p_stack"], errors="coerce").fillna(0.0).astype("float32")
    out["confidence"]= pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).clip(0.0,1.0).astype("float32")

    # Engine validasyonu
    valid = {"short","mid","long"}
    out.loc[~out["engine"].isin(valid), "engine"] = "long"  # bilinmeyenler en düşük öncelik

    # Geçersiz timestamp satırlarını at
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    return out

def read_if_exists(path: str) -> pd.DataFrame:
    df = _read_any(path)
    return _normalize_schema(df, path)

def concat_with_priority(dfs: list) -> pd.DataFrame:
    if not dfs:
        return _empty_df()
    all_df = pd.concat(dfs, ignore_index=True)
    if all_df.empty:
        return _empty_df()

    # Öncelik & bağıl sıralama:
    # 1) engine priority (short>mid>long)
    # 2) daha kısa horizon (aynı anda birden fazla aynı zamana düşerse)
    # 3) yüksek confidence
    all_df["_prio"] = all_df["engine"].map(ENGINE_PRIORITY).fillna(0).astype("int8")

    all_df = all_df.sort_values(
        ["timestamp","geoid","category","_prio","horizon_h","confidence"],
        ascending=[True, True, True, False, True, False]
    )

    # Aynı (timestamp, geoid, category) için en iyi tek satırı tut
    dedup = (all_df
             .drop_duplicates(subset=["timestamp","geoid","category"], keep="first")
             .drop(columns=["_prio"])
             .reset_index(drop=True))
    return dedup

def write_parquet(df: pd.DataFrame, path: str):
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    df.to_parquet(path, index=False)

def make_daily(df_hourly: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    Günlük skor: aynı günün 24 saatindeki p_stack değerlerinden mean/max.
    Çıkış: date (UTC), geoid, category, daily_score, engine_top, n_hours, confidence_mean
    """
    if df_hourly.empty:
        return pd.DataFrame(columns=["date","geoid","category","daily_score","engine_top","n_hours","confidence_mean"])

    tmp = df_hourly.copy()
    # UTC tabanlı gün
    tmp["date"] = tmp["timestamp"].dt.floor("D")

    # En sık görülen engine (baskın)
    eng_mode = (
        tmp.groupby(["date","geoid","category","engine"], observed=True)
           .size().reset_index(name="cnt")
           .sort_values(["date","geoid","category","cnt"], ascending=[True,True,True,False])
           .drop_duplicates(subset=["date","geoid","category"], keep="first")
           .rename(columns={"engine":"engine_top"})
           [["date","geoid","category","engine_top"]]
    )

    agg_fun = "mean" if how == "mean" else "max"
    daily = (
        tmp.groupby(["date","geoid","category"], observed=True)
           .agg(daily_score=("p_stack", agg_fun),
                n_hours=("p_stack","size"),
                confidence_mean=("confidence","mean"))
           .reset_index()
    )
    daily = daily.merge(eng_mode, on=["date","geoid","category"], how="left")

    # Tipler
    daily["geoid"] = daily["geoid"].astype(str)
    daily["category"] = daily["category"].astype(str)
    daily["daily_score"] = pd.to_numeric(daily["daily_score"], errors="coerce").fillna(0.0).astype("float32")
    daily["confidence_mean"] = pd.to_numeric(daily["confidence_mean"], errors="coerce").fillna(0.0).clip(0.0,1.0).astype("float32")

    return daily[["date","geoid","category","daily_score","engine_top","n_hours","confidence_mean"]]

def main():
    args = parse_args()

    df_short = read_if_exists(args.short)
    df_mid   = read_if_exists(args.mid)
    df_long  = read_if_exists(args.long)

    if df_short.empty and df_mid.empty and df_long.empty:
        print("Uyarı: Hiçbir giriş dosyası bulunamadı; çıkış üretilemedi.")
        return

    merged = concat_with_priority([df_short, df_mid, df_long])
    merged = merged.sort_values(["timestamp","geoid","category"]).reset_index(drop=True)

    write_parquet(merged, args.out)
    print(f"✅ Birleşik saatlik çıktı yazıldı: {args.out} | satır: {len(merged):,}")

    # Opsiyonel: günlük
    if args.out_daily:
        daily = make_daily(merged, how=args.daily_how)
        write_parquet(daily, args.out_daily)
        print(f"✅ Günlük çıktı yazıldı: {args.out_daily} | satır: {len(daily):,}")

    # QC özeti
    n_ts  = merged["timestamp"].nunique()
    n_geo = merged["geoid"].nunique()
    n_cat = merged["category"].nunique()
    eng_counts = merged["engine"].value_counts(dropna=False).to_dict()
    print(f"Özet → timestamps: {n_ts:,} | geoid: {n_geo:,} | category: {n_cat:,} | engine dağılımı: {eng_counts}")

if __name__ == "__main__":
    main()

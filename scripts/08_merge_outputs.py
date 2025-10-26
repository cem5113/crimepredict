#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_merge_outputs.py
- risk_short.parquet, risk_mid.parquet, risk_long.parquet dosyalarını
  TEK şemada birleştirir: risk_hourly_by_category.parquet
- Öncelik: short > mid > long (aynı (timestamp, geoid, category) çakışırsa)
- Confidence değerlerini 0..1 aralığında güvenli tutar (gerekirse kırpar)
- Opsiyonel: günlük agregasyon (mean veya max) üretip ayrı dosyaya yazar

Kullanım:
  python scripts/08_merge_outputs.py \
    --short data/outputs/risk_short.parquet \
    --mid   data/outputs/risk_mid.parquet \
    --long  data/outputs/risk_long.parquet \
    --out   data/outputs/risk_hourly_by_category.parquet \
    --out_daily data/outputs/risk_daily_by_category.parquet \
    --daily_how mean

Çıktı şeması (hourly birleşik):
  - timestamp (UTC)
  - horizon_h (int)
  - geoid (str)
  - category (str)
  - p_stack (float)
  - engine ∈ {'short','mid','long'}
  - confidence (float 0..1)
"""

import os
import argparse
import pandas as pd
import numpy as np

ENGINE_PRIORITY = {"short": 3, "mid": 2, "long": 1}

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

def read_if_exists(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp","horizon_h","geoid","category","p_stack","engine","confidence"])
    df = pd.read_parquet(path)
    # Şema sağlamlaştırma
    needed = ["timestamp","horizon_h","geoid","category","p_stack","engine","confidence"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{path} dosyasında '{c}' kolonu yok.")
    return df[needed].copy()

def concat_with_priority(dfs: list) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=["timestamp","horizon_h","geoid","category","p_stack","engine","confidence"])
    all_df = pd.concat(dfs, ignore_index=True)
    # Tip güvenliği
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=False)
    all_df["geoid"]     = all_df["geoid"].astype(str)
    all_df["category"]  = all_df["category"].astype(str)
    all_df["engine"]    = all_df["engine"].astype(str)
    all_df["horizon_h"] = all_df["horizon_h"].astype("int32")
    all_df["p_stack"]   = all_df["p_stack"].astype("float32")
    all_df["confidence"]= all_df["confidence"].astype("float32").clip(0.0, 1.0)

    # Öncelik skoru
    all_df["_prio"] = all_df["engine"].map(ENGINE_PRIORITY).fillna(0).astype("int8")

    # Aynı (timestamp, geoid, category) varsa en yüksek _prio seç
    # (horizon_h farklı da olsa, genelde tek saat için tek satır işimize yarar.
    #  Eğer birden çok horizon aynı timestamp'e düşmüşse en yüksek öncelik + en yüksek confidence seçelim.)
    all_df = all_df.sort_values(["timestamp","geoid","category","_prio","confidence"],
                                ascending=[True, True, True, False, False])

    dedup = (all_df
             .drop_duplicates(subset=["timestamp","geoid","category"], keep="first")
             .drop(columns=["_prio"])
             .reset_index(drop=True))
    return dedup

def write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def make_daily(df_hourly: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    Günlük skoru: aynı günün 24 saatindeki p_stack değerlerinden mean (veya max).
    Çıkış: date, geoid, category, daily_score, engine_mix, n_hours, confidence_mean
    Not: engine_mix basit bir özet (en baskın engine'i raporlar).
    """
    if df_hourly.empty:
        return pd.DataFrame(columns=["date","geoid","category","daily_score","engine_top","n_hours","confidence_mean"])

    tmp = df_hourly.copy()
    tmp["date"] = tmp["timestamp"].dt.floor("D")

    # Gün içi engine baskınlığını tespit: en çok görünen engine
    eng_mode = (tmp
                .groupby(["date","geoid","category","engine"], observed=True)
                .size()
                .reset_index(name="cnt"))
    eng_mode = (eng_mode
                .sort_values(["date","geoid","category","cnt"], ascending=[True,True,True,False])
                .drop_duplicates(subset=["date","geoid","category"], keep="first")
                .rename(columns={"engine":"engine_top"}))[["date","geoid","category","engine_top"]]

    agg_fun = {"p_stack": "mean"} if how == "mean" else {"p_stack": "max"}
    daily = (tmp
             .groupby(["date","geoid","category"], observed=True)
             .agg(daily_score=("p_stack", agg_fun["p_stack"]),
                  n_hours=("p_stack","size"),
                  confidence_mean=("confidence","mean"))
             .reset_index())
    daily = daily.merge(eng_mode, on=["date","geoid","category"], how="left")
    # Tipler
    daily["geoid"] = daily["geoid"].astype(str)
    daily["category"] = daily["category"].astype(str)
    daily["daily_score"] = daily["daily_score"].astype("float32")
    daily["confidence_mean"] = daily["confidence_mean"].astype("float32").clip(0.0,1.0)
    return daily[["date","geoid","category","daily_score","engine_top","n_hours","confidence_mean"]]

def main():
    args = parse_args()

    df_short = read_if_exists(args.short)
    df_mid   = read_if_exists(args.mid)
    df_long  = read_if_exists(args.long)

    # Boşsa uyar
    if df_short.empty and df_mid.empty and df_long.empty:
        print("Uyarı: Hiçbir giriş dosyası bulunamadı; çıkış üretilemedi.")
        return

    merged = concat_with_priority([df_short, df_mid, df_long])

    # Sıralı ve temiz
    merged = merged.sort_values(["timestamp","geoid","category"]).reset_index(drop=True)

    write_parquet(merged, args.out)
    print(f"✅ Birleşik saatlik çıktı yazıldı: {args.out} | satır: {len(merged):,}")

    # Opsiyonel: günlük
    if args.out_daily:
        daily = make_daily(merged, how=args.daily_how)
        write_parquet(daily, args.out_daily)
        print(f"✅ Günlük çıktı yazıldı: {args.out_daily} | satır: {len(daily):,}")

    # Küçük QC özeti
    n_ts = merged["timestamp"].nunique()
    n_geo = merged["geoid"].nunique()
    n_cat = merged["category"].nunique()
    eng_counts = merged["engine"].value_counts(dropna=False).to_dict()
    print(f"Özet → timestamps: {n_ts:,} | geoid: {n_geo:,} | category: {n_cat:,} | engine dağılımı: {eng_counts}")

if __name__ == "__main__":
    main()

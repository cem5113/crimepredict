#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_features.py
- Ham olay verisini (event-level) alır (ZIP/URL/yerel destekli)
- Saatlik panele (geoid × category × hourly) çevirir
- Leakage’siz lag/rolling özellikleri üretir
- SHORT / MID / LONG horizon setleri için etiketleri (Y_label_shifted) oluşturur
- Parquet çıktıları: data/features/{short,mid,long}/features_h{H}.parquet
- Ayrıca tekleştirilmiş bir "features_index.parquet" (manifest) yazar
"""

import os, sys, argparse, zipfile, requests
from io import BytesIO
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

# -----------------------
# Konfig (istersen CLI ile override)
# -----------------------
# Not: Yerel dosya kullanacaksan --input ile override et.
REPO_OWNER = "cem5113"
REPO_NAME  = "crime_prediction_data"
ASSET_ZIP  = "fr-crime-outputs-parquet.zip"
LATEST_ZIP_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/{ASSET_ZIP}"

# Varsayılan girdi: Release ZIP içindeki fr_crime_09.parquet
DEFAULT_INPUT = f"urlzip::{LATEST_ZIP_URL}::fr_crime_09.parquet"   # veya CSV/parquet/zip için --input ile değiştir
OUTPUT_DIR    = "data/features"

# Horizon tanımları (saat)
SHORT_H = [1, 2, 3, 6, 12, 24, 48, 72]
MID_H   = [96, 168, 336, 504, 720]               # ~4g, 7g, 14g, 21g, 30g
LONG_H  = [960, 1440, 2160]                      # ~40g, 60g, 90g (örnek)

# Rolling pencereler (saat)
ROLL_WINDOWS = [1, 3, 6, 12, 24, 48, 72, 168]    # 1h..7d


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=(
            "Ham veri yolu (parquet/csv) veya ZIP spec.\n"
            "Ör: /path/to/sf_crime_full.parquet\n"
            "Ör: zip::/path/to/fr-crime-outputs-parquet.zip::fr_crime_09.parquet\n"
            f"Ör: urlzip::{LATEST_ZIP_URL}::fr_crime_09.parquet"
        ),
    )
    ap.add_argument("--outdir", type=str, default=OUTPUT_DIR, help="Çıktı klasörü")
    ap.add_argument("--short", type=str, default=",".join(map(str, SHORT_H)))
    ap.add_argument("--mid", type=str, default=",".join(map(str, MID_H)))
    ap.add_argument("--long", type=str, default=",".join(map(str, LONG_H)))
    ap.add_argument("--freq", type=str, default="1H", help="Panel frekansı (varsayılan saatlik)")
    return ap.parse_args()


# -----------------------
# Okuyucular (ZIP/URL/yerel)
# -----------------------
def _read_parquet_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """
    ZIP içinden member_path'i Parquet olarak okur.
    Eğer doğrudan bulunamazsa, ZIP içindeki 'fr_parquet_outputs.zip' içinde de arar.
    """
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = set(z.namelist())
        # 1) Doğrudan arama
        if member_path in names:
            with z.open(member_path) as f:
                return pd.read_parquet(BytesIO(f.read()))
        # 2) İç ZIP: fr_parquet_outputs.zip
        nested = [n for n in names if n.endswith("fr_parquet_outputs.zip")]
        for n in nested:
            with z.open(n) as fz, zipfile.ZipFile(BytesIO(fz.read())) as z2:
                if member_path in z2.namelist():
                    with z2.open(member_path) as f2:
                        return pd.read_parquet(BytesIO(f2.read()))
    raise FileNotFoundError(f"ZIP içinde bulunamadı: {member_path}")

def _read_input(path: str) -> pd.DataFrame:
    """
    Desteklenen biçimler:
      - Düz parquet: /path/file.parquet
      - CSV:         /path/file.csv
      - Yerel ZIP:   zip::/path/file.zip::fr_crime_09.parquet
      - URL ZIP:     urlzip::https://.../fr-crime-outputs-parquet.zip::fr_crime_09.parquet
    """
    if path.startswith("urlzip::"):
        url, member = path[len("urlzip::"):].split("::", 1)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return _read_parquet_from_zip_bytes(r.content, member)

    if path.startswith("zip::"):
        zip_path, member = path[len("zip::"):].split("::", 1)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP yok: {zip_path}")
        with open(zip_path, "rb") as f:
            return _read_parquet_from_zip_bytes(f.read(), member)

    # düz dosya
    if not os.path.exists(path):
        raise FileNotFoundError(f"Girdi yok: {path}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Desteklenmeyen format (yalnızca parquet/csv/zip/urlzip).")


# -----------------------
# Yardımcılar
# -----------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Temel kolonlar için isim normalizasyonu
    lower = {c.lower(): c for c in df.columns}

    # datetime alanını adlandır
    if "datetime" not in lower:
        for cand in ["received_time", "timestamp", "event_time", "date_time"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "datetime"})
                break

    # GEOID/geo alanı
    if "geoid" not in df.columns and "GEOID" in df.columns:
        df = df.rename(columns={"GEOID": "geoid"})

    # kategori yoksa "Genel"
    if "category" not in df.columns:
        df["category"] = "Genel"

    # datetime parse
    if "datetime" not in df.columns:
        raise ValueError("Girdi verisinde datetime/timestamp alanı bulunamadı.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    df = df.dropna(subset=["datetime"])

    # Y_label adlandırma
    if "Y_label" not in df.columns and "y_label" in df.columns:
        df = df.rename(columns={"y_label": "Y_label"})

    # tipler
    df["geoid"] = df["geoid"].astype(str)
    df["category"] = df["category"].astype(str)
    return df


def _ensure_base_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Zaman türevleri (mevcut yoksa üret)
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour.astype("int16")
    if "day_of_week_x" in df.columns:
        df["dow"] = df["day_of_week_x"].astype("int16")
    else:
        df["dow"] = df["datetime"].dt.dayofweek.astype("int16")
    if "month_x" in df.columns:
        df["month"] = df["month_x"].astype("int16")
    else:
        df["month"] = df["datetime"].dt.month.astype("int16")

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0  # ileride resmi takvimle güncellenebilir
    if "is_business_hour" not in df.columns:
        df["is_business_hour"] = df["hour"].between(9, 18).astype("int8")
    if "is_night" not in df.columns:
        df["is_night"] = (~df["is_business_hour"].astype(bool)).astype("int8")

    # Season
    if "season_x" in df.columns:
        df["season"] = df["season_x"].astype(str)
    else:
        m = df["month"]
        season = np.select(
            [
                m.isin([12, 1, 2]),
                m.isin([3, 4, 5]),
                m.isin([6, 7, 8]),
            ],
            ["DJF", "MAM", "JJA"],
            default="SON",
        )
        df["season"] = season.astype(str)
    return df


def _to_hourly_panel(df: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
    """
    Olay tablosunu saatlik panele çevirir:
    - gruplama anahtarı: geoid × category
    - her saat için event_count ve (varsa) Y_label_count oluşturur
    Not: Y_label zaten 0/1 ise sum(count) == event_count olur.
    """
    keys = ["geoid", "category"]
    df["_ones"] = 1

    hourly = (
        df.set_index("datetime")
          .groupby(keys)
          .resample(freq)["_ones"].sum()
          .rename("event_count")
          .reset_index()
    )
    hourly["event_count"] = hourly["event_count"].fillna(0).astype("int32")

    if "Y_label" in df.columns:
        y_hour = (
            df.set_index("datetime")
              .groupby(keys)["Y_label"]
              .resample(freq)
              .sum()
              .rename("Y_count")
              .reset_index()
        )
        hourly = hourly.merge(y_hour, on=keys + ["datetime"], how="left")
        hourly["Y_count"] = hourly["Y_count"].fillna(0).astype("int16")
    else:
        hourly["Y_count"] = hourly["event_count"].astype("int16")

    hourly["hour"] = hourly["datetime"].dt.hour.astype("int16")
    hourly["dow"]  = hourly["datetime"].dt.dayofweek.astype("int16")
    hourly["month"] = hourly["datetime"].dt.month.astype("int16")
    hourly["is_weekend"] = hourly["dow"].isin([5, 6]).astype("int8")
    return hourly


def _add_rollings(panel: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Her geoid×category için saatlik panel üzerinde rolling/lag üretir.
    Leakage yok: rolling pencereler 'closed=left' ve shift uygulanır.
    """
    panel = panel.sort_values(["geoid", "category", "datetime"]).reset_index(drop=True)
    g = panel.groupby(["geoid", "category"], group_keys=False)

    for W in windows:
        panel[f"ev_sum_{W}h"] = g["event_count"].apply(
            lambda s, w=W: s.rolling(window=w, min_periods=1).sum().shift(1)
        ).astype("float32")
        panel[f"ev_mean_{W}h"] = g["event_count"].apply(
            lambda s, w=W: s.rolling(window=w, min_periods=1).mean().shift(1)
        ).astype("float32")
        panel[f"y_sum_{W}h"] = g["Y_count"].apply(
            lambda s, w=W: s.rolling(window=w, min_periods=1).sum().shift(1)
        ).astype("float32")
        panel[f"y_mean_{W}h"] = g["Y_count"].apply(
            lambda s, w=W: s.rolling(window=w, min_periods=1).mean().shift(1)
        ).astype("float32")

    alpha = 0.1  # ~24h yarı ömür
    panel["ev_ewm_24h"] = g["event_count"].apply(lambda s: s.shift(1).ewm(alpha=alpha).mean()).astype("float32")
    panel["y_ewm_24h"]  = g["Y_count"].apply(lambda s: s.shift(1).ewm(alpha=alpha).mean()).astype("float32")
    return panel


def _make_labels_for_horizons(panel: pd.DataFrame, horizons: List[int]) -> Dict[int, pd.DataFrame]:
    """
    Y_label_h(t) = 1{ t .. t+h-1 saatleri arasında en az 1 olay var }
    """
    out = {}
    base = panel.sort_values(["geoid", "category", "datetime"]).reset_index(drop=True)
    g = base.groupby(["geoid", "category"], group_keys=False)

    base["cumY"] = g["Y_count"].cumsum()
    for h in horizons:
        base[f"Y_label_h{h}"] = (g["cumY"].shift(-h) - g["cumY"] > 0).astype("int8")

    out_panel = base.drop(columns=["cumY"])
    for h in horizons:
        cols_keep = (
            ["geoid", "category", "datetime"] +
            [c for c in out_panel.columns if c.startswith(("ev_", "y_", "hour", "dow", "month", "is_weekend", "ev_ewm", "y_ewm"))] +
            [f"Y_label_h{h}"]
        )
        out[h] = out_panel[cols_keep].copy()
    return out


def _engine_for_horizon(h: int) -> str:
    if h <= 72:
        return "short"
    elif h <= 720:
        return "mid"
    return "long"


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Oku & normalize et
    raw = _read_input(args.input)
    raw = _normalize_columns(raw)
    raw = _ensure_base_flags(raw)

    # 2) Saatlik panele dönüştür (geoid×category×hour)
    panel = _to_hourly_panel(raw, freq=args.freq)

    # 3) Rolling / lag özellikler (leakage yok)
    panel = _add_rollings(panel, ROLL_WINDOWS)

    # 4) Horizon listelerini topla
    short_list = [int(x) for x in args.short.split(",") if x.strip()]
    mid_list   = [int(x) for x in args.mid.split(",") if x.strip()]
    long_list  = [int(x) for x in args.long.split(",") if x.strip()]
    all_h      = sorted(set(short_list + mid_list + long_list))

    # 5) Horizon bazlı etiket setleri
    label_map = _make_labels_for_horizons(panel, all_h)

    # 6) Kayıt & manifest
    manifest_rows = []
    for h, dfh in label_map.items():
        engine = _engine_for_horizon(h)
        subdir = os.path.join(args.outdir, engine)
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, f"features_h{h}.parquet")
        dfh.to_parquet(out_path, index=False)
        manifest_rows.append({"horizon_h": h, "engine": engine, "path": out_path, "n_rows": len(dfh)})

    manifest = pd.DataFrame(manifest_rows).sort_values(["engine", "horizon_h"])
    manifest_path = os.path.join(args.outdir, "features_index.parquet")
    manifest.to_parquet(manifest_path, index=False)

    print("✅ Feature üretimi tamam.")
    print(manifest)


if __name__ == "__main__":
    main()

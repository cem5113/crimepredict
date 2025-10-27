#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_features.py
- ZIP/URL/yerel okur (Actions artifact veya Release)
- İç içe ZIP yapılarını otomatik açar
- Saatlik panel + rolling/lag + horizon etiketleri üretir
"""

import os, sys, argparse, zipfile, requests, posixpath
from io import BytesIO
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

# -----------------------
# Konfig
# -----------------------
REPO_OWNER = "cem5113"
REPO_NAME  = "crime_prediction_data"

# Release fallback (varsa)
RELEASE_ASSET_ZIP = "fr-minimal-parquet.zip"

# Varsayılan girdi: Kaynağı OTOMATİK seç (artifact -> release)
# ve içinden fr_crime_09.parquet'ı oku
DEFAULT_INPUT = "urlzip::AUTO::fr_crime_09.parquet"
OUTPUT_DIR    = "data/features"

# Horizon tanımları (saat)
SHORT_H = [1, 2, 3, 6, 12, 24, 48, 72]
MID_H   = [96, 168, 336, 504, 720]
LONG_H  = [960, 1440, 2160]

# Rolling pencereler (saat)
ROLL_WINDOWS = [1, 3, 6, 12, 24, 48, 72, 168]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=(
            "Ham veri yolu (parquet/csv) veya ZIP spec.\n"
            "Ör: /path/to/sf_crime_full.parquet\n"
            "Ör: zip::/path/to/fr-minimal-parquet.zip::fr_crime_09.parquet\n"
            "Ör: urlzip::AUTO::fr_crime_09.parquet   (artifact->release otomatik)"
        ),
    )
    ap.add_argument("--outdir", type=str, default=OUTPUT_DIR, help="Çıktı klasörü")
    ap.add_argument("--short", type=str, default=",".join(map(str, SHORT_H)))
    ap.add_argument("--mid", type=str, default=",".join(map(str, MID_H)))
    ap.add_argument("--long", type=str, default=",".join(map(str, LONG_H)))
    ap.add_argument("--freq", type=str, default="1H", help="Panel frekansı (varsayılan saatlik)")
    return ap.parse_args()


# -----------------------
# Artifact/Release kaynak seçici
# -----------------------
def _resolve_artifact_zip_url(owner: str, repo: str, name_contains: str, token: str | None) -> Tuple[str | None, dict]:
    """
    Repo'daki en güncel, süresi dolmamış Actions artifact ZIP linkini döndürür.
    name_contains: artifact adını içerir (örn: 'fr-minimal-parquet').
    """
    if not token:
        return None, {}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(
        f"{base}/actions/artifacts?per_page=100",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=60,
    )
    r.raise_for_status()
    arts = r.json().get("artifacts", []) or []
    arts = [a for a in arts if (name_contains in a.get("name","")) and not a.get("expired")]
    if not arts:
        return None, {}
    art = sorted(arts, key=lambda a: a.get("created_at",""), reverse=True)[0]
    zip_url = f"{base}/actions/artifacts/{art['id']}/zip"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    return zip_url, headers

def _best_zip_url() -> Tuple[str, dict]:
    """
    1) Actions artifact (token varsa) -> önce fr-minimal-parquet, sonra eski ad
    2) Release fallback (yoksa)
    """
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("github_token")

    # Yeni artifact adı
    url, headers = _resolve_artifact_zip_url(REPO_OWNER, REPO_NAME, "fr-minimal-parquet", token)
    if url:
        return url, headers

    # Eski ad yedek
    url, headers = _resolve_artifact_zip_url(REPO_OWNER, REPO_NAME, "fr-minimal-parquet", token)
    if url:
        return url, headers 

    # Release fallback (yalnız Release asset yayınlıyorsan işe yarar)
    rel = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/{RELEASE_ASSET_ZIP}"
    return rel, {}

# -----------------------
# Okuyucular (ZIP/URL/yerel)
# -----------------------
def _read_any_table_from_bytes(raw: bytes, name_hint: str = "") -> pd.DataFrame:
    """
    ZIP değilse bytes'tan doğrudan parquet/csv okumayı dener.
    name_hint uzantı ipucu verebilir ('.parquet' / '.csv').
    """
    bio = BytesIO(raw)

    # ipucuna göre önce ilgili formatı dene
    if name_hint.lower().endswith(".csv"):
        try:
            bio.seek(0); return pd.read_csv(bio)
        except Exception:
            pass
    if name_hint.lower().endswith(".parquet"):
        try:
            bio.seek(0); return pd.read_parquet(bio)
        except Exception:
            pass

    # ipucu yoksa: önce parquet, sonra csv
    try:
        bio.seek(0); return pd.read_parquet(bio)
    except Exception:
        bio.seek(0); return pd.read_csv(bio)

def _read_table_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """ZIP/inner-ZIP içinde member_path'i CSV/Parquet olarak okur; ZIP değilse doğrudan okur."""
    # ZIP değilse: bytes'ı parquet/csv olarak dene
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as _z:
            pass
    except zipfile.BadZipFile:
        return _read_any_table_from_bytes(zip_bytes, name_hint=member_path)

    def _read(fp, name):
        return pd.read_csv(fp) if name.lower().endswith(".csv") else pd.read_parquet(fp)

    target_base = posixpath.basename(member_path)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # 1) birebir
        if member_path in names:
            with z.open(member_path) as f:
                return _read(BytesIO(f.read()), member_path)

        # 1.b) basename ile
        cand = [n for n in names if n.endswith("/"+target_base) or n == target_base]
        if cand:
            with z.open(cand[0]) as f:
                return _read(BytesIO(f.read()), cand[0])

        # 2) iç ZIP
        for n in names:
            if not n.lower().endswith(".zip"):
                continue
            with z.open(n) as fz, zipfile.ZipFile(BytesIO(fz.read())) as z2:
                inner = z2.namelist()
                if member_path in inner:
                    with z2.open(member_path) as f2:
                        return _read(BytesIO(f2.read()), member_path)
                cand2 = [m for m in inner if m.endswith("/"+target_base) or m == target_base]
                if cand2:
                    with z2.open(cand2[0]) as f2:
                        return _read(BytesIO(f2.read()), cand2[0])

    raise FileNotFoundError(f"ZIP içinde bulunamadı: {member_path}")
    
def _read_input(path: str) -> pd.DataFrame:
    """
    Desteklenen biçimler:
      - Düz parquet/csv: /path/file.parquet | /path/file.csv
      - Yerel ZIP:       zip::/path/file.zip::artifact/risk_hourly.parquet|.csv
      - URL ZIP:         urlzip::AUTO::artifact/risk_hourly.parquet|.csv
                         (AUTO = artifact->release fallback)
    """
    if path.startswith("urlzip::"):
        url, member = path[len("urlzip::"):].split("::", 1)
        headers = {}
        if url == "AUTO":
            url, headers = _best_zip_url()
        r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
        r.raise_for_status()
        return _read_table_from_zip_bytes(r.content, member)

    if path.startswith("zip::"):
        zip_path, member = path[len("zip::"):].split("::", 1)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP yok: {zip_path}")
        with open(zip_path, "rb") as f:
            return _read_table_from_zip_bytes(f.read(), member)

    # düz dosya
    if not os.path.exists(path):
        raise FileNotFoundError(f"Girdi yok: {path}")
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Desteklenmeyen format (yalnızca parquet/csv/zip/urlzip).")

# -----------------------
# Yardımcılar
# -----------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Girdiyi ortak şemaya dönüştürür:
      - datetime (varsa: datetime/timestamp/event_time/date_time; yoksa: date+hour_range -> datetime)
      - geoid (GEOID/Cell_id vb. varyantlardan)
      - category (yoksa 'Genel')
      - Y_label (y_label -> Y_label)
    """
    # --- kolon isimlerini rahat eşlemek için lowercase haritası
    lower = {c.lower(): c for c in df.columns}

    # --- GEOID / hücre kimliği
    geokey = None
    for cand in ("geoid", "GEOID", "geoid10", "geoid11", "cell_id", "id"):
        if cand in df.columns:
            geokey = cand
            break
        if cand.lower() in lower:
            geokey = lower[cand.lower()]
            break
    if geokey is not None and geokey != "geoid":
        df = df.rename(columns={geokey: "geoid"})
    if "geoid" not in df.columns:
        raise ValueError("Girdi verisinde GEOID/cell_id alanı bulunamadı.")
    df["geoid"] = df["geoid"].astype(str)

    # --- datetime alanı; yoksa date + hour_range'ten üret
    # 1) bilinen timestamp adları
    if "datetime" not in df.columns and "datetime" in lower:
        df = df.rename(columns={lower["datetime"]: "datetime"})
    if "datetime" not in df.columns:
        for cand in ("received_time", "timestamp", "event_time", "date_time"):
            if cand in df.columns:
                df = df.rename(columns={cand: "datetime"})
                break

    # 2) CSV biçimi: date + hour_range ("00-03", "3-6" vb.) -> datetime başlangıç saati
    if "datetime" not in df.columns and {"date", "hour_range"}.issubset(df.columns):
        d0 = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
        start_h = (
            df["hour_range"].astype(str)
            .str.extract(r"^(\d{1,2})")[0]
            .fillna("0").astype(int).clip(0, 23)
        )
        df["datetime"] = d0 + pd.to_timedelta(start_h, unit="h")

    # 3) yine de yoksa hata
    if "datetime" not in df.columns:
        raise ValueError("Girdi verisinde datetime/timestamp alanı (veya date+hour_range) bulunamadı.")

    # zaman tipini netleştir
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    df = df.dropna(subset=["datetime"])

    # --- kategori
    if "category" not in df.columns:
        df["category"] = "Genel"
    else:
        df["category"] = df["category"].astype(str)

    # --- etiket adı normalizasyonu
    if "Y_label" not in df.columns and ("y_label" in df.columns or "y_label" in lower):
        src = "y_label" if "y_label" in df.columns else lower["y_label"]
        df = df.rename(columns={src: "Y_label"})

    return df

def _ensure_base_flags(df: pd.DataFrame) -> pd.DataFrame:
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
        df["is_holiday"] = 0
    if "is_business_hour" not in df.columns:
        df["is_business_hour"] = df["hour"].between(9, 18).astype("int8")
    if "is_night" not in df.columns:
        df["is_night"] = (~df["is_business_hour"].astype(bool)).astype("int8")

    if "season_x" in df.columns:
        df["season"] = df["season_x"].astype(str)
    else:
        m = df["month"]
        season = np.select(
            [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        )
        df["season"] = season.astype(str)
    return df


def _to_hourly_panel(df: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
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

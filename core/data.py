from __future__ import annotations
from typing import Tuple
import json
import pandas as pd
from pathlib import Path

from config.settings import DEFAULT_PARQUET, DEFAULT_CSV, SF_CELLS_GEOJSON
from utils.constants import KEY_COL, RISK_COL, TIME_COL, DATE_COL

# -------- GEOID & risk yardımcıları --------
def fix_geoid(s: pd.Series) -> pd.Series:
    s = pd.Series(s).astype(str).str.strip()
    s = s.str.replace(r'\.0$', '', regex=True)
    s = s.str.replace(r'^\s*nan\s*$', '', regex=True)
    s = s.str.replace(r'[^0-9]', '', regex=True)
    # uzunluk sabitse (örn 11) şunu aç: s = s.str.zfill(11)
    return s

def coerce_risk(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

# -------- veri yükleme --------
def load_risk_df() -> pd.DataFrame:
    p_parq = Path(DEFAULT_PARQUET)
    p_csv = Path(DEFAULT_CSV)

    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError("Kaynak bulunamadı: risk_hourly.parquet/csv yok.")

    # beklenen kolonlar: GEOID + risk_score (+ opsiyonel timestamp/date)
    if KEY_COL not in df.columns:
        raise KeyError(f"Beklenen kolon yok: {KEY_COL}")
    if RISK_COL not in df.columns:
        # bazı setlerde "risk_level" olabilir → dönüştür
        guess = "risk_level" if "risk_level" in df.columns else None
        if guess is None:
            raise KeyError(f"Beklenen kolon yok: {RISK_COL}")
        df[RISK_COL] = df[guess]

    df[KEY_COL] = fix_geoid(df[KEY_COL])
    df[RISK_COL] = coerce_risk(df[RISK_COL])
    return df

# -------- günlük özet (aynı date ortalaması) --------
def daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if DATE_COL in d.columns:
        d[DATE_COL] = pd.to_datetime(d[DATE_COL]).dt.date
    elif TIME_COL in d.columns:
        d[DATE_COL] = pd.to_datetime(d[TIME_COL]).dt.date
    else:
        # tarih yoksa tek “gün” gibi davran: sabit bugünün tarihi
        d[DATE_COL] = pd.Timestamp("today").date()

    g = (d.groupby([KEY_COL, DATE_COL], as_index=False)
           .agg({RISK_COL: "mean"}))
    return g

# -------- GEOID → (lat, lon) centroid eşleştirme --------
def _poly_centroid(coords) -> Tuple[float, float]:
    # Koordinatları flatten edip kaba ortalama al (hızlı, depsiz)
    xs, ys = [], []
    def walk(obj):
        if isinstance(obj[0], (float, int)):
            # [lon, lat]
            xs.append(float(obj[0])); ys.append(float(obj[1]))
        else:
            for k in obj: walk(k)
    walk(coords)
    if not xs:
        return 0.0, 0.0
    return float(sum(ys)/len(ys)), float(sum(xs)/len(xs))  # (lat, lon)

def load_geoid_centroids(geojson_path: str | Path = SF_CELLS_GEOJSON) -> pd.DataFrame:
    p = Path(geojson_path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON bulunamadı: {p}")
    with open(p, "r", encoding="utf-8") as f:
        gj = json.load(f)

    rows = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        geoid = str(props.get(KEY_COL, "")).strip()
        geoid = geoid.replace(".0", "")
        if not geoid:
            continue
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", [])
        lat, lon = _poly_centroid(coords)  # (lat, lon)
        rows.append({KEY_COL: geoid, "latitude": lat, "longitude": lon})

    df = pd.DataFrame(rows)
    df[KEY_COL] = fix_geoid(df[KEY_COL])
    return df

def attach_latlon(df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    if df.empty or centroids.empty:
        return df
    out = df.merge(centroids, on=KEY_COL, how="left")
    return out

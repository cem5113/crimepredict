# crimepredict/core/mapkit.py
from __future__ import annotations
import pandas as pd
import pydeck as pdk
from pathlib import Path
from config.settings import MAP_VIEW

try:
    # repo içinden import (packaged)
    from crimepredict.utils.geo import load_geoid_layer
    from crimepredict.utils.deck import build_map_fast_deck
except ImportError:
    # local çalıştırma
    from utils.geo import load_geoid_layer
    from utils.deck import build_map_fast_deck

# -------------------------
# GeoJSON cache
# -------------------------
_geo_df_cache: pd.DataFrame | None = None
def _get_geo_df() -> pd.DataFrame:
    """sf_cells.geojson'u bir kez yükler ve cache'te tutar."""
    global _geo_df_cache
    if _geo_df_cache is None:
        geo_path = (Path(__file__).resolve().parent.parent / "data" / "sf_cells.geojson").as_posix()
        geo_df, _ = load_geoid_layer(geo_path, key_field="geoid")  # utils.geo fonksiyonu
        # Beklenen kolon adlarını güvene al
        # (geojson içinde centroid_lat/centroid_lon var)
        if "centroid_lat" not in geo_df.columns or "centroid_lon" not in geo_df.columns:
            # varsa alternatif isimlerden toparla
            for a, b in (("lat", "centroid_lat"), ("latitude", "centroid_lat"),
                         ("lon", "centroid_lon"), ("lng", "centroid_lon"), ("longitude", "centroid_lon")):
                if a in geo_df.columns and b not in geo_df.columns:
                    geo_df[b] = geo_df[a]
        _geo_df_cache = geo_df
    return _geo_df_cache


# -------------------------
# Veri hazırlama yardımcıları
# -------------------------
def _attach_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """lat/lon yoksa GEOID centroidlerini ekler."""
    if "lat" in df.columns and "lon" in df.columns:
        return df
    geo_df = _get_geo_df()
    keep = geo_df[["geoid", "centroid_lat", "centroid_lon"]].rename(
        columns={"centroid_lat": "lat", "centroid_lon": "lon"}
    )
    out = df.merge(keep, on="geoid", how="left")
    return out

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """risk_score ve temel kolonları normalize eder."""
    df = df.copy()
    # geoid stringle
    if "geoid" in df.columns:
        df["geoid"] = df["geoid"].astype(str)

    # risk_score 0-1 aralığına
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        df["risk_score"] = 0.0

    # E[olay] yoksa
    if "pred_expected" not in df.columns:
        df["pred_expected"] = 0.0
    return df

def _prep_for_layers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Katmanlara girecek veri setlerini hazırlar:
      - df_agg: GEOID bazlı risk (choropleth)
      - pts   : Heatmap için nokta seti (lat/lon zorunlu)
    """
    geo_df = _get_geo_df()
    df = _normalize_columns(df)

    # --- Choropleth için GEOID bazlı risk (örn. mean) ---
    if "geoid" in df.columns:
        df_agg = df.groupby("geoid", as_index=False)["risk_score"].mean()
    else:
        df_agg = pd.DataFrame(columns=["geoid", "risk_score"])

    # --- Heatmap noktaları (lat/lon yoksa centroid ekle) ---
    pts = _attach_centroids(df)
    # sayısallaştır + dropna
    for c in ("lat", "lon"):
        if c in pts.columns:
            pts[c] = pd.to_numeric(pts[c], errors="coerce")
    if "lat" in pts.columns and "lon" in pts.columns:
        pts = pts.dropna(subset=["lat", "lon"])
        pts = pts.rename(columns={"risk_score": "weight"})
        if "weight" not in pts.columns:
            pts["weight"] = 1.0
        pts = pts[["lat", "lon", "weight"]]
    else:
        pts = pd.DataFrame(columns=["lat", "lon", "weight"])

    return df_agg, pts


# -------------------------
# Public API
# -------------------------
def home_deck(df: pd.DataFrame) -> pdk.Deck:
    """
    Şehir Anlık Görünüm:
      • GEOID choropleth (risk katmanı)
      • Isı katmanı (centroid heatmap)
      • Hotspot marker (build_map_fast_deck içinde)
    Not: df; en azından ['geoid','risk_score'] içermelidir.
         (risk_hourly.parquet doğrudan uygundur)
    """
    geo_df = _get_geo_df()
    df_agg, pts = _prep_for_layers(df)

    # Başlangıç görünümü
    view = {"lat": MAP_VIEW["lat"], "lon": MAP_VIEW["lon"], "zoom": MAP_VIEW["zoom"]}

    deck = build_map_fast_deck(
        df_agg=df_agg,               # GEOID -> risk_score (choropleth)
        geo_df=geo_df,               # sf_cells poligon + centroidler
        show_risk_layer=True,        # renkli risk katmanı
        show_temp_hotspot=True,      # ısı katmanı (HeatmapLayer)
        temp_hotspot_points=pts,     # lat, lon, weight
        show_hotspot=True,           # %90 üzeri koyu marker (deck util içinde)
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view=view,
    )
    return deck

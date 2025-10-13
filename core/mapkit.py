# core/mapkit.py
import pydeck as pdk
import pandas as pd
from config.settings import MAP_VIEW

# Yeni importlar (mevcut utils klasöründen)
from utils.geo import load_geoid_layer
from utils.deck import build_map_fast_deck

# GEOID katmanını sadece bir kez oku
_geo_df_cache = None
def _get_geo_df():
    global _geo_df_cache
    if _geo_df_cache is None:
        geo_df, _ = load_geoid_layer("data/sf_cells.geojson", key_field="geoid")
        _geo_df_cache = geo_df
    return _geo_df_cache

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Veriyi görselleştirme için güvenli hale getirir."""
    df = df.copy()
    for c in ["lat", "lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # risk_score alanı yoksa veya boşsa 0–1 aralığına çek
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        df["risk_score"] = 1.0

    # E[olay] yoksa dummy ekle
    if "pred_expected" not in df.columns:
        df["pred_expected"] = 0.0

    # geoid eksikse boş string
    if "geoid" not in df.columns:
        df["geoid"] = ""
    return df


def home_deck(df: pd.DataFrame) -> pdk.Deck:
    """
    Şehir anlık görünüm: risk katmanı + ısı katmanı + hotspotlar.
    """
    df = _prep(df)
    geo_df = _get_geo_df()

    # Heatmap için nokta seti (risk_score -> weight)
    pts = df.rename(columns={"risk_score": "weight"})
    if "weight" not in pts.columns:
        pts["weight"] = 1.0

    # Başlangıç görünümü (MAP_VIEW ayarlarından)
    view = {
        "lat": MAP_VIEW["lat"],
        "lon": MAP_VIEW["lon"],
        "zoom": MAP_VIEW["zoom"],
    }

    # utils.deck.build_map_fast_deck fonksiyonunu kullan
    deck = build_map_fast_deck(
        df_agg=df,                    # GEOID bazlı risk verisi
        geo_df=geo_df,                # sf_cells.geojson centroid/polygon
        show_risk_layer=True,         # renkli risk katmanı
        show_temp_hotspot=True,       # ısı katmanı (HeatmapLayer)
        temp_hotspot_points=pts[["lat", "lon", "weight"]],
        show_hotspot=True,            # %90 üzeri koyu kırmızı marker
        map_style="mapbox://styles/mapbox/dark-v11",  # koyu tema
        initial_view=view,
    )
    return deck

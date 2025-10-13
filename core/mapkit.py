# core/mapkit.py
import pydeck as pdk
import pandas as pd
from config.settings import MAP_VIEW

def _base_view():
    return pdk.ViewState(
        latitude=MAP_VIEW["lat"], longitude=MAP_VIEW["lon"],
        zoom=MAP_VIEW["zoom"], pitch=MAP_VIEW["pitch"], bearing=MAP_VIEW["bearing"]
    )

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # Görselleştirme için güvenli hale getir
    need = ["geoid","lat","lon","risk_score","pred_expected"]
    df = df.copy()
    for c in ["lat","lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat","lon"])
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).clip(0, 1)
    else:
        df["risk_score"] = 1.0
    # pydeck/tooltip alanları yoksa boş string ver
    for c in need:
        if c not in df.columns:
            df[c] = ""
    return df

def heatmap_layer(df: pd.DataFrame, radius_px: int = 70):
    return pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position="[lon, lat]",
        get_weight="risk_score",
        radiusPixels=radius_px,
        aggregation="SUM",
        pickable=True,          # <-- tooltip için
        threshold=0.02,         # çok sönük kalmasın
        intensity=1.0,
    )

def hit_layer_for_tooltip(df: pd.DataFrame, size_px: int = 8):
    # Görünmez nokta katmanı; sadece hover/tooltip için
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color=[0, 0, 0, 0],
        get_line_color=[0, 0, 0, 0],
        pickable=True,
        radiusMinPixels=size_px,   # piksel bazlı
    )

def scatter_layer(df: pd.DataFrame, size_px: int = 5):
    # İstersen görünür noktalar
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color=[255, 140, 0, 120],
        get_line_color=[255, 255, 255, 60],
        pickable=True,
        radiusMinPixels=size_px,   # <- get_radius yerine
    )

def home_deck(df: pd.DataFrame):
    df = _prep(df)
    layers = [heatmap_layer(df), hit_layer_for_tooltip(df)]
    # Veriniz azsa noktaları da göster
    if len(df) < 5000:
        layers.append(scatter_layer(df, size_px=5))

    tooltip = {
        "html": "<b>GEOID:</b> {geoid}<br/><b>Risk:</b> {risk_score}<br/><b>E[olay]:</b> {pred_expected}",
        "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white"}
    }
    return pdk.Deck(layers=layers, initial_view_state=_base_view(), tooltip=tooltip)

import pydeck as pdk
import pandas as pd
from config.settings import MAP_VIEW

def _base_view():
    return pdk.ViewState(
        latitude=MAP_VIEW["lat"], longitude=MAP_VIEW["lon"],
        zoom=MAP_VIEW["zoom"], pitch=MAP_VIEW["pitch"], bearing=MAP_VIEW["bearing"]
    )

def heatmap_layer(df: pd.DataFrame, radius_px: int = 40):
    # risk_score varsa ağırlık olarak kullan, yoksa 1.0
    get_weight = "risk_score" if "risk_score" in df.columns else None
    return pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position="[lon, lat]",
        get_weight=get_weight,  # None ise default = 1
        radiusPixels=radius_px,
        aggregation="MEAN",
    )

def scatter_layer(df: pd.DataFrame, size: int = 6):
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=size,
        pickable=True,
    )

def home_deck(df: pd.DataFrame):
    # Hafif ısı + (veri azsa) nokta
    layers = [heatmap_layer(df)]
    if len(df) < 5000:
        layers.append(scatter_layer(df, size=5))
    tooltip = {
        "html": "<b>GEOID:</b> {geoid}<br/><b>Risk:</b> {risk_score}<br/><b>E[olay]:</b> {pred_expected}",
        "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white"}
    }
    return pdk.Deck(layers=layers, initial_view_state=_base_view(), tooltip=tooltip)

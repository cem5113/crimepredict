# core/mapkit.py
import pydeck as pdk
import pandas as pd
from config.settings import MAP_VIEW

def _base_view():
    return pdk.ViewState(
        latitude=MAP_VIEW["lat"], longitude=MAP_VIEW["lon"],
        zoom=MAP_VIEW["zoom"], pitch=MAP_VIEW["pitch"], bearing=MAP_VIEW["bearing"]
    )

def hex_layer(df: pd.DataFrame, radius: int = 120, elevation_scale: float = 20.0):
    return pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position="[lon, lat]",
        elevation_scale=elevation_scale,
        elevation_range=[0, 1000],
        extruded=True,
        radius=radius,
        pickable=True,
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
    # Düşük zoom’da hex, veri azsa scatter
    layer = hex_layer(df) if len(df) > 15000 else scatter_layer(df)
    tooltip = {
        "html": "<b>GEOID:</b> {geoid}<br/><b>Risk:</b> {risk_score}<br/><b>E[olay]:</b> {pred_expected}",
        "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white"}
    }
    return pdk.Deck(layers=[layer], initial_view_state=_base_view(), tooltip=tooltip)

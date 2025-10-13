# crimepredict/utils/deck.py
from __future__ import annotations
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import pydeck as pdk
from crimepredict.utils.constants import KEY_COL

_PALETTE = {
    "Çok Düşük":[198,219,239,140],"Düşük":[107,174,214,180],
    "Orta":[74,140,217,200],"Yüksek":[255,192,122,210],
    "Çok Yüksek":[239,59,44,210],
}
_DEF = [90,120,140,180]

def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "pred_expected" not in d: d["pred_expected"] = pd.to_numeric(d.get("expected",0), errors="coerce").fillna(0.0)
    d["pred_expected"] = pd.to_numeric(d["pred_expected"], errors="coerce").fillna(0.0)
    if "risk_level" not in d:
        x = d["pred_expected"].to_numpy(float)
        if np.isfinite(x).sum() >= 5:
            q = np.quantile(x[np.isfinite(x)], [0.2,0.4,0.6,0.8])
            def lvl(v): return "Çok Düşük" if v<=q[0] else "Düşük" if v<=q[1] else "Orta" if v<=q[2] else "Yüksek" if v<=q[3] else "Çok Yüksek"
            d["risk_level"] = [lvl(float(v)) for v in x]
        else:
            d["risk_level"] = "Çok Düşük"
    d[KEY_COL] = d[KEY_COL].astype(str)
    d["pred_expected_fmt"] = d["pred_expected"].round(2)
    if "neighborhood" not in d: d["neighborhood"] = ""
    return d

def build_map_fast_deck(
    df_agg: pd.DataFrame, geo_df: pd.DataFrame, *,
    show_hotspot: bool = True, show_temp_hotspot: bool = True,
    temp_hotspot_points: pd.DataFrame | None = None,
    show_risk_layer: bool = True,
    map_style: str = "mapbox://styles/mapbox/dark-v11",
    initial_view: Optional[Dict[str, float]] = None,
) -> pdk.Deck:
    if df_agg is None or df_agg.empty:
        return pdk.Deck(
            map_style=map_style,
            initial_view_state=pdk.ViewState(
                latitude=(initial_view or {}).get("lat", 37.7749),
                longitude=(initial_view or {}).get("lon", -122.4194),
                zoom=(initial_view or {}).get("zoom", 11.8),
            ),
            layers=[],
        )
    data = _ensure(df_agg)
    layers: list[pdk.Layer] = []

    if show_risk_layer:
        centers = (
            data.merge(geo_df[[KEY_COL,"centroid_lat","centroid_lon"]], on=KEY_COL, how="left")
                .dropna(subset=["centroid_lat","centroid_lon"]).copy()
        )
        centers["_color"] = centers["risk_level"].map(lambda k: _PALETTE.get(k, _DEF))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=centers, pickable=True,
            get_position="[centroid_lon, centroid_lat]",
            get_radius=80, radius_min_pixels=2, radius_max_pixels=90,
            get_fill_color="_color",
        ))

    if show_temp_hotspot and isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
        pts = temp_hotspot_points.copy()
        lc = {c.lower(): c for c in pts.columns}
        lat = lc.get("lat") or lc.get("latitude"); lon = lc.get("lon") or lc.get("longitude")
        if lat and lon:
            pts = pts.rename(columns={lat:"lat", lon:"lon"})
            if "weight" not in pts: pts["weight"] = 1.0
            layers.append(pdk.Layer(
                "HeatmapLayer", data=pts, get_position="[lon, lat]",
                get_weight="weight", radius_pixels=40, aggregation="SUM",
            ))

    if show_hotspot and "pred_expected" in data.columns:
        x = pd.to_numeric(data["pred_expected"], errors="coerce").fillna(0.0).to_numpy()
        if np.isfinite(x).sum() >= 1:
            thr = float(np.quantile(x[np.isfinite(x)], 0.90))
            strong = (data[data["pred_expected"]>=thr]
                      .merge(geo_df[[KEY_COL,"centroid_lat","centroid_lon"]], on=KEY_COL, how="left")
                      .dropna(subset=["centroid_lat","centroid_lon"]))
            if not strong.empty:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=strong, pickable=False,
                    get_position="[centroid_lon, centroid_lat]",
                    get_radius=120, get_fill_color=[139,0,0,200],
                ))

    tooltip = {
        "html": (f"<b>{KEY_COL}:</b> {{{KEY_COL}}}<br>"
                 "<b>Mahalle:</b> {neighborhood}<br>"
                 "<b>E[olay]:</b> {pred_expected_fmt}<br>"
                 "<b>Risk seviyesi:</b> {risk_level}"),
        "style": {"backgroundColor":"rgba(0,0,0,0.78)","color":"white"},
    }
    view = pdk.ViewState(
        latitude=(initial_view or {}).get("lat", 37.7749),
        longitude=(initial_view or {}).get("lon", -122.4194),
        zoom=(initial_view or {}).get("zoom", 11.8),
    )
    return pdk.Deck(layers=layers, initial_view_state=view, map_style=map_style, tooltip=tooltip)

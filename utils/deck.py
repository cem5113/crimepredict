# crimepredict/utils/deck.py
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import pandas as pd
import pydeck as pdk
from crimepredict.utils.constants import KEY_COL

# Renkler
_PALETTE = {
    "Çok Düşük":[198,219,239,140],
    "Düşük":[107,174,214,180],
    "Orta":[74,140,217,200],
    "Yüksek":[255,192,122,210],
    "Çok Yüksek":[239,59,44,210],
}
_DEF = [90,120,140,180]


def _ensure(df_like) -> pd.DataFrame:
    """
    DataFrame/dict girdiyi güvenli DataFrame'e çevirir:
      - KEY_COL stringe çevrilir
      - pred_expected yoksa expected/risk_score'dan üretilir
      - risk_level yoksa pred_expected veya risk_score quantile'ları ile kategorize edilir
      - neighborhood yoksa boş string
    """
    # 1) DataFrame'e çevir
    d = pd.DataFrame(df_like).copy()

    # 2) ana kolonlar
    if KEY_COL in d.columns:
        d[KEY_COL] = d[KEY_COL].astype(str)
    else:
        d[KEY_COL] = ""

    # 3) pred_expected
    if "pred_expected" in d.columns:
        d["pred_expected"] = pd.to_numeric(d["pred_expected"], errors="coerce").fillna(0.0)
    else:
        base = (
            d["expected"] if "expected" in d.columns
            else d["risk_score"] if "risk_score" in d.columns
            else 0.0
        )
        d["pred_expected"] = pd.to_numeric(base, errors="coerce").fillna(0.0)

    # 4) risk_score normalize (varsa)
    if "risk_score" in d.columns:
        d["risk_score"] = pd.to_numeric(d["risk_score"], errors="coerce").fillna(0.0).clip(0, 1)

    # 5) risk_level üret
    if "risk_level" not in d.columns:
        # Öncelik risk_score, yoksa pred_expected
        src = d["risk_score"] if "risk_score" in d.columns else d["pred_expected"]
        x = pd.to_numeric(src, errors="coerce").astype(float).to_numpy()
        mask = np.isfinite(x)
        if mask.sum() >= 5:
            q1, q2, q3, q4 = np.quantile(x[mask], [0.2, 0.4, 0.6, 0.8])
            def lvl(v: float) -> str:
                if v <= q1:  return "Çok Düşük"
                if v <= q2:  return "Düşük"
                if v <= q3:  return "Orta"
                if v <= q4:  return "Yüksek"
                return "Çok Yüksek"
            d["risk_level"] = [lvl(float(v)) if np.isfinite(v) else "Çok Düşük" for v in x]
        else:
            d["risk_level"] = "Çok Düşük"

    # 6) görsel formatlar
    d["pred_expected_fmt"] = pd.to_numeric(d["pred_expected"], errors="coerce").fillna(0.0).round(2)
    if "neighborhood" not in d.columns:
        d["neighborhood"] = ""

    return d

def _collapse_24h(df: pd.DataFrame) -> pd.DataFrame:
    """
    date / hour_range vb. tekrarları yok sayıp GEOID bazında 24 saatlik
    ortalama risk_score ve pred_expected üretir.
    """
    d = df.copy()
    if "risk_score" in d.columns:
        d["risk_score"] = pd.to_numeric(d["risk_score"], errors="coerce").clip(0, 1)
    if "pred_expected" not in d.columns:
        base = (
            d["expected"] if "expected" in d.columns else
            d["risk_score"] if "risk_score" in d.columns else 0.0
        )
        d["pred_expected"] = pd.to_numeric(base, errors="coerce").fillna(0.0)

    keep_first = lambda s: s.iloc[0] if len(s) else ""
    agg = {
        "risk_score": "mean",
        "pred_expected": "mean",
        "neighborhood": keep_first if "neighborhood" in d.columns else "first",
    }
    cols = [c for c in agg.keys() if c in d.columns]
    g = d.groupby(KEY_COL, as_index=False)[cols].agg(agg)
    g.rename(columns={
        "risk_score": "risk_score",          # isim aynı kalsın
        "pred_expected": "pred_expected",    # isim aynı kalsın
    }, inplace=True)
    return g


def build_map_fast_deck(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    show_hotspot: bool = True,
    show_temp_hotspot: bool = True,
    temp_hotspot_points: pd.DataFrame | None = None,
    show_risk_layer: bool = True,
    map_style: str = "mapbox://styles/mapbox/dark-v11",
    initial_view: Optional[Dict[str, float]] = None,
    aggregate_24h: bool = True,   # <<< YENİ: 24 saat agregasyon anahtarımız
) -> pdk.Deck:

    if df_agg is None or len(df_agg) == 0:
        return pdk.Deck(
            map_style=map_style,
            initial_view_state=pdk.ViewState(
                latitude=(initial_view or {}).get("lat", 37.7749),
                longitude=(initial_view or {}).get("lon", -122.4194),
                zoom=(initial_view or {}).get("zoom", 11.8),
            ),
            layers=[],
        )

    # --- 24 saatlik ortalamaya indir (date/hour_range'i yok say) ---
    data_in = pd.DataFrame(df_agg)
    if aggregate_24h:
        data_in = _collapse_24h(data_in)

    # --- downstream beklediği formatı üret ---
    data = _ensure(data_in)   # risk_level yoksa burada üretilecek

    layers: list[pdk.Layer] = []

    # --- Risk noktaları katmanı ---
    if show_risk_layer:
        centers = (
            data.merge(
                geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]],
                on=KEY_COL, how="left"
            )
            .dropna(subset=["centroid_lat", "centroid_lon"])
            .copy()
        )
        centers["_color"] = centers["risk_level"].map(lambda k: _PALETTE.get(k, _DEF))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=centers,
            pickable=True,
            get_position="[centroid_lon, centroid_lat]",
            get_radius=80,
            radius_min_pixels=2,
            radius_max_pixels=90,
            get_fill_color="_color",
        ))

    # --- Isı katmanı (varsa) ---
    if show_temp_hotspot and isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
        pts = temp_hotspot_points.copy()
        lc = {c.lower(): c for c in pts.columns}
        lat = lc.get("lat") or lc.get("latitude")
        lon = lc.get("lon") or lc.get("longitude")
        if lat and lon:
            pts = pts.rename(columns={lat: "lat", lon: "lon"})
            if "weight" not in pts.columns:
                pts["weight"] = 1.0
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=pts,
                get_position="[lon, lat]",
                get_weight="weight",
                radius_pixels=40,
                aggregation="SUM",
            ))

    # --- Üst %10 hotspot işaretleyicileri (pred_expected ort.) ---
    if show_hotspot and "pred_expected" in data.columns:
        x = pd.to_numeric(data["pred_expected"], errors="coerce").fillna(0.0).to_numpy()
        mask = np.isfinite(x)
        if mask.sum() >= 1:
            thr = float(np.quantile(x[mask], 0.90))
            strong = (
                data[data["pred_expected"] >= thr]
                .merge(geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
                .dropna(subset=["centroid_lat", "centroid_lon"])
            )
            if not strong.empty:
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=strong,
                    pickable=False,
                    get_position="[centroid_lon, centroid_lat]",
                    get_radius=120,
                    get_fill_color=[139, 0, 0, 200],
                ))

    tooltip = {
        "html": (
            f"<b>{KEY_COL}:</b> {{{KEY_COL}}}<br>"
            "<b>Mahalle:</b> {neighborhood}<br>"
            "<b>E[olay] (24s ort.):</b> {pred_expected_fmt}<br>"
            "<b>Risk seviyesi:</b> {risk_level}"
        ),
        "style": {"backgroundColor": "rgba(0,0,0,0.78)", "color": "white"},
    }

    view = pdk.ViewState(
        latitude=(initial_view or {}).get("lat", 37.7749),
        longitude=(initial_view or {}).get("lon", -122.4194),
        zoom=(initial_view or {}).get("zoom", 11.8),
    )
    return pdk.Deck(layers=layers, initial_view_state=view, map_style=map_style, tooltip=tooltip)

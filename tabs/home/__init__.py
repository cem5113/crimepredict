from __future__ import annotations
import pandas as pd
import streamlit as st
from folium import Map
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from core.data import load_risk_df, daily_mean, load_geoid_centroids, attach_latlon
from utils.constants import KEY_COL, RISK_COL, DATE_COL

def render():
    # 1) veri
    raw = load_risk_df()
    if raw is None or raw.empty:
        st.warning("Veri kaynağı boş / bulunamadı. `data/risk_hourly.parquet|csv` ekleyin.")
        return

    # 2) günlük özet
    g = daily_mean(raw).dropna(subset=[RISK_COL])
    if g.empty:
        st.warning("Özet veri boş.")
        return

    # 3) centroid ekle (GeoJSON yoksa NaN döner; app çökmez)
    cents = load_geoid_centroids()
    g = attach_latlon(g, cents)
    g = g.dropna(subset=[RISK_COL])

    # 4) gün seçimi
    dates = sorted(pd.Series(g[DATE_COL]).dropna().unique())
    sel_date = dates[-1] if dates else pd.Timestamp("today").date()
    sel_date = st.selectbox("Gün seç", options=dates or [sel_date], index=(len(dates)-1 if dates else 0))
    gday = g[g[DATE_COL] == sel_date].copy()

    st.write("Seçilen günde hücre sayısı:", len(gday))

    # 5) harita (centroid varsa)
    has_geo = gday["latitude"].notna().any() and gday["longitude"].notna().any()
    if has_geo:
        m = Map(location=[37.7749, -122.4194], zoom_start=12, control_scale=True)
        heat_data = gday[["latitude", "longitude", RISK_COL]].dropna().values.tolist()
        if heat_data:
            HeatMap(heat_data, radius=12, max_zoom=14).add_to(m)
        st_folium(m, width=1100, height=650)
    else:
        st.info("GeoJSON/centroid yok → harita pas; tablo gösteriliyor.")

    # 6) tablo
    with st.expander("Veri tablosu"):
        st.dataframe(
            gday[[KEY_COL, DATE_COL, RISK_COL, "latitude", "longitude"]]
            .sort_values(RISK_COL, ascending=False)
            .head(500)
        )

def register():
    return "🏠 Home (Harita & Tablo)", render

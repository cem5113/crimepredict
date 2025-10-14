from __future__ import annotations
import streamlit as st
import pandas as pd
from folium import Map
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from core.data import load_risk_df, daily_mean, load_geoid_centroids, attach_latlon
from utils.constants import KEY_COL, RISK_COL, DATE_COL

st.set_page_config(page_title="SF Crime Risk", layout="wide")
st.title("🗺️ SF Crime Risk — Günlük Özet")

raw = load_risk_df()
if raw.empty:
    st.warning("Veri kaynağı bulunamadı veya boş. Lütfen `data/` içine `risk_hourly.parquet` veya `risk_hourly.csv` koy.")
    st.stop()

st.success(f"Yüklendi: {len(raw):,} satır")
g = daily_mean(raw)

cents = load_geoid_centroids()          # artık FileNotFoundError atmaz
g = attach_latlon(g, cents)             # eksikse NaN ekler
g = g.dropna(subset=[RISK_COL])         # risk NaN olanları at

if g.empty:
    st.warning("Özet veri boş.")
    st.stop()

dates = sorted(g[DATE_COL].dropna().unique())
sel_date = dates[-1] if dates else pd.Timestamp("today").date()
sel_date = st.selectbox("Gün seç", options=dates or [sel_date], index=(len(dates)-1 if dates else 0))
gday = g[g[DATE_COL] == sel_date].copy()

st.write("Seçilen günde hücre sayısı:", len(gday))

# Harita sadece lat/lon mevcutsa çizilir
has_geo = gday["latitude"].notna().any() and gday["longitude"].notna().any()

if has_geo:
    m = Map(location=[37.7749, -122.4194], zoom_start=12, control_scale=True)
    heat_data = gday[["latitude", "longitude", RISK_COL]].dropna().values.tolist()
    if heat_data:
        HeatMap(heat_data, radius=12, max_zoom=14).add_to(m)
    st_folium(m, width=1100, height=650)
else:
    st.info("GeoJSON bulunamadı veya centroidler eksik → harita pas geçildi (sadece tablo).")

with st.expander("Veri tablosu"):
    st.dataframe(
        gday[[KEY_COL, DATE_COL, RISK_COL, "latitude", "longitude"]]
        .sort_values(RISK_COL, ascending=False)
        .head(500)
    )

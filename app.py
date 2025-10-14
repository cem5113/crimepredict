from __future__ import annotations
import streamlit as st
import pandas as pd
from folium import Map
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from core.data import load_risk_df, daily_mean
from utils.geo import add_centroids
from utils.constants import KEY_COL, RISK_COL, DATE_COL

st.set_page_config(page_title="SF Crime Risk", layout="wide")
st.title("🗺️ SF Crime Risk — Günlük Özet")

# 1) Veri
try:
    raw = load_risk_df()
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

if raw is None or raw.empty:
    st.warning("Veri kaynağı boş. `data/risk_hourly.parquet` veya `data/risk_hourly.csv` ekleyin.")
    st.stop()

st.success(f"Yüklendi: {len(raw):,} satır")

# 2) Günlük ortalama (aynı tarihteki skorların ortalaması)
g = daily_mean(raw)
if g is None or g.empty:
    st.warning("Günlük özet boş.")
    st.stop()

# 3) GEOID → centroid ekle (geojson yoksa NaN döner; app çökmez)
g = add_centroids(g)

# 4) Gün seçimi
dates = sorted(pd.Series(g.get(DATE_COL)).dropna().unique().tolist())
default_date = dates[-1] if dates else pd.Timestamp("today").date()
sel_date = st.selectbox("Gün seç", options=dates or [default_date], index=(len(dates)-1 if dates else 0))
gday = g[g[DATE_COL] == sel_date].copy()

st.write("Seçilen günde hücre sayısı:", len(gday))

# 5) Harita (centroid varsa)
m = Map(location=[37.7749, -122.4194], zoom_start=12, control_scale=True)
heat_src = (
    gday[["latitude", "longitude", RISK_COL]]
    .dropna(subset=["latitude", "longitude", RISK_COL])
    .values.tolist()
    if {"latitude", "longitude", RISK_COL}.issubset(gday.columns)
    else []
)
if heat_src:
    HeatMap(heat_src, radius=12, max_zoom=14).add_to(m)
else:
    st.info("Harita için yeterli coğrafi veri yok (latitude/longitude eksik).")

st_folium(m, width=1100, height=650)

# 6) Tablo
cols = [c for c in [KEY_COL, DATE_COL, RISK_COL, "latitude", "longitude"] if c in gday.columns]
with st.expander("Veri tablosu"):
    st.dataframe(
        gday[cols]
        .dropna(subset=[RISK_COL])
        .sort_values(RISK_COL, ascending=False)
        .head(500)
    )

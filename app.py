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

# 1) veri
try:
    raw = load_risk_df()
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

st.success(f"Yüklendi: {len(raw):,} satır")

# 2) günlük ortalama
g = daily_mean(raw)
st.write("Özet satır sayısı:", len(g))

# 3) centroidleri ekle
g = add_centroids(g)

# eksik centroidleri at
g = g.dropna(subset=["latitude", "longitude"])

# son günü seç (veya kullanıcı seçimi)
dates = sorted(g[DATE_COL].unique())
sel_date = st.selectbox("Gün seç", options=dates, index=len(dates)-1)
gday = g[g[DATE_COL] == sel_date].copy()

st.write("Seçilen günde hücre sayısı:", len(gday))

# 4) harita
m = Map(location=[37.7749, -122.4194], zoom_start=12, control_scale=True)
heat_data = gday[["latitude", "longitude", RISK_COL]].values.tolist()
if heat_data:
    HeatMap(heat_data, radius=12, max_zoom=14).add_to(m)
st_folium(m, width=1100, height=650)

# 5) tablo
with st.expander("Veri tablosu"):
    st.dataframe(gday[[KEY_COL, DATE_COL, RISK_COL, "latitude", "longitude"]].sort_values(RISK_COL, ascending=False).head(200))

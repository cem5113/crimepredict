# pages/3_🧭_Suç_Tahmini.py
from __future__ import annotations
import sys, pathlib

# --- bootstrap ---
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- güvenli constants import ---
try:
    from utils.constants import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES
except Exception:
    SF_TZ_OFFSET  = -7
    KEY_COL       = "geoid"
    MODEL_VERSION = "v0"
    MODEL_LAST_TRAIN = "-"
    CATEGORIES    = ["Assault","Burglary","Robbery","Theft","Vandalism","Vehicle Theft"]

import os, io, zipfile, requests
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from components.last_update import show_last_update_badge


st.set_page_config(page_title="🔮 Suç Tahmini (Stacking Model)", layout="wide")
st.title("🔮 Suç Tahmini ve Model Performansı")

# ───────────────────────────────
# 📦 GitHub artifact'tan veri yükleme
# ───────────────────────────────
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_URL = f"https://github.com/{OWNER}/{REPO}/releases/latest/download/fr-crime-outputs-parquet.zip"

@st.cache_data(show_spinner=True)
def load_artifact_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        resp = requests.get(ARTIFACT_URL, timeout=30)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        inner = [f for f in zf.namelist() if f.endswith(".csv") or f.endswith(".parquet")]
        if not inner:
            st.error("Zip içinde veri bulunamadı.")
            return pd.DataFrame(), pd.DataFrame()
        # fr_crime_09.csv’yi bul
        target_csv = next((f for f in inner if "fr_crime_09" in f.lower()), inner[0])
        with zf.open(target_csv) as f:
            df = pd.read_csv(f)
        # metrik dosyası
        metrics_file = next((f for f in inner if "metrics_stacking_ohe" in f.lower()), None)
        if metrics_file:
            with zf.open(metrics_file) as f2:
                metrics = pd.read_parquet(f2)
        else:
            metrics = pd.DataFrame()
        return df, metrics
    except Exception as e:
        st.error(f"Veri indirilemedi: {e}")
        return pd.DataFrame(), pd.DataFrame()

df, metrics = load_artifact_data()

# ───────────────────────────────
# 🕒 Güncel durum ve başlık
# ───────────────────────────────
show_last_update_badge(
    app_name="SUTAM – Suç Tahmin Modeli",
    data_upto=datetime.now(),
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
    daily_update_hour_sf=19,
    show_times=True,
)

# ───────────────────────────────
# 📊 Model Metrikleri
# ───────────────────────────────
st.subheader("📈 Model Performans Özeti")

if not metrics.empty:
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("Model metrikleri bulunamadı (metrics_stacking_ohe.parquet).")

# ───────────────────────────────
# 🔮 Tahmin Haritası
# ───────────────────────────────
st.subheader("📍 Suç Olasılığı Haritası")

if not df.empty:
    df = df.rename(columns=lambda c: c.strip())
    # GEOID ve koordinatları kontrol et
    lat_col = next((c for c in df.columns if "lat" in c.lower()), "latitude")
    lon_col = next((c for c in df.columns if "lon" in c.lower()), "longitude")
    geoid_col = next((c for c in df.columns if "geoid" in c.lower()), "GEOID")

    # Filtreler
    st.sidebar.markdown("### 🔎 Filtreler")
    cats = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []
    sel_cat = st.sidebar.selectbox("Suç Kategorisi", ["(Tümü)"] + cats)
    hour_range = st.sidebar.slider("Saat aralığı", 0, 24, (0, 24))

    # Zaman filtresi
    if "event_hour_x" in df.columns:
        df = df[df["event_hour_x"].between(hour_range[0], hour_range[1])]
    if sel_cat != "(Tümü)" and "category" in df.columns:
        df = df[df["category"] == sel_cat]

    # Risk skorlarını oluştur (örnek)
    if "Y_label" in df.columns:
        risk = df.groupby(geoid_col)["Y_label"].mean().reset_index(name="risk_score")
    else:
        risk = df.groupby(geoid_col).size().reset_index(name="risk_score")

    # Coğrafi merkez hesaplama
    if lat_col in df.columns and lon_col in df.columns:
        geo = df.groupby(geoid_col)[[lat_col, lon_col]].mean().reset_index()
        risk = risk.merge(geo, on=geoid_col, how="left")

    # PyDeck haritası
    layer = pdk.Layer(
        "HeatmapLayer",
        data=risk,
        get_position=[lon_col, lat_col],
        get_weight="risk_score",
        radiusPixels=40,
        opacity=0.6,
    )

    view = pdk.ViewState(
        latitude=risk[lat_col].mean(),
        longitude=risk[lon_col].mean(),
        zoom=11.2,
        pitch=30,
    )

    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v11", initial_view_state=view, layers=[layer]))

    # En riskli GEOID'ler
    st.subheader("🚨 En Riskli 10 GEOID")
    top10 = risk.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(top10, use_container_width=True)

    st.caption("Not: Risk skoru, GEOID bazında suç gerçekleşme olasılığının normalize edilmiş değeridir.")
else:
    st.warning("Veri yüklenemedi veya boş. Artifact bağlantısını kontrol edin.")

# ───────────────────────────────
# 🧠 Sonuç Özeti
# ───────────────────────────────
st.markdown("---")
st.markdown(
    """
    **🧠 Özet:**  
    Bu sayfa, stacking tabanlı modelin son tahmin çıktısını `fr_crime_09.csv` üzerinden yükler ve  
    model metriklerini `metrics_stacking_ohe.parquet` dosyasından alır.  
    Harita, her GEOID için normalize edilmiş suç riski yoğunluğunu gösterir.  
    """
)

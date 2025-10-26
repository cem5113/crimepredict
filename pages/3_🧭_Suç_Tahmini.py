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
# --- ayarlar ---
OWNER = "cem5113"
REPO  = "crime_prediction_data"
TARGET_ARTIFACT = "fr-crime-outputs-parquet"

def resolve_github_token() -> str | None:
    import os
    import streamlit as st
    for k in ("GITHUB_TOKEN","GH_TOKEN","github_token"):
        v = os.getenv(k) or (getattr(st, "secrets", {}).get(k) if hasattr(st, "secrets") else None)
        if v:
            os.environ["GITHUB_TOKEN"] = v
            return v
    return None

@st.cache_data(show_spinner=True)
def load_artifact_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GitHub Actions artifact'tan fr_crime_09.parquet ve metrics_stacking_ohe.parquet indirir."""
    token = resolve_github_token()
    if not token:
        st.error("GitHub Token bulunamadı (GH_TOKEN veya GITHUB_TOKEN).")
        return pd.DataFrame(), pd.DataFrame()

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        # 1️⃣ En son başarılı workflow run
        runs = requests.get(
            f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs?per_page=10",
            headers=headers, timeout=30
        ).json()
        run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]
        if not run_ids:
            st.warning("Başarılı workflow run bulunamadı.")
            return pd.DataFrame(), pd.DataFrame()

        # 2️⃣ İlgili artifact'i bul
        for rid in run_ids:
            arts = requests.get(
                f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{rid}/artifacts",
                headers=headers, timeout=30
            ).json().get("artifacts", [])
            target = next((a for a in arts if a["name"] == "fr-crime-outputs-parquet" and not a.get("expired")), None)
            if not target:
                continue

            # 3️⃣ Artifact indir
            zdata = requests.get(target["archive_download_url"], headers=headers, timeout=60).content
            outer = zipfile.ZipFile(io.BytesIO(zdata))

            # 4️⃣ İçteki zip'i aç (ör: fr_parquet_outputs.zip)
            inner_zip_name = next((n for n in outer.namelist() if n.lower().endswith(".zip")), None)
            if not inner_zip_name:
                st.error("İç zip (ör: fr_parquet_outputs.zip) bulunamadı.")
                return pd.DataFrame(), pd.DataFrame()

            nested = zipfile.ZipFile(io.BytesIO(outer.read(inner_zip_name)))
            names = nested.namelist()
            st.caption(f"Artifact içerik: {names}")

            # 5️⃣ Parquet dosyalarını bul
            crime_name = next((n for n in names if n.lower().endswith("fr_crime_09.parquet")), None)
            metrics_name = next((n for n in names if n.lower().endswith("metrics_stacking_ohe.parquet")), None)

            if not crime_name:
                st.error("fr_crime_09.parquet bulunamadı.")
                return pd.DataFrame(), pd.DataFrame()

            # 6️⃣ Parquet dosyalarını oku
            df = pd.read_parquet(io.BytesIO(nested.read(crime_name)))
            metrics = pd.DataFrame()
            if metrics_name:
                try:
                    metrics = pd.read_parquet(io.BytesIO(nested.read(metrics_name)))
                except Exception as e:
                    st.warning(f"metrics_stacking_ohe.parquet okunamadı: {e}")

            st.success(f"✅ Artifact yüklendi: {crime_name}")
            return df, metrics

        st.error("Artifact bulunamadı (adı: fr-crime-outputs-parquet).")
        return pd.DataFrame(), pd.DataFrame()

    except Exception as e:
        st.error(f"Artifact indirilemedi veya açılırken hata: {e}")
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

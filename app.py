# app.py — SUTAM (revize tam sürüm)
import streamlit as st
from components.last_update import show_last_update_badge
from components.utils import MODEL_VERSION, MODEL_LAST_TRAIN
from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH
from components.gh_data import raw_url, download_actions_artifact_zip, unzip

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption(f"Rol: {APP_ROLE}")
st.info("🔎 Harita için: **🧭 Suç Tahmini** sekmesine gidin.")

# Örnek: public raw dosyayı okuma (parquet/geojson vs.)
st.write("Veri deposu:", DATA_REPO, "-", DATA_BRANCH)
st.code(raw_url("crime_prediction_data/risk_hourly.parquet"))

# (Opsiyonel) Örnek: GitHub Actions artifact indirme ve çıkarma
with st.expander("Actions artifact indir (opsiyonel)"):
    try:
        zip_path = download_actions_artifact_zip("fr-crime-pipeline-output", "downloads")
        out_dir = unzip(zip_path, "downloads/extracted")
        st.success(f"İndirildi ve açıldı: {out_dir}")
    except Exception as e:
        st.warning(f"Artifact indirilemedi: {e}")

# Model rozeti
show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)


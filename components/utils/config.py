# components/config.py
import os
import streamlit as st

CODE_REPO      = st.secrets.get("CODE_REPO", "cem5113/crimepredict")
DATA_REPO      = st.secrets.get("DATA_REPO", "cem5113/crime_prediction_data")
DATA_BRANCH    = st.secrets.get("DATA_BRANCH", "main")
GH_TOKEN       = st.secrets.get("GH_TOKEN", None)

APP_NAME       = st.secrets.get("APP_NAME", "SUTAM – Suç Tahmin Modeli")
APP_ROLE       = st.secrets.get("APP_ROLE", "Kullanıcı")

ASSET_ZIP_1    = st.secrets.get("ASSET_ZIP_1", "fr-crime-pipeline-output.zip")
ASSET_DIR_1    = st.secrets.get("ASSET_DIR_1", "sf-crime-parquet")
ASSET_DIR_2    = st.secrets.get("ASSET_DIR_2", "sf-crime-pipeline-output")

# RAW (public) içerikler için temel URL
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_BRANCH}"

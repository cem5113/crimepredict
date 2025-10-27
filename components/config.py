# components/config.py 

import os

try:
    import streamlit as st
    _secrets = getattr(st, "secrets", {})
except Exception:
    _secrets = {}

CODE_REPO   = _secrets.get("GITHUB_REPO", "cem5113/crimepredict")
DATA_REPO   = _secrets.get("DATA_REPO", "cem5113/crime_prediction_data")
DATA_BRANCH = _secrets.get("DATA_BRANCH", "main")
GH_TOKEN    = _secrets.get("GH_TOKEN", None)

APP_NAME    = _secrets.get("APP_NAME", "SUTAM – Suç Tahmin Modeli")
APP_ROLE    = _secrets.get("APP_ROLE", "Kullanıcı")

# --- Tek artifact yapı (fr-minimal-parquet) ---
ASSET_NAME     = _secrets.get("ASSET_NAME", "fr-minimal-parquet")        # Actions artifact adı
RELEASE_ASSET  = _secrets.get("RELEASE_ASSET", "fr-minimal-parquet.zip") # Release dosya adı (indirilebilir ZIP)

# RAW (public) içerikler için temel URL
RAW_BASE = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_BRANCH}"


# components/config.py
import os
import streamlit as st

# ── Temel yapılandırma
CODE_REPO   = st.secrets.get("GITHUB_REPO", "cem5113/crimepredict")
DATA_REPO   = st.secrets.get("DATA_REPO", "cem5113/crime_prediction_data")
DATA_BRANCH = st.secrets.get("DATA_BRANCH", "main")

# ── Uygulama kimliği
APP_NAME    = st.secrets.get("APP_NAME", "SUTAM – Suç Tahmin Modeli")
APP_ROLE    = st.secrets.get("APP_ROLE", "Kullanıcı")

# ── Artifact / asset ayarları
ASSET_ZIP_1 = st.secrets.get("ASSET_ZIP_1", "fr-crime-pipeline-output.zip")
ASSET_DIR_1 = st.secrets.get("ASSET_DIR_1", "sf-crime-parquet")
ASSET_DIR_2 = st.secrets.get("ASSET_DIR_2", "sf-crime-pipeline-output")

# ── RAW (public) içerikler için temel URL
RAW_BASE    = f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_BRANCH}"

# ── GH_TOKEN çözümleme (env > secrets)
def resolve_github_token() -> str | None:
    """
    Öncelik sırası:
    1. Ortam değişkeni (GITHUB_TOKEN)
    2. Streamlit secrets (GH_TOKEN, github_token, GITHUB_TOKEN)
    """
    # 1) Ortamda varsa hemen dön
    tok = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if tok:
        return tok

    # 2) Secrets içinde ara
    try:
        for k in ("GH_TOKEN", "GITHUB_TOKEN", "github_token"):
            if k in st.secrets and st.secrets[k]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[k])  # Ortama yaz
                return os.environ["GITHUB_TOKEN"]
    except Exception:
        pass

    # 3) Yoksa None döndür
    return None

# ── Global değişken olarak da set et
GH_TOKEN = resolve_github_token()

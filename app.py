# SUTAM - Suç Tahmin Modeli
# SUTAM - Suç Tahmin Modeli

import io
import os
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

# ------------------------------------------------------------
# Ayarlar — Gerekli depolar ve artifact adı
# ------------------------------------------------------------
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"
EXPECTED_PARQUET = "risk_hourly.parquet"
EXPECTED_CSV = "risk_hourly.csv"

# GeoJSON farklı bir repoda olabilir (ör. cem5113/crimepredict)
GEOJSON_OWNER = st.secrets.get("geojson_owner", OWNER)
GEOJSON_REPO = st.secrets.get("geojson_repo", "crimepredict")

# GitHub Token (artifact erişimi için gerekir)
GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

# ------------------------------------------------------------
# Yardımcılar — Artifact indirme
# ------------------------------------------------------------

@st.cache_data(show_spinner=True, ttl=15 * 60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    """Repo'daki son **geçerli** (expired=false) artifact'i bulur ve ZIP bytes döndürür."""
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "GitHub token bulunamadı. lütfen st.secrets['github_token'] veya GITHUB_TOKEN env. değişkeni ayarlayın."
        )

    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

    r = requests.get(base, headers=headers, timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])

    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")

    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    artifact = cand[0]

    download_url = artifact.get("archive_download_url")
    if not download_url:
        raise RuntimeError("archive_download_url bulunamadı")

    r2 = requests.get(download_url, headers=headers, timeout=60)
    r2.raise_for_status()
    return r2.content


@st.cache_data(show_spinner=True, ttl=15 * 60)
def read_risk_from_artifact() -> pd.DataFrame:
    """ZIP içinden Parquet/CSV'yi okuyup kolonları normalize eder."""
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        target = (
            EXPECTED_PARQUET
            if EXPECTED_PARQUET in memlist
            else EXPECTED_CSV
            if EXPECTED_CSV in memlist
            else None
        )

        if target is None:
            raise FileNotFoundError(
                f"Zip içinde {EXPECTED_PARQUET} veya {EXPECTED_CSV} bulunamadı. İçerik: {memlist}"
            )

        with zf.open(target) as f:
            if target.endswith(".parquet"):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)

    # Kolon adlarını normalize et
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

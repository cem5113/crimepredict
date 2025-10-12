# config/settings.py
from pathlib import Path
import os

# Artefact ZIP yolu (GitHub Actions çıktısı)
ARTIFACT_ZIP = Path(os.getenv("CRIME_ARTIFACT_ZIP", "sf-crime-parquet.zip"))

# Home mini harita için hangi parquet?
DEFAULT_PARQUET_MEMBER = os.getenv("HOME_MAP_PARQUET", "risk_hourly.parquet")

# Varsayılan harita konumu (SF)
MAP_VIEW = {
    "lat": 37.7749,
    "lon": -122.4194,
    "zoom": 11,
    "pitch": 35,
    "bearing": 0,
}

# Home’da Top-K KPI (gerekirse renklendirme vs.)
RISK_LEVEL_LABELS = {3: "Very High", 2: "High", 1: "Medium", 0: "Low"}

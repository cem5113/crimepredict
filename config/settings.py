# config/settings.py
from pathlib import Path
import os

ARTIFACT_ZIP = Path(os.getenv("CRIME_ARTIFACT_ZIP", "sf-crime-parquet.zip"))
DEFAULT_PARQUET_MEMBER = os.getenv("HOME_MAP_PARQUET", "sf_crime_09.parquet") 

DEFAULT_PARQUET_MEMBER = os.getenv("HOME_MAP_PARQUET", "risk_hourly.parquet")

MAP_VIEW = {
    "lat": 37.7749,
    "lon": -122.4194,
    "zoom": 11,
    "pitch": 35,
    "bearing": 0,
}

RISK_LEVEL_LABELS = {3: "Very High", 2: "High", 1: "Medium", 0: "Low"}

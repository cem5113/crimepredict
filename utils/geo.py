from __future__ import annotations
import pandas as pd
from core.data import load_geoid_centroids, attach_latlon
from utils.constants import KEY_COL

def add_centroids(df: pd.DataFrame) -> pd.DataFrame:
    cents = load_geoid_centroids()
    return attach_latlon(df, cents)

def ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if KEY_COL not in df.columns:
        raise KeyError(f"{KEY_COL} kolonu yok.")
    return df

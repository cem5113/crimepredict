# tabs/home/__init__.py
from __future__ import annotations
import pandas as pd
import streamlit as st
from core.data import load_parquet
from core.mapkit import home_deck

TAB_KEY = "home"

def _clean_geoid(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"[^\d]", "", regex=True)
    s = s.replace("", pd.NA)
    return s

def _level_from_score(s: pd.Series) -> pd.Series:
    if s.empty:
        return s.astype("Float64")
    q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
    lvl = pd.Series(0, index=s.index, dtype="int64")
    lvl = lvl.mask(s > q1, 1)
    lvl = lvl.mask(s > q2, 2)
    lvl = lvl.mask(s > q3, 3)
    return lvl.astype("int64")

def render(state=None, services=None):
    st.title("🏠 Suç Tahmini — Ana Sayfa")

    df = load_parquet("risk_hourly.parquet", columns=["geoid", "risk_score", "date", "hour_range"])
    if df is None or df.empty:
        st.warning("⚠️ Harita için uygun veri bulunamadı.")
        return

    # GEOID fix
    df["geoid"] = _clean_geoid(df["geoid"])
    df = df.dropna(subset=["geoid", "risk_score", "date"])

    # Aynı date içindeki risk skorlarının ortalaması (saat filtresine gerek yok denmişti)
    latest_date = df["date"].max()
    df = df[df["date"] == latest_date]
    df = df.groupby("geoid", as_index=False)["risk_score"].mean()

    # Görselleştirme için seviye
    df["risk_level"] = _level_from_score(df["risk_score"]).astype("int64")

    try:
        st.pydeck_chart(home_deck(df))
    except Exception as e:
        st.error(f"Harita oluşturulamadı: {e}")

def register():
    return {
        "key": TAB_KEY,
        "title": "Ana Sayfa",
        "label": "🏠 Ana Sayfa",
        "icon": "🏠",
        "order": 0,
        "render": render,
    }

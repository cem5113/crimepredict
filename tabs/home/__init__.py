import streamlit as st
from core.data import load_parquet
from core.mapkit import home_deck

TAB_KEY = "home"

def render(state=None, services=None):
    st.title("🏠 Suç Tahmini — Ana Sayfa")
    df = load_parquet("risk_hourly.parquet", columns=["geoid","risk_score","risk_level","date","hour_range"])
    if df.empty:
        st.info("Harita için veri bulunamadı.")
        return
    st.pydeck_chart(home_deck(df))

def register():
    return {
        "key": TAB_KEY,
        "title": "Ana Sayfa",
        "icon": "🏠",
        "label": "🏠 Ana Sayfa",
        "order": 0,
        "render": render,
    }

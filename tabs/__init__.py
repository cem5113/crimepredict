import streamlit as st
from core.data import load_parquet
from core.mapkit import home_deck

TAB_KEY = "home"

def render():
    st.title("🏠 Suç Tahmini — Ana Sayfa")
    df = load_parquet("risk_hourly.parquet",
                      columns=["geoid","risk_score","risk_level","date","hour_range"])
    if df.empty:
        st.info("Harita için veri bulunamadı.")
        return
    deck = home_deck(df)
    st.pydeck_chart(deck)

def register():
    return {
        "key": TAB_KEY,
        "title": "Ana Sayfa",      
        "icon": "🏠",               
        "order": 0,
        "render": render,
    }

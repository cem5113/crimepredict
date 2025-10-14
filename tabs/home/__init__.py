import streamlit as st
from core.data import load_parquet
from core.mapkit import home_deck

TAB_KEY = "home"

def render(state=None, services=None):
    st.title("🏠 Suç Tahmini — Ana Sayfa")

    # Veri yükleme
    df = load_parquet("risk_hourly.parquet", columns=["geoid", "risk_score", "date", "hour_range"])
    if df is None or df.empty:
        st.warning("⚠️ Harita için uygun veri bulunamadı.")
        return

    # Opsiyonel: Tarih ve saat filtresi
    if "date" in df.columns and "hour_range" in df.columns:
        dates = sorted(df["date"].dropna().astype(str).unique())
        if dates:
            sel_date = st.selectbox("📅 Tarih seç", dates, index=len(dates)-1)
            hours = sorted(df.loc[df["date"].astype(str) == sel_date, "hour_range"].astype(str).unique())
            sel_hour = st.selectbox("🕐 Saat aralığı", hours)
            df = df[
                (df["date"].astype(str) == sel_date)
                & (df["hour_range"].astype(str) == sel_hour)
            ]

    # Harita
    try:
        st.pydeck_chart(home_deck(df))
    except Exception as e:
        st.error(f"Harita oluşturulamadı: {e}")

def register():
    return {
        "key": TAB_KEY,
        "title": "Ana Sayfa",
        "icon": "🏠",
        "label": "🏠 Ana Sayfa",
        "order": 0,
        "render": render,
    }

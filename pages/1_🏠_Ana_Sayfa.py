# pages/1_🏠_Ana_Sayfa.py
import streamlit as st
from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

st.set_page_config(page_title="🏠 Ana Sayfa", layout="wide")

def render_home():
    st.title("🏠 Ana Sayfa")
    st.markdown("## 👋 Hoş geldiniz — SUTAM Suç Tahmin Modeli")
    st.info(
        "Bu sistem, San Francisco suç verilerini analiz ederek günlük risk haritası ve tahminler üretir.  \n"
        "Sol menüden **Suç Risk Haritası**, **Tahmin** veya **İstatistik** sekmelerine geçebilirsiniz."
    )
    st.success("Veri pipeline’ı ve model güncellemeleri her gün otomatik olarak yapılır.")

render_home()

show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

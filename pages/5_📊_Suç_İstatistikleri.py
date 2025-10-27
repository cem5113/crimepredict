# pages/5_📊_Suç_İstatistikleri.py 

import streamlit as st
from ui.tab_stats import render_stats  # ui/tab_stats.py

from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

# NOT: st.set_page_config(...) sadece app.py'de olacak, buradan kaldırıldı.

# Sayfa başlığı
st.title("📊 Suç İstatistikleri")

# İstatistik sekmesi içeriğini oluştur
render_stats()

# Model sürümü ve son eğitim bilgisi
show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

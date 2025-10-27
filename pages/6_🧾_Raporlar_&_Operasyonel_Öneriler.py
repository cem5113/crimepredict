# pages/6_🧾_Raporlar_&_Operasyonel_Öneriler.py 

import streamlit as st
from ui.tab_reports import render_reports  # ui/tab_reports.py

from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

# NOT: st.set_page_config(...) sadece app.py'de olacak, buradan kaldırıldı.

# Sayfa başlığı
st.title("🧾 Raporlar & Operasyonel Öneriler")

# Rapor sekmesi içeriğini oluştur
render_reports()

# Model sürümü ve son eğitim bilgisi
show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

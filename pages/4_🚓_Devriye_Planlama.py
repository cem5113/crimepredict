# pages/4_👮‍♂️_Devriye_Planlama.py 
import streamlit as st
from ui.tab_planning import render_planning  # ui/tab_planning.py

from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

# NOT: st.set_page_config(...) sadece app.py'de olacak, buradan kaldırıldı.

# Sayfa başlığı
st.title("🚓 Devriye Planlama")

# Ana içeriği çağır
render_planning()

# Model sürümü ve son eğitim bilgisi
show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

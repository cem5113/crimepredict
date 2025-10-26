import streamlit as st
from ui.tab_planning import render_planning  # ui/tab_planning.py

from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

st.set_page_config(page_title="🚓 Devriye Planlama", layout="wide")

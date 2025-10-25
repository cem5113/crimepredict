import streamlit as st
from ui.tab_stats import render_stats  # ui/tab_stats.py

from components.last_update import show_last_update_badge
from components.utils import MODEL_VERSION, MODEL_LAST_TRAIN

st.set_page_config(page_title="📊 Suç İstatistikleri", layout="wide")
render_stats()

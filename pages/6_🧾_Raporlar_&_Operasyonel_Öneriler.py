import streamlit as st
from ui.tab_reports import render_reports  # ui/tab_reports.py

from components.last_update import show_last_update_badge
from components.utils import MODEL_VERSION, MODEL_LAST_TRAIN

st.set_page_config(page_title="🧾 Raporlar & Operasyonel Öneriler", layout="wide")
render_reports()

import streamlit as st
from ui.home import render_home  # ui/home.py içinde var

from components.last_update import show_last_update_badge
from components.utils import MODEL_VERSION, MODEL_LAST_TRAIN

st.set_page_config(page_title="🏠 Ana Sayfa", layout="wide")
render_home()

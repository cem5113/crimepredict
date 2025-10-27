# pages/6_🧾_Raporlar_&_Operasyonel_Öneriler.py — revize

# 0) path-fix
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

# 1) Güvenli importlar
try:
    from ui.tab_reports import render_reports  # ui/tab_reports.py
except Exception as e:
    render_reports = None
    st.error("`ui.tab_reports` modülü bulunamadı.")
    st.caption(f"Ayrıntı: {e}")

try:
    from components.last_update import show_last_update_badge
    from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN
except Exception:
    def show_last_update_badge(*a, **k): ...
    MODEL_VERSION, MODEL_LAST_TRAIN = "v0", "-"

# 2) Sayfa gövdesi
st.title("🧾 Raporlar & Operasyonel Öneriler")

if render_reports:
    render_reports()

show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

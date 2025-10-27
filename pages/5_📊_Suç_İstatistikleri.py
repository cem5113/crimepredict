# pages/5_📊_Suç_İstatistikleri.py — revize

# 0) path-fix
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

# 1) Güvenli importlar
try:
    from ui.tab_stats import render_stats  # ui/tab_stats.py
except Exception as e:
    render_stats = None
    st.error("`ui.tab_stats` modülü bulunamadı.")
    st.caption(f"Ayrıntı: {e}")

try:
    from components.last_update import show_last_update_badge
    from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN
except Exception:
    def show_last_update_badge(*a, **k): ...
    MODEL_VERSION, MODEL_LAST_TRAIN = "v0", "-"

# 2) Sayfa gövdesi
st.title("📊 Suç İstatistikleri")

if render_stats:
    render_stats()

show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

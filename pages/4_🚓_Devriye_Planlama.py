# pages/4_👮‍♂️_Devriye_Planlama.py — revize

# 0) path-fix: pages/ → kök (ui/, components/) path'e eklensin
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../crimepredict
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

# 1) Güvenli importlar
try:
    from ui.tab_planning import render_planning  # ui/tab_planning.py
except Exception as e:
    render_planning = None
    st.error("`ui.tab_planning` modülü bulunamadı.")
    st.caption(f"Ayrıntı: {e}")

try:
    from components.last_update import show_last_update_badge
    from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN
except Exception:
    # opsiyonel: modül yoksa sessiz fallback
    def show_last_update_badge(*a, **k): ...
    MODEL_VERSION, MODEL_LAST_TRAIN = "v0", "-"

# 2) Sayfa gövdesi
st.title("🚓 Devriye Planlama")

if render_planning:
    render_planning()

show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

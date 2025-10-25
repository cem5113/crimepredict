# components/last_update.py
import streamlit as st
from datetime import datetime

def show_last_update_badge(ts: str | None = None):
    # ts yoksa şimdi
    when = ts or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:.25rem .5rem;border-radius:.5rem;background:#eef; border:1px solid #ccd;">
            <b>Son Güncelleme:</b> {when}
        </div>
        """,
        unsafe_allow_html=True,
    )

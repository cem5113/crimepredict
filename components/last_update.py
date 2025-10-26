# components/last_update.py
import streamlit as st

def show_last_update_badge(data_upto=None, model_version=None, last_train=None):
    parts = []
    if data_upto: parts.append(f"Veri: {data_upto}")
    if model_version: parts.append(f"Model: {model_version}")
    if last_train: parts.append(f"Eğitim: {last_train}")
    if not parts: return
    st.markdown(
        "<div style='display:inline-block;padding:.25rem .5rem;border:1px solid #ccd;"
        "border-radius:.5rem;background:#eef;'>" + " | ".join(parts) + "</div>",
        unsafe_allow_html=True,
    )

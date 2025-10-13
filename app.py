# app.py
import streamlit as st
from pathlib import Path
import importlib, sys                     # <— util yerine standart importlib
from core.data_boot import configure_artifact_env
configure_artifact_env()

st.set_page_config(page_title="Suç Tahmini", page_icon="🔎", layout="wide")

def _discover_tabs():
    """
    Sekmeleri dosyadan exec_module ile değil, PAKET olarak import eder.
    Böylece tabs/<name>/__init__.py içindeki 'from .view import render' gibi
    göreli importlar doğru çalışır.
    """
    # crimepredict/app.py konumuna göre tabs klasörünü bul
    tabs_dir = Path(__file__).parent / "tabs"
    specs = []
    if not tabs_dir.exists():
        return specs

    # Bu dosya bir paket içindeyiz -> __package__ 'crimepredict' olmalı
    base_pkg = __package__ or "crimepredict"
    tabs_pkg = f"{base_pkg}.tabs"

    # tabs paketinin import edilebilir olması için parent'ı sys.path'te olsun
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    for sub in sorted(tabs_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "__init__.py").exists():
            continue
        mod_name = f"{tabs_pkg}.{sub.name}"      # Örn: crimepredict.tabs.home
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "register"):
                specs.append(mod.register())
        except Exception as e:
            # Hata detayı gizlenmesin diye kullanıcıya kısa bilgi verelim
            st.error(f"Sekme yüklenemedi: {sub.name} → {e}")

    # Sıralama
    order = ["home", "forecast", "planning", "stats", "reports"]
    specs.sort(key=lambda x: order.index(x["key"]) if x["key"] in order else 99)
    return specs

def main():
    tabs = _discover_tabs()
    if not tabs:
        st.error("Sekme bulunamadı. `tabs/<name>/__init__.py` içinde register() tanımlayın.")
        st.stop()

    active = st.session_state.get("__active_tab__", tabs[0]["key"])

    with st.sidebar:
        st.header("Menü")
        labels = [f"{t['icon']} {t['title']}" for t in tabs]
        keys   = [t["key"] for t in tabs]
        idx = keys.index(active) if active in keys else 0
        choice = st.radio("Sekme", labels, index=idx, label_visibility="collapsed")
        active = keys[labels.index(choice)]
        st.session_state["__active_tab__"] = active

    current = next(t for t in tabs if t["key"] == active)
    current["render"](state=None, services=None)

if __name__ == "__main__":
    main()

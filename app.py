# app.py
import streamlit as st
from pathlib import Path
import importlib.util, sys

st.set_page_config(page_title="Suç Tahmini", page_icon="🔎", layout="wide")

def _discover_tabs():
    tabs_dir = Path("tabs")
    specs = []
    if not tabs_dir.exists():
        return specs
    for sub in sorted(tabs_dir.iterdir()):
        if not sub.is_dir(): 
            continue
        init_py = sub / "__init__.py"
        if not init_py.exists():
            continue
        mod_name = f"tabs.{sub.name}.__init__"
        spec = importlib.util.spec_from_file_location(mod_name, init_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)  # type: ignore
        if hasattr(mod, "register"):
            specs.append(mod.register())
    # Sıra: home, forecast, planning, stats, reports (varsa)
    order = ["home","forecast","planning","stats","reports"]
    specs.sort(key=lambda x: order.index(x["key"]) if x["key"] in order else 99)
    return specs

def main():
    tabs = _discover_tabs()
    if not tabs:
        st.error("Sekme bulunamadı. `tabs/<name>/__init__.py` içinde register() tanımlayın.")
        st.stop()

    # Aktif sekme durumu
    active = st.session_state.get("__active_tab__", tabs[0]["key"])

    # Sidebar menü
    with st.sidebar:
        st.header("Menü")
        labels = [f"{t['icon']} {t['title']}" for t in tabs]
        keys   = [t["key"] for t in tabs]
        idx = keys.index(active) if active in keys else 0
        choice = st.radio("Sekme", labels, index=idx, label_visibility="collapsed")
        active = keys[labels.index(choice)]
        st.session_state["__active_tab__"] = active

    # Seçili sekmeyi çiz
    current = next(t for t in tabs if t["key"] == active)
    current["render"](state=None, services=None)

if __name__ == "__main__":
    main()

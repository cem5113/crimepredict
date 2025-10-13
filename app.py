# app.py  (yalnızca _discover_tabs() fonksiyonunu bu hâliyle değiştir)
from pathlib import Path
import importlib.util, sys
import streamlit as st

def _discover_tabs():
    """
    tabs/<name>/__init__.py dosyalarını **paket bağlamıyla** yükler.
    Böylece `from .view import render` güvenle çalışır.
    """
    here = Path(__file__).resolve()
    tabs_dir = here.parent / "tabs"                 # crimepredict/tabs
    specs = []
    if not tabs_dir.exists():
        return specs

    base_pkg = __package__ or "crimepredict"        # örn: 'crimepredict'
    # kökü sys.path'e ekle (import için)
    root_dir = here.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    for sub in sorted(tabs_dir.iterdir()):
        if not sub.is_dir():
            continue
        init_py = sub / "__init__.py"
        if not init_py.exists():
            continue

        # Bu modül için tam paket adı (örn: crimepredict.tabs.home)
        mod_name = f"{base_pkg}.tabs.{sub.name}"

        try:
            spec = importlib.util.spec_from_file_location(mod_name, init_py)
            mod = importlib.util.module_from_spec(spec)
            # 🔑 Paket bağlamını kur: relative importlar ('.view') çalışsın
            mod.__package__ = f"{base_pkg}.tabs.{sub.name}"
            sys.modules[mod_name] = mod
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore

            if hasattr(mod, "register"):
                specs.append(mod.register())
        except Exception as e:
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

# app.py
from __future__ import annotations
import streamlit as st, pkgutil, importlib, pathlib, importlib.util, sys
from pathlib import Path
from core.data_boot import configure_artifact_env
configure_artifact_env()

st.set_page_config(page_title="Suç Tahmini", page_icon="🔎", layout="wide")

def _discover_tabs():
    here = Path(__file__).resolve()
    pkg_root = here.parent.parent
    tabs_dir = here.parent / "tabs"
    specs = []
    if not tabs_dir.exists():
        return specs
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    pkg_name = here.parent.name

    for sub in sorted(tabs_dir.iterdir()):
        if not sub.is_dir(): continue
        init_py = sub / "__init__.py"
        if not init_py.exists(): continue

        mod = None
        # 1) crimepredict.tabs.<name>
        try:
            mod = importlib.import_module(f"crimepredict.tabs.{sub.name}")
        except Exception:
            pass
        # 2) <pkg_name>.tabs.<name>
        if mod is None:
            try:
                mod = importlib.import_module(f"{pkg_name}.tabs.{sub.name}")
            except Exception:
                pass
        # 3) path'ten yükle
        if mod is None:
            spec = importlib.util.spec_from_file_location(f"{pkg_name}.tabs.{sub.name}", init_py)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[f"{pkg_name}.tabs.{sub.name}"] = mod
                spec.loader.exec_module(mod)  # type: ignore

        if mod and hasattr(mod, "register"):
            specs.append(mod.register())

    order = ["home", "forecast", "planning", "stats", "reports", "diagnostics"]
    specs.sort(key=lambda x: order.index(x["key"]) if x["key"] in order else x.get("order", 99))
    return specs

def main():
    tabs = _discover_tabs()
    if not tabs:
        st.error("Sekme bulunamadı. `tabs/<name>/__init__.py` içinde register() tanımlayın.")
        st.stop()

    active = st.session_state.get("__active_tab__", tabs[0]["key"])
    with st.sidebar:
        st.header("Menü")
        labels = [f"{t.get('icon','🗂️')} {t.get('title', t.get('label','Sekme'))}" for t in tabs]
        keys   = [t["key"] for t in tabs]
        idx = keys.index(active) if active in keys else 0
        choice = st.radio("Sekme", labels, index=idx, label_visibility="collapsed")
        active = keys[labels.index(choice)]
        st.session_state["__active_tab__"] = active

    current = next(t for t in tabs if t["key"] == active)
    # render imzasını esnek yapıyoruz (aşağıda tabs yamaları var)
    current["render"](state=None, services=None)

if __name__ == "__main__":
    main()

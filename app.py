# app.py
import streamlit as st
from pathlib import Path
import importlib, importlib.util, sys
from core.data_boot import configure_artifact_env
configure_artifact_env()

st.set_page_config(page_title="Suç Tahmini", page_icon="🔎", layout="wide")

def _discover_tabs():
    """
    Sekmeleri sağlam biçimde bul ve yükle:
    1) crimepredict.tabs.<name>
    2) <paket_adı>.tabs.<name>  (dosya konumundan türetilir)
    3) Path'ten yükleme (spec_from_file_location) — relative importlar çalışsın
    """
    here = Path(__file__).resolve()
    pkg_root = here.parent.parent              # projenin kökü
    tabs_dir = here.parent / "tabs"            # crimepredict/tabs
    specs = []
    if not tabs_dir.exists():
        return specs

    # PYTHONPATH'e kökü ekle (import için)
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    # Dosya konumundan paket adını çıkar (örn. 'crimepredict')
    pkg_name = here.parent.name

    for sub in sorted(tabs_dir.iterdir()):
        if not sub.is_dir():
            continue
        init_py = sub / "__init__.py"
        if not init_py.exists():
            continue

        mod = None
        errors = []

        # 1) En yaygın: crimepredict.tabs.<name>
        cand1 = f"crimepredict.tabs.{sub.name}"
        try:
            mod = importlib.import_module(cand1)
        except Exception as e:
            errors.append(f"{cand1}: {e}")

        # 2) Dosya konumundan türetilen paket adı
        if mod is None:
            cand2 = f"{pkg_name}.tabs.{sub.name}"
            try:
                mod = importlib.import_module(cand2)
            except Exception as e:
                errors.append(f"{cand2}: {e}")

        # 3) Son çare: path'ten yükle (paket adını vererek)
        if mod is None:
            try:
                name = f"{pkg_name}.tabs.{sub.name}"
                spec = importlib.util.spec_from_file_location(name, init_py)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore
            except Exception as e:
                errors.append(f"spec-load {sub.name}: {e}")

        if mod is None:
            st.error(f"Sekme yüklenemedi: {sub.name} → {' | '.join(errors)}")
            continue

        if hasattr(mod, "register"):
            try:
                specs.append(mod.register())
            except Exception as e:
                st.error(f"Sekme register() hatası: {sub.name} → {e}")

    # Sıra: home, forecast, planning, stats, reports (varsa)
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

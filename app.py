# app.py
from __future__ import annotations
import inspect
import sys
import importlib
import importlib.util
from pathlib import Path

import streamlit as st

from core.data_boot import configure_artifact_env
configure_artifact_env()

# Streamlit config EN ÜSTE olmalı
st.set_page_config(page_title="Suç Tahmini", page_icon="🔎", layout="wide")


def _discover_tabs():
    """
    tabs/<name>/__init__.py içindeki register() fonksiyonlarını bulup çağırır.
    Arama sırası:
      1) crimepredict.tabs.<name>
      2) <paket_adı>.tabs.<name>  (dosya konumundan türetilen)
      3) Yol üzerinden modül yükleme
    """
    here = Path(__file__).resolve()
    pkg_root = here.parent.parent
    tabs_dir = here.parent / "tabs"
    specs = []

    if not tabs_dir.exists():
        return specs

    # PYTHONPATH'e proje kökünü ekle
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    pkg_name = here.parent.name  # örn: crimepredict

    for sub in sorted(tabs_dir.iterdir()):
        if not sub.is_dir():
            continue
        init_py = sub / "__init__.py"
        if not init_py.exists():
            continue

        mod = None

        # 1) Sabit paket adıyla dene
        try:
            mod = importlib.import_module(f"crimepredict.tabs.{sub.name}")
        except Exception:
            mod = None

        # 2) Dinamik paket adıyla dene
        if mod is None:
            try:
                mod = importlib.import_module(f"{pkg_name}.tabs.{sub.name}")
            except Exception:
                mod = None

        # 3) Yol üzerinden yükle
        if mod is None:
            spec = importlib.util.spec_from_file_location(f"{pkg_name}.tabs.{sub.name}", init_py)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[f"{pkg_name}.tabs.{sub.name}"] = mod
                spec.loader.exec_module(mod)  # type: ignore

        if mod and hasattr(mod, "register"):
            try:
                spec_dict = mod.register()
                # Asgari alanları güvenceye al
                spec_dict.setdefault("key", sub.name)
                spec_dict.setdefault("title", spec_dict.get("label", sub.name.title()))
                spec_dict.setdefault("icon", "🗂️")
                spec_dict.setdefault("order", 99)
                assert callable(spec_dict["render"]), f"{sub.name}: render callable değil."
                specs.append(spec_dict)
            except Exception as e:
                st.error(f"Sekme register() hatası: {sub.name} → {e}")

    # İstenen sıralama önceliği
    order_pref = ["home", "forecast", "planning", "stats", "reports", "diagnostics"]
    specs.sort(key=lambda x: order_pref.index(x["key"]) if x["key"] in order_pref else x.get("order", 99))
    return specs


def _safe_render(render_fn):
    """
    Sekme render fonksiyonunu imzasına göre güvenle çağırır.
    Aşağıdakilerin hepsini destekler:
      - render()
      - render(state=None)
      - render(state=None, services=None)
    """
    try:
        sig = inspect.signature(render_fn)
        params = sig.parameters
        if len(params) == 0:
            return render_fn()
        kwargs = {}
        if "state" in params:
            kwargs["state"] = None
        if "services" in params:
            kwargs["services"] = None
        return render_fn(**kwargs) if kwargs else render_fn()
    except TypeError:
        # Her ihtimale karşı basit çağrı
        return render_fn()


def main():
    tabs = _discover_tabs()
    if not tabs:
        st.error("Sekme bulunamadı. `tabs/<name>/__init__.py` içinde register() tanımlayın.")
        st.stop()

    # Aktif sekme
    active_key = st.session_state.get("__active_tab__", tabs[0]["key"])

    # Sidebar menü
    with st.sidebar:
        st.header("Menü")
        labels = [f"{t.get('icon', '🗂️')} {t.get('title', t.get('label', 'Sekme'))}" for t in tabs]
        keys = [t["key"] for t in tabs]
        idx = keys.index(active_key) if active_key in keys else 0
        choice = st.radio("Sekme", labels, index=idx, label_visibility="collapsed")
        active_key = keys[labels.index(choice)]
        st.session_state["__active_tab__"] = active_key

    # Seçili sekmeyi çalıştır
    current = next(t for t in tabs if t["key"] == active_key)
    _safe_render(current["render"])


if __name__ == "__main__":
    main()

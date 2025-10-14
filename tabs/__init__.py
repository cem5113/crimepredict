# tabs/__init__.py
from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Any
import streamlit as st

TabSpec = Dict[str, Any]  # {"key","label"/"title","render", optional: "order","icon"}

def _normalize(spec: dict) -> dict | None:
    if not isinstance(spec, dict) or "key" not in spec or "render" not in spec:
        return None
    # title/label eşitle
    if "label" not in spec and "title" in spec:
        spec["label"] = spec["title"]
    if "title" not in spec and "label" in spec:
        spec["title"] = spec["label"]
    return spec

def _try_import(fullname: str) -> dict | None:
    try:
        mod = importlib.import_module(fullname)
    except Exception as e:
        st.warning(f"'{fullname}' yüklenemedi: {e}")
        return None
    reg = getattr(mod, "register", None)
    if not callable(reg):
        return None
    try:
        spec = reg()
    except Exception as e:
        st.warning(f"'{fullname}.register()' hata verdi: {e}")
        return None
    return _normalize(spec)

def load_tabs(package: str = "tabs") -> List[TabSpec]:
    loaded: List[TabSpec] = []
    pkg = importlib.import_module(package)

    # 1) Paket altındaki *paket* sekmeleri (tabs/<name>/__init__.py)
    for info in pkgutil.iter_modules(pkg.__path__):
        name = info.name
        if name.startswith("_") and name != "_template":
            continue
        fullname = f"{package}.{name}"
        if info.ispkg:
            spec = _try_import(fullname)
            if spec:
                loaded.append(spec)

    # 2) Paket altındaki *modül* sekmeleri (tabs/<name>.py)
    pkg_path = Path(pkg.__path__[0])
    for py in pkg_path.glob("*.py"):
        if py.name in {"__init__.py"}:
            continue
        name = py.stem
        if name.startswith("_"):
            continue
        fullname = f"{package}.{name}"
        spec = _try_import(fullname)
        if spec:
            loaded.append(spec)

    # 3) _template istenirse dahil edilebilir (opsiyonel)
    # spec = _try_import(f"{package}._template")
    # if spec: loaded.append(spec)

    if not loaded:
        st.error("Yüklenecek geçerli sekme bulunamadı (register() yok).")
        return []

    # Sıralama
    loaded.sort(key=lambda x: (int(x.get("order", 999)), str(x.get("title", ""))))
    # Tanı için kısa liste göster
    st.caption("Yüklenen sekmeler: " + ", ".join([t.get("key", "?") for t in loade]()

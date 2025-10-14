# tabs/__init__.py
from __future__ import annotations
import importlib
import pkgutil
from typing import Dict, List, Callable, Any
import streamlit as st

TabSpec = Dict[str, Any]  # {"key","label"/"title","render", optional: "order","icon"}

def _safe_register(mod) -> TabSpec | None:
    reg = getattr(mod, "register", None)
    if not callable(reg):
        return None
    try:
        spec = reg()
    except Exception as e:
        st.warning(f"'{mod.__name__}' register() hatası: {e}")
        return None

    if not isinstance(spec, dict) or "key" not in spec or "render" not in spec:
        st.warning(f"'{mod.__name__}' register() çıktısı eksik (key/render).")
        return None

    # title/label uyumluluğu
    if "label" not in spec and "title" in spec:
        spec["label"] = spec["title"]
    if "title" not in spec and "label" in spec:
        spec["title"] = spec["label"]

    return spec

def load_tabs(package: str = "tabs") -> List[TabSpec]:
    loaded: List[TabSpec] = []
    pkg = importlib.import_module(package)
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        if not modinfo.ispkg:
            continue
        name = modinfo.name
        try:
            mod = importlib.import_module(f"{package}.{name}")
        except Exception as e:
            st.warning(f"'{name}' sekmesi yüklenemedi: {e}")
            continue
        spec = _safe_register(mod)
        if spec:
            loaded.append(spec)

    if not loaded:
        st.error("Yüklenecek geçerli sekme bulunamadı.")
        return []

    loaded.sort(key=lambda x: (int(x.get("order", 999)), str(x.get("title", ""))))
    return loaded

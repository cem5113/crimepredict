# app.py
from __future__ import annotations

# --- SAFE IMPORT SHIM (kritik) ---
import sys, importlib.util
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]  # repo kökü
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from crimepredict.dataio.loaders import (
        load_sf_crime_latest,
        load_metadata_or_default,
        _validate_schema,
    )
except ModuleNotFoundError:
    _cand = _THIS.parent / "dataio" / "loaders.py"
    if not _cand.exists():
        _cand = _REPO_ROOT / "crimepredict" / "dataio" / "loaders.py"
    spec = importlib.util.spec_from_file_location("crimepredict.dataio.loaders", _cand)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader, "loaders.py bulunamadı veya yüklenemedi."
    spec.loader.exec_module(mod)  # type: ignore
    load_sf_crime_latest = mod.load_sf_crime_latest
    load_metadata_or_default = mod.load_metadata_or_default
    _validate_schema = mod._validate_schema
# --- SAFE IMPORT SHIM SONU ---

import inspect
import importlib
import importlib.util
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

# --- Ortam/Artifact ayarı ---
from core.data_boot import configure_artifact_env

# Streamlit config EN ÜSTE olmalı
configure_artifact_env()
st.set_page_config(page_title="Suç Tahmini", page_icon="🔎", layout="wide")


# ========================= yardımcılar =========================
# === DEBUG & FIX HELPERS ===
import pandas as pd
import streamlit as st

def debug_geoid(df: pd.DataFrame, geoid_col="GEOID", sample=5) -> dict:
    out = {"geoid_present": geoid_col in df.columns}
    if not out["geoid_present"]:
        return out
    s = df[geoid_col]
    out["geoid_dtype"] = str(s.dtype)
    bad_float_mask = s.apply(lambda x: isinstance(x, float))
    out["geoid_float_count"] = int(bad_float_mask.sum())
    out["geoid_float_samples"] = s[bad_float_mask].head(sample).tolist()
    try:
        str_s = s.astype(str)
        dot_mask = str_s.str.contains(r"\.0$", regex=True)
        out["geoid_str_dotzero_count"] = int(dot_mask.sum())
        out["geoid_str_dotzero_samples"] = str_s[dot_mask].head(sample).tolist()
    except Exception as e:
        out["geoid_str_check_error"] = repr(e)
    return out

def coerce_geoid(df: pd.DataFrame, col="GEOID") -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = (df[col].astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)  # sondaki .0'ı at
                .str.replace(".", "", regex=False))    # varsa nokta temizle
    df[col] = s
    return df

def series_or_default_debug(df: pd.DataFrame, cols, default=0.0, label="pred_expected") -> pd.Series:
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    # hiçbir kolon yoksa index uzunluğunda default seri döndür
    return pd.Series(default, index=df.index, dtype="float64")

def _discover_tabs() -> List[Dict[str, Any]]:
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
    specs: List[Dict[str, Any]] = []

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
                # Asgari alanlar
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


def _safe_render(render_fn, services: Dict[str, Any] | None = None):
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
        kwargs: Dict[str, Any] = {}
        if "state" in params:
            kwargs["state"] = None
        if "services" in params:
            kwargs["services"] = services
        return render_fn(**kwargs) if kwargs else render_fn()
    except TypeError:
        # Her ihtimale karşı basit çağrı
        return render_fn()

    # ========================= ana uygulama =========================
    
def main():
    tabs = _discover_tabs()
    if not tabs:
        st.error("Sekme bulunamadı. `tabs/<name>/__init__.py` içinde register() tanımlayın.")
        st.stop()

    # --- VERİ & METADATA (tek sefer) ---
    with st.spinner("Veri yükleniyor..."):
        df, src_tag = load_sf_crime_latest()
        meta = load_metadata_or_default()
        ok, missing = _validate_schema(df)

    # ==== DIAGNOSTIC & FIXES ====
    with st.expander("🧪 Veri Teşhisi / GEOID & kolon kontrolleri", expanded=False):
        rep = debug_geoid(df, "GEOID")
        st.write("GEOID raporu:", rep)

        if rep.get("geoid_present", False):
            if rep.get("geoid_float_count", 0) > 0 or rep.get("geoid_str_dotzero_count", 0) > 0:
                st.warning("GEOID’de float veya '.0' uçlu değerler tespit edildi → string’e çevrilecek.")
            df = coerce_geoid(df, "GEOID")
            st.caption("GEOID string’e dönüştürüldü.")

        # pred_expected/expected/risk_score → güvenli seri (float üstünde .fillna hatasını engeller)
        cand_cols = ["pred_expected", "expected", "risk_score"]
        have = [c for c in cand_cols if c in df.columns]
        st.write("Risk kolon adayları (mevcut):", have if have else "Yok")
        df["pred_expected"] = series_or_default_debug(
            df, cand_cols, default=0.0, label="pred_expected"
        ).fillna(0.0)

        # pydeck öncesi numerik/NaN temizlik
        for c in ["latitude", "longitude", "pred_expected"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if {"latitude","longitude"} <= set(df.columns):
            miss = df[["latitude","longitude"]].isna().sum().to_dict()
            st.write("Lat/Lon NaN sayıları:", miss)
    # ==== DIAGNOSTIC & FIXES SONU ====

    # services nesnesi: sekmelere dağıtılacak ortak kaynaklar
    services: Dict[str, Any] = {
        "data": df,               # ana DataFrame (düzeltilmiş)
        "source": src_tag,        # verinin geldiği katman (artifact/release/...)
        "meta": meta,             # üretim zamanı, kolonlar, vs.
        "schema_ok": ok,
        "schema_missing": missing,
    }

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

        # ---- veri durumu
        st.divider()
        st.caption("**Veri Durumu**")
        st.write(f"Kaynak: `{services['source']}`")
        try:
            st.write(f"Satır: {len(services['data']):,}")
        except Exception:
            st.write("Satır: -")
        gen_at = services["meta"].get("generated_at")
        if gen_at:
            st.write(f"Üretim: {gen_at}")
        if not services["schema_ok"]:
            st.warning(f"Beklenen şema eksik: {services['schema_missing']}")

    # Boş/eksik veri için ana ekranda da uyarı
    if not services["schema_ok"]:
        st.info(
            "Minimum şema `['GEOID','date','event_hour']` olmalı. "
            "Eksikler nedeniyle bazı sekmeler sınırlı çalışabilir."
        )

    # Seçili sekmeyi çalıştır
    current = next(t for t in tabs if t["key"] == active_key)
    _safe_render(current["render"], services=services)

if __name__ == "__main__":
    main()

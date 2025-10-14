from __future__ import annotations
import traceback
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st

# --- core.data bağımlılıklarını güvenli içe aktar ---
try:
    # Public API
    from core.data import list_members, load_parquet, get_latest_kpis  # type: ignore
except Exception as e:
    st.warning(f"core.data public API import edilemedi: {e}")
    def list_members() -> List[str]:
        return []
    def load_parquet(member: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return pd.DataFrame()
    def get_latest_kpis() -> Dict[str, Any]:
        return {}

# Opsiyonel (private) yardımcılar — yoksa graceful degrade
try:
    from core.data import _read_schema, _pick_best_member  # type: ignore
except Exception:
    _read_schema = None
    _pick_best_member = None

TAB_KEY = "diagnostics"

def _geoid_report(df: pd.DataFrame) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"geoid_present": False}
    if df is None or df.empty or "geoid" not in df.columns:
        return rep
    s = df["geoid"]
    rep["geoid_present"] = True
    rep["geoid_dtype"] = str(s.dtype)
    # float kaynaklı sorunlar
    as_str = s.astype("string")
    rep["geoid_float_count"] = int((s.astype(str).str.contains(r"\.0$", regex=True, na=False)).sum())
    rep["geoid_float_samples"] = list(as_str[s.astype(str).str.contains(r"\.0$", regex=True, na=False)].head(5))
    rep["geoid_str_dotzero_count"] = int((as_str.str.endswith(".0", na=False)).sum())
    rep["geoid_str_dotzero_samples"] = list(as_str[as_str.str.endswith(".0", na=False)].head(5))
    return rep

def _nan_counts(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cols:
        if c in df.columns:
            out[c] = int(df[c].isna().sum())
    return out

def render(state=None, services=None) -> None:
    st.title("🩺 Teşhis (Diagnostics)")

    st.subheader("Veri üyeleri")
    members = list_members() or []
    if not members:
        st.info("Paket içinde veri üyesi (parquet/karar) bulunamadı.")
        return
    st.write(members)

    # En iyi aday üye: risk_hourly varsa onu seç
    candidate = "risk_hourly.parquet" if "risk_hourly.parquet" in members else members[0]
    if _pick_best_member:
        try:
            candidate = _pick_best_member(members) or candidate  # type: ignore
        except Exception:
            pass

    st.subheader("Örnek veri")
    try:
        df = load_parquet(candidate, columns=["geoid", "risk_score", "risk_level", "date", "hour_range", "latitude", "longitude"])
    except Exception as e:
        st.error(f"Veri okunamadı: {e}")
        st.code(traceback.format_exc())
        return

    if df is None or df.empty:
        st.warning("Veri boş görünüyor.")
        return

    st.write(df.head(10))

    st.subheader("GEOID Raporu")
    rep = _geoid_report(df)
    st.json(rep)

    st.subheader("Lat/Lon NaN sayıları")
    st.json(_nan_counts(df, ["latitude", "longitude"]))

    st.subheader("Şema")
    if _read_schema:
        try:
            schema = _read_schema(candidate)  # type: ignore
            st.json({f.name: str(f.type) for f in schema})
        except Exception as e:
            st.info(f"Şema okunamadı: {e}")
    else:
        st.caption("(_read_schema mevcut değil)")

    st.subheader("Son KPI'lar")
    try:
        st.json(get_latest_kpis() or {})
    except Exception as e:
        st.info(f"KPI okunamadı: {e}")

def register() -> Dict[str, Any]:
    return {
        "key": TAB_KEY,
        "title": "Teşhis",
        "label": "🩺 Teşhis",
        "icon": "🩺",
        "order": 98,
        "render": render,
    }

from __future__ import annotations
import streamlit as st
import pandas as pd
from core.data import (list_members, load_parquet, _read_schema, _pick_best_member, get_latest_kpis)
from config.settings import ARTIFACT_ZIP

TAB_KEY = "diagnostics"

def _bool_emoji(v: bool) -> str:
    return "✅" if v else "❌"

def render(state=None, services=None):
    st.title("🔎 Veri Teşhis")
    st.caption(f"📦 Zip yolu: {ARTIFACT_ZIP}")

    members = list_members()
    if not members:
        st.error("Zip bulunamadı veya içinde .parquet üye yok.")
        return
    st.success(f"{len(members)} parquet bulundu.")
    with st.expander("Üyeleri göster (ilk 30)"):
        st.code("\n".join(members[:30]), language="text")

    member = _pick_best_member() or members[0]
    st.info(f"Seçilen üye: {member}")

    schema = _read_schema(member)
    with st.expander("Şema sütunları"):
        st.code(", ".join(schema), language="text")

    wanted = ["geoid","lat","lon","timestamp","risk_score","risk_level","pred_expected","date","hour_range"]
    df = load_parquet(member, columns=wanted)
    if df.empty:
        st.error("Seçilen üyeden veri okunamadı (boş DataFrame).")
        return

    st.subheader("Örnek veri (ilk 10)")
    st.dataframe(df.head(10), use_container_width=True)

    has_lat = "lat" in df.columns and pd.to_numeric(df["lat"], errors="coerce").notna().any()
    has_lon = "lon" in df.columns and pd.to_numeric(df["lon"], errors="coerce").notna().any()
    st.write(f"🧭 Lat/Lon mevcut mu? lat: {_bool_emoji(has_lat)} • lon: {_bool_emoji(has_lon)}")

    ts_ok = False
    if "timestamp" in df.columns:
        ts_s = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        ts_ms = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        ts_ok = ts_s.notna().any() or ts_ms.notna().any()
        st.write(f"⏱️ timestamp epoch mu? {_bool_emoji(ts_ok)}")
        with st.expander("timestamp örnekleri"):
            st.write(df["timestamp"].head().tolist())
    else:
        st.write("⏱️ timestamp: bulunamadı")

    if "risk_score" in df.columns:
        rs = pd.to_numeric(df["risk_score"], errors="coerce")
        nan_rate = float(rs.isna().mean())
        st.write(f"📊 risk_score min–max: {float(rs.min(skipna=True)):.3f} → {float(rs.max(skipna=True)):.3f} • NaN oranı: {nan_rate:.2%}")
        if not (rs.min(skipna=True) >= 0 and rs.max(skipna=True) <= 1):
            st.warning("risk_score 0–1 aralığında değil; normalize edilmeli.")

    if "risk_level" in df.columns:
        st.write("🏷️ risk_level örnekleri:", df["risk_level"].astype(str).dropna().unique()[:10])

    k = get_latest_kpis()
    st.subheader("KPI")
    st.info(f"rows={k['rows']} • last_update={k['last_update']} • avg_risk={{k['avg_risk']}} • high_rate={k['high_rate']} • member={k['member']}")

def register():
    return {
        "key": TAB_KEY,
        "title": "Veri Teşhis",
        "icon": "🔎",
        "label": "🔎 Veri Teşhis",   # eski keşif kodları için emniyet supabı
        "order": 99,
        "render": render,
    }

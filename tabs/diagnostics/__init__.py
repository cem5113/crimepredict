# tabs/diagnostics/__init__.py
import streamlit as st
import pandas as pd
from core.data import (
    list_members, load_parquet, _read_schema, _pick_best_member,
    COL_ALIASES, get_latest_kpis
)
from config.settings import ARTIFACT_ZIP

TAB_KEY = "diagnostics"

def render():
    st.title("🔎 Veri Teşhis")
    st.write("Zip yolu:", ARTIFACT_ZIP)

    members = list_members()
    if not members:
        st.error("Zip bulunamadı veya içinde .parquet yok.")
        return
    st.success(f"{len(members)} parquet bulundu.")
    st.code("\n".join(members[:30]), language="text")

    member = _pick_best_member() or members[0]
    st.caption(f"Seçilen üye: {member}")

    schema = _read_schema(member)
    st.code(", ".join(schema), language="text")

    wanted = ["geoid","lat","lon","timestamp","risk_score","risk_level","pred_expected","date","hour_range"]
    df = load_parquet(member, columns=wanted)
    st.dataframe(df.head(10))

    # timestamp kontrolü
    if "timestamp" in df.columns:
        ts_s = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        ts_ms = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        st.write("TS saniye OK?", ts_s.notna().any(), "• ms OK?", ts_ms.notna().any())

    # risk_score
    if "risk_score" in df.columns:
        rs = pd.to_numeric(df["risk_score"], errors="coerce")
        st.write("risk min–max:", float(rs.min(skipna=True)), "→", float(rs.max(skipna=True)))
        st.write("NaN oranı:", round(float(rs.isna().mean()), 3))

    k = get_latest_kpis()
    st.info(f"rows={k['rows']} • last_update={k['last_update']} • avg_risk={k['avg_risk']} • high_rate={k['high_rate']} • member={k['member']}")

def register():
    return {
        "key": TAB_KEY,
        "label": "🔎 Veri Teşhis",
        "order": 99,
        "render": render,
    }

import streamlit as st
import pandas as pd
from core.data import (
    list_members, load_parquet, _read_schema, _pick_best_member,
    COL_ALIASES, get_latest_kpis
)
from config.settings import ARTIFACT_ZIP

st.set_page_config(page_title="Veri Teşhis", page_icon="🔎", layout="wide")
st.title("🔎 Veri Teşhis Paneli")

if st.button("Kontrolleri Çalıştır"):
    st.write("**1️⃣ ARTIFACT_ZIP:**", ARTIFACT_ZIP)
    members = list_members()
    if not members:
        st.error("❌ Zip bulunamadı veya içinde .parquet yok.")
        st.stop()
    st.success(f"Zip OK • {len(members)} üye bulundu.")
    st.code("\n".join(members[:20]), language="text")

    member = _pick_best_member() or members[0]
    st.info(f"Seçilen üye: {member}")

    schema_cols = _read_schema(member)
    st.write("**Şema kolonları:**", len(schema_cols))
    st.code(", ".join(schema_cols), language="text")

    wanted = ["geoid","lat","lon","timestamp","risk_score","risk_level","pred_expected"]
    df = load_parquet(member, columns=wanted)
    st.dataframe(df.head(10))

    # lat/lon
    has_lat = "lat" in df.columns and df["lat"].notna().any()
    has_lon = "lon" in df.columns and df["lon"].notna().any()
    if has_lat and has_lon:
        st.success("✅ lat/lon OK")
    else:
        st.error("❌ lat/lon eksik veya tamamen NaN.")

    # timestamp kontrol
    if "timestamp" in df.columns:
        ts_s = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        ts_ms = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        if ts_s.notna().any() or ts_ms.notna().any():
            st.success("✅ timestamp epoch formatında")
        else:
            st.warning("⚠️ timestamp string formatında olabilir.")
    else:
        st.error("❌ timestamp kolonu yok.")

    # risk_score kontrol
    if "risk_score" in df.columns:
        rs = pd.to_numeric(df["risk_score"], errors="coerce")
        st.write(f"Min={rs.min(skipna=True):.3f}, Max={rs.max(skipna=True):.3f}, NaN oranı={rs.isna().mean():.2%}")
        if rs.min(skipna=True) >= 0 and rs.max(skipna=True) <= 1:
            st.success("✅ risk_score 0–1 aralığında")
        else:
            st.warning("⚠️ risk_score 0–1 aralığında değil.")
    else:
        st.error("❌ risk_score kolonu yok.")

    # KPI özeti
    kpis = get_latest_kpis()
    st.info(f"🧾 rows={kpis['rows']} • last_update={kpis['last_update']} • avg_risk={kpis['avg_risk']} • high_rate={kpis['high_rate']} • member={kpis['member']}")

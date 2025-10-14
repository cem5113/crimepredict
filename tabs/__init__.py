# tabs/home/__init__.py  (Şehir Anlık Görünüm kısmında)
import streamlit as st
import pandas as pd
import pydeck as pdk
from core.data import load_parquet
from core.mapkit import load_cells_geojson, join_risk_to_cells

st.markdown("### Şehir Anlık Görünüm")

# 1) risk verisini oku (GEOID + risk + zaman)
risk = load_parquet(
    "risk_hourly.parquet",
    columns=["geoid","risk_score","risk_level","date","hour_range"]
)
if risk.empty:
    st.info("Harita için veri bulunamadı.")
else:
    # 2) zaman filtreleri
    dates = sorted(risk["date"].dropna().astype(str).unique())
    sel_date = st.selectbox("Tarih", options=dates, index=len(dates)-1 if dates else 0)
    hours = risk.loc[risk["date"].astype(str).eq(sel_date), "hour_range"].dropna().astype(str).unique()
    sel_hour = st.selectbox("Saat dilimi", options=sorted(hours))

    view_df = risk[(risk["date"].astype(str)==sel_date) & (risk["hour_range"].astype(str)==sel_hour)]
    if view_df.empty:
        st.info("Seçilen tarih-saat için veri yok.")
    else:
        # 3) geojson'u yükle + riski join et
        gj = load_cells_geojson("data/sf_cells.geojson")
        gj = join_risk_to_cells(gj, view_df, geoid_col="geoid", risk_col="risk_score")

        # 4) choropleth layer
        layer = pdk.Layer(
            "GeoJsonLayer",
            gj,
            pickable=True,
            opacity=0.6,
            get_line_color=[80,80,80],
            lineWidthMinPixels=0.5,
            autoHighlight=True,
            # risk_score ∈ [0,1] varsayımıyla kırmızı→mavi geçiş
            get_fill_color="""
                d => {
                  const r = d.properties.risk_score;
                  if (!Number.isFinite(r)) return [200,200,200,60];
                  return [Math.round(255*r), 40, Math.round(200*(1-r)), 160];
                }
            """,
        )
        view_state = pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=11)
        tooltip = {
            "html": "<b>GEOID:</b> {geoid}<br/><b>Risk:</b> {risk_score}",
            "style": {"backgroundColor": "rgba(30,30,30,0.8)", "color": "white"}
        }
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))



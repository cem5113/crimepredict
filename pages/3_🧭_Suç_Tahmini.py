# -*- coding: utf-8 -*-
"""
V1 — SUÇ TAHMİN (yalnız kolluk için yararlı çıktılar)
-----------------------------------------------------
Bu sayfa, tahmin skorlarını harita üzerinde gösterir; yalnızca kolluğu ilgilendiren
(Yüksek/Riskli veya Anomali) noktaları öne çıkarır ve planlama sekmesi için JSON/CSV
export üretir.

Nasıl bağlanır?
- model_predict_proba(): senin stacking modelinden (XGB/LGBM/Cat) proba döndürecek
- get_daily_quantiles(): günün Q25/Q50/Q75 değerlerini döndürür (kalibrasyon şart)
- data kaynağı: son 5 yıl verinden güncel saatlik satırlar (geoid, lat, lon, hour,...)

Not: Kod, veri/Model entegrasyonunu kolaylaştırmak için modüler yazıldı.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from datetime import datetime, timedelta

# =============================
# ------- CONFIG & THEME ------
# =============================
st.set_page_config(
    page_title="Suç Tahmin (Kolluk)",
    page_icon="🚨",
    layout="wide",
)

PRIMARY_COLOR = "#d7263d"  # pin vurgu
RISK_COLORS = {
    "Düşük": [180, 220, 180, 90],
    "Orta": [255, 220, 130, 110],
    "Riskli": [252, 140, 80, 150],
    "Yüksek": [215, 38, 61, 190],
}

# =============================
# --------- HELPERS -----------
# =============================
@dataclass
class Hotspot:
    geoid: str
    lat: float
    lon: float
    p_crime: float
    risk_level: str
    time_window: Tuple[str, str]
    category_pred: str | None = None
    category_conf: float | None = None
    anomaly: bool = False
    reason_triple: List[str] = None


def load_operational_frame(date_from: datetime, date_to: datetime,
                           categories: List[str] | None = None) -> pd.DataFrame:
    """Uygulama, buradan veriyi çeker. Yerine kendi kaynağını bağla.
    Beklenen kolonlar (en az): geoid, latitude, longitude, datetime, hour,
    category (ops.), neighbor_crime_24h, 911_request_count_hour_range, ...
    """
    # ÖRNEK: Yer tutucu (kendi veri yoluna bağla)
    # df = pd.read_parquet("data/operational_frame.parquet")
    # df = df[(df["datetime"] >= date_from) & (df["datetime"] < date_to)]
    # if categories:
    #     df = df[df["category"].isin(categories)]
    # return df.copy()

    # Demo amaçlı sentetik veri (sil/degistir)
    rng = pd.date_range(date_from, date_to, freq="H", inclusive="left")
    np.random.seed(7)
    rows = []
    geoids = [f"6075{1000+i}" for i in range(60)]
    base_lat, base_lon = 37.77, -122.42
    for ts in rng:
        for g in geoids:
            lat = base_lat + np.random.normal(0, 0.03)
            lon = base_lon + np.random.normal(0, 0.03)
            nc24 = max(0, int(np.random.normal(2.5, 1.3)))
            r911 = max(0, int(np.random.normal(8, 3)))
            rows.append({
                "geoid": g,
                "latitude": lat,
                "longitude": lon,
                "datetime": ts,
                "hour": ts.hour,
                "neighbor_crime_24h": nc24,
                "911_request_count_hour_range": r911,
                "category": np.random.choice(["Theft", "Assault", "Other"]),
                "is_night": int(ts.hour >= 20 or ts.hour < 6),
                "distance_to_police_range": np.random.choice(["<150m","150-300m",">300m"], p=[.3,.5,.2])
            })
    df = pd.DataFrame(rows)
    if categories:
        df = df[df["category"].isin(categories)]
    return df


def model_predict_proba(batch_df: pd.DataFrame) -> np.ndarray:
    """Stacking modelinin predict_proba(:,1) çıktısı.
    Buraya senin gerçek model yükleme/çağırma kodun gelecek.
    """
    # TODO: gerçek modele bağla. Şimdilik yalın bir skorlayıcı:
    # Normalizasyon ve basit lineer kombo (demo)
    s1 = (batch_df["neighbor_crime_24h"].astype(float) + 1) / (batch_df["neighbor_crime_24h"].max() + 1)
    s2 = (batch_df["911_request_count_hour_range"].astype(float) + 1) / (batch_df["911_request_count_hour_range"].max() + 1)
    s3 = 0.15 + 0.25 * batch_df["is_night"].astype(float)
    s = 0.5*s1 + 0.3*s2 + 0.2*s3
    return np.clip(s.values, 0.01, 0.98)


@st.cache_data(ttl=600)
def get_daily_quantiles(day: datetime, scores: pd.Series | None = None) -> Tuple[float, float, float]:
    """Günün Q25/Q50/Q75 değerleri. Eğer scores verilmişse ondan, yoksa
    dosyadan/DB'den oku (burada sentetik hesap)."""
    if scores is not None and len(scores) > 20:
        q25, q50, q75 = np.quantile(scores, [0.25, 0.5, 0.75])
        return float(q25), float(q50), float(q75)
    # Yerine: pd.read_parquet("daily_quantiles.parquet"); filtrele day
    return 0.25, 0.5, 0.75


def label_risk(p: float, q25: float, q50: float, q75: float) -> str:
    if p <= q25:
        return "Düşük"
    elif p <= q50:
        return "Orta"
    elif p <= q75:
        return "Riskli"
    return "Yüksek"


def detect_anomaly(df: pd.DataFrame) -> pd.Series:
    """Basit z-score ile 911 anomali tespiti (satır bazlı). V1 için yeterli.
    Gerçek uygulamada: saat penceresine göre z-score / CUSUM / STL trend çıkarımı.
    """
    x = df["911_request_count_hour_range"].astype(float)
    mu, sigma = x.mean(), x.std(ddof=1) if len(x) > 1 else (x.mean(), 1.0)
    z = (x - mu) / (sigma if sigma > 1e-6 else 1.0)
    return (z >= 2.0)


def reason_triple(row: pd.Series) -> List[str]:
    r = []
    if int(row.get("is_night", 0)) == 1:
        r.append("gece")
    if row.get("neighbor_crime_24h", 0) >= 3:
        r.append("komşu-24h yüksek")
    if str(row.get("distance_to_police_range", "")).endswith(">300m"):
        r.append(">300m polis")
    return r[:3] if r else ["genel risk"]


# =============================
# --------- SIDEBAR -----------
# =============================
st.sidebar.header("Filtreler")
now_utc = datetime.utcnow()
start_dt = st.sidebar.datetime_input("Başlangıç", value=(now_utc.replace(minute=0, second=0, microsecond=0)))
end_dt = st.sidebar.datetime_input("Bitiş", value=(now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=6)))

selected_categories = st.sidebar.multiselect("Suç kategorisi (opsiyonel)", ["Theft","Assault","Other"], default=[])

col_a, col_b = st.sidebar.columns(2)
with col_a:
    top_n = st.number_input("Top-N sıcak nokta", min_value=3, max_value=50, value=10, step=1)
with col_b:
    useful_only = st.toggle("Sadece kolluğa yararlı olanı göster", value=True)

show_neighbors = st.sidebar.toggle("Komşuları göster (harita)", value=False)

# =============================
# -------- MAIN STAGE ---------
# =============================
st.title("🚨 Suç Tahmin (Kolluk Odaklı)")

# 1) Veri yükle
with st.spinner("Veri hazırlanıyor..."):
    df = load_operational_frame(start_dt, end_dt, categories=selected_categories or None)
    if df.empty:
        st.warning("Seçilen aralıkta veri bulunamadı.")
        st.stop()

# 2) Tahmin
with st.spinner("Tahminler hesaplanıyor..."):
    scores = model_predict_proba(df)
    df = df.assign(p_crime=scores)
    q25, q50, q75 = get_daily_quantiles(start_dt, scores=pd.Series(scores))
    df["risk_level"] = [label_risk(p, q25, q50, q75) for p in df["p_crime"]]
    df["anomaly"] = detect_anomaly(df)
    df["time_start"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%dT%H:00:00Z")
    df["time_end"] = (pd.to_datetime(df["datetime"]) + pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:00:00Z")
    df["reason_triple"] = df.apply(reason_triple, axis=1)

# 3) Sadece yararlı olanı göster/gizle
if useful_only:
    mask_useful = (df["risk_level"].isin(["Riskli","Yüksek"])) | (df["anomaly"] == True)
    df_view = df[mask_useful].copy()
else:
    df_view = df.copy()

if df_view.empty:
    st.info("Eşiklere göre gösterilecek kritik nokta yok. (Filtreleri veya eşiği gevşetin.)")
    st.stop()

# 4) GEOID bazlı en yüksek saat dilimini seç (Top-1 per GEOID), sonra Top-N
agg = (
    df_view.sort_values(["geoid","p_crime"], ascending=[True, False])
          .groupby("geoid", as_index=False)
          .first()
)
agg = agg.sort_values("p_crime", ascending=False).head(int(top_n)).reset_index(drop=True)

# 5) Harita — pin + ısı
mid_lat = float(agg["latitude"].mean())
mid_lon = float(agg["longitude"].mean())

# Scatter pin katmanı
pin_layer = pdk.Layer(
    "ScatterplotLayer",
    data=agg,
    get_position="[longitude, latitude]",
    get_radius=120,
    get_fill_color=[
        "risk_level == 'Yüksek' ? 215 : risk_level == 'Riskli' ? 252 : risk_level == 'Orta' ? 255 : 180",
        "risk_level == 'Yüksek' ? 38 : risk_level == 'Riskli' ? 140 : risk_level == 'Orta' ? 220 : 220",
        "risk_level == 'Yüksek' ? 61 : risk_level == 'Riskli' ? 80 : risk_level == 'Orta' ? 130 : 180",
        190
    ],
    pickable=True,
)

# (Opsiyonel) ısı katmanı — tüm görünür kayıtlar
heat_layer = pdk.Layer(
    "HeatmapLayer",
    data=df_view,
    get_position="[longitude, latitude]",
    get_weight="p_crime",
    radius_pixels=40,
)

initial_view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11, pitch=0)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=initial_view_state,
    layers=[heat_layer, pin_layer],
    tooltip={
        "html": "<b>GEOID</b>: {geoid}<br/>"
                "<b>Risk</b>: {p_crime} <br/>"
                "<b>Seviye</b>: {risk_level}<br/>"
                "<b>Saat</b>: {time_start} – {time_end}<br/>"
                "<b>Gerekçe</b>: {reason_triple}",
        "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}
    }
))

# 6) Kolluk Kartları — tablo görünümü (yalın)
st.markdown("### 🎯 Kritik Noktalar (Top-N)")

def format_pct(x):
    return f"{x*100:.1f}%"

view_cols = [
    "geoid", "time_start", "time_end", "p_crime", "risk_level", "category",
    "anomaly", "reason_triple"
]

show_df = agg[view_cols].copy()
show_df["p_crime"] = show_df["p_crime"].apply(format_pct)
st.dataframe(show_df, use_container_width=True, hide_index=True)

# 7) Export — Planlama sekmesi için JSON/CSV
st.markdown("#### ⤵️ Dışa Aktar (Planlama için)")
export_records: List[Dict] = []
for _, row in agg.iterrows():
    export_records.append({
        "geoid": str(row["geoid"]),
        "time_window": f"{row['time_start']}/{row['time_end']}",
        "p_crime": float(row["p_crime"]),
        "risk_level": str(row["risk_level"]),
        "category_pred": str(row.get("category", "")),  # gerçek sınıflandırıcıya bağlanınca güncelle
        "category_conf": None,  # ileride eklenecek
        "anomaly": bool(row.get("anomaly", False)),
        "reason_triple": list(row.get("reason_triple", [])),
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
    })

json_bytes = json.dumps(export_records, ensure_ascii=False, indent=2).encode("utf-8")
st.download_button("JSON indir", data=json_bytes, file_name="hotspots_export.json", mime="application/json")

csv_bytes = pd.DataFrame(export_records).to_csv(index=False).encode("utf-8")
st.download_button("CSV indir", data=csv_bytes, file_name="hotspots_export.csv", mime="text/csv")

# 8) Not — kullanıcıya yalnız faydalı içerik kuralı
st.caption(
    "*Bu ekranda yalnızca kolluk için anlamlı görülen (Riskli/Yüksek veya Anomali) noktalar varsayılan olarak gösterilir.*"
)

# --- app.py (baş kısım) ---------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime
import os, sys

# 1) components/ yolu mevcutsa ekle (farklı dizin düzenlerinde yardımcı olur)
_here = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.join(_here, "components")
if os.path.isdir(components_dir) and components_dir not in sys.path:
    sys.path.append(components_dir)

# 2) config'i koşullu içe aktar: yoksa varsayılanları kullan
try:
    # DİKKAT: Satırda fazladan karakter yok (özellikle 'ü' vs.)
    from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH  # type: ignore
except Exception:
    APP_NAME = "Crime Prediction Dashboard"
    APP_ROLE = "Kolluk Kuvvetleri için Suç Tahmini"
    DATA_REPO = "cem5113/crime_prediction_data"
    DATA_BRANCH = "main"
    st.sidebar.info("⚙️ components/config.py bulunamadı. Varsayılan yapılandırma kullanılıyor.")
# ---------------------------------------------------------

# ---------------------------------------------------------

# --- 1) Veri yükleme ---
# Kaynaklar:
# - Suç verisi: cem5113/crime_prediction_data/artifact/fr-crime-pipeline-output.zip → fr_crime_09.csv
# - Stacking metrikleri: cem5113/crime_prediction_data/artifact/sf-crime-pipeline-output.zip → metrics_stacking_ohe.csv

@st.cache_data(show_spinner=False)
def load_data():
    """Veriyi sağlam şekilde yükler. Önce raw CSV, olmazsa ZIP (repo/raw veya releases).
    Dönenler: (df, metrics)
    """
    import io, zipfile, requests

    RAW_FR_CSV = (
        "https://github.com/cem5113/crime_prediction_data/raw/main/"
        "artifact/fr-crime-pipeline-output/fr_crime_09.csv"
    )
    RAW_METRICS_CSV = (
        "https://github.com/cem5113/crime_prediction_data/raw/main/"
        "artifact/sf-crime-pipeline-output/metrics_stacking_ohe.csv"
    )

    ZIP_FR_RAW = "https://github.com/cem5113/crime_prediction_data/raw/main/artifact/fr-crime-pipeline-output.zip"
    ZIP_SF_RAW = "https://github.com/cem5113/crime_prediction_data/raw/main/artifact/sf-crime-pipeline-output.zip"

    ZIP_FR_REL = "https://github.com/cem5113/crime_prediction_data/releases/latest/download/fr-crime-pipeline-output.zip"
    ZIP_SF_REL = "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf-crime-pipeline-output.zip"

    def _read_csv_or_from_zip(primary_csv_url: str, fallback_zips: list[str], wanted_name_endswith: str) -> pd.DataFrame:
        # 1) doğrudan CSV dene
        try:
            return pd.read_csv(primary_csv_url)
        except Exception:
            pass
        # 2) ZIP'lerden sırayla dene
        for url in fallback_zips:
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    member = None
                    for name in zf.namelist():
                        if name.endswith(wanted_name_endswith):
                            member = name
                            break
                    if member is None:
                        continue
                    with zf.open(member) as f:
                        return pd.read_csv(f)
            except Exception:
                continue
        raise FileNotFoundError(f"{wanted_name_endswith} indirilemedi veya ZIP içinde bulunamadı.")

    df = _read_csv_or_from_zip(RAW_FR_CSV, [ZIP_FR_RAW, ZIP_FR_REL], "fr_crime_09.csv")
    metrics = _read_csv_or_from_zip(RAW_METRICS_CSV, [ZIP_SF_RAW, ZIP_SF_REL], "metrics_stacking_ohe.csv")

    return df, metrics


df, metrics = load_data()

# --- 2) Yardımcılar ---

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun adları farklı varyantlarla gelirse, standart alias kolonlar oluştur.
    Beklenen: GEOID, Category, hour, latitude, longitude, risk_score, date/datetime
    """
    cols = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}

    def alias(src_opts: list[str], target: str):
        for s in src_opts:
            if s in df.columns:
                if target not in df.columns:
                    df[target] = df[s]
                return
            if s.lower() in lower:
                orig = lower[s.lower()]
                if target not in df.columns:
                    df[target] = df[orig]
                return
        # yoksa dokunma

    alias(["GEOID", "geoid", "Geoid", "id"], "GEOID")
    alias(["Category", "category", "crime_category"], "Category")
    alias(["Subcategory", "subcategory", "crime_subcategory"], "Subcategory")
    alias(["hour", "event_hour", "event_hour_x", "event_hour_y"], "hour")
    alias(["latitude", "lat", "Latitude"], "latitude")
    alias(["longitude", "lon", "Longitude"], "longitude")
    alias(["risk_score", "p_crime", "prob", "score"], "risk_score")
    alias(["date", "Date"], "date")
    alias(["datetime", "ts", "timestamp", "Datetime"], "datetime")

    # Eğer hour yok ama datetime varsa çıkar
    if "hour" not in df.columns and "datetime" in df.columns:
        try:
            df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        except Exception:
            pass

    # risk_score yoksa basit yedek skor üret (kalibre değil)
    if "risk_score" not in df.columns:
        # bulunabilirse basit normalize kombine skor
        parts = []
        for c in ["neighbor_crime_24h", "911_request_count_hour_range", "crime_count", "daily_cnt"]:
            if c in df.columns:
                x = (pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float) + 1)
                parts.append(x / (x.max() if x.max() > 0 else 1))
        if parts:
            df["risk_score"] = np.clip(0.4 * parts[0] + sum(parts[1:]) * 0.2, 0.01, 0.99)
        else:
            df["risk_score"] = 0.5  # düz sabit
    return df


df = ensure_columns(df)

# --- 3) Başlık ---
st.title("🔎 Suç Tahmin Modülü (Yalnız Kolluğa Yararlı)")
st.markdown("Zaman, mekân ve kategori bazlı risk tahminleri — yalnız kolluk için anlamlı sonuçlar gösterilir.")

# --- 4) Filtreler ---
col1, col2, col3 = st.columns(3)
with col1:
    default_date = None
    if "date" in df.columns:
        try:
            default_date = pd.to_datetime(df["date"], errors="coerce").dropna().dt.date.max()
        except Exception:
            default_date = None
    elif "datetime" in df.columns:
        try:
            default_date = pd.to_datetime(df["datetime"], errors="coerce").dropna().dt.date.max()
        except Exception:
            default_date = None
    date_selected = st.date_input("Tarih seçin", value=default_date)

with col2:
    min_h = int(df["hour"].min()) if "hour" in df.columns else 0
    max_h = int(df["hour"].max()) if "hour" in df.columns else 23
    hour_selected = st.slider("Saat aralığı seçin", 0, 23, (max(min_h, 0), min(max_h, 23)))

with col3:
    if "Category" in df.columns:
        cats = sorted([c for c in df["Category"].dropna().unique().tolist() if str(c).strip() != ""])
    else:
        cats = []
    category_selected = st.selectbox("Suç kategorisi", ["(Hepsi)"] + cats)

show_only_relevant = st.toggle("🔒 Yalnız kolluğa yararlı sonuçları göster", value=True)

# --- 5) Tarih filtresi ---
mask = pd.Series(True, index=df.index)
if date_selected is not None:
    if "date" in df.columns:
        try:
            mask &= (pd.to_datetime(df["date"], errors="coerce").dt.date == date_selected)
        except Exception:
            pass
    elif "datetime" in df.columns:
        try:
            mask &= (pd.to_datetime(df["datetime"], errors="coerce").dt.date == date_selected)
        except Exception:
            pass

view = df[mask].copy() if mask.any() else df.copy()

# --- 6) Risk eşiği ve diğer filtreler ---
if "risk_score" in view.columns:
    q75 = view["risk_score"].quantile(0.75)
else:
    q75 = 0.5

if show_only_relevant:
    view = view[view["risk_score"] >= q75]

if "hour" in view.columns:
    view = view[(view["hour"] >= hour_selected[0]) & (view["hour"] <= hour_selected[1])]

if category_selected and category_selected != "(Hepsi)" and "Category" in view.columns:
    view = view[view["Category"] == category_selected]

if view.empty:
    st.info("Eşiklere veya filtrelere göre gösterilecek kritik nokta yok.")
    st.stop()

# --- 7) Harita ---
center = [float(view["latitude"].mean()), float(view["longitude"].mean())] if {"latitude","longitude"}.issubset(view.columns) else [37.77, -122.42]
m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

if {"latitude","longitude"}.issubset(view.columns):
    q75_local = view["risk_score"].quantile(0.75)
    for _, row in view.iterrows():
        try:
            popup_text = (
                f"GEOID: {row.get('GEOID','?')}<br>"
                f"Risk: {float(row['risk_score']):.2f}<br>"
                f"Saat: {int(row.get('hour',-1)) if pd.notna(row.get('hour',np.nan)) else '-'}<br>"
                f"Kategori: {row.get('Category','-')}"
            )
            color = 'red' if row['risk_score'] >= q75_local else 'orange'
            folium.CircleMarker(
                location=[float(row['latitude']), float(row['longitude'])],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(m)
        except Exception:
            continue

st_folium(m, width=800, height=560)

# --- 8) Tablo ---
st.subheader("📊 Yüksek Riskli Noktalar")
cols_to_show = [c for c in ["GEOID","hour","Category","risk_score","latitude","longitude"] if c in view.columns]
st.dataframe(view[cols_to_show].sort_values(by="risk_score", ascending=False).head(50), use_container_width=True)

# --- 9) Stacking metrikleri ---
st.subheader("📈 Model Performans Özeti (Stacking)")
st.dataframe(metrics, use_container_width=True)

# --- 10) Export ---
st.download_button(
    "⬇️ Hotspot verisini indir (CSV)",
    view.to_csv(index=False).encode("utf-8"),
    "high_risk_hotspots.csv",
    "text/csv"
)

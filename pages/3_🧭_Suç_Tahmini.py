# 3_🧭_Suç_Tahmini.py
import io, zipfile
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime

# --- Güvenli import (hata ekranı ile) ---
for name, stmt in [
    ("components.config", "from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH, GH_TOKEN"),
    ("components.gh_data", "from components.gh_data import download_actions_artifact_zip"),
    ("streamlit", "import streamlit as st"),
    ("pandas", "import pandas as pd"),
    ("numpy", "import numpy as np"),
    ("folium", "import folium"),
    ("streamlit_folium", "from streamlit_folium import st_folium"),
]:
    try:
        exec(stmt, {})
    except Exception as e:
        import streamlit as st
        st.error(f"Import hatası: {name} → {type(e).__name__}: {e}")
        st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 1) Artifact'ten PARQUET/PAQUET okuma
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def _read_parquet_from_zip(zip_bytes: bytes, candidate_names: list[str]) -> pd.DataFrame:
    """
    GitHub Actions artifact zip baytlarından, verilen son-ad eşleşmelerine göre
    ilk uygun PARQUET/PAQUET dosyasını bulup DataFrame döndürür.
    Eşleşme: önce tam uçtan (case-insensitive), sonra kısmi içerir eşleşmesi.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        # 1) Tam uçtan eşleşme (case-insensitive)
        low_names = [n.lower() for n in names]
        for cand in candidate_names:
            c = cand.lower()
            for i, ln in enumerate(low_names):
                if ln.endswith(c):
                    with zf.open(names[i]) as f:
                        return pd.read_parquet(f)

        # 2) Kısmi içerir eşleşmesi
        bases = [cand.split("/")[-1].lower() for cand in candidate_names]
        for i, ln in enumerate(low_names):
            if any(b in ln for b in bases):
                with zf.open(names[i]) as f:
                    return pd.read_parquet(f)

        # Bulunamadı
        raise FileNotFoundError(
            "ZIP içinde beklenen PARQUET/PAQUET bulunamadı.\n"
            f"Aranan: {candidate_names}\n"
            f"Örnek içerik: {names[:30]}"
        )

@st.cache_data(show_spinner=False)
def load_data():
    """
    - fr-crime-pipeline-output → fr_crime_09.parquet (veya .paquet)
    - sf-crime-parquet        → metrics_stacking_ohe.parquet (veya .paquet)
    """
    # config
    try:
        from components.config import DATA_REPO, GH_TOKEN
    except Exception:
        st.error("components.config içinden DATA_REPO / GH_TOKEN okunamadı.")
        st.stop()

    try:
        owner, repo = DATA_REPO.split("/", 1)
    except ValueError:
        st.error(f"DATA_REPO beklenen formatta değil: {DATA_REPO} (örn. cem5113/crime_prediction_data)")
        st.stop()

    if not GH_TOKEN:
        st.error("GH_TOKEN gerekli. components.config içinde GH_TOKEN tanımlayın.")
        st.stop()

    # 1) FR verisi — fr-crime-pipeline-output
    fr_zip = download_actions_artifact_zip(
        owner=owner,
        repo=repo,
        artifact_name="fr-crime-pipeline-output",
        token=GH_TOKEN,
    )
    df_fr = _read_parquet_from_zip(
        fr_zip,
        candidate_names=[
            "fr_crime_09.parquet",
            "fr-crime_09.parquet",
            "fr_crime_09.paquet",
            "fr-crime_09.paquet",
        ],
    )

    # 2) Stacking metrikleri — sf-crime-parquet
    sf_zip = download_actions_artifact_zip(
        owner=owner,
        repo=repo,
        artifact_name="sf-crime-parquet",
        token=GH_TOKEN,
    )
    metrics = _read_parquet_from_zip(
        sf_zip,
        candidate_names=[
            "metrics_stacking_ohe.parquet",
            "metrics_stacking.parquet",
            "metrics_stacking_ohe.paquet",
            "metrics_stacking.paquet",
        ],
    )

    return df_fr, metrics

df, metrics = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# 2) Yardımcılar
# ──────────────────────────────────────────────────────────────────────────────
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sütun adları farklı varyantlarla gelirse, standart alias kolonlar oluştur.
    Beklenen: GEOID, Category, hour, latitude, longitude, risk_score, date/datetime
    """
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

    alias(["GEOID", "geoid", "Geoid", "id", "cell_id"], "GEOID")
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

# ──────────────────────────────────────────────────────────────────────────────
# 3) Başlık
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔎 Suç Tahmin Modülü (Yalnız Kolluğa Yararlı)")
st.markdown("Zaman, mekân ve kategori bazlı risk tahminleri — yalnız kolluk için anlamlı sonuçlar gösterilir.")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Filtreler
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 5) Tarih filtresi
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 6) Risk eşiği ve diğer filtreler
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 7) Harita
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 8) Tablo
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Yüksek Riskli Noktalar")
cols_to_show = [c for c in ["GEOID","hour","Category","risk_score","latitude","longitude"] if c in view.columns]
st.dataframe(view[cols_to_show].sort_values(by="risk_score", ascending=False).head(50), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 9) Stacking metrikleri
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Model Performans Özeti (Stacking)")
st.dataframe(metrics, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 10) Export
# ──────────────────────────────────────────────────────────────────────────────
st.download_button(
    "⬇️ Hotspot verisini indir (CSV)",
    view.to_csv(index=False).encode("utf-8"),
    "high_risk_hotspots.csv",
    "text/csv"
)

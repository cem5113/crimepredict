# pages/3_🧭_Suç_Tahmini.py
import io, os, zipfile
import requests
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# ──────────────────────────────────────────────────────────────────────────────
# 0) Config & Token çözümleme (harici modüle bağımlı değil)
# ──────────────────────────────────────────────────────────────────────────────
def resolve_repo_and_token():
    owner_repo = None
    token = None

    # 1) components.config
    try:
        from components.config import DATA_REPO, GH_TOKEN
        owner_repo = DATA_REPO
        token = GH_TOKEN
    except Exception:
        pass

    # 2) secrets
    if not token:
        for k in ("GH_TOKEN", "github_token", "GITHUB_TOKEN"):
            try:
                if k in st.secrets and st.secrets[k]:
                    token = str(st.secrets[k])
                    break
            except Exception:
                pass

    # 3) env
    if not token:
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("github_token")

    # 4) owner/repo yoksa fallback
    if not owner_repo:
        # İstediğin repo'yu buraya yazabilirsin:
        owner_repo = "cem5113/crime_prediction_data"

    # doğrula
    if "/" not in owner_repo:
        st.error(f"DATA_REPO beklenen formatta değil: {owner_repo} (örn. cem5113/crime_prediction_data)")
        st.stop()
    return owner_repo, token

def gh_headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

# ──────────────────────────────────────────────────────────────────────────────
# 1) Actions artifact'ten ZIP indirme (yerel implementasyon)
# ──────────────────────────────────────────────────────────────────────────────
def download_actions_artifact_zip(owner: str, repo: str, artifact_name: str, token: str | None) -> bytes:
    """
    En güncel, süresi dolmamış artifact'i bulur ve ZIP baytlarını döndürür.
    """
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=gh_headers(token), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")
    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    url = cand[0].get("archive_download_url")
    if not url:
        raise RuntimeError("archive_download_url bulunamadı")
    r2 = requests.get(url, headers=gh_headers(token), timeout=60)
    r2.raise_for_status()
    return r2.content

@st.cache_data(show_spinner=True)
def _read_parquet_from_zip(zip_bytes: bytes, candidate_names: list[str]) -> pd.DataFrame:
    """
    ZIP içinden .parquet/.paquet dosyayı bulup okur.
    Önce tam uçtan (case-insensitive), sonra kısmi içerir eşleşmesi.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        low_names = [n.lower() for n in names]

        # 1) Tam uçtan eşleşme
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
    owner_repo, token = resolve_repo_and_token()
    owner, repo = owner_repo.split("/", 1)

    if not token:
        st.error("GitHub token bulunamadı. `components.config.GH_TOKEN`, `st.secrets`, ya da `GITHUB_TOKEN`/`GH_TOKEN` ortam değişkenlerinden biri gerekli.")
        st.stop()

    # FR verisi
    fr_zip = download_actions_artifact_zip(owner, repo, "fr-crime-pipeline-output", token)
    df_fr = _read_parquet_from_zip(
        fr_zip,
        candidate_names=[
            "fr_crime_09.parquet",
            "fr-crime_09.parquet",
            "fr_crime_09.paquet",
            "fr-crime_09.paquet",
        ],
    )

    # Stacking metrikleri (SF)
    sf_zip = download_actions_artifact_zip(owner, repo, "sf-crime-parquet", token)
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
    lower = {c.lower(): c for c in df.columns}

    def alias(src_opts: list[str], target: str):
        for s in src_opts:
            if s in df.columns:
                if target not in df.columns:
                    df[target] = df[s]
                return
            s_low = s.lower()
            if s_low in lower:
                orig = lower[s_low]
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

    if "hour" not in df.columns and "datetime" in df.columns:
        try:
            df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        except Exception:
            pass

    if "risk_score" not in df.columns:
        parts = []
        for c in ["neighbor_crime_24h", "911_request_count_hour_range", "crime_count", "daily_cnt"]:
            if c in df.columns:
                x = (pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float) + 1)
                parts.append(x / (x.max() if x.max() > 0 else 1))
        if parts:
            df["risk_score"] = np.clip(0.4 * parts[0] + sum(parts[1:]) * 0.2, 0.01, 0.99)
        else:
            df["risk_score"] = 0.5
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

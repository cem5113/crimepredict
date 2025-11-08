# 3_🧭_Suç_Tahmini 
# Tam kapsamlı: saatlik (≤7 gün) ve günlük (≤365 gün) risk görünümleri
# Not: Değişken/işlev adlarında kısaltma yok; açıklayıcı yorumlar eklendi.

import os
import io
import posixpath
import zipfile
from io import BytesIO
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------
# ️⚙️ GitHub repo ve artifact bilgisi
# ------------------------------------------------------------
REPOSITORY_OWNER = "cem5113"
REPOSITORY_NAME  = "crime_prediction_data"
ARTIFACT_NAME_SHOULD_CONTAIN = "sf-crime-parquet"  # Artifact adında bu ifade geçmeli

# Artifact içindeki beklenen dosya adları (üye/"member")
ARTIFACT_MEMBER_HOURLY  = "risk_hourly_grid_full_labeled.parquet"   # parquet yoksa .csv denenir
ARTIFACT_MEMBER_DAILY   = "risk_daily_grid_full_labeled.parquet"    # parquet yoksa .csv denenir

# (Opsiyonel) GEOID → (lat, lon) eşlemesi için aday dosyalar (artifact içinde aranır)
CENTROID_FILE_CANDIDATES = [
    "geoid_centroids.parquet",
    "sf_geoid_centroids.parquet",
    "geoid_centroids.csv",
    "sf_geoid_centroids.csv",
    # Son çare: grid CSV (içinde lat/lon olabilir)
    "sf_crime_grid_full_labeled.csv",
]

# ------------------------------------------------------------
# 🔑 Token / Header
# ------------------------------------------------------------
def resolve_github_token() -> str | None:
    """Önce ortam değişkeni, sonra Streamlit secrets üzerinden GitHub token getirir."""
    if os.getenv("GITHUB_TOKEN"):
        return os.getenv("GITHUB_TOKEN")
    for key in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if key in st.secrets and st.secrets[key]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[key])
                return os.environ["GITHUB_TOKEN"]
        except Exception:
            pass
    return None

def github_api_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

# ------------------------------------------------------------
# 📦 Artifact ZIP alma (en güncel ve süresi dolmamış)
# ------------------------------------------------------------
def resolve_latest_artifact_zip_url(owner: str, repo: str, name_contains: str):
    """Adı belirli bir deseni içeren en güncel (expired olmayan) artifact ZIP URL’sini döndürür."""
    token = resolve_github_token()
    if not token:
        return None, {}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(f"{base}/actions/artifacts?per_page=100", headers=github_api_headers(), timeout=60)
    response.raise_for_status()
    artifacts = (response.json() or {}).get("artifacts", []) or []
    artifacts = [a for a in artifacts if (name_contains in a.get("name", "")) and not a.get("expired")]
    if not artifacts:
        return None, {}
    artifacts.sort(key=lambda a: a.get("updated_at", ""), reverse=True)
    url = f"{base}/actions/artifacts/{artifacts[0]['id']}/zip"
    return url, github_api_headers()

# ------------------------------------------------------------
# 🧰 ZIP içinden üye okuma (parquet→csv fallback)
# ------------------------------------------------------------
def read_member_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """ZIP içinden dosya okur; parquet bulunamazsa aynı adın .csv versiyonunu dener."""
    def read_any_table(raw_bytes: bytes, name_hint: str) -> pd.DataFrame:
        buffer = BytesIO(raw_bytes)
        if name_hint.lower().endswith(".csv"):
            return pd.read_csv(buffer)
        try:
            buffer.seek(0)
            return pd.read_parquet(buffer)
        except Exception:
            buffer.seek(0)
            return pd.read_csv(buffer)

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zip_file:
        zip_member_names = zip_file.namelist()

        # 1) Tam yol eşleşmesi
        if member_path in zip_member_names:
            with zip_file.open(member_path) as file_obj:
                return read_any_table(file_obj.read(), member_path)

        # 2) Sadece basename ile eşleşme
        base_name = posixpath.basename(member_path)
        candidate_names = [n for n in zip_member_names if n.endswith("/" + base_name) or n == base_name]
        if candidate_names:
            with zip_file.open(candidate_names[0]) as file_obj:
                return read_any_table(file_obj.read(), candidate_names[0])

        # 3) parquet ↔ csv fallback
        alternative = None
        if member_path.lower().endswith(".parquet"):
            alternative = member_path[:-8] + ".csv"
        elif member_path.lower().endswith(".csv"):
            alternative = member_path[:-4] + ".parquet"
        if alternative:
            return read_member_from_zip_bytes(zip_bytes, alternative)

    raise FileNotFoundError(f"ZIP içinde bulunamadı: {member_path}")

@st.cache_data(show_spinner=False)
def load_artifact_member(member: str) -> pd.DataFrame:
    url, headers = resolve_latest_artifact_zip_url(REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN)
    if not url:
        raise RuntimeError("Artifact bulunamadı veya GITHUB_TOKEN yok.")
    response = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    response.raise_for_status()
    return read_member_from_zip_bytes(response.content, member)

# ------------------------------------------------------------
# 🧭 Şema doğrulayıcılar (hourly/daily)
# ------------------------------------------------------------

def normalize_hourly_schema(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Saatlik veriyi standart sütunlara dönüştürür: timestamp, date, hour, geoid, risk_score"""
    column_lookup = {c.lower(): c for c in hourly_df.columns}

    def pick_column(*names: str) -> str | None:
        for name in names:
            if name in hourly_df.columns:
                return name
            if name.lower() in column_lookup:
                return column_lookup[name.lower()]
        return None

    date_column  = pick_column("date")
    hour_column  = pick_column("hour")
    geoid_column = pick_column("geoid", "GEOID", "cell_id", "id")
    risk_column  = pick_column("risk_score", "p_stack", "prob", "probability", "score", "risk")

    if not (date_column and hour_column and geoid_column and risk_column):
        raise ValueError("Saatlik veri için 'date, hour, geoid, risk_score' sütunları gerekli.")

    normalized = pd.DataFrame({
        "date": pd.to_datetime(hourly_df[date_column], errors="coerce"),
        "hour": pd.to_numeric(hourly_df[hour_column], errors="coerce").astype("Int64").clip(0, 23),
        "geoid": hourly_df[geoid_column].astype(str),
        "risk_score": pd.to_numeric(hourly_df[risk_column], errors="coerce"),
    }).dropna(subset=["date", "hour", "geoid"]).copy()

    normalized["timestamp"] = normalized["date"].dt.floor("D") + pd.to_timedelta(normalized["hour"].fillna(0).astype(int), unit="h")
    return normalized[["timestamp", "date", "hour", "geoid", "risk_score"]]


def normalize_daily_schema(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Günlük veriyi standart sütunlara dönüştürür: date, geoid, risk_score"""
    column_lookup = {c.lower(): c for c in daily_df.columns}

    def pick_column(*names: str) -> str | None:
        for name in names:
            if name in daily_df.columns:
                return name
            if name.lower() in column_lookup:
                return column_lookup[name.lower()]
        return None

    date_column  = pick_column("date")
    geoid_column = pick_column("geoid", "GEOID", "cell_id", "id")
    risk_column  = pick_column("risk_score", "p_stack", "prob", "probability", "score", "risk")

    if not (date_column and geoid_column and risk_column):
        raise ValueError("Günlük veri için 'date, geoid, risk_score' sütunları gerekli.")

    normalized = pd.DataFrame({
        "date": pd.to_datetime(daily_df[date_column], errors="coerce").dt.floor("D"),
        "geoid": daily_df[geoid_column].astype(str),
        "risk_score": pd.to_numeric(daily_df[risk_column], errors="coerce"),
    }).dropna(subset=["date", "geoid"]).copy()
    return normalized[["date", "geoid", "risk_score"]]

# ------------------------------------------------------------
# 🗺️ Centroid yükleyici (opsiyonel)
# ------------------------------------------------------------

def coerce_centroid_dataframe(any_df: pd.DataFrame) -> pd.DataFrame | None:
    """Herhangi bir centroid benzeri tabloyu standart {geoid, lat, lon} formuna getirir."""
    column_lookup = {c.lower(): c for c in any_df.columns}

    def pick_column(*candidates: str) -> str | None:
        for name in candidates:
            if name in any_df.columns:
                return name
            if name.lower() in column_lookup:
                return column_lookup[name.lower()]
        return None

    geoid_column = pick_column("geoid", "GEOID", "cell_id", "id")
    lat_column   = pick_column("lat", "latitude", "y")
    lon_column   = pick_column("lon", "lng", "longitude", "x")

    if not (geoid_column and lat_column and lon_column):
        return None

    standardized = pd.DataFrame({
        "geoid": any_df[geoid_column].astype(str),
        "lat": pd.to_numeric(any_df[lat_column], errors="coerce"),
        "lon": pd.to_numeric(any_df[lon_column], errors="coerce"),
    }).dropna(subset=["lat", "lon"]).copy()

    return standardized.drop_duplicates("geoid")

@st.cache_data(show_spinner=False)
def load_centroids_optional() -> pd.DataFrame | None:
    try:
        url, headers = resolve_latest_artifact_zip_url(REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN)
        if not url:
            return None
        response = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            for candidate in CENTROID_FILE_CANDIDATES:
                try:
                    with zip_file.open(candidate) as file_obj:
                        raw_bytes = file_obj.read()
                    try:
                        centroid_df = pd.read_parquet(BytesIO(raw_bytes))
                    except Exception:
                        centroid_df = pd.read_csv(BytesIO(raw_bytes))
                    standardized = coerce_centroid_dataframe(centroid_df)
                    if standardized is not None and len(standardized):
                        return standardized
                except KeyError:
                    continue
        return None
    except Exception:
        return None

# ------------------------------------------------------------
# 📥 Veri yükleme (cache'li)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_hourly_dataframe() -> pd.DataFrame:
    raw_hourly_df = load_artifact_member(ARTIFACT_MEMBER_HOURLY)
    return normalize_hourly_schema(raw_hourly_df)

@st.cache_data(show_spinner=False)
def load_daily_dataframe() -> pd.DataFrame:
    raw_daily_df = load_artifact_member(ARTIFACT_MEMBER_DAILY)
    return normalize_daily_schema(raw_daily_df)

# ------------------------------------------------------------
# 🧮 Risk bucket fonksiyonları (sabit eşikler)
# ------------------------------------------------------------
RISK_BUCKET_DEFINITIONS = [
    (0.0, 0.20, "Çok Düşük"),
    (0.20, 0.40, "Düşük"),
    (0.40, 0.60, "Orta"),
    (0.60, 0.80, "Yüksek"),
    (0.80, 1.01, "Çok Yüksek"),
]

RISK_BUCKET_COLORS_RGBA = {
    "Çok Düşük": [220, 220, 220, 160],
    "Düşük":     [180, 210, 255, 200],
    "Orta":      [255, 220, 130, 210],
    "Yüksek":    [255, 170, 110, 220],
    "Çok Yüksek": [255, 90, 90, 240],
}

def map_value_to_risk_bucket(probability_value: float) -> str:
    value = 0.0 if pd.isna(probability_value) else float(probability_value)
    for lower, upper, name in RISK_BUCKET_DEFINITIONS:
        if lower <= value < upper:
            return name
    return "Çok Düşük"

# ------------------------------------------------------------
# 🧾 CSV indir yardımcı
# ------------------------------------------------------------

def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ------------------------------------------------------------
# 🎛️ UI — Ayarlar
# ------------------------------------------------------------
st.set_page_config(page_title="🌀 Suç Tahmini", layout="wide")
st.sidebar.header("⚙️ Ayarlar")

selected_resolution_mode = st.sidebar.radio(
    "Zaman çözünürlüğü",
    ["Saatlik (≤7 gün)", "Günlük (≤365 gün)"],
    index=0,
)

current_datetime = datetime.now()
if selected_resolution_mode.startswith("Saatlik"):
    maximum_selectable_days = 7
    st.sidebar.caption("Saatlik görünümde en fazla 7 gün seçebilirsiniz.")
else:
    maximum_selectable_days = 365
    st.sidebar.caption("Günlük görünümde en fazla 365 gün seçebilirsiniz.")

selected_start_date = st.sidebar.date_input(
    "Başlangıç tarihi",
    value=(current_datetime - timedelta(days=1)).date(),
)
selected_end_date = st.sidebar.date_input(
    "Bitiş tarihi",
    value=current_datetime.date(),
)

# Tarih aralığını güvenli biçimde sınırla
if (pd.to_datetime(selected_end_date) - pd.to_datetime(selected_start_date)).days > maximum_selectable_days:
    selected_end_date = (pd.to_datetime(selected_start_date) + pd.Timedelta(days=maximum_selectable_days)).date()
    st.sidebar.warning(f"Seçim {maximum_selectable_days} günü aşamaz; bitiş {selected_end_date} olarak güncellendi.")

# Opsiyonel GEOID filtre girişi
geoid_filter_text = st.sidebar.text_input("GEOID filtre (virgülle ayır)", value="")
selected_geoids = [g.strip() for g in geoid_filter_text.split(",") if g.strip()]

# Tablo boyutu (Top-K)
selected_top_k = st.sidebar.slider("Top-K (tablo)", 10, 200, 50, step=10)

# ------------------------------------------------------------
# 📥 Veri yükleme ve filtreleme
# ------------------------------------------------------------
with st.spinner("Veriler yükleniyor…"):
    if selected_resolution_mode.startswith("Saatlik"):
        hourly_dataframe = load_hourly_dataframe()
        start_ts = pd.to_datetime(selected_start_date)
        end_ts   = pd.to_datetime(selected_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        hourly_dataframe = hourly_dataframe[(hourly_dataframe["timestamp"] >= start_ts) & (hourly_dataframe["timestamp"] <= end_ts)].copy()
        if selected_geoids:
            hourly_dataframe = hourly_dataframe[hourly_dataframe["geoid"].isin(selected_geoids)].copy()
        # Seçilen pencerede GEOID bazında ortalama risk
        aggregated_dataframe = (
            hourly_dataframe.groupby("geoid", as_index=False)["risk_score"].mean().rename(columns={"risk_score": "risk_mean"})
        )
        data_used_for_views = hourly_dataframe
    else:
        daily_dataframe = load_daily_dataframe()
        start_day = pd.to_datetime(selected_start_date).floor("D")
        end_day   = pd.to_datetime(selected_end_date).floor("D")
        daily_dataframe = daily_dataframe[(daily_dataframe["date"] >= start_day) & (daily_dataframe["date"] <= end_day)].copy()
        if selected_geoids:
            daily_dataframe = daily_dataframe[daily_dataframe["geoid"].isin(selected_geoids)].copy()
        aggregated_dataframe = (
            daily_dataframe.groupby("geoid", as_index=False)["risk_score"].mean().rename(columns={"risk_score": "risk_mean"})
        )
        data_used_for_views = daily_dataframe

# Risk bucket ve sıralama
if len(aggregated_dataframe):
    aggregated_dataframe["risk_bucket"] = aggregated_dataframe["risk_mean"].map(map_value_to_risk_bucket)
    aggregated_sorted = aggregated_dataframe.sort_values("risk_mean", ascending=False).reset_index(drop=True)
else:
    aggregated_sorted = aggregated_dataframe

# ------------------------------------------------------------
# 🧠 Özet kartlar
# ------------------------------------------------------------
st.title("🌀 Suç Tahmini — rev3")
st.caption("Saatlik (≤7 gün) / Günlük (≤365 gün) pencerede GEOID bazlı ortalama risk.")

summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
summary_col_1.metric("Kapsanan kayıt", f"{len(data_used_for_views):,}")
summary_col_2.metric("GEOID sayısı", f"{aggregated_sorted['geoid'].nunique():,}" if len(aggregated_sorted) else "0")
summary_col_3.metric("Ortalama risk", f"{data_used_for_views['risk_score'].mean():.3f}" if len(data_used_for_views) else "—")

# ------------------------------------------------------------
# 🔝 Top-K tablo + indir
# ------------------------------------------------------------
st.subheader("🔝 Top-K GEOID")
visible_topk_dataframe = aggregated_sorted.head(selected_top_k).copy()
st.dataframe(visible_topk_dataframe, use_container_width=True, height=420)
st.download_button(
    "⬇️ CSV indir (Top-K)",
    data=dataframe_to_csv_bytes(visible_topk_dataframe),
    file_name="risk_topk.csv",
    mime="text/csv",
)

# ------------------------------------------------------------
# 📈 Zaman serisi (seçili GEOID'ler)
# ------------------------------------------------------------
st.subheader("📈 Zaman serisi (risk_score)")
if len(data_used_for_views) == 0:
    st.info("Seçilen aralık için veri yok.")
else:
    default_geoids_for_plot = visible_topk_dataframe["geoid"].head(3).tolist() if len(visible_topk_dataframe) else []
    selectable_geoids = sorted(data_used_for_views["geoid"].unique().tolist())
    chosen_geoids_for_plot = st.multiselect("Grafik GEOID seç", options=selectable_geoids, default=default_geoids_for_plot)
    if selected_resolution_mode.startswith("Saatlik"):
        time_series_pivot = (
            data_used_for_views[data_used_for_views["geoid"].isin(chosen_geoids_for_plot)]
            .pivot_table(index="timestamp", columns="geoid", values="risk_score", aggfunc="mean")
            .sort_index()
        )
    else:
        time_series_pivot = (
            data_used_for_views[data_used_for_views["geoid"].isin(chosen_geoids_for_plot)]
            .pivot_table(index="date", columns="geoid", values="risk_score", aggfunc="mean")
            .sort_index()
        )
    if len(time_series_pivot):
        st.line_chart(time_series_pivot, height=360)
    else:
        st.caption("Seçilen GEOID'ler için veri yok.")

# ------------------------------------------------------------
# 🗺️ Harita (opsiyonel centroid bulunursa)
# ------------------------------------------------------------
st.subheader("🗺️ Harita — 5 seviye risk renklendirme")
centroid_dataframe = load_centroids_optional()
if centroid_dataframe is None or len(centroid_dataframe) == 0:
    st.info("Centroid verisi (geoid→lat/lon) artifact içinde bulunamadı. Harita devre dışı.")
else:
    map_dataframe = aggregated_sorted.merge(centroid_dataframe, on="geoid", how="left").dropna(subset=["lat", "lon"]).copy()
    if len(map_dataframe) == 0:
        st.info("Harita için geçerli nokta yok (lat/lon eşleşmedi).")
    else:
        map_dataframe["color"] = map_dataframe["risk_bucket"].map(RISK_BUCKET_COLORS_RGBA)
        legend_markdown = (
            "**Lejand:** "
            "<span style='background:#ddd;padding:2px 6px;border-radius:4px;'>Çok Düşük</span> "
            "<span style='background:#b4d2ff;padding:2px 6px;border-radius:4px;'>Düşük</span> "
            "<span style='background:#ffdc82;padding:2px 6px;border-radius:4px;'>Orta</span> "
            "<span style='background:#ffaa6e;padding:2px 6px;border-radius:4px;'>Yüksek</span> "
            "<span style='background:#ff5a5a;padding:2px 6px;border-radius:4px;'>Çok Yüksek</span> "
        )
        st.markdown(legend_markdown, unsafe_allow_html=True)

        import pydeck as pdk
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_dataframe,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=80 if selected_resolution_mode.startswith("Saatlik") else 120,
            pickable=True,
            radius_min_pixels=2,
            radius_max_pixels=20,
            auto_highlight=True,
        )
        initial_view_state = pdk.ViewState(
            latitude=float(map_dataframe["lat"].median()),
            longitude=float(map_dataframe["lon"].median()),
            zoom=11,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[scatter_layer],
            initial_view_state=initial_view_state,
            tooltip={
                "text": "GEOID {geoid}\\nRisk {risk_mean:.3f}\\nSeviye {risk_bucket}"
            },
        ))
# ------------------------------------------------------------
# 📝 Dipnot
# ------------------------------------------------------------
st.caption(
    "Kaynak: artifact 'sf-crime-parquet' → "
    "risk_hourly_grid_full_labeled / risk_daily_grid_full_labeled.\n"
    "Harita, artifact içinde geoid→lat/lon eşleşmesi bulunduğu durumda etkinleşir."
)


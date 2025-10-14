# SUTAM - Suç Tahmin Modeli

import io
import os
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

# ------------------------------------------------------------
# Ayarlar — Gerekli depolar ve artifact adı
# ------------------------------------------------------------
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"
EXPECTED_PARQUET = "risk_hourly.parquet"
EXPECTED_CSV = "risk_hourly.csv"

# GitHub Token:
# - Public repo olsa bile, Actions artifact indirmek için **token gerekir**.
# - Token'ı Streamlit Secrets'a `github_token` olarak ekleyin
#   (Settings → Secrets → github_token)
GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

# ------------------------------------------------------------
# Yardımcılar — İndirme & Okuma
# ------------------------------------------------------------

@st.cache_data(show_spinner=True, ttl=15*60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    """Repo'daki son **geçerli** (expired=false) artifact'i bulur ve ZIP bytes döndürür."""
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "GitHub token bulunamadı. lütfen st.secrets['github_token'] veya GITHUB_TOKEN env. değişkeni ayarlayın.")

    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

    # Sayfalanmış olabilir; basitçe ilk sayfayı alıp isim eşleyenleri filtreleriz
    r = requests.get(base, headers=headers, timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    # İsim eşleşen, süresi dolmamış olanları tarihine göre sırala
    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")
    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    artifact = cand[0]

    download_url = artifact.get("archive_download_url")
    if not download_url:
        raise RuntimeError("archive_download_url bulunamadı")

    r2 = requests.get(download_url, headers=headers, timeout=60)
    r2.raise_for_status()
    return r2.content


@st.cache_data(show_spinner=True, ttl=15*60)
def read_risk_from_artifact() -> pd.DataFrame:
    """ZIP içinden Parquet/CSV'yi okuyup kolonları normalize eder."""
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        target = EXPECTED_PARQUET if EXPECTED_PARQUET in memlist else (
            EXPECTED_CSV if EXPECTED_CSV in memlist else None)
        if target is None:
            raise FileNotFoundError(
                f"Zip içinde {EXPECTED_PARQUET} veya {EXPECTED_CSV} bulunamadı. İçerik: {memlist}")
        with zf.open(target) as f:
            if target.endswith(".parquet"):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)

    # kolon isimleri
    df.columns = [c.strip().lower() for c in df.columns]
    # esnek yeniden adlandırma
    rename_map = {}
    for src, dst in [
        ('geoid', 'geoid'),
        ('date', 'date'),
        ('hour_range', 'hour_range'),
        ('risk_score', 'risk_score'),
    ]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    if rename_map:
        df = df.rename(columns=rename_map)

    # tarih
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date

    return df


def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    """geoid×date bazında günlük ortalama risk skoru."""
    if df.empty:
        return df
    needed = {'geoid', 'date', 'risk_score'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {', '.join(sorted(missing))}")

    daily = (
        df.groupby(['geoid', 'date'], as_index=False)['risk_score']
          .mean()
          .rename(columns={'risk_score': 'risk_score_daily'})
    )
    return daily


def classify_quantiles(daily_df: pd.DataFrame, day: date) -> pd.DataFrame:
    """Seçilen gün için low/medium/high/critical etiketleri (Q25/Q50/Q75)."""
    one = daily_df[daily_df['date'] == day].copy()
    if one.empty:
        return one

    q25, q50, q75 = one['risk_score_daily'].quantile([0.25, 0.50, 0.75]).tolist()

    def labeler(x):
        if x <= q25:
            return 'low'
        elif x <= q50:
            return 'medium'
        elif x <= q75:
            return 'high'
        else:
            return 'critical'

    one['risk_level'] = one['risk_score_daily'].apply(labeler)
    one['q25'] = q25
    one['q50'] = q50
    one['q75'] = q75
    return one


@st.cache_data(show_spinner=False)
def load_geojson(uploaded_geojson) -> dict:
    # Manuel yükleme yolunu koruyoruz (yan panelde opsiyonel)
    if uploaded_geojson is None:
        return {}
    return json.load(uploaded_geojson)


def inject_properties(geojson_dict: dict, day_df: pd.DataFrame) -> dict:
    """Seçilen gün risk metriklerini GeoJSON özelliklerine ekler (GEOID eşlemesi)."""
    if not geojson_dict:
        return {}
    props_key_candidates = ['GEOID', 'geoid']

    dmap = day_df.set_index(day_df['geoid'].astype(str))
    features_out = []
    for feat in geojson_dict.get('features', []):
        props = feat.get('properties', {}) or {}
        geoid_prop = None
        for k in props_key_candidates:
            if k in props:
                geoid_prop = props[k]
                break
        if geoid_prop is None:
            features_out.append(feat)
            continue
        key = str(geoid_prop)
        if key in dmap.index:
            row = dmap.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            props = {
                **props,
                'risk_score_daily': float(row['risk_score_daily']),
                'risk_level': row['risk_level'],
            }
        feat_out = {**feat, 'properties': props}
        features_out.append(feat_out)

    return {**geojson_dict, 'features': features_out}


def make_map(geojson_enriched: dict, initial_view=None):
    if not geojson_enriched:
        st.info("Haritayı görmek için GeoJSON yükleyin.")
        return

    # Renkler: low=yeşil, medium=sarı, high=turuncu, critical=kırmızı
    color_expr = [
        'case',
        ['==', ['get', 'risk_level'], 'low'], [56, 168, 0],
        ['==', ['get', 'risk_level'], 'medium'], [255, 221, 0],
        ['==', ['get', 'risk_level'], 'high'], [255, 140, 0],
        ['==', ['get', 'risk_level'], 'critical'], [204, 0, 0],
        [200, 200, 200],
    ]

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_enriched,
        stroked=False,
        filled=True,
        get_fill_color=color_expr,
        pickable=True,
        extruded=False,
        opacity=0.6,
    )

    if initial_view is None:
        initial_view = pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10)

    tooltip = {
        "html": "<b>GEOID:</b> {GEOID}{geoid}<br/>"
                "<b>Risk:</b> {risk_level}<br/>"
                "<b>Skor:</b> {risk_score_daily}",
        "style": {"backgroundColor": "#262730", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=initial_view,
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.set_page_config(page_title="Suç Risk Haritası (Günlük)", layout="wide")
st.title("Suç Risk Haritası — Günlük Ortalama (low / medium / high / critical)")

st.sidebar.header("GitHub Artifact (otomatik)")
refresh = st.sidebar.button("Veriyi Yenile (artifact'ten yeniden çek)")

# Veri çek
try:
    if refresh:
        read_risk_from_artifact.clear()
        fetch_latest_artifact_zip.clear()
    risk_df = read_risk_from_artifact()
except Exception as e:
    st.error(f"Artifact indirilemedi/okunamadı: {e}")
    st.stop()

# Günlük ortalama
risk_daily = daily_average(risk_df)

# Tarih seçimi
dates = sorted(risk_daily['date'].unique())
sel_date = st.sidebar.selectbox("Gün seçin", dates, index=len(dates)-1 if dates else 0, format_func=lambda d: str(d))

# Sınıflandırma (per-gün çeyrekler)
st.sidebar.subheader("Sınıflandırma")
one_day = classify_quantiles(risk_daily, sel_date)

# Eşikler
if not one_day.empty:
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Q25", f"{one_day['q25'].iloc[0]:.4f}")
    with c2: st.metric("Q50", f"{one_day['q50'].iloc[0]:.4f}")
    with c3: st.metric("Q75", f"{one_day['q75'].iloc[0]:.4f}")

# GeoJSON — otomatik getir + manuel seçenek
@st.cache_data(show_spinner=True, ttl=60*60)
def fetch_geojson_auto() -> dict:
    """Sırayla dener:
    1) Artifact ZIP içinde `sf_tracts.geojson` (veya secrets: geojson_path)
    2) Repo contents API (OWNER/REPO içindeki geojson_path)
    3) Secrets: geojson_url doğrudan indir
    """
    path = st.secrets.get("geojson_path", "sf_tracts.geojson")
    url_override = st.secrets.get("geojson_url", "")

    # 1) Artifact içinden dene
    try:
        zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            memlist = zf.namelist()
            cand = path if path in memlist else None
            if cand:
                with zf.open(cand) as f:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception:
        pass

    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

    # 2) Repo contents API
    try:
        api = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}"
        r = requests.get(api, headers=headers, timeout=30)
        if r.status_code == 200:
            b64 = r.json().get('content', '')
            if b64:
                import base64
                data = base64.b64decode(b64)
                return json.loads(data.decode('utf-8'))
    except Exception:
        pass

    # 3) Direkt URL override
    if url_override:
        r = requests.get(url_override, headers=headers if GITHUB_TOKEN else None, timeout=30)
        if r.status_code == 200:
            return r.json()

    return {}

# Manuel yükleme opsiyonu
st.sidebar.header("Harita Sınırları (GeoJSON)")
geojson_file = st.sidebar.file_uploader(
    "(Opsiyonel) GEOID içeren GeoJSON yükle",
    type=["json", "geojson"],
)

geojson = load_geojson(geojson_file) if geojson_file else fetch_geojson_auto()

st.subheader(f"Harita — {sel_date}")
if not geojson:
    st.warning("GeoJSON yüklenmedi. Yine de tablo ve indirilebilir çıktıyı görebilirsiniz.")
else:
    enriched = inject_properties(geojson, one_day)
    make_map(enriched)

# Tablo + indirme
st.subheader("Seçilen Gün Tablosu")
st.dataframe(one_day.sort_values('risk_score_daily', ascending=False), use_container_width=True)

csv = one_day.drop(columns=['q25','q50','q75'], errors='ignore').to_csv(index=False).encode('utf-8')
st.download_button(
    label="Günlük tabloyu CSV indir",
    data=csv,
    file_name=f"risk_daily_{sel_date}.csv",
    mime="text/csv",
)

with st.expander("Nasıl çalışır?"):
    st.markdown(
        """
        - **Veri Kaynağı:** GitHub Actions artifact → **`sf-crime-parquet.zip`** (repo: `cem5113/crime_prediction_data`).
        - **Kimlik Doğrulama:** Streamlit secrets içine `github_token` ekleyin (Actions → Artifacts okuma yetkisiyle).
        - **Günlük Ortalama:** `hour_range` göz ardı edilerek aynı gün içindeki kayıtların ortalaması alınır.
        - **4 Seviye:** Günlük dağılıma göre çeyrekler: `low (≤Q25)`, `medium (Q25–Q50)`, `high (Q50–Q75)`, `critical (>Q75)`.
        - **Harita:** GeoJSON'daki **GEOID** eşleşir; renkler: low=yeşil, medium=sarı, high=turuncu, critical=kırmızı.
        - **Yenile:** Sol menüden artifact'i yeniden çekebilirsiniz.
        """
    )

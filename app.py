# SUTAM - Suç Tahmin Modeli (Parquet-only, CSV fallback kaldırıldı)

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
# Ayarlar — default değerler (secrets ile override edilir)
# ------------------------------------------------------------
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"
EXPECTED_PARQUET = "risk_hourly.parquet"

# GeoJSON repo/yol (harita sınırları)
GEOJSON_OWNER = st.secrets.get("geojson_owner", OWNER)
GEOJSON_REPO = st.secrets.get("geojson_repo", "crimepredict")
GEOJSON_PATH = st.secrets.get("geojson_path", "data/sf_cells.geojson")

# Artifact repo ayarlarını secrets'tan okumaya izin ver
ARTIFACT_OWNER = st.secrets.get("artifact_owner", OWNER)
ARTIFACT_REPO = st.secrets.get("artifact_repo", REPO)
ARTIFACT_NAME = st.secrets.get("artifact_name", ARTIFACT_NAME)

# GitHub Token (artifact erişimi için)
GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

# ------------------------------------------------------------
# UI Ayarları
# ------------------------------------------------------------
st.set_page_config(page_title="Suç Risk Haritası (Günlük)", layout="wide")
st.title("Suç Risk Haritası — Günlük Ortalama (low / medium / high / critical)")

# ------------------------------------------------------------
# Yardımcılar — HTTP
# ------------------------------------------------------------
def _gh_headers():
    hdrs = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        hdrs["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdrs

# ------------------------------------------------------------
# Artifact indirme ve okuma
# ------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=15 * 60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    """Repo'daki son geçerli (expired=false) artifact'i indirip ZIP bytes döndürür."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GitHub token yok. Secrets veya env olarak ekleyin.")

    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=_gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")

    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    artifact = cand[0]
    url = artifact.get("archive_download_url")
    if not url:
        raise RuntimeError("archive_download_url bulunamadı")

    r2 = requests.get(url, headers=_gh_headers(), timeout=60)
    r2.raise_for_status()
    return r2.content


@st.cache_data(show_spinner=True, ttl=15 * 60)
def read_from_artifact(file_names: list[str]) -> dict[str, bytes]:
    """Artifact ZIP içinden istenen dosyaları döndürür: {ad: bytes}"""
    zip_bytes = fetch_latest_artifact_zip(ARTIFACT_OWNER, ARTIFACT_REPO, ARTIFACT_NAME)
    results: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        for fname in file_names:
            matches = [n for n in memlist if n.endswith("/" + fname) or n.endswith(fname)]
            if matches:
                with zf.open(matches[0]) as f:
                    results[fname] = f.read()
            else:
                st.warning(f"{fname} ZIP içinde bulunamadı. İlk 10 dosya: {memlist[:10]}")
    return results


@st.cache_data(show_spinner=True, ttl=15 * 60)
def read_risk_from_artifact() -> pd.DataFrame:
    """Artifact ZIP içindeki risk_hourly.parquet -> DataFrame"""
    files = read_from_artifact([EXPECTED_PARQUET])

    if EXPECTED_PARQUET not in files:
        raise FileNotFoundError(f"Artifact'te {EXPECTED_PARQUET} bulunamadı.")

    df = pd.read_parquet(io.BytesIO(files[EXPECTED_PARQUET]))
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

# ------------------------------------------------------------
# GeoJSON getir (ayrı repo)
# ------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_geojson() -> dict:
    """Önce GitHub API, sonra raw fallback"""
    api = f"https://api.github.com/repos/{GEOJSON_OWNER}/{GEOJSON_REPO}/contents/{GEOJSON_PATH}"
    try:
        r = requests.get(api, headers=_gh_headers(), timeout=30)
        if r.status_code == 200:
            import base64
            data = base64.b64decode(r.json().get("content", ""))
            return json.loads(data.decode("utf-8"))
    except Exception:
        pass

    raw = f"https://raw.githubusercontent.com/{GEOJSON_OWNER}/{GEOJSON_REPO}/main/{GEOJSON_PATH}"
    r = requests.get(raw, timeout=30)
    if r.status_code == 200:
        return r.json()
    return {}

# ------------------------------------------------------------
# Veri işleme
# ------------------------------------------------------------
def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    """geoid×date bazında günlük ortalama risk skoru"""
    if df.empty:
        return df
    needed = {"geoid", "date", "risk_score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Eksik kolonlar: {sorted(list(needed - set(df.columns)))}")
    out = (
        df.groupby(["geoid", "date"], as_index=False)["risk_score"]
        .mean()
        .rename(columns={"risk_score": "risk_score_daily"})
    )
    return out


def classify_quantiles(daily_df: pd.DataFrame, day: date) -> pd.DataFrame:
    """Seçilen gün için low/medium/high/critical sınıflandırması"""
    one = daily_df[daily_df["date"] == day].copy()
    if one.empty:
        return one
    q25, q50, q75 = one["risk_score_daily"].quantile([0.25, 0.5, 0.75]).tolist()

    def lab(x: float) -> str:
        if x <= q25:
            return "low"
        elif x <= q50:
            return "medium"
        elif x <= q75:
            return "high"
        return "critical"

    one["risk_level"] = one["risk_score_daily"].apply(lab)
    one["q25"] = q25
    one["q50"] = q50
    one["q75"] = q75
    return one


def _detect_geojson_id_len(gj: dict) -> int | None:
    """GeoJSON'da en sık görülen sayısal ID uzunluğu"""
    if not gj:
        return None
    lens = []
    for feat in gj.get("features", [])[:200]:
        props = feat.get("properties", {}) or {}
        for k in ("geoid", "GEOID", "cell_id", "id"):
            if k in props:
                digits = "".join(ch for ch in str(props[k]) if ch.isdigit())
                if digits:
                    lens.append(len(digits))
                break
    if not lens:
        return None
    return max(set(lens), key=lens.count)


def inject_properties(geojson_dict: dict, day_df: pd.DataFrame) -> dict:
    """Günlük risk metriklerini GeoJSON'a ekler"""
    if not geojson_dict or day_df.empty:
        return geojson_dict

    target_len = _detect_geojson_id_len(geojson_dict)

    day_df = day_df.copy()
    day_df["geoid"] = (
        day_df["geoid"].astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(target_len or 0)
    )
    dmap = day_df.set_index(day_df["geoid"])

    features_out = []
    for feat in geojson_dict.get("features", []):
        props = feat.get("properties", {}) or {}
        key_raw = None
        for cand in ("geoid", "GEOID", "cell_id", "id"):
            if cand in props:
                key_raw = str(props[cand])
                break
        if not key_raw:
            features_out.append(feat)
            continue

        key_norm = "".join(ch for ch in key_raw if ch.isdigit())
        if target_len:
            key_norm = key_norm.zfill(target_len)

        if key_norm in dmap.index:
            row = dmap.loc[key_norm]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        
            # gösterimde kullanacağımız tekil ID
            disp = props.get("GEOID") or props.get("geoid") or props.get("cell_id") or props.get("id")
        
            props.update({
                "display_id": disp,                                        # <-- eklendi
                "risk_score_daily": float(row["risk_score_daily"]),
                "risk_level": row["risk_level"],
                "risk_score_txt": f"{float(row['risk_score_daily']):.4f}", # <-- eklendi (tooltip için)
            })
        
        features_out.append({**feat, "properties": props})

    return {**geojson_dict, "features": features_out}


def make_map(geojson_enriched: dict):
    if not geojson_enriched:
        st.info("Haritayı görmek için GeoJSON bulunamadı.")
        return

    color_expr = [
        "case",
        ["==", ["get", "properties.risk_level"], "low"], [56, 168, 0],
        ["==", ["get", "properties.risk_level"], "medium"], [255, 221, 0],
        ["==", ["get", "properties.risk_level"], "high"], [255, 140, 0],
        ["==", ["get", "properties.risk_level"], "critical"], [204, 0, 0],
        [200, 200, 200],
    ]

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_enriched,
        stroked=False,
        filled=True,
        get_fill_color=color_expr,
        pickable=True,
        opacity=0.6,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip={
            "html": (
                "<b>ID:</b> {properties.display_id}<br/>"
                "<b>Risk:</b> {properties.risk_level}<br/>"
                "<b>Skor:</b> {properties.risk_score_txt}"
            ),
            "style": {"backgroundColor": "#262730", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)

# ------------------------------------------------------------
# UI Akışı
# ------------------------------------------------------------
st.sidebar.header("Veri")
refresh = st.sidebar.button("Veriyi Yenile (artifact'i tazele)")

try:
    if refresh:
        read_risk_from_artifact.clear()
        read_from_artifact.clear()
        fetch_latest_artifact_zip.clear()
    risk_df = read_risk_from_artifact()
except Exception as e:
    st.error(f"Risk verisi indirilemedi: {e}")
    st.stop()

risk_daily = daily_average(risk_df)

dates = sorted(risk_daily["date"].unique())
sel_date = st.sidebar.selectbox(
    "Gün seçin",
    dates,
    index=len(dates) - 1 if dates else 0,
    format_func=lambda d: str(d)
)

one_day = classify_quantiles(risk_daily, sel_date)

if not one_day.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Q25", f"{one_day['q25'].iloc[0]:.4f}")
    c2.metric("Q50", f"{one_day['q50'].iloc[0]:.4f}")
    c3.metric("Q75", f"{one_day['q75'].iloc[0]:.4f}")

geojson = fetch_geojson()

st.subheader(f"Harita — {sel_date}")
if not geojson:
    st.warning("GeoJSON bulunamadı.")
else:
    enriched = inject_properties(geojson, one_day)
    make_map(enriched)

# Tablo + indirme
st.subheader("Seçilen Gün Tablosu")
st.dataframe(
    one_day.drop(columns=["q25", "q50", "q75"], errors="ignore").sort_values(
        "risk_score_daily", ascending=False
    ),
    use_container_width=True,
)

csv = one_day.drop(columns=["q25", "q50", "q75"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    "Günlük tabloyu CSV indir",
    csv,
    file_name=f"risk_daily_{sel_date}.csv",
    mime="text/csv",
)

with st.expander("Teşhis (GeoJSON)"):
    st.write({
        "geojson_owner": GEOJSON_OWNER,
        "geojson_repo": GEOJSON_REPO,
        "geojson_path": GEOJSON_PATH,
    })
st.caption("GeoJSON private ise veya rate limit varsa token gereklidir.")

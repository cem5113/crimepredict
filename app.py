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
# Ayarlar — default değerler (secrets ile override edilir)
# ------------------------------------------------------------
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"
EXPECTED_PARQUET = "risk_hourly.parquet"
EXPECTED_CSV = "risk_hourly.csv"

# GeoJSON repo/yol (harita sınırları)
GEOJSON_OWNER = st.secrets.get("geojson_owner", OWNER)
GEOJSON_REPO = st.secrets.get("geojson_repo", "crimepredict")  # <- ayrı repo
GEOJSON_PATH = st.secrets.get("geojson_path", "data/sf_cells.geojson")

# Artifact repo ayarlarını secrets'tan okumaya izin ver
ARTIFACT_OWNER = st.secrets.get("artifact_owner", OWNER)
ARTIFACT_REPO = st.secrets.get("artifact_repo", REPO)
ARTIFACT_NAME = st.secrets.get("artifact_name", ARTIFACT_NAME)

# GitHub Token (artifact ve/veya private repo erişimi için gerekir)
GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

# ------------------------------------------------------------
# Genel UI Ayarları
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
        raise RuntimeError(
            "GitHub token yok. Lütfen st.secrets['github_token'] veya GITHUB_TOKEN env ayarlayın."
        )
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
    """Artifact ZIP içinden istenen dosyaları (list halinde) döndürür: {ad: bytes}."""
    zip_bytes = fetch_latest_artifact_zip(ARTIFACT_OWNER, ARTIFACT_REPO, ARTIFACT_NAME)
    results: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        for fname in file_names:
            # Tam eşleşme yoksa sondan eşleşeni bul (alt klasörlerde olabilir)
            matches = [n for n in memlist if n.endswith("/" + fname) or n.endswith(fname)]
            if matches:
                with zf.open(matches[0]) as f:
                    results[fname] = f.read()
            else:
                st.warning(f"{fname} ZIP içinde bulunamadı. İlk 10 dosya: {memlist[:10]}")
    return results


@st.cache_data(show_spinner=True, ttl=15 * 60)
def read_risk_from_artifact() -> pd.DataFrame:
    """risk_hourly.parquet (yoksa CSV) -> DataFrame (kolonlar normalize)."""
    wanted = [EXPECTED_PARQUET, EXPECTED_CSV]
    files = read_from_artifact(wanted)
    buf = None
    target = None
    for name in wanted:
        if name in files:
            buf = io.BytesIO(files[name])
            target = name
            break
    if buf is None:
        raise FileNotFoundError(f"Artifact'te {EXPECTED_PARQUET} veya {EXPECTED_CSV} bulunamadı.")
    if target.endswith(".parquet"):
        df = pd.read_parquet(buf)
    else:
        df = pd.read_csv(buf)

    # Kolon adlarını normalize et
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ------------------------------------------------------------
# GeoJSON getir (ayrı repo) + fallback raw
# ------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_geojson() -> dict:
    """Önce GitHub API (contents), olmazsa raw.githubusercontent; public değilse token'lı içerik."""
    # 1) Repo contents API
    api = f"https://api.github.com/repos/{GEOJSON_OWNER}/{GEOJSON_REPO}/contents/{GEOJSON_PATH}"
    try:
        r = requests.get(api, headers=_gh_headers(), timeout=30)
        if r.status_code == 200:
            b64 = r.json().get("content", "")
            if b64:
                import base64

                data = base64.b64decode(b64)
                return json.loads(data.decode("utf-8"))
    except Exception:
        pass

    # 2) Raw (public ise)
    raw = f"https://raw.githubusercontent.com/{GEOJSON_OWNER}/{GEOJSON_REPO}/main/{GEOJSON_PATH}"
    r = requests.get(raw, timeout=30)
    if r.status_code == 200:
        return r.json()

    return {}  # bulunamadı


# ------------------------------------------------------------
# Veri işleme
# ------------------------------------------------------------
def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    """geoid×date bazında günlük ortalama risk skoru (hour_range dikkate alınmaz)."""
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
    """Seçilen gün için low/medium/high/critical (Q25/Q50/Q75)."""
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
    """GeoJSON'da en sık görülen sayısal ID uzunluğunu (örn. 12) bulur."""
    if not gj:
        return None
    lens = []
    for feat in gj.get("features", [])[:200]:
        props = feat.get("properties", {}) or {}
        for k in ("geoid", "GEOID", "cell_id", "id"):
            if k in props:
                val = str(props[k])
                digits = "".join(ch for ch in val if ch.isdigit())
                if digits:
                    lens.append(len(digits))
                break
    if not lens:
        return None
    # moda
    return max(set(lens), key=lens.count)


def inject_properties(geojson_dict: dict, day_df: pd.DataFrame) -> dict:
    """Günlük risk metriklerini GeoJSON'a ekler (ID normalize + zfill)."""
    if not geojson_dict or day_df.empty:
        return geojson_dict

    target_len = _detect_geojson_id_len(geojson_dict)
    # risk tarafını normalize et
    day_df = day_df.copy()
    day_df["geoid"] = (
        day_df["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(target_len or 0)
    )
    dmap = day_df.set_index(day_df["geoid"])

    features_out = []
    for feat in geojson_dict.get("features", []):
        props = feat.get("properties", {}) or {}
        # olası alan adları
        key_raw = None
        for cand in ("geoid", "GEOID", "cell_id", "id"):
            if cand in props:
                key_raw = str(props[cand])
                break
        if key_raw is None:
            features_out.append(feat)
            continue

        key_norm = "".join(ch for ch in key_raw if ch.isdigit())
        if target_len:
            key_norm = key_norm.zfill(target_len)

        if key_norm in dmap.index:
            row = dmap.loc[key_norm]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            props = {
                **props,
                "risk_score_daily": float(row["risk_score_daily"]),
                "risk_level": row["risk_level"],
            }
        features_out.append({**feat, "properties": props})

    return {**geojson_dict, "features": features_out}


def make_map(geojson_enriched: dict):
    if not geojson_enriched:
        st.info("Haritayı görmek için GeoJSON bulunamadı.")
        return
    # low=yeşil, medium=sarı, high=turuncu, critical=kırmızı
    color_expr = [
        "case",
        ["==", ["get", "risk_level"], "low"], [56, 168, 0],
        ["==", ["get", "risk_level"], "medium"], [255, 221, 0],
        ["==", ["get", "risk_level"], "high"], [255, 140, 0],
        ["==", ["get", "risk_level"], "critical"], [204, 0, 0],
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
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip={
            "html": "<b>GEOID:</b> {GEOID}{geoid}{cell_id}{id}<br/>"
                    "<b>Risk:</b> {risk_level}<br/>"
                    "<b>Skor:</b> {risk_score_daily}",
            "style": {"backgroundColor": "#262730", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)


# ------------------------------------------------------------
# UI — veri çekme ve akış
# ------------------------------------------------------------
st.sidebar.header("Veri")
refresh = st.sidebar.button("Veriyi Yenile (artifact'i tazele)")

err = None
try:
    if refresh:
        read_risk_from_artifact.clear()
        read_from_artifact.clear()
        fetch_latest_artifact_zip.clear()
    risk_df = read_risk_from_artifact()
except Exception as e:
    err = f"Risk verisi indirilemedi: {e}"
    risk_df = pd.DataFrame()

if err:
    st.error(err)
    st.stop()

# Günlük ortalama
risk_daily = daily_average(risk_df)

# Tarih seçimi
dates = sorted(risk_daily["date"].unique())
sel_date = st.sidebar.selectbox(
    "Gün seçin", dates, index=len(dates) - 1 if dates else 0, format_func=lambda d: str(d)
)

# Günün sınıflandırması
one_day = classify_quantiles(risk_daily, sel_date)

# Eşikler
if not one_day.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Q25", f"{one_day['q25'].iloc[0]:.4f}")
    c2.metric("Q50", f"{one_day['q50'].iloc[0]:.4f}")
    c3.metric("Q75", f"{one_day['q75'].iloc[0]:.4f}")

# GeoJSON getir
geojson = fetch_geojson()

st.subheader(f"Harita — {sel_date}")
if not geojson:
    st.warning("GeoJSON bulunamadı. Secrets'taki `geojson_owner/repo/path` değerlerini kontrol edin.")
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
    st.caption("GeoJSON public değilse veya rate limit varsa, token (secrets) gereklidir.")

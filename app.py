# Streamlit — Suç Risk Haritası (Günlük)
# ------------------------------------------------------------
# Bu uygulama, veriyi GitHub Actions artifact'inden okur:
#   repo: cem5113/crime_prediction_data
#   artifact: sf-crime-parquet.zip
# Zip içinden:
#   - risk_hourly.parquet -> günlük (geoid×date) risk ortalaması + çeyreklerle sınıflandırma
#   - data/sf_cells.geojson -> harita sınırları (aynı zip içinden)
# ------------------------------------------------------------

import io
import os
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

# =========================
# Ayarlar
# =========================
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-parquet"
EXPECTED_PARQUET = "risk_hourly.parquet"         # artifact içinde
GEOJSON_IN_ZIP_PATH_DEFAULT = "data/sf_cells.geojson"  # artifact içinde

# Actions artifact indirmek için token şart
GITHUB_TOKEN = st.secrets.get("github_token", os.environ.get("GITHUB_TOKEN", ""))

# UI config (ilk satırlarda olsun)
st.set_page_config(page_title="Suç Risk Haritası (Günlük)", layout="wide")
st.title("Suç Risk Haritası — Günlük Ortalama (low / medium / high / critical)")

# =========================
# Yardımcılar
# =========================
def _gh_headers():
    hdrs = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        hdrs["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdrs

@st.cache_data(show_spinner=True, ttl=15 * 60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    """Repo'daki son geçerli (expired=false) artifact'i indirip ZIP bytes döndürür."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GitHub token yok. `st.secrets['github_token']` veya GITHUB_TOKEN env ayarlayın.")

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
def read_risk_from_artifact() -> pd.DataFrame:
    """Artifact ZIP içindeki risk_hourly.parquet -> DataFrame (kolonlar normalize)."""
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        if EXPECTED_PARQUET not in memlist:
            # alt klasör olasılığına karşı sondan eşleşme ara
            matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
            if not matches:
                raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} bulunamadı. İçerik örneği: {memlist[:15]}")
            target = matches[0]
        else:
            target = EXPECTED_PARQUET

        with zf.open(target) as f:
            df = pd.read_parquet(f)

    # kolon adlarını normalize et
    df.columns = [c.strip().lower() for c in df.columns]
    # tarih formatı
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    """geoid×date bazında günlük ortalama risk skoru (hour_range dikkate alınmaz)."""
    if df.empty:
        return df
    needed = {"geoid", "date", "risk_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {', '.join(sorted(missing))}")
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
                digits = "".join(ch for ch in str(props[k]) if ch.isdigit())
                if digits:
                    lens.append(len(digits))
                break
    if not lens:
        return None
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
    dmap = day_df.set_index("geoid")

    features_out = []
    for feat in geojson_dict.get("features", []):
        props = (feat.get("properties") or {}).copy()

        # GeoJSON tarafındaki kimlik anahtarı
        key_raw = None
        for cand in ("geoid", "GEOID", "cell_id", "id"):
            if cand in props:
                key_raw = str(props[cand])
                break
        if not key_raw:
            features_out.append(feat)
            continue

        # normalize + zfill
        key_norm = "".join(ch for ch in key_raw if ch.isdigit())
        if target_len:
            key_norm = key_norm.zfill(target_len)

        # eşleşme varsa metrikleri enjekte et
        if key_norm in dmap.index:
            row = dmap.loc[key_norm]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            disp = props.get("GEOID") or props.get("geoid") or props.get("cell_id") or props.get("id")
            props.update({
                "display_id": disp,
                "risk_score_daily": float(row["risk_score_daily"]),
                "risk_level": row["risk_level"],
                "risk_score_txt": f"{float(row['risk_score_daily']):.4f}",
            })

        features_out.append({**feat, "properties": props})

    return {**geojson_dict, "features": features_out}

@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_geojson_from_artifact(path_in_zip: str) -> dict:
    """Aynı Actions artifact (sf-crime-parquet.zip) içinde verilen yolu arar ve GeoJSON döndürür."""
    zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        # tam/sondan eşleşme
        candidates = [n for n in memlist if n.endswith("/" + path_in_zip) or n.endswith(path_in_zip)]
        if not candidates:
            st.warning(f"GeoJSON '{path_in_zip}' ZIP içinde yok. İlk 20 dosya: {memlist[:20]}")
            return {}
        with zf.open(candidates[0]) as f:
            return json.load(io.TextIOWrapper(f, encoding="utf-8"))

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
        opacity=0.6,
    )

    tooltip = {
        "text": "ID: {display_id}\nRisk: {risk_level}\nSkor: {risk_score_txt}",
        "style": {"backgroundColor": "#262730", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

# =========================
# UI Akışı
# =========================
st.sidebar.header("GitHub Artifact")
refresh = st.sidebar.button("Veriyi Yenile (artifact'i tazele)")

# Risk verisini çek
try:
    if refresh:
        # cache temizle
        fetch_latest_artifact_zip.clear()
        read_risk_from_artifact.clear()
        fetch_geojson_from_artifact.clear()
    risk_df = read_risk_from_artifact()
except Exception as e:
    st.error(f"Artifact indirilemedi/okunamadı: {e}")
    st.stop()

# Günlük ortalama + tarih seçimi
risk_daily = daily_average(risk_df)
dates = sorted(risk_daily["date"].unique())
sel_date = st.sidebar.selectbox(
    "Gün seçin",
    dates,
    index=len(dates) - 1 if dates else 0,
    format_func=lambda d: str(d),
)

one_day = classify_quantiles(risk_daily, sel_date)

# Eşikler (Q25/Q50/Q75)
if not one_day.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Q25", f"{one_day['q25'].iloc[0]:.4f}")
    c2.metric("Q50", f"{one_day['q50'].iloc[0]:.4f}")
    c3.metric("Q75", f"{one_day['q75'].iloc[0]:.4f}")

# GeoJSON yolunu artifact içinden seçtir
st.sidebar.header("Harita Sınırları (GeoJSON)")
geojson_path_in_zip = st.sidebar.text_input(
    "Artifact ZIP içindeki GeoJSON yolu",
    value=GEOJSON_IN_ZIP_PATH_DEFAULT,
    help="Örn: data/sf_cells.geojson",
)

# GeoJSON'u artifact'ten al, risk metriklerini enjekte et ve haritayı çiz
geojson = fetch_geojson_from_artifact(geojson_path_in_zip)
st.subheader(f"Harita — {sel_date}")
if not geojson:
    st.warning("GeoJSON artifact içinde bulunamadı. Yan paneldeki yolu kontrol edin.")
else:
    enriched = inject_properties(geojson, one_day)
    make_map(enriched)

# Teşhis bilgileri
with st.expander("Teşhis (Artifact İçeriği)"):
    try:
        zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            mem = zf.namelist()
        st.write({
            "artifact": f"{OWNER}/{REPO} → {ARTIFACT_NAME}.zip",
            "geojson_path_in_zip": geojson_path_in_zip,
            "zip_manifest_head": mem[:20],
        })
    except Exception as e:
        st.write({"artifact_error": str(e)})

# Tablo + indirme (ekrandaki DF'ten CSV üretimi)
st.subheader("Seçilen Gün Tablosu")
st.dataframe(
    one_day.drop(columns=["q25", "q50", "q75"], errors="ignore").sort_values(
        "risk_score_daily", ascending=False
    ),
    use_container_width=True,
)
csv = one_day.drop(columns=["q25", "q50", "q75"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    label="Günlük tabloyu CSV indir",
    data=csv,
    file_name=f"risk_daily_{sel_date}.csv",
    mime="text/csv",
)

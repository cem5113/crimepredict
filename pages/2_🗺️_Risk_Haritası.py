# pages/2_🗺️_Risk_Haritası.py
import io, os, json, zipfile
from datetime import date
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests
from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

st.set_page_config(page_title="🗺️ Risk Haritası (Günlük)", layout="wide")
st.title("🕒 Anlık Suç Risk Haritası")
st.markdown(
    "<p style='font-size:14px; font-style:italic;'>Bu harita, en güncel veriler üzerinden her GEOID bazında suç gerçekleşme olasılıklarını göstermektedir.</p>",
    unsafe_allow_html=True
)

# ── Ayarlar
cfg = st.secrets if hasattr(st, "secrets") else {}
OWNER = cfg.get("artifact_owner", "cem5113")
REPO = cfg.get("artifact_repo", "crime_prediction_data")
ARTIFACT_NAME = cfg.get("artifact_name", "sf-crime-parquet")
EXPECTED_PARQUET = "risk_hourly.parquet"

GEOJSON_PATH_LOCAL_DEFAULT = cfg.get("geojson_path", "data/sf_cells.geojson")
RAW_GEOJSON_OWNER = cfg.get("geojson_owner", "cem5113")
RAW_GEOJSON_REPO = cfg.get("geojson_repo", "crimepredict")

# ── Token çözümleme (env > secrets)
def resolve_github_token() -> str | None:
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        return tok
    for k in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if k in st.secrets and st.secrets[k]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[k])  # env'e yazarak tek kaynak
                return os.environ["GITHUB_TOKEN"]
        except Exception:
            pass
    return None

def gh_headers() -> dict:
    hdrs = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    return hdrs

@st.cache_data(show_spinner=True, ttl=15*60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")
    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    url = cand[0].get("archive_download_url")
    if not url:
        raise RuntimeError("archive_download_url bulunamadı")
    r2 = requests.get(url, headers=gh_headers(), timeout=60)
    r2.raise_for_status()
    return r2.content

@st.cache_data(show_spinner=True, ttl=15*60)
def read_risk_from_artifact(owner: str, repo: str, artifact_name: str) -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} yok. Örnek içerik: {memlist[:15]}")
        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)

    df.columns = [c.strip().lower() for c in df.columns]

    # risk_score kolonunu esnek eşle
    if "risk_score" not in df.columns:
        for alt in ("risk", "score", "prob", "probability"):
            if alt in df.columns:
                df = df.rename(columns={alt: "risk_score"})
                break
    if "risk_score" not in df.columns:
        raise ValueError("Beklenen kolon yok: risk_score")

    # GEOID üret
    if "geoid" not in df.columns:
        for alt in ("cell_id", "geoid10", "geoid11", "geoid_10", "geoid_11", "id"):
            if alt in df.columns:
                df["geoid"] = df[alt]
                break
    if "geoid" not in df.columns:
        raise ValueError("Beklenen kolon yok: geoid / cell_id")

    df["geoid"] = (
        df["geoid"].astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(11)
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        raise ValueError("Beklenen kolon yok: date")

    return df

def daily_average(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    needed = {"geoid", "date", "risk_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {', '.join(sorted(missing))}")
    return (
        df.groupby(["geoid", "date"], as_index=False)["risk_score"]
        .mean()
        .rename(columns={"risk_score": "risk_score_daily"})
    )

def only_digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

def classify_quantiles(daily_df: pd.DataFrame, day: date) -> pd.DataFrame:
    one = daily_df[daily_df["date"] == day].copy()
    if one.empty:
        return one
    q25, q50, q75 = one["risk_score_daily"].quantile([0.25, 0.5, 0.75]).tolist()

    def lab(x: float) -> str:
        if x <= q25: return "low"
        elif x <= q50: return "medium"
        elif x <= q75: return "high"
        return "critical"

    one["risk_level"] = one["risk_score_daily"].apply(lab)
    one["q25"], one["q50"], one["q75"] = q25, q50, q75
    return one

@st.cache_data(show_spinner=True, ttl=60*60)
def fetch_geojson_smart(path_local: str, path_in_zip: str, raw_owner: str, raw_repo: str) -> dict:
    # 1) Local
    try:
        if os.path.exists(path_local):
            with open(path_local, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass

    # 2) Artifact (varsa)
    try:
        zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            memlist = zf.namelist()
            candidates = [n for n in memlist if n.endswith("/" + path_in_zip) or n.endswith(path_in_zip)]
            if candidates:
                with zf.open(candidates[0]) as f:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception:
        pass

    # 3) Raw GitHub (public)
    try:
        raw = f"https://raw.githubusercontent.com/{raw_owner}/{raw_repo}/main/{path_local}"
        r = requests.get(raw, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    return {}

def inject_properties(geojson_dict: dict, day_df: pd.DataFrame) -> dict:
    if not geojson_dict or day_df.empty:
        return geojson_dict

    df = day_df.copy()
    df["geoid_digits"] = df["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
    dmap = (
        df.groupby("geoid_digits", as_index=True)["risk_score_daily"]
        .mean()
        .to_frame()
        .rename_axis("match_key")
    )

    feats = geojson_dict.get("features", [])
    enriched = 0

    q25 = float(df["risk_score_daily"].quantile(0.25))
    q50 = float(df["risk_score_daily"].quantile(0.50))
    q75 = float(df["risk_score_daily"].quantile(0.75))
    EPS = 1e-12

    COLOR_MAP = {
        "zero": [200, 200, 200],
        "low": [56, 168, 0],
        "medium": [255, 221, 0],
        "high": [255, 140, 0],
        "critical": [204, 0, 0],
    }

    out = []
    for feat in feats:
        props = (feat.get("properties") or {}).copy()
        raw = next((props[k] for k in ("geoid", "GEOID", "cell_id", "id") if k in props), None)
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v
                    break

        props.setdefault("display_id", str(raw or ""))

        key = only_digits(raw)[:11] if raw is not None else ""
        lvl = None
        if key and key in dmap.index:
            val = float(dmap.loc[key, "risk_score_daily"])
            props["risk_score_daily"] = val
            props["risk_score_txt"] = f"{val:.4f}"
            if abs(val) <= EPS: lvl = "zero"
            elif val <= q25:   lvl = "low"
            elif val <= q50:   lvl = "medium"
            elif val <= q75:   lvl = "high"
            else:              lvl = "critical"
            enriched += 1

        if lvl is None:
            lvl = props.get("risk_level", "zero")
        props["risk_level"] = lvl
        props["fill_color"] = COLOR_MAP.get(lvl, [220, 220, 220])

        out.append({**feat, "properties": props})
    return {**geojson_dict, "features": out}

def make_map(geojson_enriched: dict):
    if not geojson_enriched:
        st.info("Haritayı görmek için GeoJSON bulunamadı.")
        return
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_enriched,
        stroked=True,
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )
    tooltip = {
        "html": "<b>GEOID:</b> {display_id}<br/><b>Risk:</b> {risk_level}<br/><b>Skor:</b> {risk_score_txt}",
        "style": {"backgroundColor": "#262730", "color": "white"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.markdown(
        """
        <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:14px;">
          <div><span style="display:inline-block;width:14px;height:14px;background:#C8C8C8;border:1px solid #666;"></span> zero</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:rgb(56,168,0);border:1px solid #666;"></span> low</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:rgb(255,221,0);border:1px solid #666;"></span> medium</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:rgb(255,140,0);border:1px solid #666;"></span> high</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:rgb(204,0,0);border:1px solid #666;"></span> critical</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── UI
TOKEN = resolve_github_token()

st.sidebar.header("GitHub Artifact")
with st.sidebar.expander("🔐 Token Durumu", expanded=TOKEN is None):
    st.write("Env GITHUB_TOKEN:", "✅" if os.getenv("GITHUB_TOKEN") else "❌")
    has_secret = False
    try:
        has_secret = any(k in st.secrets for k in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"))
    except Exception:
        pass
    st.write("Secrets'ta Token:", "✅" if has_secret else "❌")

refresh = st.sidebar.button("Veriyi Yenile (artifact'i tazele)")
if refresh:
    fetch_latest_artifact_zip.clear()
    read_risk_from_artifact.clear()
    fetch_geojson_smart.clear()

try:
    if not TOKEN:
        st.error("GitHub token yok. `st.secrets['github_token']` veya GITHUB_TOKEN env ayarlayın.")
        st.stop()
    risk_df = read_risk_from_artifact(OWNER, REPO, ARTIFACT_NAME)
except Exception as e:
    st.error(f"Artifact indirilemedi/okunamadı: {e}")
    st.stop()

risk_daily = daily_average(risk_df)
dates = sorted(risk_daily["date"].unique())
sel_date = st.sidebar.selectbox("Gün seçin", dates, index=len(dates) - 1, format_func=str) if dates else None
one_day = classify_quantiles(risk_daily, sel_date) if sel_date else pd.DataFrame()

if not one_day.empty:
    q25 = one_day['q25'].iloc[0] * 100
    q50 = one_day['q50'].iloc[0] * 100
    q75 = one_day['q75'].iloc[0] * 100

    st.markdown(
        f"""
        <div style="font-size:16px; font-weight:600;">Risk Sınıflandırması</div>
        <div style="font-size:14px; margin-top:4px;">
            🟢 <b>Düşük Riskli:</b> &lt; %{q25:.2f}<br>
            🟡 <b>Orta Riskli:</b> &gt; %{q25:.2f}<br>
            🟠 <b>Riskli:</b> &gt; %{q50:.2f}<br>
            🔴 <b>Yüksek Riskli:</b> &gt; %{q75:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    gj = fetch_geojson_smart(
        GEOJSON_PATH_LOCAL_DEFAULT,
        GEOJSON_PATH_LOCAL_DEFAULT,
        RAW_GEOJSON_OWNER,
        RAW_GEOJSON_REPO
    )
    enriched = inject_properties(gj, one_day)
    make_map(enriched)
else:
    st.info("Seçili tarih için veri yok.")

show_last_update_badge(data_upto=None, model_version=MODEL_VERSION, last_train=MODEL_LAST_TRAIN)

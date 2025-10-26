# pages/2_🗺️_Risk_Haritası.py
import io, os, json, zipfile
from typing import Optional
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
    "<p style='font-size:14px; font-style:italic;'>Bu harita, en güncel veriler üzerinden her GEOID bazında 24 saat içerisinde suç gerçekleşme olasılıklarını göstermektedir. Harita, model tarafından son güncellenen tahmin skorları üzerinden oluşturulmuştur. Gerçek suç verileriyle birebir eşleşmeyebilir.</p>",
    unsafe_allow_html=True
)

# ── Ayarlar / varsayılanlar
cfg = getattr(st, "secrets", {}) if hasattr(st, "secrets") else {}
# Eğer DATA_REPO verilmişse OWNER/REPO’yu oradan ayıkla
DATA_REPO = cfg.get("DATA_REPO", os.getenv("DATA_REPO", "cem5113/crime_prediction_data"))
DATA_BRANCH = cfg.get("DATA_BRANCH", os.getenv("DATA_BRANCH", "main"))
if "/" in DATA_REPO:
    OWNER, REPO = DATA_REPO.split("/", 1)
else:
    OWNER, REPO = cfg.get("artifact_owner", "cem5113"), cfg.get("artifact_repo", "crime_prediction_data")

ARTIFACT_NAME = cfg.get("artifact_name", "sf-crime-parquet")
EXPECTED_PARQUET = "risk_hourly.parquet"

# Release fallback için (public)
ASSET_ZIP_1 = cfg.get("ASSET_ZIP_1", os.getenv("ASSET_ZIP_1", "sf-crime-parquet.zip"))
ASSET_DIR_1 = cfg.get("ASSET_DIR_1", os.getenv("ASSET_DIR_1", "sf-crime-parquet"))

GEOJSON_PATH_LOCAL_DEFAULT = cfg.get("geojson_path", "data/sf_cells.geojson")
RAW_GEOJSON_OWNER = cfg.get("geojson_owner", "cem5113")
RAW_GEOJSON_REPO = cfg.get("geojson_repo", "crimepredict")

# ── Yardımcılar: token çözümleme, başlıklar, maskeleme
def _secret_lookup_in_secrets(keys=("GITHUB_TOKEN", "GH_TOKEN", "github_token")) -> Optional[str]:
    try:
        sec = getattr(st, "secrets", None)
        if not sec:
            return None

        # 1) düz anahtarlar
        for k in keys:
            v = sec.get(k)
            if v:
                v = str(v).strip()
                if v:
                    return v
        # 2) olası alt sözlükler
        for bucket in ("github", "tokens", "secrets", "config"):
            sub = sec.get(bucket)
            if isinstance(sub, dict):
                for k in list(keys) + [k.lower() for k in keys]:
                    v = sub.get(k)
                    if v:
                        v = str(v).strip()
                        if v:
                            return v
    except Exception:
        pass
    return None

def resolve_github_token() -> Optional[str]:
    tok = (os.getenv("GITHUB_TOKEN")
           or os.getenv("GH_TOKEN")
           or os.getenv("github_token"))
    if not tok:
        tok = _secret_lookup_in_secrets()
    if tok:
        tok = str(tok).strip()
        if tok:
            os.environ["GITHUB_TOKEN"] = tok  # tek kaynak
            return tok
    return None

def gh_headers() -> dict:
    hdrs = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN") or resolve_github_token()
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    return hdrs

def _mask(tok: Optional[str]) -> str:
    if not tok: return "—"
    t = str(tok)
    if len(t) <= 12: return t[:3] + "…" + t[-2:]
    return t[:6] + "…" + t[-4:]

# ── Artifact / Release indirme
@st.cache_data(show_spinner=True, ttl=15*60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    """
    1) Token varsa: Actions Artifacts (en güncel).
    2) Yoksa veya başarısızsa: Releases/latest/download/{ASSET_ZIP_1} (public).
    """
    tok = resolve_github_token()
    if tok:
        try:
            base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
            r = requests.get(base, headers=gh_headers(), timeout=30)
            r.raise_for_status()
            items = r.json().get("artifacts", [])
            cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
            if cand:
                cand.sort(key=lambda x: x.get("updated_at", "") or "", reverse=True)
                url = cand[0].get("archive_download_url")
                if url:
                    r2 = requests.get(url, headers=gh_headers(), timeout=60)
                    r2.raise_for_status()
                    return r2.content
        except Exception as e:
            st.warning(f"Artifact API erişimi başarısız; Release fallback deneniyor… ({e})")

    # Release fallback (public)
    rel_url = f"https://github.com/{owner}/{repo}/releases/latest/download/{ASSET_ZIP_1}"
    r3 = requests.get(rel_url, timeout=60)
    if r3.status_code == 200 and r3.content:
        return r3.content
    raise FileNotFoundError(
        f"İndirilemedi: Artifact API ya da Release asset (denenen: {rel_url})."
    )

@st.cache_data(show_spinner=True, ttl=15*60)
def read_risk_from_artifact(owner: str, repo: str, artifact_name: str) -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()

        # Önce doğrudan EXPECTED_PARQUET (risk_hourly.parquet)
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]

        # Bulunamazsa ASSET_DIR_1 altında ara (ör. sf-crime-parquet/risk_hourly.parquet)
        if not matches and ASSET_DIR_1:
            matches = [n for n in memlist if n.endswith(f"{ASSET_DIR_1}/{EXPECTED_PARQUET}")]

        if not matches:
            sample = ", ".join(memlist[:10])
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} bulunamadı. Örnek içerik: [{sample}]")

        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)

    # kolon normalizasyonu
    df.columns = [c.strip().lower() for c in df.columns]

    # risk_score esnek eşle
    if "risk_score" not in df.columns:
        for alt in ("risk", "score", "prob", "probability"):
            if alt in df.columns:
                df = df.rename(columns={alt: "risk_score"})
                break
    if "risk_score" not in df.columns:
        raise ValueError("Beklenen kolon yok: risk_score")

    # geoid türetme
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

# ── Dönüşümler ve harita
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
        if x <= q25: return "düşük riskli"
        elif x <= q50: return "orta riskli"
        elif x <= q75: return "riskli"
        return "yüksek riskli"

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
    q25 = float(df["risk_score_daily"].quantile(0.25))
    q50 = float(df["risk_score_daily"].quantile(0.50))
    q75 = float(df["risk_score_daily"].quantile(0.75))
    EPS = 1e-12
    COLOR_MAP = {
        "çok düşük riskli": [200, 200, 200],
        "düşük riskli":     [56, 168, 0],
        "orta riskli":      [255, 221, 0],
        "riskli":           [255, 140, 0],
        "yüksek riskli":    [204, 0, 0],
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
            disp = min(val, 0.999)
            props["risk_score_txt"] = f"{disp:.3f}"
            if abs(val) <= EPS: lvl = "çok düşük riskli"
            elif val <= q25:   lvl = "düşük riskli"
            elif val <= q50:   lvl = "orta riskli"
            elif val <= q75:   lvl = "riskli"
            else:              lvl = "yüksek riskli"
        if lvl is None:
            lvl = props.get("risk_level", "çok düşük riskli")
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

# ── UI / Diagnostik
TOKEN = resolve_github_token()

st.sidebar.header("GitHub Bağlantı")
with st.sidebar.expander("🔐 Token Durumu", expanded=TOKEN is None):
    env_tok = os.getenv("GITHUB_TOKEN")
    st.write("ENV GITHUB_TOKEN:", "✅" if env_tok else "❌")
    try:
        sec = getattr(st, "secrets", None)
        s_flat = bool(sec and any(k in sec and sec[k] for k in ("GITHUB_TOKEN","GH_TOKEN","github_token")))
        s_nested = bool(
            sec and any(isinstance(sec.get(b), dict) and any(x in sec[b] for x in ("GITHUB_TOKEN","GH_TOKEN","github_token"))
                        for b in ("github","tokens","secrets","config"))
        )
        st.write("secrets (düz):", "✅" if s_flat else "❌")
        st.write("secrets (iç içe):", "✅" if s_nested else "❌")
        st.write("Token (maskeli):", _mask(env_tok))
    except Exception:
        st.write("secrets erişimi: ❌ (lokal olabilir)")

refresh = st.sidebar.button("Veriyi Yenile (artefact/asset)")
if refresh:
    fetch_latest_artifact_zip.clear()
    read_risk_from_artifact.clear()
    fetch_geojson_smart.clear()

# ── Veri yükleme
try:
    if not TOKEN:
        st.warning("GitHub token bulunamadı — Actions artifact yerine Release yedeği deneniyor…")
    risk_df = read_risk_from_artifact(OWNER, REPO, ARTIFACT_NAME)
except Exception as e:
    st.error(f"Veri indirilemedi: {e}")
    st.stop()

# ── Günlük ortalama ve harita
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
        <div style="font-size:17px; margin-top:10px; line-height:1.6;">
            🟢 <b>Düşük Riskli:</b> &lt; %{q25:.2f}<br>
            🟡 <b>Orta Riskli:</b> &gt; %{q25:.2f}<br>
            🟠 <b>Riskli:</b> &gt; %{q50:.2f}<br>
            🔴 <b>Yüksek Riskli:</b> &gt; %{q75:.2f}
        </div>
        <div style="font-size:13px; font-style:italic; color:#666; margin-top:8px;">
            Bu sınıflandırma, GEOID alanlarını dört risk seviyesine ayırmak için belirlenen günlük risk skorlarından elde edilen değişken eşiklere dayanmaktadır.
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

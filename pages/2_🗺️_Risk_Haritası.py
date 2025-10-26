# pages/2_🗺️_Risk_Haritası.py
from __future__ import annotations

import io, os, json, zipfile
from typing import Optional, Dict, Any, List
from datetime import date
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

# ---- Meta/rozet ----
from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

# ─────────────────────────────────────────────────────────────────────────────
# 0) Genel sayfa ayarları
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🗺️ Risk Haritası (Günlük)", layout="wide")
st.title("🕒 Anlık Suç Risk Haritası")
st.markdown(
    "<p style='font-size:14px; font-style:italic;'>Bu harita, en güncel veriler üzerinden her GEOID bazında 24 saat içerisinde suç gerçekleşme olasılıklarını göstermektedir. Harita, model tarafından son güncellenen tahmin skorları üzerinden oluşturulmuştur. Gerçek suç verileriyle birebir eşleşmeyebilir.</p>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Konfigürasyon çözümü (Secrets > Env > Defaults)
# ─────────────────────────────────────────────────────────────────────────────
def _secrets_dict() -> Dict[str, Any]:
    try:
        return getattr(st, "secrets", {}) or {}
    except Exception:
        return {}

def _pick(sec: Dict[str, Any], *keys: str, default=None):
    # secrets doğrudan → env → nested buckets
    for k in keys:
        v = (sec.get(k) if hasattr(sec, "get") else None) or os.getenv(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    # nested dictionaries (opsiyonel)
    for bucket in ("github", "tokens", "secrets", "config"):
        sub = sec.get(bucket)
        if isinstance(sub, dict):
            for k in list(keys) + [k.lower() for k in keys]:
                v = sub.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
    return default

def get_cfg() -> Dict[str, Any]:
    sec = _secrets_dict()

    # Token
    tok = _pick(sec, "GITHUB_TOKEN", "GH_TOKEN", "github_token", default=None)
    if tok:
        os.environ["GITHUB_TOKEN"] = tok  # sonraki fonk. tek kaynaktan okusun

    # Veri deposu
    data_repo   = _pick(sec, "DATA_REPO",   default="cem5113/crime_prediction_data")
    data_branch = _pick(sec, "DATA_BRANCH", default="main")

    # OWNER/REPO çıkar
    if "/" in (data_repo or ""):
        owner, repo = data_repo.split("/", 1)
    else:
        owner = _pick(sec, "artifact_owner", default="cem5113")
        repo  = _pick(sec, "artifact_repo",  default="crime_prediction_data")

    # Release fallback varlık adları
    asset_zip_1 = _pick(sec, "ASSET_ZIP_1", default="sf-crime-parquet.zip")
    asset_dir_1 = _pick(sec, "ASSET_DIR_1", default="sf-crime-parquet")

    # GeoJSON kaynakları
    geojson_local = _pick(sec, "geojson_path", default="data/sf_cells.geojson")
    raw_gj_owner  = _pick(sec, "geojson_owner", default="cem5113")
    raw_gj_repo   = _pick(sec, "geojson_repo",  default="crimepredict")

    # Artifact adı (Actions)
    artifact_name = _pick(sec, "artifact_name", default="sf-crime-parquet")

    return {
        "TOKEN": tok,
        "DATA_REPO": data_repo,
        "DATA_BRANCH": data_branch,
        "OWNER": owner,
        "REPO": repo,
        "ASSET_ZIP_1": asset_zip_1,
        "ASSET_DIR_1": asset_dir_1,
        "GEOJSON_PATH": geojson_local,
        "RAW_GJ_OWNER": raw_gj_owner,
        "RAW_GJ_REPO": raw_gj_repo,
        "ARTIFACT_NAME": artifact_name,
    }

CFG = get_cfg()

# Beklenen dosya adı (ZIP içinde)
EXPECTED_PARQUET = "risk_hourly.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# 2) GitHub HTTP yardımcıları
# ─────────────────────────────────────────────────────────────────────────────
def gh_headers() -> Dict[str, str]:
    hdrs = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    return hdrs

def _mask(tok: Optional[str]) -> str:
    if not tok: return "—"
    t = str(tok)
    if len(t) <= 10: return t[:3] + "…" + t[-2:]
    return t[:6] + "…" + t[-4:]

# ─────────────────────────────────────────────────────────────────────────────
# 3) Artifact / Release indirme + okuma
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True, ttl=15*60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str, asset_zip_1: str) -> bytes:
    """
    1) Token varsa: Actions Artifacts (en güncel).
    2) Yoksa/başarısızsa: Releases/latest/download/{asset_zip_1} (public).
    """
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        try:
            base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts?per_page=100"
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

    # Release fallback
    rel_url = f"https://github.com/{owner}/{repo}/releases/latest/download/{asset_zip_1}"
    r3 = requests.get(rel_url, timeout=60)
    if r3.status_code == 200 and r3.content:
        return r3.content
    raise FileNotFoundError(
        f"İndirilemedi: Artifact API ya da Release asset (denenen: {rel_url})."
    )

@st.cache_data(show_spinner=True, ttl=15*60)
def read_risk_from_artifact(owner: str, repo: str, artifact_name: str, asset_zip_1: str) -> pd.DataFrame:
    """
    ZIP içinden EXPECTED_PARQUET (risk_hourly.parquet) dosyasını bulup okur.
    """
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name, asset_zip_1)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()

        # 1) Doğrudan dosya adı
        matches = [n for n in memlist if n.endswith("/" + EXPECTED_PARQUET) or n.endswith(EXPECTED_PARQUET)]
        # 2) Asset dizini altında (örn. sf-crime-parquet/risk_hourly.parquet)
        if not matches and CFG["ASSET_DIR_1"]:
            matches = [n for n in memlist if n.endswith(f"{CFG['ASSET_DIR_1'].rstrip('/')}/{EXPECTED_PARQUET}")]

        if not matches:
            sample = ", ".join(memlist[:10])
            raise FileNotFoundError(f"Zip içinde {EXPECTED_PARQUET} bulunamadı. Örnek içerik: [{sample}]")

        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)

    # Kolon normalizasyonu
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

    # tarih alanı
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        raise ValueError("Beklenen kolon yok: date")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4) Dönüşümler, sınıflandırma, GeoJSON
# ─────────────────────────────────────────────────────────────────────────────
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

def _only_digits(s: Any) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

@st.cache_data(show_spinner=True, ttl=60*60)
def fetch_geojson_smart(path_local: str, path_in_zip: str, raw_owner: str, raw_repo: str) -> dict:
    # 1) Local
    try:
        if os.path.exists(path_local):
            with open(path_local, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # 2) Artifact (ZIP içinde)
    try:
        zip_bytes = fetch_latest_artifact_zip(CFG["OWNER"], CFG["REPO"], CFG["ARTIFACT_NAME"], CFG["ASSET_ZIP_1"])
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            memlist = zf.namelist()
            candidates = [n for n in memlist if n.endswith("/" + path_in_zip) or n.endswith(path_in_zip)]
            if candidates:
                with zf.open(candidates[0]) as f:
                    return json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception:
        pass
    # 3) Raw GitHub
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
        raw = next((props.get(k) for k in ("geoid", "GEOID", "cell_id", "id") if k in props), None)
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v
                    break
        props.setdefault("display_id", str(raw or ""))
        key = _only_digits(raw)[:11] if raw is not None else ""
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

# ─────────────────────────────────────────────────────────────────────────────
# 5) Sidebar: GitHub bağlantı ve refresh
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("GitHub Bağlantı")
with st.sidebar.expander("🔐 Token Durumu", expanded=CFG["TOKEN"] is None):
    env_tok = os.getenv("GITHUB_TOKEN")
    st.write("ENV GITHUB_TOKEN:", "✅" if env_tok else "❌")
    sec = _secrets_dict()
    s_flat = bool(sec and any(k in sec and sec[k] for k in ("GITHUB_TOKEN","GH_TOKEN","github_token")))
    s_nested = bool(
        sec and any(isinstance(sec.get(b), dict) and any(x in sec[b] for x in ("GITHUB_TOKEN","GH_TOKEN","github_token"))
                    for b in ("github","tokens","secrets","config"))
    )
    st.write("secrets (düz):", "✅" if s_flat else "❌")
    st.write("secrets (iç içe):", "✅" if s_nested else "❌")
    st.write("Token (maskeli):", _mask(env_tok))

refresh = st.sidebar.button("Veriyi Yenile (artifact/asset)")
if refresh:
    fetch_latest_artifact_zip.clear()
    read_risk_from_artifact.clear()
    fetch_geojson_smart.clear()
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 6) Veri yükleme ve görselleştirme
# ─────────────────────────────────────────────────────────────────────────────
try:
    risk_df = read_risk_from_artifact(
        CFG["OWNER"], CFG["REPO"], CFG["ARTIFACT_NAME"], CFG["ASSET_ZIP_1"]
    )
except Exception as e:
    st.error(f"Veri indirilemedi: {e}")
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
        <div style="font-size:17px; margin-top:10px; line-height:1.6;">
            🟢 <b>Düşük Riskli:</b> &lt; %{q25:.2f}<br>
            🟡 <b>Orta Riskli:</b> &gt; %{q25:.2f}<br>
            🟠 <b>Riskli:</b> &gt; %{q50:.2f}<br>
            🔴 <b>Yüksek Riskli:</b> &gt; %{q75:.2f}
        </div>
        <div style="font-size:13px; font-style:italic; color:#666; margin-top:8px;">
            Bu sınıflandırma, GEOID alanlarını dört risk seviyesine ayırmak için günlük risk skorlarının kendi çeyreklerine göre dinamik eşiklerle belirlenir.
        </div>
        """,
        unsafe_allow_html=True
    )

    gj = fetch_geojson_smart(
        CFG["GEOJSON_PATH"],               # local path
        CFG["GEOJSON_PATH"],               # path in zip (aynı isimle)
        CFG["RAW_GJ_OWNER"],               # raw fallback owner
        CFG["RAW_GJ_REPO"],                # raw fallback repo
    )
    enriched = inject_properties(gj, one_day)
    make_map(enriched)
else:
    st.info("Seçili tarih için veri yok.")

show_last_update_badge(data_upto=None, model_version=MODEL_VERSION, last_train=MODEL_LAST_TRAIN)

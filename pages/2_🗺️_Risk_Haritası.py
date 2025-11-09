# pages/2_🗺️_Risk_Haritası.py — ANLIK görünüm (CSV: risk_hourly_grid_full_labeled.csv)

import io, os, json, zipfile
from typing import Optional, Iterable
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

# ──────────────────────────────────────────────────────────────────────────────
# SAYFA BAŞLIĞI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🕒 Anlık Suç Risk Haritası")
st.caption(
    "Bu sayfa açıldığı anda (SF yerel saatine göre) geçerli **hour_range** otomatik seçilir "
    "ve sadece o dilime ait riskler gösterilir. Veriler doğrudan CSV’den okunur; "
    "**risk_level** yeniden hesaplanmaz."
)

# ──────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ──────────────────────────────────────────────────────────────────────────────
cfg = st.secrets if hasattr(st, "secrets") else {}
OWNER = cfg.get("artifact_owner", "cem5113")
REPO = cfg.get("artifact_repo", "crime_prediction_data")
ARTIFACT_NAME = cfg.get("artifact_name", "sf-crime-pipeline-output")  # Actions artifact adı
CSV_TARGET_NAME = "risk_hourly_grid_full_labeled.csv"                  # Zip içindeki dosya
TARGET_TZ = cfg.get("risk_timezone", "America/Los_Angeles")            # Anlık saat TZ

# GeoJSON (önce local → artifact → raw github)
GEOJSON_PATH_LOCAL_DEFAULT = cfg.get("geojson_path", "data/sf_cells.geojson")
RAW_GEOJSON_OWNER = cfg.get("geojson_owner", "cem5113")
RAW_GEOJSON_REPO  = cfg.get("geojson_repo",  "crimepredict")

# ──────────────────────────────────────────────────────────────────────────────
# GITHUB API YARDIMCILARI
# ──────────────────────────────────────────────────────────────────────────────
def resolve_github_token() -> Optional[str]:
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        return tok
    for k in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if k in st.secrets and st.secrets[k]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[k])
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

@st.cache_data(show_spinner=True, ttl=15 * 60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> bytes:
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        cand = [a for a in items if a.get("name","").startswith(artifact_name) and not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError(f"Artifact bulunamadı: {artifact_name}")
    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    url = cand[0].get("archive_download_url")
    if not url:
        raise RuntimeError("archive_download_url bulunamadı")
    r2 = requests.get(url, headers=gh_headers(), timeout=60)
    r2.raise_for_status()
    return r2.content

# ──────────────────────────────────────────────────────────────────────────────
# CSV OKU (HEDEF: risk_hourly_grid_full_labeled.csv)
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = {
    "geoid", "hour_range", "risk_score", "risk_level",
    "expected_count",
    "top1_category","top1_prob","top1_expected",
    "top2_category","top2_prob","top2_expected",
    "top3_category","top3_prob","top3_expected",
}

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # GEOID eşadlıları
    if "geoid" not in df.columns:
        for alt in ("cell_id","geoid11","geoid_11","geoid10","geoid_10","id"):
            if alt in df.columns:
                df.rename(columns={alt:"geoid"}, inplace=True); break
    # Risk skoru eşadlıları (gerekirse)
    if "risk_score" not in df.columns:
        for alt in ("risk","score","prob","probability"):
            if alt in df.columns:
                df.rename(columns={alt:"risk_score"}, inplace=True); break
    # GEOID 11 haneye zorla
    if "geoid" in df.columns:
        df["geoid"] = (
            df["geoid"].astype(str)
            .str.replace(r"\D","", regex=True)
            .str.zfill(11)
        )
    # hour_range stringe zorla
    if "hour_range" in df.columns:
        df["hour_range"] = df["hour_range"].astype(str)
    return df

def _has_required_cols(df: pd.DataFrame) -> bool:
    return REQUIRED_COLS.issubset(set(df.columns))

@st.cache_data(show_spinner=True, ttl=15 * 60)
def load_hourly_csv(owner: str, repo: str, artifact_name: str, target_csv_name: str) -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Zip içinden isim/son ek eşleşmesi
        cand = [n for n in zf.namelist() if n.endswith("/"+target_csv_name) or n.endswith(target_csv_name)]
        if not cand:
            raise FileNotFoundError(f"Zip içinde {target_csv_name} bulunamadı.")
        with zf.open(cand[0]) as f:
            df = pd.read_csv(f)
    df = _normalize_cols(df)
    if not _has_required_cols(df):
        missing = REQUIRED_COLS - set(df.columns)
        raise ValueError(f"CSV zorunlu kolonları eksik: {', '.join(sorted(missing))}")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# GEOJSON (local → artifact → raw github)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_geojson_smart(path_local: str, path_in_zip: str, raw_owner: str, raw_repo: str) -> dict:
    # 1) Local
    try:
        if os.path.exists(path_local):
            with open(path_local, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # 2) Artifact
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
    # 3) Raw GitHub
    try:
        raw = f"https://raw.githubusercontent.com/{raw_owner}/{raw_repo}/main/{path_local}"
        r = requests.get(raw, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# ──────────────────────────────────────────────────────────────────────────────
# HOUR-RANGE SEÇİMİ (ANLIK) — sadece CSV’deki etiketlerden biri
# ──────────────────────────────────────────────────────────────────────────────
def parse_range_token(tok: str) -> Optional[tuple[int,int]]:
    if not isinstance(tok, str) or "-" not in tok:
        return None
    a, b = tok.split("-", 1)
    try:
        s = int(a.strip())
        e = int(b.strip())
        s = max(0, min(23, s))
        e = 24 if e == 24 else max(1, min(24, e))
        return (s, e)
    except Exception:
        return None

def hour_to_bucket(hour: int, candidates: Iterable[str]) -> Optional[str]:
    parsed = []
    for c in candidates:
        rng = parse_range_token(str(c))
        if rng:
            parsed.append((c, rng[0], rng[1]))
    # 1) Doğrudan kapsama
    for label, s, e in parsed:
        if s <= hour < (e if e < 24 else 24):
            return label
    # 2) Sarma aralık
    for label, s, e in parsed:
        if s > e and (hour >= s or hour < e):
            return label
    # 3) Fallback
    if parsed:
        parsed.sort(key=lambda x: (abs(x[1]-hour), x[2]-x[1]))
        return parsed[0][0]
    return None

# ──────────────────────────────────────────────────────────────────────────────
# RENKLER / TOOLTIP (risk_level CSV’den → Türkçe gösterim)
# ──────────────────────────────────────────────────────────────────────────────
# risk_level CSV’de: low / medium / high / critical (örneklerde böyle görünüyor)
# Türkçe gösterim ve renk eşlemeleri:
LEVEL_TR = {
    "low":      ("düşük riskli",   [56, 168, 0]),
    "medium":   ("orta riskli",    [255, 221, 0]),
    "high":     ("yüksek riskli",  [255, 140, 0]),
    "critical": ("kritik riskli",  [160, 0, 0]),
    # Alternatif anahtarlar (TR/EN karışık gelirse)
    "düşük":    ("düşük riskli",   [56, 168, 0]),
    "orta":     ("orta riskli",    [255, 221, 0]),
    "yüksek":   ("yüksek riskli",  [255, 140, 0]),
    "kritik":   ("kritik riskli",  [160, 0, 0]),
}
DEFAULT_FILL = [220, 220, 220]

def inject_properties(geojson_dict: dict, df_hr: pd.DataFrame) -> dict:
    if not geojson_dict or df_hr.empty:
        return geojson_dict

    df = df_hr.copy()
    df["geoid"] = df["geoid"].astype(str).str.replace(r"\D","", regex=True).str.zfill(11)
    # risk_level → lowercase
    df["risk_level"] = df["risk_level"].astype(str).str.strip().str.lower()

    # Tooltip’te Türkçe gösterim için ön hesap
    def _fmt_prob(x):
        try: return f"{float(x):.3f}"
        except: return ""
    def _fmt_num(x):
        try: return f"{float(x):.3f}"
        except: return ""

    # İndeks: GEOID
    dmap = df.set_index("geoid")

    feats = geojson_dict.get("features", [])
    out = []
    for feat in feats:
        props = dict((feat.get("properties") or {}))
        raw = None
        for k in ("geoid","GEOID","cell_id","id"):
            if k in props:
                raw = props[k]; break
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v; break
        key = str(raw) if raw is not None else ""
        key = "".join(ch for ch in key if ch.isdigit()).zfill(11)

        props.setdefault("display_id", str(raw or ""))

        if key in dmap.index:
            row = dmap.loc[key]
            # risk_score (0..1 arası varsayıyoruz)
            try:
                rscore = float(row["risk_score"])
            except Exception:
                rscore = None
            props["risk_score_txt"] = (f"{min(max(rscore,0.0),0.999):.3f}" if rscore is not None else "")

            # risk_level (TR çeviri + renk)
            lvl_key = str(row["risk_level"]).lower()
            tr_label, color = LEVEL_TR.get(lvl_key, ("bilinmiyor", DEFAULT_FILL))
            props["risk_level_tr"] = tr_label
            props["fill_color"] = color

            # expected_count ve Top1-Top3
            props["expected_count_txt"] = _fmt_num(row.get("expected_count", ""))
            for i in (1,2,3):
                c = row.get(f"top{i}_category", "")
                p = row.get(f"top{i}_prob", "")
                e = row.get(f"top{i}_expected", "")
                props[f"top{i}_category"] = (str(c) if pd.notna(c) and str(c).strip() else "")
                props[f"top{i}_prob_txt"]  = _fmt_prob(p)
                props[f"top{i}_exp_txt"]   = _fmt_num(e)
        else:
            props.setdefault("risk_level_tr","veri yok")
            props.setdefault("risk_score_txt","")
            props.setdefault("expected_count_txt","")
            props.setdefault("fill_color", DEFAULT_FILL)

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
    # TR açıklamalı tooltip
    tooltip = {
        "html": (
            "<b>GEOID:</b> {display_id}"
            "<br/><b>Risk düzeyi:</b> {risk_level_tr}"
            "<br/><b>Risk skoru (0-1):</b> {risk_score_txt}"
            "<br/><b>Beklenen toplam olay (bu saat dilimi):</b> {expected_count_txt}"
            "<hr style='opacity:0.3'/>"
            "<b>En olası suç tipleri</b>"
            "<br/>1) {top1_category} — olasılık: {top1_prob_txt} — beklenen: {top1_exp_txt}"
            "<br/>2) {top2_category} — olasılık: {top2_prob_txt} — beklenen: {top2_exp_txt}"
            "<br/>3) {top3_category} — olasılık: {top3_prob_txt} — beklenen: {top3_exp_txt}"
        ),
        "style": {"backgroundColor": "#262730", "color": "white"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# AKIŞ: Token → CSV’yi yükle → anlık hour_range → filtrele → GeoJSON zenginleştir → HARİTA
# ──────────────────────────────────────────────────────────────────────────────
if not resolve_github_token():
    st.error("GitHub token yok. `st.secrets['github_token']` veya GITHUB_TOKEN env ayarlayın.")
    st.stop()

try:
    df_all = load_hourly_csv(OWNER, REPO, ARTIFACT_NAME, CSV_TARGET_NAME)
except Exception as e:
    st.error(f"Artifact/CSV okunamadı: {e}")
    st.stop()

# Mevcut hour_range adayları (CSV’den)
hour_opts = sorted([str(x) for x in df_all["hour_range"].dropna().astype(str).unique()])

# SF saatine göre anlık hour_range
try:
    tz = ZoneInfo(TARGET_TZ)
except Exception:
    tz = ZoneInfo("America/Los_Angeles")

now_local = datetime.now(tz)
current_hour = now_local.hour
selected_hr = hour_to_bucket(current_hour, hour_opts) or (hour_opts[0] if hour_opts else None)

# Başlık altı küçük bilgi
st.caption(f"SF yerel zamanı: **{now_local.strftime('%Y-%m-%d %H:%M')} ({tz.key})** — "
           f"seçilen saat dilimi: **{selected_hr}**")

if not selected_hr:
    st.info("Bu veri kümesinde hour_range bulunamadı.")
    st.stop()

# Yalnızca ANLIK dilimi göster
df_hr = df_all[df_all["hour_range"].astype(str) == str(selected_hr)].copy()

# Hızlı özet
c1, c2, c3 = st.columns(3)
c1.metric("GEOID sayısı", f"{df_hr['geoid'].nunique():,}")
c2.metric("Risk skoru medyanı", f"{df_hr['risk_score'].median():.3f}" if not df_hr.empty else "—")
c3.metric("En yüksek skor", f"{df_hr['risk_score'].max():.3f}" if not df_hr.empty else "—")

# GeoJSON → enrich → harita
gj = fetch_geojson_smart(
    GEOJSON_PATH_LOCAL_DEFAULT,
    GEOJSON_PATH_LOCAL_DEFAULT,
    RAW_GEOJSON_OWNER,
    RAW_GEOJSON_REPO
)
enriched = inject_properties(gj, df_hr)
make_map(enriched)

# Alt bilgi / rozet
show_last_update_badge(data_upto=None, model_version=MODEL_VERSION, last_train=MODEL_LAST_TRAIN)

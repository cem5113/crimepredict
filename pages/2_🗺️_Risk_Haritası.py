# pages/2_🗺️_Risk_Haritası.py

import io, os, json, zipfile
from typing import Optional
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

# NOT: st.set_page_config(...) sadece app.py'de olmalı — buradan kaldırıldı.

st.title("🕒 Anlık Suç Risk Haritası")
st.markdown(
    "<p style='font-size:14px; font-style:italic;'>Bu harita, model çıktılarındaki saatlik risk skorlarını GEOID çokgenlerinin içine yansıtır. "
    "Boyamalar, seçtiğiniz saat aralığındaki <b>risk düzeyi (risk_level)</b> alanına göre yapılır. "
    "İmleci bir alana götürdüğünüzde risk puanı ve düzeyi (ve varsa en olası suç kategorileri) görünür.</p>",
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Ayarlar (secrets → env fallback)
# ──────────────────────────────────────────────────────────────────────────────
cfg = st.secrets if hasattr(st, "secrets") else {}
OWNER = cfg.get("artifact_owner", "cem5113")
REPO = cfg.get("artifact_repo", "crime_prediction_data")

# Artifact adı ve içinde arayacağımız dosya örnekleri:
# Kullanıcı: "sf-crime-pipeline-output.csv dosyası kullanılsın"
# Esnek davranalım: .zip içindeki CSV'leri tarayıp zorunlu kolonları taşıyan ilk dosyayı alalım.
ARTIFACT_NAME = cfg.get("artifact_name", "sf-crime-pipeline-output")  # Actions'da artifact "adı"
PREFERRED_FILE_HINTS = [
    "sf-crime-pipeline-output.csv",
    "risk_hourly_grid_full_labeled.csv",
    "risk_hourly.csv",
]  # Dosya adı ipuçları (esnek tarama)

# GeoJSON (önce local → artifact → raw github)
GEOJSON_PATH_LOCAL_DEFAULT = cfg.get("geojson_path", "data/sf_cells.geojson")
RAW_GEOJSON_OWNER = cfg.get("geojson_owner", "cem5113")
RAW_GEOJSON_REPO  = cfg.get("geojson_repo",  "crimepredict")

# ──────────────────────────────────────────────────────────────────────────────
# GitHub API yardımcıları
# ──────────────────────────────────────────────────────────────────────────────
def resolve_github_token() -> Optional[str]:
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        return tok
    for k in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if k in st.secrets and st.secrets[k]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[k])  # env'e set edelim
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
    # Actions → Artifacts listesi
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])

    # İsim tam eşleşme ve süresi dolmamış olanlar
    cand = [a for a in items if a.get("name") == artifact_name and not a.get("expired", False)]
    if not cand:
        # Bazen artifact adı tam tutmayabilir; "startswith" ile de deneriz.
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
# CSV okuma (artifact içinden esnek dosya bulma)
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"geoid", "hour_range", "risk_score", "risk_level"}

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # GEOID eşadlıları
    if "geoid" not in df.columns:
        for alt in ("cell_id", "geoid11", "geoid_11", "geoid10", "geoid_10", "id"):
            if alt in df.columns:
                df.rename(columns={alt: "geoid"}, inplace=True)
                break
    # risk_score eşadlıları
    if "risk_score" not in df.columns:
        for alt in ("risk", "score", "prob", "probability"):
            if alt in df.columns:
                df.rename(columns={alt: "risk_score"}, inplace=True)
                break
    # hour_range zorunlu — yoksa hour, hour_bin gibi alanlardan üretebilirdik
    # risk_level zorunlu — yoksa quantile ile hesaplayacağız (aşağıda)
    # GEOID'i 11 haneye zorla (sadece rakam)
    if "geoid" in df.columns:
        df["geoid"] = (
            df["geoid"].astype(str)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(11)
        )
    return df

def _has_required_cols(df: pd.DataFrame) -> bool:
    return REQUIRED_COLS.issubset(set(df.columns))

def _pick_best_csv_name(namelist: list[str]) -> Optional[str]:
    # Önce ipuçları ile sırala
    for hint in PREFERRED_FILE_HINTS:
        cand = [n for n in namelist if n.lower().endswith(hint.lower()) or hint.lower() in n.lower()]
        if cand:
            # Zip'lerde klasör/stage ile gelebilir — ilkini alırız
            return sorted(cand)[0]
    # Olmadı → Her CSV'yi deneyeceğiz (kolon kontrolü ile)
    any_csv = [n for n in namelist if n.lower().endswith(".csv")]
    return sorted(any_csv)[0] if any_csv else None

@st.cache_data(show_spinner=True, ttl=15 * 60)
def read_risk_csv_from_artifact(owner: str, repo: str, artifact_name: str) -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        target = _pick_best_csv_name(names)
        if not target:
            raise FileNotFoundError("Artifact zip içinde CSV dosyası bulunamadı.")
        # İlk seçilen dosyayı oku; gerekirse fallback: diğer CSV'leri sırayla dene
        tried = []
        for name in [target] + [n for n in names if n.endswith(".csv") and n != target]:
            try:
                with zf.open(name) as f:
                    df = pd.read_csv(f)
                df = _normalize_cols(df)
                if _has_required_cols(df):
                    return df
                tried.append(name)
            except Exception:
                tried.append(name)
        raise ValueError(f"Uygun kolonları taşıyan CSV bulunamadı. Denenenler: {tried[:10]}")

# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON temini (local → artifact → raw github)
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

    # 2) Artifact içinden dene
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

    # 3) Public raw GitHub
    try:
        raw = f"https://raw.githubusercontent.com/{raw_owner}/{raw_repo}/main/{path_local}"
        r = requests.get(raw, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    return {}

# ──────────────────────────────────────────────────────────────────────────────
# Boyama/tooltip için zenginleştirme
# ──────────────────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "çok düşük riskli": [200, 200, 200],
    "düşük":            [56, 168, 0],
    "düşük riskli":     [56, 168, 0],
    "orta":             [255, 221, 0],
    "orta riskli":      [255, 221, 0],
    "riskli":           [255, 140, 0],
    "yüksek":           [204, 0, 0],
    "yüksek riskli":    [204, 0, 0],
}
DEFAULT_FILL = [220, 220, 220]

def compute_level_by_quantile(df_hr: pd.DataFrame) -> pd.DataFrame:
    """risk_level yoksa quantile ile üret (seçili hour_range için)."""
    out = df_hr.copy()
    q25, q50, q75 = out["risk_score"].quantile([0.25, 0.50, 0.75]).tolist()
    def lab(x: float) -> str:
        if x <= q25: return "düşük riskli"
        elif x <= q50: return "orta riskli"
        elif x <= q75: return "riskli"
        return "yüksek riskli"
    if "risk_level" not in out.columns or out["risk_level"].isna().all():
        out["risk_level"] = out["risk_score"].apply(lab)
    return out

def inject_properties(geojson_dict: dict, df_hr: pd.DataFrame) -> dict:
    if not geojson_dict or df_hr.empty:
        return geojson_dict

    df = df_hr.copy()
    df["geoid"] = df["geoid"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
    dmap = df.set_index("geoid")

    feats = geojson_dict.get("features", [])
    out = []
    for feat in feats:
        props = dict((feat.get("properties") or {}))
        raw = None
        for k in ("geoid", "GEOID", "cell_id", "id"):
            if k in props:
                raw = props[k]
                break
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v
                    break
        key = str(raw) if raw is not None else ""
        key = "".join(ch for ch in key if ch.isdigit()).zfill(11)

        # Varsayılan
        props.setdefault("display_id", str(raw or ""))

        if key in dmap.index:
            row = dmap.loc[key]
            risk_score = float(row["risk_score"])
            props["risk_score_txt"] = f"{min(max(risk_score, 0.0), 0.999):.3f}"
            lvl = str(row.get("risk_level", "çok düşük riskli")).strip().lower()
            props["risk_level"] = lvl
            props["fill_color"] = COLOR_MAP.get(lvl, DEFAULT_FILL)

            # Opsiyonel: top kategoriler
            for i in (1, 2, 3):
                c = row.get(f"top{i}_category", "")
                p = row.get(f"top{i}_prob", "")
                e = row.get(f"top{i}_expected", "")
                if pd.notna(c) and str(c).strip():
                    props[f"top{i}_category"] = str(c)
                    props[f"top{i}_prob_txt"] = (f"{float(p):.3f}" if pd.notna(p) else "")
                    props[f"top{i}_exp_txt"]  = (f"{float(e):.3f}" if pd.notna(e) else "")
        else:
            # Veri yoksa çok düşük risk gibi nötr tonda boya
            props.setdefault("risk_level", "çok düşük riskli")
            props.setdefault("risk_score_txt", "")
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
    tooltip = {
        "html": (
            "<b>GEOID:</b> {display_id}"
            "<br/><b>Risk:</b> {risk_level}"
            "<br/><b>Skor:</b> {risk_score_txt}"
            "<br/>{top1_category} {top1_prob_txt} {top1_exp_txt}"
            "<br/>{top2_category} {top2_prob_txt} {top2_exp_txt}"
            "<br/>{top3_category} {top3_prob_txt} {top3_exp_txt}"
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
# UI — Artifact → CSV oku → Saat dilimi seç → Harita
# ──────────────────────────────────────────────────────────────────────────────
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
    read_risk_csv_from_artifact.clear()
    fetch_geojson_smart.clear()

if not TOKEN:
    st.error("GitHub token yok. `st.secrets['github_token']` veya GITHUB_TOKEN env ayarlayın.")
    st.stop()

try:
    df_all = read_risk_csv_from_artifact(OWNER, REPO, ARTIFACT_NAME)
except Exception as e:
    st.error(f"Artifact indirilemedi/okunamadı: {e}")
    st.stop()

# Yalnızca gerekli kolonlar (fazlalıkları tutabiliriz ama görünüme gerek yok)
# date kolonu boş geliyor — hiç kullanmıyoruz.
needed = ["geoid", "hour_range", "risk_score", "risk_level",
          "top1_category", "top1_prob", "top1_expected",
          "top2_category", "top2_prob", "top2_expected",
          "top3_category", "top3_prob", "top3_expected"]
for c in needed:
    if c not in df_all.columns:
        df_all[c] = None

# Saat dilimi seçimi
hours = sorted([str(h) for h in df_all["hour_range"].dropna().astype(str).unique()])
default_idx = 0
if "00-03" in hours:
    default_idx = hours.index("00-03")
sel_hr = st.sidebar.selectbox("Saat aralığı", hours, index=default_idx, help="Örn. 00-03, 03-06, ...")

# Seçilen saat aralığına göre filtrele
df_hr = df_all[df_all["hour_range"].astype(str) == str(sel_hr)].copy()
df_hr = compute_level_by_quantile(df_hr)  # risk_level yoksa üretir; varsa dokunmaz

# Kısa bir özet kutusu
c1, c2, c3 = st.columns(3)
c1.metric("GEOID sayısı", f"{df_hr['geoid'].nunique():,}")
c2.metric("Risk skoru medyanı", f"{df_hr['risk_score'].median():.3f}" if not df_hr.empty else "—")
c3.metric("En yüksek skor", f"{df_hr['risk_score'].max():.3f}" if not df_hr.empty else "—")

# GeoJSON’ı getir → özellik enjekte et → haritayı çiz
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

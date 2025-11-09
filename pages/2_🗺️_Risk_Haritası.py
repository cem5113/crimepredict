# pages/2_🗺️_Risk_Haritası.py — ANLIK (sadece hour_range; CSV: risk_hourly_grid_full_labeled.csv)

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
# Sabitler (KULLANILACAK TEK ARTIFACT ve TEK CSV)
# ──────────────────────────────────────────────────────────────────────────────
OWNER = "cem5113"
REPO = "crime_prediction_data"
ARTIFACT_NAME = "sf-crime-pipeline-output"            # ← sadece bunu ara
CSV_TARGET_NAME = "risk_hourly_grid_full_labeled.csv" # ← zip içindeki hedef dosya
TARGET_TZ = "America/Los_Angeles"                      # anlık saat SF
GEOJSON_LOCAL = "data/sf_cells.geojson"               # yerel dosya (varsa)

# ──────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ──────────────────────────────────────────────────────────────────────────────
def gh_headers() -> dict:
    hdrs = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    return hdrs

def digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def parse_range(tok: str) -> Optional[tuple[int,int]]:
    if not isinstance(tok, str) or "-" not in tok: return None
    a,b = tok.split("-",1)
    s = max(0, min(23, int(a.strip())))
    e = int(b.strip()); e = 24 if e==24 else max(1, min(24, e))
    return (s,e)

def hour_to_bucket(h: int, labels: Iterable[str]) -> Optional[str]:
    parsed = []
    for lab in labels:
        rg = parse_range(lab)
        if rg: parsed.append((lab, rg[0], rg[1]))
    for lab,s,e in parsed:
        if s <= h < (e if e<24 else 24): return lab
    for lab,s,e in parsed:                   # s>e saran aralıklar
        if s > e and (h >= s or h < e): return lab
    return parsed[0][0] if parsed else None

# ──────────────────────────────────────────────────────────────────────────────
# Artifact → CSV yükle (SADECE hedef isimler)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True, ttl=15*60)
def load_hourly_csv() -> pd.DataFrame:
    base = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/artifacts"
    r = requests.get(base, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    arts = [a for a in r.json().get("artifacts", []) if a.get("name")==ARTIFACT_NAME and not a.get("expired", False)]
    if not arts:
        raise FileNotFoundError(f"Artifact bulunamadı: {ARTIFACT_NAME}")
    arts.sort(key=lambda x: x.get("updated_at",""), reverse=True)
    dl = arts[0]["archive_download_url"]

    z = requests.get(dl, headers=gh_headers(), timeout=60)
    z.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        names = zf.namelist()
        cand = [n for n in names if n.endswith("/"+CSV_TARGET_NAME) or n.endswith(CSV_TARGET_NAME)]
        if not cand:
            raise FileNotFoundError(f"Zip içinde {CSV_TARGET_NAME} yok.")
        with zf.open(cand[0]) as f:
            df = pd.read_csv(f)

    # Zorunlu kolonlar ve normalizasyon
    req = {
        "GEOID","hour_range","risk_score","risk_level","expected_count",
        "top1_category","top1_prob","top1_expected",
        "top2_category","top2_prob","top2_expected",
        "top3_category","top3_prob","top3_expected",
    }
    miss = req - set(df.columns)
    if miss:
        # kolonlar farklı büyük/küçük olabilir → küçük harfe indirip eşle
        df.columns = [c.strip().lower() for c in df.columns]
        req = {c.lower() for c in req}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"CSV eksik kolon: {', '.join(sorted(miss))}")

    # GEOID 11 haneye zorla, hour_range stringe zorla
    col = "geoid" if "geoid" in df.columns else "GEOID"
    df["geoid"] = df[col].map(digits11)
    df["hour_range"] = df["hour_range"].astype(str)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON yükle (yalın)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True, ttl=60*60)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_LOCAL):
        with open(GEOJSON_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ──────────────────────────────────────────────────────────────────────────────
# Harita özellik zenginleştirme
# ──────────────────────────────────────────────────────────────────────────────
LEVEL_TR = {
    "low": ("düşük riskli", [56,168,0]),
    "medium": ("orta riskli", [255,221,0]),
    "high": ("yüksek riskli", [255,140,0]),
    "critical": ("kritik riskli", [160,0,0]),
}
DEFAULT_FILL = [220,220,220]

def enrich_geojson(gj: dict, df: pd.DataFrame) -> dict:
    if not gj or df.empty: return gj

    df = df.copy()
    df["geoid"] = df["geoid"].map(digits11)
    df["risk_level"] = df["risk_level"].astype(str).str.lower()
    dmap = df.set_index("geoid")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})
        # ID adayları
        raw = None
        for k in ("geoid","GEOID","cell_id","id","geoid11","geoid10","geoid_11","geoid_10"):
            if k in props:
                raw = props[k]; break
        if raw is None:
            for k,v in props.items():
                if "geoid" in str(k).lower(): raw = v; break
        key = digits11(raw)
        props["display_id"] = str(raw) if raw not in (None,"") else key

        # Varsayılan boşluklar
        props.setdefault("risk_level_tr","")
        props.setdefault("risk_score_txt","")
        props.setdefault("expected_count_txt","")
        for i in (1,2,3):
            props.setdefault(f"top{i}_category","")
            props.setdefault(f"top{i}_prob_txt","")
            props.setdefault(f"top{i}_exp_txt","")
        props.setdefault("fill_color", DEFAULT_FILL)

        if key and key in dmap.index:
            row = dmap.loc[key]
            # seviye + renk
            tr, col = LEVEL_TR.get(str(row["risk_level"]).lower(), ("", DEFAULT_FILL))
            props["risk_level_tr"] = tr
            props["fill_color"] = col
            # skor
            try:
                r = float(row["risk_score"])
                props["risk_score_txt"] = f"{min(max(r,0.0),0.999):.3f}"
            except:
                pass
            # expected & topN
            def f3(x): 
                try: return f"{float(x):.3f}"
                except: return ""
            props["expected_count_txt"] = f3(row.get("expected_count",""))
            for i in (1,2,3):
                props[f"top{i}_category"] = str(row.get(f"top{i}_category","") or "")
                props[f"top{i}_prob_txt"]  = f3(row.get(f"top{i}_prob",""))
                props[f"top{i}_exp_txt"]   = f3(row.get(f"top{i}_expected",""))

        feats_out.append({**feat, "properties": props})
    return {**gj, "features": feats_out}

def hr_label_to_human(lab: str) -> str:
    # "3-6" → "03:00-05:59"
    if not lab or "-" not in lab: return lab
    a,b = lab.split("-",1)
    s = int(a.strip())
    e = int(b.strip())
    end_h = e-1 if e>0 else 23
    return f"{s:02d}:00-{end_h:02d}:59"

def draw_map(gj: dict):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[80,80,80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )
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
        "style": {"backgroundColor":"#262730","color":"white"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Çalıştır (yalnızca ANLIK hour_range)
# ──────────────────────────────────────────────────────────────────────────────

st.title("🕒 Anlık Suç Risk Haritası")

st.caption(
    f"Bu harita anlık tahmin edilen suç riskini gösterir.\n\n"
    "Başka bir güne/saate ait tahmin görmek isterseniz → **Suç Tahmini** sekmesini kullanın."
)

# Token kontrol
if not os.getenv("GITHUB_TOKEN"):
    st.error("GITHUB_TOKEN ayarlı değil. Secrets veya env üzerinden tanımlayın.")
    st.stop()

# CSV yükle
try:
    df_all = load_hourly_csv()
except Exception as e:
    st.error(f"Artifact/CSV okunamadı: {e}")
    st.stop()

# Anlık saat → CSV’deki hour_range etiketleri
labels = sorted(df_all["hour_range"].dropna().astype(str).unique().tolist())

tz = ZoneInfo(TARGET_TZ)
now_sf = datetime.now(tz)

hr_label = hour_to_bucket(now_sf.hour, labels) or (labels[0] if labels else None)

if not hr_label:
    st.error("CSV’de hour_range bulunamadı.")
    st.stop()

st.caption(
    f"SF saati: **{now_sf:%Y-%m-%d %H:%M}** — gösterilen dilim: **{hr_label}**"
)

# Filtrele → sadece anlık hour_range göster
df_hr = df_all[df_all["hour_range"].astype(str) == hr_label].copy()

# Küçük özet
c1, c2, c3 = st.columns(3)

c1.metric(
    "GEOID sayısı",
    f"{df_hr['geoid'].nunique():,}"
)

c2.metric(
    "Risk medyanı",
    f"{df_hr['risk_score'].median():.3f}" if not df_hr.empty else "—"
)

c3.metric(
    "Maks skor",
    f"{df_hr['risk_score'].max():.3f}" if not df_hr.empty else "—"
)

# GeoJSON → enrich → çiz
gj = load_geojson()
gj_enriched = enrich_geojson(gj, df_hr)
draw_map(gj_enriched)

# Alt rozet
show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN
)


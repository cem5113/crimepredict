# pages/3_🔮_Suç_Tahmini_ve_Forecast.py
# --- Amaç ---
# Bu sayfa, birden fazla/tüm suç kategorisi ve GEOID seçimiyle geçmiş 5 yıllık veriden
# saatlik pattern çıkarır, basit fakat açıklanabilir bir forecast üretir ve
# harita üzerinde (hover) her hücre için en olası 3 suç türü ve en olası zaman aralıklarını gösterir.
# Veri kaynakları GitHub Actions artifact'larından okunur:
#  - fr-crime-pipeline-output: fr_crime_09.parquet  (geçmiş olay+özellikler)
#  - crime_prediction_data / sf-crime-parquet: risk_hourly.parquet (varsa genel risk/şimdiki skor)
#
# Not: Bu forecast, kolluk için yalın ve hızlı çalışacak şekilde tasarlandı:
#  - “Son X gün + saat-of-day + gün-of-week” ağırlıklı bir profil hesabı (recency-weighting)
#  - Aynı saat dilimi paternine göre önümüzdeki 24/48/168 saat için beklenen olasılık dağılımları
#  - Top 3 kategori ve bu kategorilerin %95 kitlesini taşıyan pik saat aralıkları
#  - Harita hover tooltip: Top-3 + tavsiye saat blokları
#
# Geliştirme noktaları (ileride):
#  - Mevcut stacking model skorlarını kategori-bazlı hale getirmek
#  - 911/311 kısa vadeli sinyalleriyle nowcast düzeltmesi
#  - ETS/Prophet/TFT gibi modellerle GEOID×kategori bazlı ileri forecast

import io, os, json, zipfile
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import requests
import altair as alt

# Yerel bileşenler
from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN

# ---------- Sayfa başlık & ayarlar ----------
st.set_page_config(page_title="🔮 Suç Tahmini ve Forecast", layout="wide")
st.title("🔮 Suç Tahmini • Kategori & GEOID Seçimli")
st.caption("5 yıllık tarihsel desenlere dayalı hızlı forecast ve görsel özetler")

# ---------- Konfig ----------
cfg = st.secrets if hasattr(st, "secrets") else {}
# Artifacts
OWNER_MAIN = cfg.get("artifact_owner", "cem5113")
REPO_MAIN = cfg.get("artifact_repo", "crime_prediction_data")
ARTIFACT_RISK = cfg.get("artifact_name", "sf-crime-parquet")
EXPECTED_RISK_PARQUET = "risk_hourly.parquet"  # opsiyonel

OWNER_FR = cfg.get("fr_owner", OWNER_MAIN)
REPO_FR = cfg.get("fr_repo", "fr-crime-pipeline-output")
ARTIFACT_FR = cfg.get("fr_artifact", "fr-crime-pipeline-output")
EXPECTED_FR_PARQUET = cfg.get("fr_file", "fr_crime_09.parquet")

# GeoJSON
GEOJSON_PATH_LOCAL_DEFAULT = cfg.get("geojson_path", "data/sf_cells.geojson")
RAW_GEOJSON_OWNER = cfg.get("geojson_owner", "cem5113")
RAW_GEOJSON_REPO = cfg.get("geojson_repo", "crimepredict")

# ---------- Yardımcılar: Token & GitHub ----------
def resolve_github_token() -> str | None:
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

@st.cache_data(show_spinner=True, ttl=15*60)
def fetch_latest_artifact_zip(owner: str, repo: str) -> bytes:
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    # En güncel ve süresi geçmemiş olanı seç
    cand = [a for a in items if not a.get("expired", False)]
    if not cand:
        raise FileNotFoundError("Uygun artifact bulunamadı.")
    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    url = cand[0].get("archive_download_url")
    if not url:
        raise RuntimeError("archive_download_url yok")
    r2 = requests.get(url, headers=gh_headers(), timeout=60)
    r2.raise_for_status()
    return r2.content

@st.cache_data(show_spinner=True, ttl=15*60)
def read_parquet_from_artifact(owner: str, repo: str, wanted_suffix: str) -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(owner, repo)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + wanted_suffix) or n.endswith(wanted_suffix)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {wanted_suffix} yok. Örnek içerik: {memlist[:15]}")
        with zf.open(matches[0]) as f:
            df = pd.read_parquet(f)
    # Kolon isimlerini normalize et
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- GeoJSON getir ----------
@st.cache_data(show_spinner=True, ttl=60*60)
def fetch_geojson_smart(path_local: str, path_in_zip: str, raw_owner: str, raw_repo: str) -> dict:
    # 1) Local
    try:
        if os.path.exists(path_local):
            with open(path_local, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    # 2) risk artifact içinde olabilir
    try:
        zip_bytes = fetch_latest_artifact_zip(OWNER_MAIN, REPO_MAIN)
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

# ---------- Veri hazırlık ----------
def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns and "event_hour" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["event_hour"].astype(str) + ":00:00", errors="coerce")
    else:
        # En azından date alanı olsun
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            # son çare: index
            df["datetime"] = pd.to_datetime(df.index, errors="coerce")
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek  # 0=Mon
    return df

def normalize_geoid(x) -> str:
    s = ''.join(ch for ch in str(x) if str(ch).isdigit())
    return s.zfill(11)[:11]

# Recency ağırlığı (ör: yarı-ömür 14 gün)
def recency_weight(ts: pd.Series, halflife_days: float = 14.0) -> pd.Series:
    if ts.empty:
        return ts
    max_t = ts.max()
    dt_days = (max_t - ts).dt.total_seconds()/86400.0
    lam = np.log(2.0)/max(halflife_days, 1e-6)
    return np.exp(-lam * dt_days)

# GEOID×kategori×saat bazlı profile çıkar
@st.cache_data(show_spinner=True, ttl=10*60)
def build_hourly_profile(df: pd.DataFrame, last_n_days: int = 90, halflife_days: float = 14.0,
                         use_y_label: bool = True) -> pd.DataFrame:
    base = df.copy()
    base = base.sort_values("datetime")
    # Son N gün filtresi
    if last_n_days:
        cutoff = base["datetime"].max() - pd.Timedelta(days=last_n_days)
        base = base.loc[base["datetime"] >= cutoff]
    # Ağırlık: zaman yakınlığı
    w = recency_weight(base["datetime"], halflife_days=halflife_days)
    base["w"] = w.values
    # Olay sinyali: Y_label varsa onu kullan, yoksa 1
    if use_y_label and "Y_label" in base.columns:
        sig = (base["Y_label"].astype(float).clip(0,1))
    else:
        sig = 1.0
    base["signal"] = sig

    # GEOID normalizasyonu
    if "GEOID" in base.columns:
        base["geoid"] = base["GEOID"].map(normalize_geoid)
    elif "geoid" in base.columns:
        base["geoid"] = base["geoid"].map(normalize_geoid)
    elif "id" in base.columns:
        base["geoid"] = base["id"].map(normalize_geoid)
    else:
        base["geoid"] = ""

    # Eksik kategori -> "Unknown"
    if "category" not in base.columns:
        base["category"] = "Unknown"

    # Gün/ saat
    base = ensure_datetime(base)

    # Gün-of-week conditioning (dilersek benzer günleri ağır basacak şekilde)
    base["dow"] = base["dow"].fillna(-1).astype(int)

    # Weighted olay beklentisi: geoid×category×hour (ve opsiyonel DOW)
    grp = base.groupby(["geoid", "category", "dow", "hour"], as_index=False).agg(
        exp_events=("signal", lambda x: float(np.sum(x))) ,
        w_mean=("w", "mean"),
        n=("signal", "size")
    )
    # beklenen skor = exp_events * w_mean  (basitçe ağırlıklandırma)
    grp["score"] = grp["exp_events"] * grp["w_mean"]
    # Normalizasyon: her geoid×dow için kategori×saat skoru -> olasılığa ölçekle
    norm = grp.groupby(["geoid", "dow"], as_index=False)["score"].sum().rename(columns={"score":"tot"})
    out = grp.merge(norm, on=["geoid","dow"], how="left")
    out["prob"] = np.where(out["tot"]>0, out["score"]/out["tot"], 0.0)
    return out  # kolonlar: geoid, category, dow, hour, prob

# Seçili DOW senaryosuna göre (ör. forecast başlangıç gününün DOW'u) next H saat için dağıtım
@st.cache_data(show_spinner=True, ttl=10*60)
def forecast_next_hours(profile: pd.DataFrame, start_dt: datetime, horizon_h: int = 24) -> pd.DataFrame:
    if profile.empty:
        return profile
    rows = []
    for h in range(horizon_h):
        t = start_dt + timedelta(hours=h)
        dow = t.weekday()
        hr = t.hour
        # geoid×category×dow×hour prob
        sl = profile[(profile["dow"]==dow) & (profile["hour"]==hr)]
        if sl.empty:
            # fallback: dow bağımsız
            sl = profile[profile["hour"]==hr].copy()
        if sl.empty:
            continue
        tmp = sl.copy()
        tmp["ts"] = t
        rows.append(tmp)
    fc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(profile.columns)+["ts"]) 
    return fc  # kolonlar: geoid, category, dow, hour, prob, ts

# Top-3 kategori ve saat bloklarını biçimle

def contiguous_blocks(hours: List[int]) -> List[Tuple[int,int]]:
    if not hours: return []
    hours = sorted(set(int(h)%24 for h in hours))
    blocks = []
    start = prev = hours[0]
    for h in hours[1:]+[hours[0]]:  # wrap check
        if (h-prev) % 24 != 1:
            blocks.append((start, prev))
            start = h
        prev = h
    if blocks and blocks[0] == (start, prev):
        return [blocks[0]]
    blocks.append((start, prev))
    # En fazla ilk 2 blok
    return blocks[:2]

def format_block(b: Tuple[int,int]) -> str:
    a,bh = b
    if a==bh:
        return f"{a:02d}:00"
    # sarma varsa iki parça göster
    if (bh - a) % 24 < (a - bh) % 24:
        return f"{a:02d}:00–{bh:02d}:59"
    return f"{a:02d}:00–{bh:02d}:59"

@st.cache_data(show_spinner=True, ttl=10*60)
def summarize_top3(fc: pd.DataFrame, geoid_list: List[str]) -> pd.DataFrame:
    if fc.empty:
        return pd.DataFrame(columns=["geoid","rank","category","share","peak_hours"])
    sl = fc[fc["geoid"].isin(geoid_list)].copy()
    # geoid×category toplam olasılık payı (horizon içinde)
    g = sl.groupby(["geoid","category"], as_index=False)["prob"].sum().rename(columns={"prob":"prob_sum"})
    # geoid bazında normalize
    tot = g.groupby("geoid", as_index=False)["prob_sum"].sum().rename(columns={"prob_sum":"tot"})
    g = g.merge(tot, on="geoid", how="left")
    g["share"] = np.where(g["tot"]>0, g["prob_sum"]/g["tot"], 0.0)
    # Top-3 seç
    ranks = []
    for geoid, sub in g.groupby("geoid"):
        sub = sub.sort_values("share", ascending=False).head(3)
        # her kategori için pik saatleri bul: prob'a göre ilk %30 kitleyi taşıyan saatler
        peaks = []
        for _, r in sub.iterrows():
            cat = r["category"]
            hh = (
                sl[(sl["geoid"]==geoid) & (sl["category"]==cat)]
                .groupby("hour", as_index=False)["prob"].mean()
                .sort_values("prob", ascending=False)
            )
            if not hh.empty:
                csum = (hh["prob"].cumsum()/hh["prob"].sum()).fillna(0)
                top_hours = hh.loc[csum<=0.30, "hour"].astype(int).tolist()
                # saat bloklarına dönüştür (en fazla 2 blok)
                blks = contiguous_blocks(top_hours) if top_hours else []
                label = ", ".join(format_block(b) for b in blks) if blks else "—"
            else:
                label = "—"
            peaks.append(label)
        sub["peak_hours"] = peaks
        sub["rank"] = range(1, len(sub)+1)
        ranks.append(sub[["geoid","rank","category","share","peak_hours"]])
    return pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame(columns=["geoid","rank","category","share","peak_hours"])

# ---------- Harita zenginleştirme ----------
COLOR_MAP = {
    "çok düşük": [200,200,200],
    "düşük": [56,168,0],
    "orta": [255,221,0],
    "yüksek": [204,0,0],
}

def only_digits(s: str) -> str:
    return ''.join(ch for ch in str(s) if ch.isdigit())

def inject_properties_top3(geojson_dict: dict, top3_df: pd.DataFrame, risk_daily_df: pd.DataFrame | None = None) -> dict:
    if not geojson_dict:
        return geojson_dict
    feats = geojson_dict.get("features", [])
    # quick lookup
    tmap: Dict[str, pd.DataFrame] = {
        g: sub.sort_values("rank").reset_index(drop=True)
        for g, sub in top3_df.groupby("geoid")
    }
    rmap = None
    if risk_daily_df is not None and not risk_daily_df.empty:
        rmap = risk_daily_df.set_index("geoid")["risk_score_daily"].to_dict()
        # eşiklere göre renk
        if not risk_daily_df.empty:
            q25, q50, q75 = (
                risk_daily_df["risk_score_daily"].quantile([0.25,0.5,0.75]).tolist()
            )
    out = []
    for feat in feats:
        props = (feat.get("properties") or {}).copy()
        raw = next((props[k] for k in ("geoid","GEOID","cell_id","id") if k in props), None)
        geoid = only_digits(raw)[:11] if raw is not None else ""
        props.setdefault("display_id", str(raw or ""))

        # risk renkleri (varsa)
        lvl = "orta"
        if rmap is not None and geoid in rmap:
            val = float(rmap[geoid])
            props["risk_score_daily"] = val
            # basit renkleme: q25/50/75
            if val <= q25: lvl = "düşük"
            elif val <= q50: lvl = "orta"
            elif val <= q75: lvl = "yüksek"
            else: lvl = "yüksek"
        props["fill_color"] = COLOR_MAP.get(lvl, [220,220,220])

        # Top-3 kategori + saat blokları
        if geoid in tmap:
            sub = tmap[geoid]
            lines = []
            for i, rr in sub.iterrows():
                pct = f"{rr['share']*100:.1f}%"
                ph = rr["peak_hours"]
                lines.append(f"{rr['rank']}. {rr['category']} • {pct} • {ph}")
            props["top3_html"] = "<br/>".join(lines)
        else:
            props["top3_html"] = "Veri yok"

        out.append({**feat, "properties": props})
    return {**geojson_dict, "features": out}

# ---------- UI: Yan panel ----------
TOKEN = resolve_github_token()
st.sidebar.header("Veri Kaynakları")
with st.sidebar.expander("🔐 Token Durumu", expanded=TOKEN is None):
    st.write("Env GITHUB_TOKEN:", "✅" if os.getenv("GITHUB_TOKEN") else "❌")
    has_secret = False
    try:
        has_secret = any(k in st.secrets for k in ("github_token","GH_TOKEN","GITHUB_TOKEN"))
    except Exception:
        pass
    st.write("Secrets'ta Token:", "✅" if has_secret else "❌")

refresh = st.sidebar.button("Veriyi Yenile (artifact'ları tazele)")
if refresh:
    fetch_latest_artifact_zip.clear()
    read_parquet_from_artifact.clear()
    fetch_geojson_smart.clear()
    build_hourly_profile.clear()
    forecast_next_hours.clear()
    summarize_top3.clear()

if not TOKEN:
    st.error("GitHub token yok. `st.secrets['github_token']` veya GITHUB_TOKEN env ayarlayın.")
    st.stop()

# ---------- Veri yükleme ----------
try:
    df_events = read_parquet_from_artifact(OWNER_FR, REPO_FR, EXPECTED_FR_PARQUET)
except Exception as e:
    st.error(f"Olay verisi (fr_crime_09.parquet) okunamadı: {e}")
    st.stop()

# Opsiyonel risk dosyası (genel renk için)
try:
    df_risk = read_parquet_from_artifact(OWNER_MAIN, REPO_MAIN, EXPECTED_RISK_PARQUET)
    # günlük ortalama
    if not df_risk.empty:
        cols = [c.lower().strip() for c in df_risk.columns]
        df_risk.columns = cols
        # risk_score/ geoid / date esnek eşleme
        if "risk_score" not in df_risk.columns:
            for alt in ("risk","score","prob","probability"):
                if alt in df_risk.columns:
                    df_risk = df_risk.rename(columns={alt:"risk_score"}); break
        if "geoid" not in df_risk.columns:
            for alt in ("cell_id","geoid10","geoid_10","id"):
                if alt in df_risk.columns:
                    df_risk["geoid"] = df_risk[alt]; break
        if "date" in df_risk.columns:
            df_risk["date"] = pd.to_datetime(df_risk["date"]).dt.date
            risk_daily = (
                df_risk.groupby([df_risk["geoid"].map(lambda x: ''.join(ch for ch in str(x) if str(ch).isdigit()).zfill(11)[:11]), "date"], as_index=False)["risk_score"].mean()
                .rename(columns={"geoid":"geoid_norm","risk_score":"risk_score_daily"})
            )
            # Son gün
            if not risk_daily.empty:
                last_day = max(risk_daily["date"])
                risk_today = risk_daily[risk_daily["date"]==last_day].copy()
                risk_today = risk_today.rename(columns={"geoid_norm":"geoid"})
            else:
                risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"]) 
        else:
            risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"]) 
    else:
        risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"]) 
except Exception:
    risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"]) 

# ---------- Veri temizleme ----------
df_events = ensure_datetime(df_events)
# GEOID
if "geoid" in df_events.columns:
    df_events["geoid"] = df_events["geoid"].map(normalize_geoid)
elif "GEOID" in df_events.columns:
    df_events["geoid"] = df_events["GEOID"].map(normalize_geoid)
elif "id" in df_events.columns:
    df_events["geoid"] = df_events["id"].map(normalize_geoid)
else:
    df_events["geoid"] = ""

# Kategoriler
if "category" not in df_events.columns:
    df_events["category"] = "Unknown"

# Kullanıcıya seçimler için listeler
all_categories = sorted([str(x) for x in df_events["category"].dropna().unique().tolist()])
all_geoids = sorted(df_events["geoid"].dropna().unique().tolist())

# ---------- Sidebar Filtreleri ----------
st.sidebar.header("Filtreler")
sel_cats = st.sidebar.multiselect("Suç kategorileri", options=all_categories, default=[])
sel_geoids = st.sidebar.multiselect("GEOID seçimi", options=all_geoids[:5000], default=[], help="Arama kutusunu kullanarak GEOID filtreleyin")

min_dt = df_events["datetime"].min()
max_dt = df_events["datetime"].max()
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Başlangıç tarihi (profil için)", value=(max_dt.date() - timedelta(days=90)), min_value=min_dt.date(), max_value=max_dt.date())
with col2:
    halflife = st.number_input("Yarı-ömür (gün)", min_value=3.0, max_value=60.0, value=14.0, step=1.0, help="Recency ağırlığı")

horizon = st.sidebar.select_slider("Horizon (saat)", options=[24, 48, 72, 168], value=24)
fc_start = st.sidebar.datetime_input("Forecast başlangıcı", value=datetime.combine(max_dt.date(), datetime.min.time()), help="Varsayılan: veri son gününün 00:00")

# Alt filtreler
if sel_cats:
    df_events = df_events[df_events["category"].isin(sel_cats)]
if sel_geoids:
    df_events = df_events[df_events["geoid"].isin(sel_geoids)]

# Profili üret
last_n_days = (max_dt.date() - start_date).days if start_date else 90
profile = build_hourly_profile(df_events, last_n_days=last_n_days, halflife_days=float(halflife), use_y_label=True)

if profile.empty:
    st.warning("Profil üretilemedi. Filtreleri gevşetin veya veri kaynağını kontrol edin.")
    st.stop()

# Forecast
fc = forecast_next_hours(profile, start_dt=fc_start, horizon_h=int(horizon))

# Top-3 özet
geoids_in_scope = sel_geoids if sel_geoids else all_geoids
summary_top3 = summarize_top3(fc, geoid_list=geoids_in_scope)

# ---------- Harita ----------
col_map, col_tbl = st.columns([2,1])
with col_map:
    gj = fetch_geojson_smart(GEOJSON_PATH_LOCAL_DEFAULT, GEOJSON_PATH_LOCAL_DEFAULT, RAW_GEOJSON_OWNER, RAW_GEOJSON_REPO)
    enriched = inject_properties_top3(gj, summary_top3, risk_today if not risk_today.empty else None)
    layer = pdk.Layer(
        "GeoJsonLayer",
        enriched,
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
            "<b>GEOID:</b> {display_id}<br/>"
            "<b>Top-3 (pay • saat blokları):</b><br/>{top3_html}"
        ),
        "style": {"backgroundColor":"#262730", "color":"white"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

with col_tbl:
    st.subheader("Seçim Özeti – Top 3")
    if not summary_top3.empty:
        show = summary_top3.copy()
        show["share"] = (show["share"]*100).map(lambda x: f"{x:.1f}%")
        st.dataframe(show.rename(columns={"rank":"Sıra","category":"Kategori","share":"Pay","peak_hours":"Pik Saatler"}), use_container_width=True)
    else:
        st.info("Top-3 özeti için veri yok.")

# ---------- Detay Grafikler ----------
st.markdown("---")
st.subheader("Detay: GEOID × Kategori × Saat Profili / Forecast")

# Grafik için seçim
sel_geo_for_plot = st.multiselect("Grafik için GEOID seçin", options=geoids_in_scope[:1000], default=geoids_in_scope[:1])

if sel_geo_for_plot:
    # Saat bazlı ısı haritası (kategori×saat)
    plot_df = (
        fc[fc["geoid"].isin(sel_geo_for_plot)]
        .groupby(["category","hour"], as_index=False)["prob"].mean()
    )
    if not plot_df.empty:
        heat = alt.Chart(plot_df).mark_rect().encode(
            x=alt.X("hour:O", title="Saat"),
            y=alt.Y("category:N", title="Kategori"),
            tooltip=[alt.Tooltip("category", title="Kategori"), alt.Tooltip("hour", title="Saat"), alt.Tooltip("prob", title="Olasılık", format=".3f")],
            color=alt.Color("prob:Q", title="Olasılık", scale=alt.Scale(scheme="orangered"))
        ).properties(height=320)
        st.altair_chart(heat, use_container_width=True)

    # Top kategoriler için saatlik çizgi
    topk = (
        summary_top3[summary_top3["geoid"].isin(sel_geo_for_plot)]
        .sort_values(["geoid","rank"]) 
        .groupby("geoid").head(3)[["geoid","category"]].drop_duplicates()
    )
    if not topk.empty:
        lines = (
            fc.merge(topk, on=["geoid","category"], how="inner")
              .groupby(["category","hour"], as_index=False)["prob"].mean()
        )
        line = alt.Chart(lines).mark_line(point=True).encode(
            x=alt.X("hour:O", title="Saat"),
            y=alt.Y("prob:Q", title="Olasılık"),
            color=alt.Color("category:N", title="Kategori"),
            tooltip=["category","hour","prob"]
        ).properties(height=320)
        st.altair_chart(line, use_container_width=True)
else:
    st.info("Grafik için en az bir GEOID seçin.")

# ---------- Açıklamalar ----------
with st.expander("ℹ️ Metodoloji"):
    st.markdown(
        """
        **Özet**: Forecast, *son N gün* verisi üzerinden saat-of-day ve gün-of-week paternlerini **zaman yakınlığına göre ağırlıklandırarak** çıkarır. 
        Seçilen horizon (24/48/72/168 saat) boyunca her GEOID×kategori×saat için beklenen olasılık dağılımı hesaplanır. 
        Hover'da gösterilen *Top-3* listesi, bu horizon içindeki toplam olasılık payına göre belirlenir ve her kategori için olasılığın %30'unu taşıyan **pik saat blokları** verilir.

        **Parametreler**:
        - *Başlangıç tarihi*: profile dahil edilecek son gün sayısını belirler (ör. 90 gün).
        - *Yarı-ömür (gün)*: recency ağırlığında yarı-ömür. Daha küçük değer, son günleri daha baskın kılar.
        - *Horizon*: ileriye dönük saat sayısı (24/48/72/168).

        **Sınırlamalar / Geliştirme Fikirleri**:
        - Kategori-bazlı model skorlarıyla nowcast düzeltmesi (911/311 kısa vadeli sinyaller).
        - Mevsimsellik + tatil etkisi için ayrı profiller.
        - İleri seviye zaman serisi modelleri (ETS, Prophet, TFT) ile GEOID×kategori forecast.
        """
    )

show_last_update_badge(data_upto=None, model_version=MODEL_VERSION, last_train=MODEL_LAST_TRAIN)

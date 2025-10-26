# pages/3_🔮_Suç_Tahmini_ve_Forecast.py
# --- Amaç ---
# Bu sayfa, birden fazla/tüm suç kategorisi ve GEOID seçimiyle geçmiş 5 yıllık veriden
# saatlik pattern çıkarır, basit fakat açıklanabilir bir forecast üretir ve
# harita üzerinde (hover) her hücre için en olası 3 suç türü ve en olası zaman aralıklarını gösterir.
# Veri kaynakları GitHub Actions artifact'larından okunur:
#  - fr-crime-outputs-parquet: fr_crime_10.parquet  (geçmiş olay+özellikler)
#  - crime_prediction_data / sf-crime-parquet: risk_hourly.parquet ve/veya stacking nowcast/parquet (opsiyonel)
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

import io, os, json, zipfile, re
from datetime import datetime, timedelta, date, time
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
# --- Ana artifact (risk) bilgileri ---
OWNER_MAIN = cfg.get("artifact_owner", "cem5113")
REPO_MAIN  = cfg.get("artifact_repo",  "crime_prediction_data")
ARTIFACT_RISK = cfg.get("artifact_name", "sf-crime-parquet")
EXPECTED_RISK_PARQUET = cfg.get("risk_file", "risk_hourly.parquet")  # opsiyonel

# --- FR artifact bilgileri (ZIP ve iç yollar) ---
OWNER_FR   = cfg.get("fr_owner", OWNER_MAIN)
REPO_FR    = cfg.get("fr_repo",  REPO_MAIN)             # FR artifact bu repoda
ARTIFACT_FR = cfg.get("fr_artifact", "fr-crime-outputs-parquet")

# ZIP dosyasının adı (artifact içeriği)
FR_ZIP_NAME = cfg.get("fr_zip_name", "fr_parquet_outputs.zip")

# ZIP içindeki dosya yolları (inner paths)
EXPECTED_FR_PARQUET = cfg.get("fr_file", "fr_crime_10.parquet")
EXPECTED_METRICS_OHE = cfg.get("fr_metrics_file", "artifact/metrics_stacking_ohe.parquet")

# GeoJSON
GEOJSON_PATH_LOCAL_DEFAULT = cfg.get("geojson_path", "data/sf_cells.geojson")
RAW_GEOJSON_OWNER = cfg.get("geojson_owner", "cem5113")
RAW_GEOJSON_REPO = cfg.get("geojson_repo", "crimepredict")

# ---------- Yardımcılar: Token & GitHub ----------
def fr_locator_for(inner_path: str) -> dict:
    """
    FR artifact içindeki bir dosyayı bulmak için gerekli tüm bilgileri döndürür.
    inner_path: ZIP içindeki görece yol (örn. 'fr_crime_10.parquet' ya da 'artifact/metrics_stacking_ohe.parquet')
    """
    return {
        "owner": OWNER_FR,                      # 'cem5113'
        "repo": REPO_FR,                        # 'crime_prediction_data'
        "artifact": ARTIFACT_FR,                # 'fr-crime-outputs-parquet'
        "zip_name": FR_ZIP_NAME,                # 'fr_parquet_outputs.zip'
        "inner_path": inner_path,               # 'fr_crime_10.parquet' veya 'artifact/metrics_stacking_ohe.parquet'
        # Tam mantıksal yol (okunabilirlik için)
        "logical_path": f"{OWNER_FR}/{REPO_FR}/{ARTIFACT_FR}/{FR_ZIP_NAME}/{inner_path}"
    }

FR_CRIME_10 = fr_locator_for(EXPECTED_FR_PARQUET)
METRICS_STACKING_OHE = fr_locator_for(EXPECTED_METRICS_OHE)

def resolve_github_token() -> str | None:
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        return tok
    # Streamlit secrets güvenli kontrol
    try:
        for k in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
            if k in st.secrets and st.secrets[k]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[k])
                return os.environ["GITHUB_TOKEN"]
    except Exception:
        pass
    return None

def gh_headers() -> dict:
    hdrs = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "sutam-forecast-ui"
    }
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    return hdrs

def _read_parquet_or_raise(filelike) -> pd.DataFrame:
    try:
        return pd.read_parquet(filelike)
    except Exception as e:
        raise RuntimeError(
            "Parquet okunamadı. PyArrow/fastparquet eksik olabilir. "
            "requirements.txt içine 'pyarrow' ekleyin. Orijinal hata: " + str(e)
        )

@st.cache_data(show_spinner=True, ttl=15*60)
def read_parquet_from_artifact(owner: str, repo: str, wanted_suffix: str, artifact_name: str | None = None) -> pd.DataFrame:
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name)
    def _read_from_zip(zb: bytes) -> pd.DataFrame | None:
        with zipfile.ZipFile(io.BytesIO(zb)) as zf:
            memlist = zf.namelist()

            # 1) Parquet arayalım
            matches = [n for n in memlist if n.endswith("/"+wanted_suffix) or n.endswith(wanted_suffix)]
            if matches:
                with zf.open(matches[0]) as f:
                    return _read_parquet_or_raise(f)

            # 2) İç ZIP'lere dal
            inner_zips = [n for n in memlist if n.lower().endswith(".zip")]
            for iz in inner_zips:
                with zf.open(iz) as f:
                    inner_bytes = f.read()
                got = _read_from_zip(inner_bytes)
                if got is not None:
                    return got

            # 3) Parquet yoksa CSV fallback (aynı isim gövdesiyle)
            csv_cand = [n for n in memlist if n.lower().endswith(".csv")]
            if csv_cand:
                with zf.open(csv_cand[0]) as f:
                    return pd.read_csv(f)
        return None
    df = _read_from_zip(zip_bytes)
    if df is None:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            raise FileNotFoundError(f"Zip içinde {wanted_suffix} veya CSV fallback bulunamadı. Örnek içerik: {zf.namelist()[:20]}")
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=True, ttl=15*60)
def fetch_latest_artifact_zip(owner: str, repo: str, artifact_name: str | None = None) -> bytes:
    """
    İlgili repodaki en güncel (expire olmamış) artifact'ı indirir ve ZIP bytes döner.
    artifact_name verilirse sadece adı eşleşenler arasından seçer.
    """
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts"
    r = requests.get(base, headers=gh_headers(), timeout=30)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    cand = [a for a in items if not a.get("expired", False)]
    if artifact_name:
        cand = [a for a in cand if a.get("name") == artifact_name]
    if not cand:
        raise FileNotFoundError("Uygun artifact bulunamadı.")
    cand.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    url = cand[0].get("archive_download_url")
    if not url:
        raise RuntimeError("archive_download_url yok.")
    r2 = requests.get(url, headers=gh_headers(), timeout=60)
    r2.raise_for_status()
    return r2.content

# --- İsteğe bağlı: İç ZIP adı + iç yol biliniyorsa doğrudan okuyucu ---
@st.cache_data(show_spinner=True, ttl=15*60)
def read_parquet_from_nested_artifact(
    owner: str,
    repo: str,
    inner_zip_name: str,           # örn. 'fr_parquet_outputs.zip'
    inner_path_in_zip: str,        # örn. 'fr_crime_10.parquet'
    artifact_name: str | None = None
) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer:
        members = outer.namelist()
        cand = [n for n in members if n.endswith("/"+inner_zip_name) or n.endswith(inner_zip_name)]
        if not cand:
            raise FileNotFoundError(f"Artifact içinde {inner_zip_name} yok. İçerik örneği: {members[:20]}")
        with outer.open(cand[0]) as inner_zip_file:
            inner_bytes = inner_zip_file.read()

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        inner_members = inner.namelist()
        targets = [n for n in inner_members if n.endswith("/"+inner_path_in_zip) or n.endswith(inner_path_in_zip)]
        if not targets:
            raise FileNotFoundError(f"{inner_zip_name} içinde {inner_path_in_zip} yok. İçerik örneği: {inner_members[:20]}")
        with inner.open(targets[0]) as f:
            df = _read_parquet_or_raise(f)

    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=True, ttl=15*60)
def list_zip_members(owner: str, repo: str, artifact_name: str | None = None) -> list[str]:
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return zf.namelist()

@st.cache_data(show_spinner=True, ttl=15*60)
def read_json_from_artifact(owner: str, repo: str, wanted_suffix: str, artifact_name: str | None = None) -> dict:
    zip_bytes = fetch_latest_artifact_zip(owner, repo, artifact_name)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        memlist = zf.namelist()
        matches = [n for n in memlist if n.endswith("/" + wanted_suffix) or n.endswith(wanted_suffix)]
        if not matches:
            raise FileNotFoundError(f"Zip içinde {wanted_suffix} yok.")
        with zf.open(matches[0]) as f:
            return json.load(io.TextIOWrapper(f, encoding="utf-8"))

def _find_first_by_patterns(names: list[str], patterns: list[str]) -> str | None:
    for p in patterns:
        rx = re.compile(p)
        for n in names:
            if rx.search(n):
                return n
    return None

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
        zip_bytes = fetch_latest_artifact_zip(OWNER_MAIN, REPO_MAIN, ARTIFACT_RISK)
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
    d = df.copy()
    # 1) datetime varsa doğrudan
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
    else:
        # 2) date + time
        if "date" in d.columns and "time" in d.columns:
            d["datetime"] = pd.to_datetime(
                d["date"].astype(str) + " " + d["time"].astype(str),
                utc=True, errors="coerce"
            )
        # 3) date + event_hour_x/y
        elif "date" in d.columns and any(c in d.columns for c in ("event_hour","event_hour_x","event_hour_y")):
            eh = next((c for c in ("event_hour","event_hour_x","event_hour_y") if c in d.columns), None)
            d["datetime"] = pd.to_datetime(
                d["date"].astype(str) + " " + d[eh].astype(str) + ":00:00",
                utc=True, errors="coerce"
            )
        else:
            # 4) son çare: index'ten dene
            d["datetime"] = pd.to_datetime(d.index, utc=True, errors="coerce")

    d["date"] = d["datetime"].dt.date
    d["hour"] = d["datetime"].dt.hour
    d["dow"] = d["datetime"].dt.dayofweek
    return d

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
    base = df.copy().sort_values("datetime")
    if last_n_days:
        cutoff = base["datetime"].max() - pd.Timedelta(days=last_n_days)
        base = base.loc[base["datetime"] >= cutoff]
    w = recency_weight(base["datetime"], halflife_days=halflife_days)
    base["w"] = w.values
    if use_y_label and "Y_label" in base.columns:
        sig = (base["Y_label"].astype(float).clip(0,1))
    else:
        sig = 1.0
    base["signal"] = sig

    if "GEOID" in base.columns:
        base["geoid"] = base["GEOID"].map(normalize_geoid)
    elif "geoid" in base.columns:
        base["geoid"] = base["geoid"].map(normalize_geoid)
    elif "id" in base.columns:
        base["geoid"] = base["id"].map(normalize_geoid)
    else:
        base["geoid"] = ""

    if "category" not in base.columns:
        base["category"] = "Unknown"

    base = ensure_datetime(base)
    base["dow"] = base["dow"].fillna(-1).astype(int)

    grp = base.groupby(["geoid", "category", "dow", "hour"], as_index=False).agg(
        exp_events=("signal", lambda x: float(np.sum(x))),
        w_mean=("w", "mean"),
        n=("signal", "size")
    )
    grp["score"] = grp["exp_events"] * grp["w_mean"]
    norm = grp.groupby(["geoid", "dow"], as_index=False)["score"].sum().rename(columns={"score":"tot"})
    out = grp.merge(norm, on=["geoid","dow"], how="left")
    out["prob"] = np.where(out["tot"]>0, out["score"]/out["tot"], 0.0)
    return out  # geoid, category, dow, hour, prob

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
        sl = profile[(profile["dow"]==dow) & (profile["hour"]==hr)]
        if sl.empty:
            sl = profile[profile["hour"]==hr].copy()
        if sl.empty:
            continue
        tmp = sl.copy()
        tmp["ts"] = t
        rows.append(tmp)
    fc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(profile.columns)+["ts"])
    return fc  # geoid, category, dow, hour, prob, ts

# Top-3 kategori ve saat bloklarını biçimle
def contiguous_blocks(hours: List[int]) -> List[Tuple[int,int]]:
    if not hours: return []
    hours = sorted(set(int(h)%24 for h in hours))
    blocks = []
    start = prev = hours[0]
    for h in hours[1:]+[hours[0]]:
        if (h-prev) % 24 != 1:
            blocks.append((start, prev))
            start = h
        prev = h
    if blocks and blocks[0] == (start, prev):
        return [blocks[0]]
    blocks.append((start, prev))
    return blocks[:2]

def format_block(b: Tuple[int,int]) -> str:
    a,bh = b
    if a==bh:
        return f"{a:02d}:00"
    if (bh - a) % 24 < (a - bh) % 24:
        return f"{a:02d}:00–{bh:02d}:59"
    return f"{a:02d}:00–{bh:02d}:59"

@st.cache_data(show_spinner=True, ttl=10*60)
def summarize_top3(fc: pd.DataFrame, geoid_list: List[str]) -> pd.DataFrame:
    if fc.empty:
        return pd.DataFrame(columns=["geoid","rank","category","share","peak_hours"])
    sl = fc[fc["geoid"].isin(geoid_list)].copy()
    g = sl.groupby(["geoid","category"], as_index=False)["prob"].sum().rename(columns={"prob":"prob_sum"})
    tot = g.groupby("geoid", as_index=False)["prob_sum"].sum().rename(columns={"prob_sum":"tot"})
    g = g.merge(tot, on="geoid", how="left")
    g["share"] = np.where(g["tot"]>0, g["prob_sum"]/g["tot"], 0.0)
    ranks = []
    for geoid, sub in g.groupby("geoid"):
        sub = sub.sort_values("share", ascending=False).head(3)
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
    "düşük":     [56,168,0],
    "orta":      [255,221,0],
    "yüksek":    [255,140,0],
    "çok yüksek":[204,0,0],
}

def only_digits(s: str) -> str:
    return ''.join(ch for ch in str(s) if ch.isdigit())

def inject_properties_top3(geojson_dict: dict, top3_df: pd.DataFrame, risk_daily_df: pd.DataFrame | None = None) -> dict:
    if not geojson_dict:
        return geojson_dict
    feats = geojson_dict.get("features", [])
    tmap = {g: sub.sort_values("rank").reset_index(drop=True) for g, sub in top3_df.groupby("geoid")}
    rmap, q25, q50, q75 = None, None, None, None
    if risk_daily_df is not None and not risk_daily_df.empty:
        rmap = risk_daily_df.set_index("geoid")["risk_score_daily"].to_dict()
        q25, q50, q75 = risk_daily_df["risk_score_daily"].quantile([0.25,0.5,0.75]).tolist()

    def level_of(v: float) -> str:
        if q25 is None: return "orta"
        if v <= q25: return "düşük"
        if v <= q50: return "orta"
        if v <= q75: return "yüksek"
        return "çok yüksek"

    out = []
    for feat in feats:
        props = (feat.get("properties") or {}).copy()
        raw = next((props[k] for k in ("geoid","GEOID","cell_id","id") if k in props), None)
        geoid = ''.join(ch for ch in str(raw or "") if str(ch).isdigit())[:11]
        props.setdefault("display_id", str(raw or ""))

        if rmap is not None and geoid in rmap:
            val = float(rmap[geoid])
            props["risk_score_daily"] = val
            lvl = level_of(val)
        else:
            lvl = "orta"
        props["fill_color"] = COLOR_MAP.get(lvl, [220,220,220])

        if geoid in tmap:
            sub = tmap[geoid]
            lines = []
            for _, rr in sub.iterrows():
                pct = f"{rr['share']*100:.1f}%"
                ph = rr.get("peak_hours","—")
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
    read_parquet_from_nested_artifact.clear()
    fetch_geojson_smart.clear()
    build_hourly_profile.clear()
    forecast_next_hours.clear()
    summarize_top3.clear()


if not TOKEN:
    st.error("GitHub token yok. `st.secrets['github_token']` veya GITHUB_TOKEN env ayarlayın.")
    st.stop()

# ---------- Veri yükleme ----------
try:
    df_metrics_ohe = read_parquet_from_nested_artifact(
        OWNER_FR, REPO_FR,
        inner_zip_name=FR_ZIP_NAME,
        inner_path_in_zip=EXPECTED_METRICS_OHE,  # 'artifact/metrics_stacking_ohe.parquet'
        artifact_name=ARTIFACT_FR
    )
except Exception:
    df_metrics_ohe = pd.DataFrame()

try:
    df_events = read_parquet_from_nested_artifact(
        OWNER_FR, REPO_FR,
        inner_zip_name=FR_ZIP_NAME,              # 'fr_parquet_outputs.zip'
        inner_path_in_zip=EXPECTED_FR_PARQUET,   # 'fr_crime_10.parquet'
        artifact_name=ARTIFACT_FR                # 'fr-crime-outputs-parquet'
    )
except Exception as e:
    st.error(f"Olay verisi ({EXPECTED_FR_PARQUET}) okunamadı: {e}")
    st.stop()

# Opsiyonel risk dosyası (genel renk için)
try:
    df_risk = read_parquet_from_artifact(OWNER_MAIN, REPO_MAIN, EXPECTED_RISK_PARQUET, ARTIFACT_RISK)
    if not df_risk.empty:
        cols = [c.lower().strip() for c in df_risk.columns]
        df_risk.columns = cols
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
                df_risk.groupby(
                    [df_risk["geoid"].map(lambda x: ''.join(ch for ch in str(x) if str(ch).isdigit()).zfill(11)[:11]), "date"],
                    as_index=False
                )["risk_score"].mean().rename(columns={"geoid":"geoid_norm","risk_score":"risk_score_daily"})
            )
            if not risk_daily.empty:
                last_day = max(risk_daily["date"])
                risk_today = risk_daily[risk_daily["date"]==last_day].copy().rename(columns={"geoid_norm":"geoid"})
            else:
                risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"])
        else:
            risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"])
    else:
        risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"])
except Exception:
    risk_today = pd.DataFrame(columns=["geoid","risk_score_daily"])

# ---------- (Opsiyonel) Stacking nowcast + metrikler ----------
try:
    sf_members = list_zip_members(OWNER_MAIN, REPO_MAIN, ARTIFACT_RISK)
except Exception:
    sf_members = []

# Nowcast/parquet dosyasını desenlerle bul
df_nowcast = pd.DataFrame()
stacking_metrics = {}

try:
    if sf_members:
        pred_name = _find_first_by_patterns(
            sf_members,
            patterns=[
                r"(?:^|/)(stacking|model|pred).*hour.*\.parquet$",
                r"(?:^|/)predictions?_hourly\.parquet$",
                r"(?:^|/)risk_hourly\.parquet$"
            ]
        )
        if pred_name:
            with zipfile.ZipFile(io.BytesIO(fetch_latest_artifact_zip(OWNER_MAIN, REPO_MAIN, ARTIFACT_RISK))) as zf:
                with zf.open(pred_name) as f:
                    df_nowcast = pd.read_parquet(f)
            df_nowcast.columns = [str(c).strip().lower() for c in df_nowcast.columns]
            if "geoid" not in df_nowcast.columns:
                for alt in ("cell_id","geoid10","geoid_10","id"):
                    if alt in df_nowcast.columns:
                        df_nowcast["geoid"] = df_nowcast[alt]
                        break
            if "geoid" not in df_nowcast.columns:
                df_nowcast["geoid"] = ""
            # zaman alanları
            if "datetime" in df_nowcast.columns:
                df_nowcast["datetime"] = pd.to_datetime(df_nowcast["datetime"], errors="coerce")
                df_nowcast["hour"] = df_nowcast["datetime"].dt.hour
            elif "date" in df_nowcast.columns and "hour" in df_nowcast.columns:
                df_nowcast["datetime"] = pd.to_datetime(df_nowcast["date"].astype(str)) + pd.to_timedelta(df_nowcast["hour"].astype(int), unit="h")
                df_nowcast["hour"] = df_nowcast["hour"].astype(int)
            elif "hour" in df_nowcast.columns:
                df_nowcast["hour"] = df_nowcast["hour"].astype(int)
            # skor sütunu
            score_col = None
            for c in ("score","risk_score","prob","p","y_pred","prediction","stacking_prob"):
                if c in df_nowcast.columns:
                    score_col = c; break
            if score_col is None:
                numeric_cols = [c for c in df_nowcast.columns if pd.api.types.is_numeric_dtype(df_nowcast[c])]
                score_col = numeric_cols[0] if numeric_cols else None
            df_nowcast["geoid"] = df_nowcast["geoid"].map(normalize_geoid)
            _invalid_all = df_nowcast["geoid"].isna().all() or (df_nowcast["geoid"] == "00000000000").all()
            if _invalid_all:
                df_nowcast = pd.DataFrame()  # böylece aşağıda use_nowcast koşulu ile harman yapılmaz
            
            # metrikler
            metrics_name = _find_first_by_patterns(
                sf_members,
                patterns=[
                    r"(?:^|/)(stacking|model).*metrics.*\.json$",
                    r"(?:^|/)metrics.*\.json$"
                ]
            )
        # metrikler
        metrics_name = _find_first_by_patterns(
            sf_members,
            patterns=[
                r"(?:^|/)(stacking|model).*metrics.*\.json$",
                r"(?:^|/)metrics.*\.json$"
            ]
        )
        if metrics_name:
            stacking_metrics = read_json_from_artifact(OWNER_MAIN, REPO_MAIN, metrics_name, ARTIFACT_RISK)
except Exception:
    df_nowcast = pd.DataFrame()
    stacking_metrics = {}

# ---------- Veri temizleme ----------
df_events = ensure_datetime(df_events)
if "geoid" in df_events.columns:
    df_events["geoid"] = df_events["geoid"].map(normalize_geoid)
elif "GEOID" in df_events.columns:
    df_events["geoid"] = df_events["GEOID"].map(normalize_geoid)
elif "id" in df_events.columns:
    df_events["geoid"] = df_events["id"].map(normalize_geoid)
else:
    df_events["geoid"] = ""

if "category" not in df_events.columns:
    df_events["category"] = "Unknown"

# Kullanıcıya seçimler için listeler
all_categories = sorted([str(x) for x in df_events["category"].dropna().unique().tolist()])
all_geoids = sorted(df_events["geoid"].dropna().unique().tolist())

# ---------- Sidebar Filtreleri ----------
st.sidebar.header("Filtreler")

sel_cats = st.sidebar.multiselect(
    "Suç kategorileri",
    options=all_categories,
    default=[]
)

# PATCH 8: GEOID seçiminde performans koruması
MAX_GEOIDS_IN_WIDGET = 5000
geo_options = all_geoids if len(all_geoids) <= MAX_GEOIDS_IN_WIDGET else all_geoids[:MAX_GEOIDS_IN_WIDGET]

sel_geoids = st.sidebar.multiselect(
    "GEOID seçimi",
    options=geo_options,
    default=[],
    help=f"Arama kutusuyla filtreleyin. Listede ilk {MAX_GEOIDS_IN_WIDGET} gösteriliyor."
)

# --- Güvenli min/max datetime ---
dt_series = pd.to_datetime(df_events["datetime"], errors="coerce")
min_dt_raw = dt_series.min()
max_dt_raw = dt_series.max()

if pd.isna(min_dt_raw) or pd.isna(max_dt_raw):
    min_dt_raw = pd.Timestamp.utcnow().normalize()
    max_dt_raw = min_dt_raw

# Eğer veri NaT ise bugünün tarihi ile güvenli fallback oluştur
today_date = datetime.utcnow().date()
safe_min_date = (min_dt_raw.date() if pd.notna(min_dt_raw) else today_date)
safe_max_date = (max_dt_raw.date() if pd.notna(max_dt_raw) else today_date)

# Uç durum: min > max ise takas et
if safe_min_date > safe_max_date:
    safe_min_date, safe_max_date = safe_max_date, safe_min_date

# --- Tarih girişleri ---
col1, col2 = st.sidebar.columns(2)
with col1:
    default_start_date = max(safe_min_date, safe_max_date - timedelta(days=90))
    start_date = st.date_input(
        "Başlangıç tarihi (profil için)",
        value=default_start_date,
        min_value=safe_min_date,
        max_value=safe_max_date,
    )
with col2:
    halflife = st.number_input(
        "Yarı-ömür (gün)",
        min_value=3.0, max_value=60.0, value=14.0, step=1.0,
        help="Recency ağırlığı"
    )

horizon = st.sidebar.select_slider("Horizon (saat)", options=[24, 48, 72, 168], value=24)

# Forecast başlangıcı için güvenli varsayılan + açık limitler (min/max) + clamp
# min/max'ı saat bilgisiyle oluştur (günün başı/sonu)
min_dt_bound = datetime.combine(safe_min_date, time(0, 0, 0))
max_dt_bound = datetime.combine(safe_max_date, time(23, 59, 59))

# default değeri datetime olarak kur (verinin son gününün 00:00'ı)
fc_start_default = datetime.combine(safe_max_date, time(0, 0, 0))

# Eğer default, aralık dışına düşerse kıskaçla (clamp)
if fc_start_default < min_dt_bound:
    fc_start_default = min_dt_bound
elif fc_start_default > max_dt_bound:
    fc_start_default = max_dt_bound

fc_date = st.sidebar.date_input(
    "Forecast tarihi",
    value=fc_start_default.date(),
    min_value=safe_min_date,
    max_value=safe_max_date,
    key="fc_date_widget"
)

# Saat parçası (Streamlit time_input min/max desteklemiyor; gerekirse kıskaç uygularız)
fc_time = st.sidebar.time_input(
    "Forecast saati",
    value=fc_start_default.time().replace(microsecond=0),
    step=3600,  # saatlik adım
    key="fc_time_widget"
)

# Birleştir
fc_start = datetime.combine(fc_date, fc_time)

# Güvenlik: min/max sınırına kıskaç (clamp)
if fc_start < min_dt_bound:
    fc_start = min_dt_bound
elif fc_start > max_dt_bound:
    fc_start = max_dt_bound

# Nowcast düzeltmesi ayarları
st.sidebar.header("Nowcast düzeltmesi")
use_nowcast = st.sidebar.checkbox(
    "Stacking/nowcast skorlarıyla düzelt",
    value=True,
    help="Saatlik stacking risk skorları varsa forecast olasılıklarını hafifçe ayarlar."
)
alpha = st.sidebar.slider(
    "Forecast ağırlığı (α)", 0.0, 1.0, 0.7, 0.05,
    help="0=yalnız nowcast, 1=yalnız forecast. Öneri: 0.6–0.8"
)

# Alt filtreler (seçim sonrası veri daraltma)
if sel_cats:
    df_events = df_events[df_events["category"].isin(sel_cats)]
if sel_geoids:
    df_events = df_events[df_events["geoid"].isin(sel_geoids)]

# Profili üret
last_n_days = (safe_max_date - start_date).days if start_date else 90
profile = build_hourly_profile(
    df_events, last_n_days=last_n_days,
    halflife_days=float(halflife), use_y_label=True
)

if profile.empty:
    st.warning("Profil üretilemedi. Filtreleri gevşetin veya veri kaynağını kontrol edin.")
    st.stop()

# Forecast
fc = forecast_next_hours(profile, start_dt=fc_start, horizon_h=int(horizon))
if "hour" not in fc.columns:
    fc["hour"] = pd.to_datetime(fc["ts"]).dt.hour if "ts" in fc.columns else 0

# ---------- Nowcast ile harmanlama ----------
if use_nowcast and not df_nowcast.empty:
    tmp = df_nowcast.copy()
    cat_col = next((c for c in ("category","cat","type","crime_category") if c in tmp.columns), None)
    score_col = next((c for c in ("score","risk_score","prob","p","y_pred","prediction","stacking_prob") if c in tmp.columns), None)
    if score_col is None:
        num_cols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
        score_col = num_cols[0] if num_cols else None

    if score_col:
        if cat_col is None:
            tmp["nowcast_norm"] = tmp.groupby(["geoid","hour"])[score_col].transform(
                lambda s: 0.0 if s.max()==s.min() else (s - s.min())/(s.max()-s.min())
            ).fillna(0.0)
            fc = fc.merge(tmp[["geoid","hour","nowcast_norm"]].drop_duplicates(), on=["geoid","hour"], how="left")
        else:
            tmp = tmp.rename(columns={cat_col:"category"})
            tmp["nowcast_norm"] = tmp.groupby(["geoid","category","hour"])[score_col].transform(
                lambda s: 0.0 if s.max()==s.min() else (s - s.min())/(s.max()-s.min())
            ).fillna(0.0)
            fc = fc.merge(
                tmp[["geoid","category","hour","nowcast_norm"]].drop_duplicates(),
                on=["geoid","category","hour"], how="left"
            )
        med = 0.0 if fc["nowcast_norm"].isna().all() else fc["nowcast_norm"].median(skipna=True)
        fc["nowcast_norm"] = fc["nowcast_norm"].fillna(0.0 if np.isnan(med) else med)
        fc["prob_adj"] = alpha*fc["prob"] + (1-alpha)*fc["nowcast_norm"]
    else:
        fc["prob_adj"] = fc["prob"]
else:
    fc["prob_adj"] = fc["prob"]

if "prob_adj" not in fc.columns:
    fc["prob_adj"] = fc.get("prob", 0.0)

# Top-3 özet (ayarlı olasılık ile)
geoids_in_scope = sel_geoids if sel_geoids else all_geoids
fc_for_summary = fc.copy()
fc_for_summary["prob"] = fc_for_summary["prob_adj"]
summary_top3 = summarize_top3(fc_for_summary, geoid_list=geoids_in_scope)

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
            "<br/><i>Nowcast düzeltmesi aktifse: Top-3 payları nowcast ile harmanlanmıştır.</i>"
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

st.markdown("----")
st.subheader("🧪 Stacking Model Metrikleri (sf-artifact)")
if stacking_metrics:
    overview_rows = []
    for k in ("auc","roc_auc","pr_auc","f1","f1_macro","f1_weighted","accuracy","balanced_accuracy"):
        if k in stacking_metrics:
            overview_rows.append({"Metric": k, "Value": stacking_metrics[k]})
    if overview_rows:
        st.table(pd.DataFrame(overview_rows))
    by_cat = stacking_metrics.get("by_category") or stacking_metrics.get("per_class") or {}
    if isinstance(by_cat, dict) and by_cat:
        cat_rows = []
        for cat, mm in by_cat.items():
            if isinstance(mm, dict):
                cat_rows.append({
                    "Kategori": cat,
                    "AUC": mm.get("auc"),
                    "F1": mm.get("f1"),
                    "Precision": mm.get("precision"),
                    "Recall": mm.get("recall")
                })
        if cat_rows:
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)
else:
    st.info("Stacking metrikleri bulunamadı (artifact içinde *metrics*.json yok veya isimler farklı).")

# ---------- Detay Grafikler ----------
st.markdown("---")
st.subheader("Detay: GEOID × Kategori × Saat Profili / Forecast")

sel_geo_for_plot = st.multiselect("Grafik için GEOID seçin", options=geoids_in_scope[:1000], default=geoids_in_scope[:1])

if sel_geo_for_plot:
    # Saat bazlı ısı haritası (kategori×saat) — nowcast harmanlı
    plot_df = (
        fc_for_summary[fc_for_summary["geoid"].isin(sel_geo_for_plot)]
        .groupby(["category","hour"], as_index=False)["prob"].mean()
    )
    if not plot_df.empty:
        heat = alt.Chart(plot_df).mark_rect().encode(
            x=alt.X("hour:O", title="Saat"),
            y=alt.Y("category:N", title="Kategori"),
            tooltip=[alt.Tooltip("category", title="Kategori"),
                     alt.Tooltip("hour", title="Saat"),
                     alt.Tooltip("prob", title="Olasılık", format=".3f")],
            color=alt.Color("prob:Q", title="Olasılık", scale=alt.Scale(scheme="oranges"))
        ).properties(height=320)
        st.altair_chart(heat, use_container_width=True)

    # Top kategoriler için saatlik çizgi — nowcast harmanlı
    topk = (
        summary_top3[summary_top3["geoid"].isin(sel_geo_for_plot)]
        .sort_values(["geoid","rank"])
        .groupby("geoid").head(3)[["geoid","category"]].drop_duplicates()
    )
    if not topk.empty:
        lines = (
            fc_for_summary.merge(topk, on=["geoid","category"], how="inner")
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

        **Nowcast harmanı**: Eğer artifact'ta saatlik stacking skorları (nowcast) mevcutsa,
        olasılıklar `prob_adj = α · forecast + (1−α) · nowcast_norm` olarak harmanlanır.
        α kaydırıcısı ile ayarlanabilir (öneri: 0.6–0.8).

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

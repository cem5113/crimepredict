# 3_🧭_Suç_Tahmini — Haritalı görünüm (GEOID + centroid)
# Saatlik (≤24 saat; tek saat / saat aralığı) ve Günlük (≤365 gün) risk görünümleri
# Kaynak: artifact 'fr-crime-outputs-parquet' → risk_hourly_next24h_top3 / risk_daily_next365d_top5
# Not: Harita için centroid yalnızca artifact içindeki adaylardan bulunur (upload yok).

import os
import io
import posixpath
import zipfile
from io import BytesIO

from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo 
except Exception:
    ZoneInfo = None 

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------
# ⚙️ GitHub repo ve artifact bilgisi
# ------------------------------------------------------------
REPOSITORY_OWNER = "cem5113"
REPOSITORY_NAME  = "crime_prediction_data"
ARTIFACT_NAME_SHOULD_CONTAIN = "fr-crime-outputs-parquet"  # FR risk çıktıları artifact'i

# Artifact içindeki beklenen dosyalar (FR pipeline risk çıktıları)
ARTIFACT_MEMBER_HOURLY = "risk_hourly_next24h_top3.parquet"    # parquet yoksa .csv denenir
ARTIFACT_MEMBER_DAILY  = "risk_daily_next365d_top5.parquet"    # parquet yoksa .csv denenir

# Centroid için otomatik adaylar (artifact içinde aranır)
CENTROID_FILE_CANDIDATES = [
    "geoid_centroids.parquet",
    "sf_geoid_centroids.parquet",
    "geoid_centroids.csv",
    "sf_geoid_centroids.csv",
    # grid dosyasında lat/lon varsa onu da dener
    "sf_crime_grid_full_labeled.parquet",
    "sf_crime_grid_full_labeled.csv",
]

# ------------------------------------------------------------
# 🔑 Token / Header
# ------------------------------------------------------------
def resolve_github_token() -> str | None:
    if os.getenv("GITHUB_TOKEN"):
        return os.getenv("GITHUB_TOKEN")
    for key in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if key in st.secrets and st.secrets[key]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[key])
                return os.environ["GITHUB_TOKEN"]
        except Exception:
            pass
    return None

def github_api_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

# ------------------------------------------------------------
# 📦 Artifact ZIP alma (en güncel ve süresi dolmamış)
# ------------------------------------------------------------
def resolve_latest_artifact_zip_url(owner: str, repo: str, name_contains: str):
    token = resolve_github_token()
    if not token:
        return None, {}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(f"{base}/actions/artifacts?per_page=100",
                            headers=github_api_headers(), timeout=60)
    response.raise_for_status()
    artifacts = (response.json() or {}).get("artifacts", []) or []
    artifacts = [a for a in artifacts if (name_contains in a.get("name", "")) and not a.get("expired")]
    if not artifacts:
        return None, {}
    artifacts.sort(key=lambda a: a.get("updated_at", ""), reverse=True)
    url = f"{base}/actions/artifacts/{artifacts[0]['id']}/zip"
    return url, github_api_headers()

# ------------------------------------------------------------
# 🧰 ZIP içinden üye okuma (parquet→csv fallback)
# ------------------------------------------------------------
def read_member_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    def read_any_table(raw_bytes: bytes, name_hint: str) -> pd.DataFrame:
        buf = BytesIO(raw_bytes)
        if name_hint.lower().endswith(".csv"):
            return pd.read_csv(buf)
        try:
            buf.seek(0)
            return pd.read_parquet(buf)
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf)

    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # 1) tam eşleşme
        if member_path in names:
            with z.open(member_path) as f:
                return read_any_table(f.read(), member_path)

        # 2) basename eşleşmesi
        base = posixpath.basename(member_path)
        cands = [n for n in names if n.endswith("/"+base) or n == base]
        if cands:
            with z.open(cands[0]) as f:
                return read_any_table(f.read(), cands[0])

        # 3) parquet<->csv fallback
        alt = None
        if member_path.lower().endswith(".parquet"):
            alt = member_path[:-8] + ".csv"
        elif member_path.lower().endswith(".csv"):
            alt = member_path[:-4] + ".parquet"
        if alt:
            return read_member_from_zip_bytes(zip_bytes, alt)

    raise FileNotFoundError(f"ZIP içinde bulunamadı: {member_path}")

@st.cache_data(show_spinner=False)
def load_artifact_member(member: str) -> pd.DataFrame:
    url, headers = resolve_latest_artifact_zip_url(REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN)
    if not url:
        raise RuntimeError("Artifact bulunamadı veya GITHUB_TOKEN yok.")
    r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return read_member_from_zip_bytes(r.content, member)

# ------------------------------------------------------------
# 🧭 Şema doğrulayıcılar (hourly/daily)
# ------------------------------------------------------------
def normalize_hourly_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in cols: return cols[n.lower()]
        return None

    c_date  = pick("date")
    c_hour  = pick("hour")
    c_geoid = pick("geoid","GEOID","cell_id","id")
    c_risk  = pick("risk_score","p_stack","prob","probability","score","risk")
    if not (c_date and c_hour and c_geoid and c_risk):
        raise ValueError("Saatlik veri için 'date, hour, geoid, risk_score' zorunlu.")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[c_date], errors="coerce"),
        "hour": pd.to_numeric(df[c_hour], errors="coerce").astype("Int64").clip(0,23),
        "geoid": df[c_geoid].astype(str),
        "risk_score": pd.to_numeric(df[c_risk], errors="coerce"),
    }).dropna(subset=["date","hour","geoid"]).copy()
    out["timestamp"] = out["date"].dt.floor("D") + pd.to_timedelta(out["hour"].fillna(0).astype(int), unit="h")
    return out[["timestamp","date","hour","geoid","risk_score"]]

def normalize_daily_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in cols: return cols[n.lower()]
        return None

    c_date  = pick("date")
    c_geoid = pick("geoid","GEOID","cell_id","id")
    c_risk  = pick("risk_score","p_stack","prob","probability","score","risk")
    if not (c_date and c_geoid and c_risk):
        raise ValueError("Günlük veri için 'date, geoid, risk_score' zorunlu.")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[c_date], errors="coerce").dt.floor("D"),
        "geoid": df[c_geoid].astype(str),
        "risk_score": pd.to_numeric(df[c_risk], errors="coerce"),
    }).dropna(subset=["date","geoid"]).copy()
    return out[["date","geoid","risk_score"]]

# ------------------------------------------------------------
# 🗺️ Centroid yükleyici (yalnızca artifact içi otomatik arama)
# ------------------------------------------------------------
def coerce_centroids(any_df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {c.lower(): c for c in any_df.columns}
    def pick(*names):
        for n in names:
            if n in any_df.columns: return n
            if n.lower() in cols: return cols[n.lower()]
        return None

    c_geoid = pick("geoid","GEOID","cell_id","id")
    c_lat   = pick("lat","latitude","y")
    c_lon   = pick("lon","lng","longitude","x")
    if not (c_geoid and c_lat and c_lon):
        return None
    out = pd.DataFrame({
        "geoid": any_df[c_geoid].astype(str),
        "lat": pd.to_numeric(any_df[c_lat], errors="coerce"),
        "lon": pd.to_numeric(any_df[c_lon], errors="coerce"),
    }).dropna(subset=["lat","lon"]).copy()
    return out.drop_duplicates("geoid")

@st.cache_data(show_spinner=False)
def load_centroids_from_artifact() -> pd.DataFrame | None:
    try:
        url, headers = resolve_latest_artifact_zip_url(REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN)
        if not url:
            return None
        resp = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
        resp.raise_for_status()
        with zipfile.ZipFile(BytesIO(resp.content)) as z:
            for cand in CENTROID_FILE_CANDIDATES:
                try:
                    with z.open(cand) as f:
                        b = f.read()
                    try:
                        dfm = pd.read_parquet(BytesIO(b))
                    except Exception:
                        dfm = pd.read_csv(BytesIO(b))
                    c = coerce_centroids(dfm)
                    if c is not None and len(c):
                        return c
                except KeyError:
                    continue
        return None
    except Exception:
        return None

# ------------------------------------------------------------
# 🧮 Risk bucket (sabit eşikler)
# ------------------------------------------------------------
RISK_BUCKETS = [
    (0.00, 0.20, "Çok Düşük", [220,220,220,160]),
    (0.20, 0.40, "Düşük",     [180,210,255,200]),
    (0.40, 0.60, "Orta",      [255,220,130,210]),
    (0.60, 0.80, "Yüksek",    [255,170,110,220]),
    (0.80, 1.01, "Çok Yüksek",[255, 90, 90,240]),
]
def bucket_of(v: float) -> str:
    x = 0.0 if pd.isna(v) else float(v)
    for lo, hi, name, _ in RISK_BUCKETS:
        if lo <= x < hi: return name
    return "Çok Düşük"
COLOR_MAP = {name: rgba for _,_,name,rgba in RISK_BUCKETS}

def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")

# ------------------------------------------------------------
# 🎛️ UI — Ayarlar
# ------------------------------------------------------------
st.set_page_config(page_title="🌀 Suç Tahmini", layout="wide")
st.sidebar.header("⚙️ Ayarlar")

# Zaman modu
mode = st.sidebar.radio("Zaman çözünürlüğü", ["Saatlik (≤7 gün)", "Günlük (≤365 gün)"], index=0)

# Saatlik modda SADECE saat aralığı seçimi
def default_hour_block_label(hour_blocks: dict) -> str:
    """
    San Francisco yerel saatine göre şu an hangi blok içindeysek
    o blok etiketini döndürür. Hata olursa '18–21' fallback.
    """
    fallback = "18–21"
    try:
        if ZoneInfo is None:
            raise RuntimeError("ZoneInfo yok")
        now_sf = datetime.now(ZoneInfo("America/Los_Angeles"))
        h = now_sf.hour  # 0–23
        for label, (h0, h1) in hour_blocks.items():
            if h0 <= h <= h1:
                return label
        return fallback
    except Exception:
        return fallback

if mode.startswith("Saatlik"):
    st.sidebar.subheader("Saat Aralığı")
    
    # 3 saatlik bloklar (istersen değiştirilebilir: 2-4-6 saat)
    hour_blocks = {
        "00–03": (0, 2),
        "03–06": (3, 5),
        "06–09": (6, 8),
        "09–12": (9,11),
        "12–15": (12,14),
        "15–18": (15,17),
        "18–21": (18,20),
        "21–24": (21,23),
    }

    # Varsayılan: San Francisco'da şu an hangi bloksa o
    default_label = default_hour_block_label(hour_blocks)

    selected_label = st.sidebar.select_slider(
        "Saat aralığı",
        options=list(hour_blocks.keys()),
        value=default_label,
    )

    h0, h1 = hour_blocks[selected_label]
    selected_hours = list(range(h0, h1+1))

else:
    selected_hours = []

# Tarih aralığı
now = datetime.now()
max_days = 7 if mode.startswith("Saatlik") else 365
st.sidebar.caption(f"{'Saatlik' if max_days==7 else 'Günlük'} görünümde en fazla {max_days} gün seçebilirsiniz.")
d_start = st.sidebar.date_input("Başlangıç tarihi", value=(now - timedelta(days=1)).date())
d_end   = st.sidebar.date_input("Bitiş tarihi",     value=now.date())
if (pd.to_datetime(d_end) - pd.to_datetime(d_start)).days > max_days:
    d_end = (pd.to_datetime(d_start) + pd.Timedelta(days=max_days)).date()
    st.sidebar.warning(f"Seçim {max_days} günü aşamaz; bitiş {d_end} olarak güncellendi.")

# GEOID filtre
geof_txt = st.sidebar.text_input("GEOID filtre (virgülle ayır)", value="")
geoids_sel = [g.strip() for g in geof_txt.split(",") if g.strip()]

# Top-K (tablo)
top_k = st.sidebar.slider("Top-K (tablo)", 10, 200, 50, step=10)

# ------------------------------------------------------------
# 📥 Veri yükleme ve filtre
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_hourly_dataframe() -> pd.DataFrame:
    raw = load_artifact_member(ARTIFACT_MEMBER_HOURLY); return normalize_hourly_schema(raw)

@st.cache_data(show_spinner=False)
def load_daily_dataframe() -> pd.DataFrame:
    raw = load_artifact_member(ARTIFACT_MEMBER_DAILY);  return normalize_daily_schema(raw)

with st.spinner("Veriler yükleniyor…"):
    if mode.startswith("Saatlik"):
        src = load_hourly_dataframe()
        t0 = pd.to_datetime(d_start)
        t1 = pd.to_datetime(d_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = src[(src["timestamp"] >= t0) & (src["timestamp"] <= t1)].copy()
        if geoids_sel: df = df[df["geoid"].isin(geoids_sel)].copy()
        if selected_hours: df = df[df["hour"].isin(selected_hours)].copy()
        agg = df.groupby("geoid", as_index=False)["risk_score"].mean().rename(columns={"risk_score":"risk_mean"})
        view_df = df; time_col = "timestamp"
    else:
        src = load_daily_dataframe()
        t0 = pd.to_datetime(d_start).floor("D"); t1 = pd.to_datetime(d_end).floor("D")
        df = src[(src["date"] >= t0) & (src["date"] <= t1)].copy()
        if geoids_sel: df = df[df["geoid"].isin(geoids_sel)].copy()
        agg = df.groupby("geoid", as_index=False)["risk_score"].mean().rename(columns={"risk_score":"risk_mean"})
        view_df = df; time_col = "date"

if len(agg):
    agg["risk_bucket"] = agg["risk_mean"].map(bucket_of)
    agg_sorted = agg.sort_values("risk_mean", ascending=False).reset_index(drop=True)
else:
    agg_sorted = agg

# ------------------------------------------------------------
# 🗺️ HARİTA — EN ÜSTE
# ------------------------------------------------------------
st.subheader("🗺️ Harita — 5 seviye risk renklendirme")
centroids = load_centroids_from_artifact()
if centroids is None or len(centroids) == 0:
    st.info("Centroid (geoid→lat/lon) artifact içinde bulunamadı. Harita devre dışı.")
else:
    map_df = agg_sorted.merge(centroids, on="geoid", how="left").dropna(subset=["lat","lon"]).copy()
    if len(map_df) == 0:
        st.info("Harita için lat/lon eşleşmesi bulunamadı.")
    else:
        map_df["color"] = map_df["risk_bucket"].map(COLOR_MAP)
        st.markdown(
            "**Lejand:** "
            "<span style='background:#ddd;padding:2px 6px;border-radius:4px;'>Çok Düşük</span> "
            "<span style='background:#b4d2ff;padding:2px 6px;border-radius:4px;'>Düşük</span> "
            "<span style='background:#ffdc82;padding:2px 6px;border-radius:4px;'>Orta</span> "
            "<span style='background:#ffaa6e;padding:2px 6px;border-radius:4px;'>Yüksek</span> "
            "<span style='background:#ff5a5a;padding:2px 6px;border-radius:4px;'>Çok Yüksek</span> ",
            unsafe_allow_html=True
        )
        import pydeck as pdk
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=80 if mode.startswith("Saatlik") else 120,
            pickable=True,
            radius_min_pixels=2,
            radius_max_pixels=20,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(
            latitude=float(map_df["lat"].median()),
            longitude=float(map_df["lon"].median()),
            zoom=11,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "GEOID {geoid}\\nRisk {risk_mean:.3f}\\nSeviye {risk_bucket}"}
        ))

# ------------------------------------------------------------
# 🧠 Özet kartlar
# ------------------------------------------------------------
st.title("🌀 Suç Tahmini — Haritalı GEOID görünüm")
st.caption("Saatlik (tek saat / saat aralığı, ≤7 gün) veya günlük (≤365 gün) pencerede GEOID bazlı ortalama risk.")

c1, c2, c3 = st.columns(3)
c1.metric("Kapsanan kayıt", f"{len(view_df):,}")
c2.metric("GEOID sayısı", f"{agg_sorted['geoid'].nunique():,}" if len(agg_sorted) else "0")
c3.metric("Ortalama risk", f"{view_df['risk_score'].mean():.3f}" if len(view_df) else "—")

# ------------------------------------------------------------
# 🔝 Top-K tablo + indir
# ------------------------------------------------------------
st.subheader("🔝 Top-K GEOID")
topk = agg_sorted.head(top_k).copy()
st.dataframe(topk, use_container_width=True, height=420)
st.download_button("⬇️ CSV indir (Top-K)", data=csv_bytes(topk), file_name="risk_topk.csv", mime="text/csv")

# ------------------------------------------------------------
# 📈 Zaman serisi (seçili GEOID'ler)
# ------------------------------------------------------------
st.subheader("📈 Zaman serisi (risk_score)")
if len(view_df) == 0:
    st.info("Seçilen aralık için veri yok.")
else:
    default_geoids = topk["geoid"].head(3).tolist() if len(topk) else []
    options_geoids = sorted(view_df["geoid"].unique().tolist())
    chosen = st.multiselect("Grafik GEOID seç", options=options_geoids, default=default_geoids)
    if len(chosen):
        piv = (
            view_df[view_df["geoid"].isin(chosen)]
            .pivot_table(index=time_col, columns="geoid", values="risk_score", aggfunc="mean")
            .sort_index()
        )
        if len(piv):
            st.line_chart(piv, height=360)
        else:
            st.caption("Seçilen GEOID'ler için veri yok.")
    else:
        st.caption("Grafik için en az bir GEOID seçin.")

# ------------------------------------------------------------
# 🔥 Isı haritası (GEOID × Saat / Gün)
# ------------------------------------------------------------
st.subheader("🔥 Isı haritası (GEOID × Zaman)")
if len(view_df) == 0:
    st.info("Seçilen aralık için veri yok.")
else:
    heat_index = "hour" if mode.startswith("Saatlik") else "date"
    heat = (
        view_df.groupby([heat_index,"geoid"], as_index=False)["risk_score"].mean()
        .pivot(index=heat_index, columns="geoid", values="risk_score")
        .sort_index()
    )
    visible_cols = topk["geoid"].tolist() if len(topk) else heat.columns.tolist()
    heat = heat[[g for g in heat.columns if g in visible_cols]]
    st.dataframe(heat.style.format("{:.3f}"), use_container_width=True, height=420)

# ------------------------------------------------------------
# 🧾 Dipnot
# ------------------------------------------------------------
st.caption(
    "Kaynak: artifact 'fr-crime-outputs-parquet' → risk_hourly_next24h_top3 / risk_daily_next365d_top5 (parquet). "
    "Harita, centroid (GEOID→lat/lon) dosyası artifact içinde bulunursa otomatik etkinleşir."
)

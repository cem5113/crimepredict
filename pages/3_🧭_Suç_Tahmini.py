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
ARTIFACT_MEMBER_HOURLY = "risk_hourly_next24h_top3"
ARTIFACT_MEMBER_DAILY  = "risk_daily_next365d_top5"

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
    response = requests.get(
        f"{base}/actions/artifacts?per_page=100",
        headers=github_api_headers(),
        timeout=60,
    )
    response.raise_for_status()
    artifacts = (response.json() or {}).get("artifacts", []) or []
    artifacts = [
        a for a in artifacts
        if (name_contains in a.get("name", "")) and not a.get("expired")
    ]
    if not artifacts:
        return None, {}
    artifacts.sort(key=lambda a: a.get("updated_at", ""), reverse=True)
    url = f"{base}/actions/artifacts/{artifacts[0]['id']}/zip"
    return url, github_api_headers()

# ------------------------------------------------------------
# 🧰 ZIP içinden üye okuma (nested zip + parquet/csv fallback)
# ------------------------------------------------------------
def read_member_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """
    Artifact ZIP'inde:
      - önce doğrudan dosyayı arar
      - yoksa içerdeki .zip (örn. fr_parquet_outputs.zip) dosyalarını açıp orada arar.

    member_path: "risk_hourly_next24h_top3" gibi gövde adı.
    """

    def read_any_table(raw_bytes: bytes, name_hint: str) -> pd.DataFrame:
        buf = BytesIO(raw_bytes)
        name_l = name_hint.lower()
        if name_l.endswith(".csv"):
            return pd.read_csv(buf)
        # Önce parquet dene, hata olursa csv'e düş
        try:
            buf.seek(0)
            return pd.read_parquet(buf)
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf)

    def scan_zip(zf: zipfile.ZipFile, member_path: str) -> pd.DataFrame | None:
        """Verilen ZipFile içinde stem'i geçen ilk dosyayı bulup DataFrame döndürür."""
        names = zf.namelist()
        base  = posixpath.basename(member_path)
        stem  = base.split(".")[0]
        stemL = stem.lower()

        for n in names:
            bn = posixpath.basename(n)
            if stemL in bn.lower():
                with zf.open(n) as f:
                    return read_any_table(f.read(), bn)
        return None

    # 1) Dış ZIP'i aç
    with zipfile.ZipFile(BytesIO(zip_bytes)) as outer:
        # Önce dış zip içinde ara
        df = scan_zip(outer, member_path)
        if df is not None:
            return df

        # 2) Bulunamazsa: içerdeki .zip dosyalarını sırayla dene (örneğin fr_parquet_outputs.zip)
        for name in outer.namelist():
            if name.lower().endswith(".zip"):
                with outer.open(name) as f_z:
                    inner_bytes = f_z.read()
                try:
                    with zipfile.ZipFile(BytesIO(inner_bytes)) as inner:
                        df_inner = scan_zip(inner, member_path)
                        if df_inner is not None:
                            return df_inner
                except zipfile.BadZipFile:
                    continue

    # Hiçbir eşleşme bulunamadıysa:
    raise FileNotFoundError(
        f"ZIP içinde '{member_path}' gövdesini içeren bir CSV/PARQUET dosyası bulunamadı."
    )

@st.cache_data(show_spinner=False)
def load_artifact_member(member: str) -> pd.DataFrame:
    url, headers = resolve_latest_artifact_zip_url(
        REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN
    )
    if not url:
        raise RuntimeError("Artifact bulunamadı veya GITHUB_TOKEN yok.")
    r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return read_member_from_zip_bytes(r.content, member)

# ------------------------------------------------------------
# 🧭 Şema doğrulayıcılar (hourly/daily)
#    NOT: Artık tüm FR kolonları korunuyor, sadece zorunlu
#         kolonlar normalize ediliyor.
# ------------------------------------------------------------
def normalize_hourly_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date  = pick("date")
    c_hour  = pick("hour", "hour_idx", "hour_of_day", "hour_index")
    c_geoid = pick("geoid", "GEOID", "cell_id", "id")
    c_risk  = pick("risk_score", "p_stack", "prob", "probability", "score", "risk")

    if not (c_date and c_hour and c_geoid and c_risk):
        raise ValueError("Saatlik veri için 'date, hour, geoid, risk_score' zorunlu.")

    df["date"] = pd.to_datetime(df[c_date], errors="coerce")
    df["hour"] = pd.to_numeric(df[c_hour], errors="coerce").astype("Int64").clip(0, 23)
    df["geoid"] = df[c_geoid].astype(str)
    df["risk_score"] = pd.to_numeric(df[c_risk], errors="coerce")

    df = df.dropna(subset=["date", "hour", "geoid"]).copy()
    df["timestamp"] = df["date"].dt.floor("D") + pd.to_timedelta(
        df["hour"].fillna(0).astype(int), unit="h"
    )
    return df

def normalize_daily_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date  = pick("date")
    c_geoid = pick("geoid", "GEOID", "cell_id", "id")
    c_risk  = pick("risk_score", "p_stack", "prob", "probability", "score", "risk")

    if not (c_date and c_geoid and c_risk):
        raise ValueError("Günlük veri için 'date, geoid, risk_score' zorunlu.")

    df["date"] = pd.to_datetime(df[c_date], errors="coerce").dt.floor("D")
    df["geoid"] = df[c_geoid].astype(str)
    df["risk_score"] = pd.to_numeric(df[c_risk], errors="coerce")

    df = df.dropna(subset=["date", "geoid"]).copy()
    return df

# ------------------------------------------------------------
# 🗺️ Centroid yükleyici (yalnızca artifact içi otomatik arama)
# ------------------------------------------------------------
def coerce_centroids(any_df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {c.lower(): c for c in any_df.columns}

    def pick(*names):
        for n in names:
            if n in any_df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_geoid = pick("geoid", "GEOID", "cell_id", "id")
    c_lat   = pick("lat", "latitude", "y")
    c_lon   = pick("lon", "lng", "longitude", "x")
    if not (c_geoid and c_lat and c_lon):
        return None
    out = pd.DataFrame({
        "geoid": any_df[c_geoid].astype(str),
        "lat": pd.to_numeric(any_df[c_lat], errors="coerce"),
        "lon": pd.to_numeric(any_df[c_lon], errors="coerce"),
    }).dropna(subset=["lat", "lon"]).copy()
    return out.drop_duplicates("geoid")

@st.cache_data(show_spinner=False)
def load_centroids_from_artifact() -> pd.DataFrame | None:
    """
    Artifact ZIP'i içinden centroidleri yükler.

    Önce doğrudan:
        geoid_centroids
        sf_geoid_centroids
    sonra fallback olarak:
        sf_crime_grid_full_labeled

    isimli dosyaların *stem*'ini kullanır, yani zip içindeki
    klasör yapısı ve .csv/.parquet uzantısı fark etmez.
    """
    try:
        url, headers = resolve_latest_artifact_zip_url(
            REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN
        )
        if not url:
            return None

        resp = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
        resp.raise_for_status()
        zip_bytes = resp.content

        # Önce net centroid dosyalarını dene
        centroid_stems = [
            "geoid_centroids",
            "sf_geoid_centroids",
        ]

        for stem in centroid_stems:
            try:
                dfm = read_member_from_zip_bytes(zip_bytes, stem)
                c = coerce_centroids(dfm)
                if c is not None and len(c):
                    return c
            except FileNotFoundError:
                continue

        # Fallback: grid dosyasında lat/lon varsa oradan centroid üret
        grid_stems = [
            "sf_crime_grid_full_labeled",
        ]
        for stem in grid_stems:
            try:
                dfm = read_member_from_zip_bytes(zip_bytes, stem)
                c = coerce_centroids(dfm)
                if c is not None and len(c):
                    return c
            except FileNotFoundError:
                continue

        return None

    except Exception:
        return None

# ------------------------------------------------------------
# 🧮 Risk bucket (sabit eşikler)
# ------------------------------------------------------------
RISK_BUCKETS = [
    (0.00, 0.20, "Çok Düşük", [220, 220, 220, 160]),
    (0.20, 0.40, "Düşük",     [180, 210, 255, 200]),
    (0.40, 0.60, "Orta",      [255, 220, 130, 210]),
    (0.60, 0.80, "Yüksek",    [255, 170, 110, 220]),
    (0.80, 1.01, "Çok Yüksek",[255,  90,  90, 240]),
]

def bucket_of(v: float) -> str:
    x = 0.0 if pd.isna(v) else float(v)
    for lo, hi, name, _ in RISK_BUCKETS:
        if lo <= x < hi:
            return name
    return "Çok Düşük"

COLOR_MAP = {name: rgba for _, _, name, rgba in RISK_BUCKETS}

def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

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

    # 3 saatlik bloklar
    hour_blocks = {
        "00–03": (0, 2),
        "03–06": (3, 5),
        "06–09": (6, 8),
        "09–12": (9, 11),
        "12–15": (12, 14),
        "15–18": (15, 17),
        "18–21": (18, 20),
        "21–24": (21, 23),
    }

    default_label = default_hour_block_label(hour_blocks)

    selected_label = st.sidebar.select_slider(
        "Saat aralığı",
        options=list(hour_blocks.keys()),
        value=default_label,
    )

    h0, h1 = hour_blocks[selected_label]
    selected_hours = list(range(h0, h1 + 1))
else:
    selected_hours = []

# Tarih aralığı
now = datetime.now()
max_days = 7 if mode.startswith("Saatlik") else 365
st.sidebar.caption(f"{'Saatlik' if max_days == 7 else 'Günlük'} görünümde en fazla {max_days} gün seçebilirsiniz.")

# 🔁 MOD: Saatlik ve Günlük mod için farklı varsayılan tarih aralığı
if mode.startswith("Saatlik"):
    # Saatlik görünüm: dün–bugün (eski davranış korunuyor)
    d_start_default = (now - timedelta(days=1)).date()
    d_end_default   = now.date()
else:
    # Günlük (365 gün) görünüm: sadece "bugün"
    d_start_default = now.date()
    d_end_default   = now.date()

d_start = st.sidebar.date_input("Başlangıç tarihi", value=d_start_default)
d_end   = st.sidebar.date_input("Bitiş tarihi",     value=d_end_default)

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
    raw = load_artifact_member(ARTIFACT_MEMBER_HOURLY)
    return normalize_hourly_schema(raw)

@st.cache_data(show_spinner=False)
def load_daily_dataframe() -> pd.DataFrame:
    raw = load_artifact_member(ARTIFACT_MEMBER_DAILY)
    return normalize_daily_schema(raw)

agg = pd.DataFrame()
view_df = pd.DataFrame()
view_df_city = pd.DataFrame()   # geoid == "0" (şehir geneli)
view_df_cells = pd.DataFrame()  # geoid != "0" (hücreler)
time_col = "timestamp"

with st.spinner("Veriler yükleniyor…"):
    if mode.startswith("Saatlik"):
        src = load_hourly_dataframe()
        t0 = pd.to_datetime(d_start)
        t1 = pd.to_datetime(d_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = src[(src["timestamp"] >= t0) & (src["timestamp"] <= t1)].copy()
        if geoids_sel:
            df = df[df["geoid"].isin(geoids_sel)].copy()
        if selected_hours:
            df = df[df["hour"].isin(selected_hours)].copy()

        # Eğer bu pencere için hiç kayıt yoksa → tüm saatlik risk çıktısını kullan
        if df.empty:
            st.info(
                "Seçilen tarih/saat aralığı için kayıt bulunamadı; "
                "en güncel saatlik risk çıktısı gösteriliyor."
            )
            df = src.copy()

        view_df = df
        time_col = "timestamp"

    else:
        src = load_daily_dataframe()
        t0 = pd.to_datetime(d_start).floor("D")
        t1 = pd.to_datetime(d_end).floor("D")
        df = src[(src["date"] >= t0) & (src["date"] <= t1)].copy()
        if geoids_sel:
            df = df[df["geoid"].isin(geoids_sel)].copy()

        # Eğer bu pencere için hiç kayıt yoksa → tüm günlük risk çıktısını kullan
        if df.empty:
            st.info(
                "Seçilen tarih aralığı için kayıt bulunamadı; "
                "en güncel günlük risk çıktısı gösteriliyor."
            )
            df = src.copy()

        view_df = df
        time_col = "date"

    if len(view_df):
        # GEOID=0 → şehir geneli, diğerleri hücreler
        mask_city = view_df["geoid"].astype(str) == "0"
        view_df_city = view_df[mask_city].copy()
        view_df_cells = view_df[~mask_city].copy()

        if len(view_df_cells):
            # Temel GEOID bazlı risk ortalaması (sadece hücreler)
            agg = (
                view_df_cells.groupby("geoid", as_index=False)["risk_score"]
                .mean()
                .rename(columns={"risk_score": "risk_mean"})
            )
        else:
            view_df_cells = pd.DataFrame()
            agg = pd.DataFrame()

        # Opsiyonel kolonları GEOID bazında özetle (risk_prob, expected_crimes, top1_category vs.)
        def safe_mean(col_name: str):
            if len(view_df_cells) and col_name in view_df_cells.columns:
                return view_df_cells.groupby("geoid", as_index=False)[col_name].mean()
            return None

        def safe_first(col_name: str):
            if len(view_df_cells) and col_name in view_df_cells.columns:
                tmp = (
                    view_df_cells.sort_values(time_col)
                    .groupby("geoid", as_index=False)[col_name]
                    .first()
                )
                return tmp
            return None

        for c in ["risk_prob", "expected_crimes", "expected_count"]:
            tmp = safe_mean(c)
            if tmp is not None and len(tmp):
                agg = agg.merge(tmp, on="geoid", how="left")

        for c in ["risk_level", "risk_decile", "top1_category"]:
            tmp = safe_first(c)
            if tmp is not None and len(tmp):
                agg = agg.merge(tmp, on="geoid", how="left")

# ------------------------------------------------------------
# 🔎 DEBUG — Artifact ZIP içindeki dosya isimlerini göster
# ------------------------------------------------------------
with st.expander("🔎 Artifact içindeki dosya isimleri (debug)", expanded=False):
    try:
        url, headers = resolve_latest_artifact_zip_url(
            REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN
        )
        if url:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            with zipfile.ZipFile(BytesIO(r.content)) as outer:
                names_outer = outer.namelist()
            st.write(f"Dış ZIP toplam dosya: {len(names_outer)}")
            st.write(names_outer)

            with zipfile.ZipFile(BytesIO(r.content)) as outer2:
                for name in outer2.namelist():
                    if name.lower().endswith(".zip"):
                        with outer2.open(name) as f_z:
                            inner_bytes = f_z.read()
                        try:
                            with zipfile.ZipFile(BytesIO(inner_bytes)) as inner:
                                st.write(f"İç ZIP: {name}")
                                st.write(inner.namelist())
                        except zipfile.BadZipFile:
                            st.write(f"İç ZIP açılamadı: {name}")
        else:
            st.warning("Artifact bulunamadı veya token eksik.")
    except Exception as e:
        st.error(f"Debug sırasında hata: {e}")

# ------------------------------------------------------------
# 🗺️ HARİTA — EN ÜSTE
# ------------------------------------------------------------
if len(agg):
    agg["risk_bucket"] = agg["risk_mean"].map(bucket_of)
    agg_sorted = agg.sort_values("risk_mean", ascending=False).reset_index(drop=True)
else:
    agg_sorted = agg

st.subheader("🗺️ Harita — 5 seviye risk renklendirme")
centroids = load_centroids_from_artifact()

if centroids is None or len(centroids) == 0 or len(agg_sorted) == 0:
    if len(view_df_city) and not len(agg_sorted):
        st.info(
            "Bu aralıkta sadece şehir geneli (GEOID=0) için risk üretilmiş; "
            "hücre (GEOID) bazlı risk olmadığı için harita devre dışı."
        )
    else:
        st.info("Centroid (geoid→lat/lon) veya hücre bazlı risk verisi bulunamadı. Harita devre dışı.")

else:
    map_df = (
        agg_sorted.merge(centroids, on="geoid", how="left")
        .dropna(subset=["lat", "lon"])
        .copy()
    )
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
            unsafe_allow_html=True,
        )

        import pydeck as pdk

        # Tooltip'te opsiyonel alanlar varsa göster
        tooltip_text = "GEOID {geoid}\\nOrtalama risk {risk_mean:.3f}\\nSeviye {risk_bucket}"
        if "expected_crimes" in map_df.columns:
            tooltip_text += "\\nBeklenen suç {expected_crimes:.3f}"
        elif "expected_count" in map_df.columns:
            tooltip_text += "\\nBeklenen suç {expected_count:.3f}"
        if "top1_category" in map_df.columns:
            tooltip_text += "\\nEn olası tür {top1_category}"

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
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": tooltip_text},
            )
        )

# ------------------------------------------------------------
# 🧠 Özet kartlar
# ------------------------------------------------------------
st.title("🌀 Suç Tahmini — Haritalı GEOID görünüm")
st.caption(
    "Saatlik (tek saat / saat aralığı, ≤7 gün) veya günlük (≤365 gün) pencerede GEOID bazlı ortalama risk."
)

c1, c2, c3 = st.columns(3)
c1.metric("Kapsanan kayıt", f"{len(view_df):,}")
c2.metric(
    "GEOID sayısı",
    f"{agg_sorted['geoid'].nunique():,}" if len(agg_sorted) else "0",
)
c3.metric(
    "Ortalama risk",
    f"{view_df['risk_score'].mean():.3f}" if len(view_df) else "—",
)

# GEOID etiketleyici (0 için özel label)
def geoid_label(g: str) -> str:
    return "Şehir geneli (GEOID=0)" if str(g) == "0" else str(g)

# GEOID seçimi (detay sekmeleri için)
options = []
# Önce şehir geneli (varsa)
if len(view_df_city):
    options.append("0")
# Sonra hücreler (harita/Top-K ile tutarlı)
if len(view_df_cells):
    options.extend(sorted(view_df_cells["geoid"].unique().tolist()))

if options:
    selected_geoid = st.selectbox(
        "Detay göstermek için GEOID seç:",
        options=options,
        index=0,
        format_func=geoid_label,
    )
else:
    selected_geoid = None

# Top-K her halükârda hesaplayalım (sadece hücreler üzerinden)
topk = agg_sorted.head(top_k).copy() if len(agg_sorted) else pd.DataFrame()

# ------------------------------------------------------------
# 🔍 Sekmeli görünüm: Özet & nedenler / Zaman serisi / Isı haritası & Top-K
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Özet & Nedenler", "Zaman Serisi", "Isı Haritası / Top-K"])

# --------------------------- TAB 1: Özet & Nedenler ---------------------------
with tab1:
    st.subheader("📌 Seçili GEOID için risk özeti ve nedenler")

    if selected_geoid is None or len(view_df) == 0:
        st.info("Görüntülenecek veri bulunamadı.")
    else:
        df_sel = (
            view_df[view_df["geoid"] == selected_geoid]
            .sort_values(time_col)
            .copy()
        )
        if len(df_sel) == 0:
            st.info("Seçili GEOID için veri yok.")
        else:
            # 🔁 MOD: Günlük (365 gün) modunda mümkünse "bugün" satırını kullan
            if time_col == "date":
                today_dt = pd.to_datetime(datetime.now().date())
                mask_today = df_sel[time_col] == today_dt
                if mask_today.any():
                    latest = df_sel[mask_today].iloc[0]
                else:
                    # Bugün yoksa, eski davranış: en son satır
                    latest = df_sel.iloc[-1]
            else:
                # Saatlik modda olduğu gibi son satırı kullan
                latest = df_sel.iloc[-1]
        
            def gv(col, default="—"):
                return latest[col] if col in df_sel.columns and pd.notna(latest[col]) else default

            # Üstte küçük metrik kartlar
            c1, c2, c3 = st.columns(3)
            c1.metric("GEOID", selected_geoid)
            c2.metric("Son pencere risk skoru", f"{gv('risk_score', np.nan):.4f}" if gv("risk_score", np.nan) == gv("risk_score", np.nan) else "—")
            if "risk_prob" in df_sel.columns:
                c3.metric("Risk olasılığı", f"{gv('risk_prob', np.nan):.4f}" if gv("risk_prob", np.nan) == gv("risk_prob", np.nan) else "—")
            elif "risk_mean" in agg_sorted.columns:
                c3.metric("Ortalama risk", f"{float(agg_sorted.loc[agg_sorted['geoid']==selected_geoid, 'risk_mean'].iloc[0]):.4f}")
            else:
                c3.metric("Ortalama risk", "—")

            # İkinci satır: beklenen suç, komşu suç, 911/311
            c4, c5, c6 = st.columns(3)
            if "expected_crimes" in df_sel.columns:
                c4.metric("Beklenen suç (son pencere)", f"{gv('expected_crimes', np.nan):.4f}" if gv("expected_crimes", np.nan) == gv("expected_crimes", np.nan) else "—")
            elif "expected_count" in df_sel.columns:
                c4.metric("Beklenen suç (son pencere)", f"{gv('expected_count', np.nan):.4f}" if gv("expected_count", np.nan) == gv("expected_count", np.nan) else "—")
            else:
                c4.metric("Beklenen suç", "—")

            if "neighbor_crime_7d" in df_sel.columns:
                c5.metric("Komşu suç (7gün)", f"{gv('neighbor_crime_7d', 0):.1f}")
            elif "neighbor_crime_24h" in df_sel.columns:
                c5.metric("Komşu suç (24s)", f"{gv('neighbor_crime_24h', 0):.1f}")
            else:
                c5.metric("Komşu suç", "—")

            if "911_request_count_hour_range" in df_sel.columns:
                c6.metric("911 çağrıları (saat aralığı)", f"{gv('911_request_count_hour_range', 0):.1f}")
            elif "911_geo_last3d" in df_sel.columns:
                c6.metric("911 çağrıları (3gün)", f"{gv('911_geo_last3d', 0):.1f}")
            else:
                c6.metric("911 çağrıları", "—")

            # POI / ulaşım / demografi
            c7, c8, c9 = st.columns(3)
            if "poi_risk_score" in df_sel.columns:
                c7.metric("POI risk skoru", f"{gv('poi_risk_score', 0):.2f}")
            elif "poi_total_count" in df_sel.columns:
                c7.metric("POI sayısı", f"{gv('poi_total_count', 0):.0f}")
            else:
                c7.metric("POI", "—")

            if "bus_stop_count" in df_sel.columns:
                c8.metric("Otobüs durağı sayısı", f"{gv('bus_stop_count', 0):.0f}")
            elif "train_stop_count" in df_sel.columns:
                c8.metric("Tren durağı sayısı", f"{gv('train_stop_count', 0):.0f}")
            else:
                c8.metric("Toplu taşıma", "—")

            if "population" in df_sel.columns:
                c9.metric("Nüfus", f"{gv('population', 0):,.0f}")
            else:
                c9.metric("Nüfus", "—")

            # Hava durumu / zaman bayrakları mini satır
            flags = []
            if "wx_tavg" in df_sel.columns:
                flags.append(f"Ortalama sıcaklık: {gv('wx_tavg', '—')}")
            if "wx_prcp" in df_sel.columns:
                flags.append(f"Yağış (mm): {gv('wx_prcp', '—')}")
            if "wx_is_rainy" in df_sel.columns:
                if gv("wx_is_rainy", 0) == 1:
                    flags.append("Yağışlı gün")
            if "wx_is_hot_day" in df_sel.columns:
                if gv("wx_is_hot_day", 0) == 1:
                    flags.append("Sıcak gün")

            if "is_night" in df_sel.columns:
                flags.append("Gece" if gv("is_night", 0) == 1 else "Gündüz")
            if "is_weekend" in df_sel.columns:
                flags.append("Hafta sonu" if gv("is_weekend", 0) == 1 else "Hafta içi")
            if "is_holiday" in df_sel.columns:
                if gv("is_holiday", 0) == 1:
                    flags.append("Resmî tatil")
            if "is_business_hour" in df_sel.columns:
                if gv("is_business_hour", 0) == 1:
                    flags.append("Mesai saatleri")
            if "is_school_hour" in df_sel.columns:
                if gv("is_school_hour", 0) == 1:
                    flags.append("Okul saatleri")

            # Sezon / gün / saat aralığı ek bilgi
            if "season_x" in df_sel.columns:
                flags.append(f"Mevsim: {gv('season_x', '—')}")
            if "day_of_week_x" in df_sel.columns:
                flags.append(f"Gün: {gv('day_of_week_x', '—')}")
            if "hour_range_x" in df_sel.columns:
                flags.append(f"Saat aralığı: {gv('hour_range_x', '—')}")

            if len(flags):
                st.markdown(
                    "<br>".join([f"• {f}" for f in flags]),
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ----------------- NEDENLER / AÇIKLAMA BLOĞU -----------------
            st.markdown("### 🧠 Modelin öne çıkardığı nedenler")

            reasons = []
            for i in range(1, 6):
                col = f"reason_{i}"
                if col in df_sel.columns:
                    txt = gv(col, "")
                    if isinstance(txt, str) and txt.strip():
                        reasons.append(txt.strip())

            col_left, col_right = st.columns([1, 1])

            with col_left:
                if reasons:
                    st.markdown("**Ana nedenler (otomatik açıklamalar):**")
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.info("Bu GEOID için kayıtlı ayrıntılı 'reason_1–5' açıklaması bulunamadı.")

            with col_right:
                if "explanation_report" in df_sel.columns:
                    rep = gv("explanation_report", "")
                    if isinstance(rep, str) and rep.strip():
                        st.markdown("**Detaylı açıklama raporu:**")
                        st.markdown(
                            f"<div style='max-height:260px; overflow:auto; padding:6px; "
                            f"border-radius:6px; border:1px solid #ddd; background-color:#fafafa;'>"
                            f"{rep}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Detaylı açıklama raporu boş.")
                else:
                    st.caption("`explanation_report` alanı bu dataset içinde yok.")

            st.markdown("---")

            # ----------------- SUÇ TÜRÜ KOMPOZİSYONU TABLOSU -----------------
            st.markdown("### 🧬 Beklenen suç türü kompozisyonu")

            # Top1–Top5 ve pay/olasılık/expected sütunlarını derle
            rows = []
            for k in range(1, 6):
                cat_col = f"top{k}_category"
                share_col = f"top{k}_share"
                prob_col = f"top{k}_prob"
                exp_col = f"top{k}_expected"

                if cat_col not in df_sel.columns:
                    continue

                cat = gv(cat_col, "")
                if not isinstance(cat, str) or not cat.strip():
                    continue

                row = {"Sıra": k, "Suç türü": cat}

                if share_col in df_sel.columns:
                    row["Pay (share)"] = gv(share_col, np.nan)
                if prob_col in df_sel.columns:
                    row["Olasılık (prob)"] = gv(prob_col, np.nan)
                if exp_col in df_sel.columns:
                    row["Beklenen sayı"] = gv(exp_col, np.nan)

                rows.append(row)

            if rows:
                df_comp = pd.DataFrame(rows)
                st.dataframe(
                    df_comp.style.format(
                        {
                            "Pay (share)": "{:.3f}",
                            "Olasılık (prob)": "{:.3f}",
                            "Beklenen sayı": "{:.3f}",
                        }
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Top1–Top5 suç türü kompozisyon bilgisi bu GEOID için bulunamadı.")

with tab2:
    st.subheader("📈 Zaman serisi (risk_score)")

    if len(view_df) == 0:
        st.info("Seçilen tarih/saat aralığı için veri yok.")
    else:
        # Varsayılan: varsa önce şehir geneli, sonra Top-K içindeki ilk 3 hücre
        default_geoids = []
        if len(view_df_city):
            default_geoids.append("0")
        if len(topk):
            default_geoids.extend(topk["geoid"].head(3).tolist())

        # Seçilebilir GEOID listesi: şehir geneli + hücreler
        options_geoids = []
        if len(view_df_city):
            options_geoids.append("0")
        # Hücreler (0 hariç)
        options_geoids.extend(
            sorted([g for g in view_df["geoid"].astype(str).unique().tolist() if g != "0"])
        )

        chosen = st.multiselect(
            "Grafikte gösterilecek GEOID'ler",
            options=options_geoids,
            default=default_geoids,
            format_func=geoid_label,
        )

        if len(chosen):
            piv = (
                view_df[view_df["geoid"].isin(chosen)]
                .pivot_table(
                    index=time_col,
                    columns="geoid",
                    values="risk_score",
                    aggfunc="mean",
                )
                .sort_index()
            )
            if len(piv):
                st.line_chart(piv, height=360)
            else:
                st.caption("Seçilen GEOID'ler için veri yok.")
        else:
            st.caption("Grafik için en az bir GEOID seçin.")

# --------------------------- TAB 3: Isı Haritası & Top-K ---------------------------
with tab3:
    st.subheader("🔥 Isı haritası (GEOID × Zaman)")

    if len(view_df) == 0:
        st.info("Seçilen aralık için veri yok.")
    else:
        heat_index = "hour" if mode.startswith("Saatlik") else "date"
        heat = (
            view_df.groupby([heat_index, "geoid"], as_index=False)["risk_score"]
            .mean()
            .pivot(index=heat_index, columns="geoid", values="risk_score")
            .sort_index()
        )

        # 🔥 Isı haritası TÜM GEOID'leri kapsar (şehir geneli = 0 dahil)
        st.dataframe(
            heat.style.format("{:.3f}"),
            use_container_width=True,
            height=420,
        )

        st.markdown("---")
        st.subheader("🔝 Top-K GEOID tablo & indir")

        if len(topk):
            st.dataframe(topk, use_container_width=True, height=320)
            st.download_button(
                "⬇️ CSV indir (Top-K)",
                data=csv_bytes(topk),
                file_name="risk_topk.csv",
                mime="text/csv",
            )
        else:
            st.caption("Top-K tablosu için yeterli veri yok.")

# ------------------------------------------------------------
# 🧾 Dipnot
# ------------------------------------------------------------
st.caption(
    "Kaynak: artifact 'fr-crime-outputs-parquet' → "
    "risk_hourly_next24h_top3 / risk_daily_next365d_top5 (parquet veya csv). "
    "Harita, centroid (GEOID→lat/lon) dosyası artifact içinde bulunursa otomatik etkinleşir."
)

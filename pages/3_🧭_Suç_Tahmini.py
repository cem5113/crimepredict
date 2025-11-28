# 3_🧭_Suç_Tahmini — Haritalı görünüm (GEOID + centroid) 
# 3 Saatlik Bloklar (≤7 gün; 3-saatlik aralık) ve Günlük (≤365 gün) risk görünümleri
# Kaynak: artifact 'fr-crime-outputs-parquet' → risk_hourly_next24h_top3 / risk_daily_next365d_top5
# Not: Harita için centroid yalnızca artifact içindeki adaylardan bulunur (upload yok).

import os
from streamlit_folium import st_folium
import folium
import io
import json
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
import pydeck as pdk

# ------------------------------------------------------------
# ⚙️ GitHub repo ve artifact bilgisi
# ------------------------------------------------------------
REPOSITORY_OWNER = "cem5113"
REPOSITORY_NAME  = "crime_prediction_data"
ARTIFACT_NAME_SHOULD_CONTAIN = "fr-crime-outputs-parquet"  # FR risk çıktıları artifact'i

# Artifact içindeki beklenen dosyalar (FR pipeline risk çıktıları)
ARTIFACT_MEMBER_HOURLY = "risk_3h_next7d_top3"
ARTIFACT_MEMBER_DAILY  = "risk_daily_next365d_top5"

# 🔁 Yeni 3-saatlik CSV (FR style) için yerel yol
CSV_HOURLY_FRSTYLE = "data/crime_forecast_7days_all_geoids_FRstyle.csv"

# Yerel GeoJSON (2_🗺️_Risk_Haritası.py ile aynı)
GEOJSON_LOCAL = "data/sf_cells.geojson"

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
    """
    risk_3h_next7d_top3 veya crime_forecast_7days_all_geoids_FRstyle.csv için
    saatlik (3-saatlik blok) şema normalizasyonu.

    Desteklenen kolonlar:
      - date
      - geoid
      - risk_score / p_stack / prob / probability / score / risk
      - hour  veya  hour_range_3h / hour_range / hour_block

    Eğer hour yoksa, hour_range_3h içinden başlangıç saati (0,3,6,...) çıkarılır
    ve 'hour' kolonuna yazılır. 'timestamp' = date + hour (saat) olarak üretilir.
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date   = pick("date")
    c_hour   = pick("hour", "hour_idx", "hour_of_day", "hour_index")
    c_hrange = pick("hour_range_3h", "hour_range", "hour_block")
    c_geoid  = pick("geoid", "GEOID", "cell_id", "id")
    c_risk   = pick("risk_score", "p_stack", "prob", "probability", "score", "risk")

    if not (c_date and c_geoid and c_risk and (c_hour or c_hrange)):
        raise ValueError(
            "Saatlik veri için 'date, geoid, risk_score' ve 'hour' veya "
            "'hour_range_3h' benzeri bir kolon zorunlu."
        )

    # Tarih
    df["date"] = pd.to_datetime(df[c_date], errors="coerce")

    # GEOID ve risk skoru
    df["geoid"] = df[c_geoid].astype(str)
    df["risk_score"] = pd.to_numeric(df[c_risk], errors="coerce")

    # Saat: varsa doğrudan 'hour', yoksa hour_range_3h içinden başlangıç saati
    if c_hour:
        df["hour"] = (
            pd.to_numeric(df[c_hour], errors="coerce")
            .astype("Int64")
            .clip(0, 23)
        )
    else:
        def parse_start_hour(val) -> float:
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            # farklı tire karakterlerini normalize et
            s = s.replace("–", "-").replace("—", "-")
            if "-" not in s:
                return np.nan
            a, _ = s.split("-", 1)
            try:
                h0 = int(a.strip())
                # 0–23 aralığına zorla
                h0 = max(0, min(23, h0))
                return h0
            except Exception:
                return np.nan

        df["hour"] = df[c_hrange].map(parse_start_hour).astype("Int64")

    # İsteğe bağlı: hour_range stringini de sakla (ileride lazım olursa)
    if c_hrange:
        df["hour_range_3h"] = df[c_hrange].astype(str)

    # Geçersiz satırları at
    df = df.dropna(subset=["date", "hour", "geoid"]).copy()

    # Zaman damgası: tarih + saat
    df["timestamp"] = df["date"].dt.floor("D") + pd.to_timedelta(
        df["hour"].fillna(0).astype(int),
        unit="h",
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

def rgba_to_hex(rgba):
    """[r,g,b,a] → '#rrggbb'"""
    try:
        r, g, b, _ = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    except Exception:
        return "#dddddd"
        
# ------------------------------------------------------------
# 🧩 GEOID normalizasyonu (harita için 11 haneli + şehir geneli = '0')
# ------------------------------------------------------------
def normalize_geoid_for_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 'geoid' kolonu varsa:
      * GEOID=0 → '0' olarak kalır (şehir geneli)
      * Diğer tüm değerler → sayıya çevrilip 11 haneli zero-pad yapılır
        (örn. 6075010101 → '06075010101')
    """
    df = df.copy()
    if "geoid" not in df.columns:
        return df

    # Önce hepsini string yap
    df["geoid"] = df["geoid"].astype(str)

    # Şehir geneli satırlar
    mask_city = df["geoid"].isin(["0", "0.0"])

    # Hücre satırları
    mask_cells = ~mask_city

    if mask_cells.any():
        df.loc[mask_cells, "geoid"] = (
            pd.to_numeric(df.loc[mask_cells, "geoid"], errors="coerce")
              .astype("Int64")
              .astype(str)
              .str.zfill(11)
        )

    # Şehir geneli satırları tek tip olsun
    if mask_city.any():
        df.loc[mask_city, "geoid"] = "0"

    return df
    
# ------------------------------------------------------------
# 🗺️ GeoJSON yükleyici & özellik zenginleştirme
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    """
    Yerel sf_cells.geojson dosyasını okur.
    2_🗺️_Risk_Haritası.py ile aynı mantık.
    """
    if os.path.exists(GEOJSON_LOCAL):
        with open(GEOJSON_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _digits11(x) -> str:
    """
    GeoJSON properties içindeki GEOID adayını 11 haneye zorlar.
    """
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def enrich_geojson_with_risk(gj: dict, agg_df: pd.DataFrame) -> dict:
    """
    sf_cells.geojson içindeki her hücreye:
      - risk_mean
      - risk_bucket
      - expected_count (varsa)
      - top1_category (varsa)
    gibi özet bilgileri yazar ve fill_color atar.
    """
    if not gj or agg_df is None or agg_df.empty:
        return gj

    agg_df = agg_df.copy()
    # GEOID'ler zaten normalize_geoid_for_map ile gelmiş olmalı
    agg_df["geoid"] = agg_df["geoid"].astype(str)
    risk_map = agg_df.set_index("geoid")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        # GeoJSON içinden GEOID adayı bul
        raw = None
        for k in ("geoid", "GEOID", "cell_id", "id", "geoid11", "geoid_11"):
            if k in props:
                raw = props[k]
                break
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v
                    break

        key = _digits11(raw)
        props["display_id"] = str(raw) if raw not in (None, "") else key
        props["geoid_norm"] = key

        # Varsayılan boş değerler
        props.setdefault("risk_mean_txt", "")
        props.setdefault("risk_bucket", "")
        props.setdefault("expected_count_txt", "")
        props.setdefault("top1_category", "")
        props.setdefault("fill_color", [220, 220, 220, 160])  # Çok düşük default

        if key and key in risk_map.index:
            row = risk_map.loc[key]

            # Risk bucket ve renk
            bucket = row.get("risk_bucket", "")
            if not bucket and "risk_mean" in row:
                bucket = bucket_of(row["risk_mean"])
            props["risk_bucket"] = str(bucket)

            color = COLOR_MAP.get(bucket, [220, 220, 220, 160])
            props["fill_color"] = color

            # Ortalama risk
            try:
                r = float(row.get("risk_mean", np.nan))
                if r == r:
                    props["risk_mean_txt"] = f"{min(max(r, 0.0), 0.999):.3f}"
            except Exception:
                pass

            # Beklenen suç (gün/saat başı)
            def f3(x):
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return ""

            if "expected_crimes" in row.index:
                props["expected_count_txt"] = f3(row["expected_crimes"])
            elif "expected_count" in row.index:
                props["expected_count_txt"] = f3(row["expected_count"])

            # En olası suç türü
            if "top1_category" in row.index:
                props["top1_category"] = str(row["top1_category"] or "")

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}

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
mode = st.sidebar.radio(
    "Zaman çözünürlüğü",
    ["3 Saatlik Bloklar (≤7 gün)", "Günlük (≤365 gün)"],
    index=0,
)

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

if mode.startswith("3 Saatlik"):
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

# Tarih aralığı  ➜ SF yerel zamanına göre
if ZoneInfo is not None:
    now_sf = datetime.now(ZoneInfo("America/Los_Angeles"))
else:
    # Fallback: UTC / sistem zamanı
    now_sf = datetime.utcnow()

max_days = 7 if mode.startswith("3 Saatlik") else 365
st.sidebar.caption(
    f"{'3 Saatlik' if max_days == 7 else 'Günlük'} görünümde en fazla {max_days} gün seçebilirsiniz. "
    "(San Francisco yerel zamanı baz alınır.)"
)

# 🔁 MOD: Saatlik ve Günlük mod için farklı varsayılan tarih aralığı
if mode.startswith("3 Saatlik"):
    # 3 Saatlik görünüm: SF bugün (sadece bugünün blokları)
    d_start_default = now_sf.date()
    d_end_default   = now_sf.date()
else:
    # Günlük (365 gün) görünüm: SF bugün
    d_start_default = now_sf.date()
    d_end_default   = now_sf.date()

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
    """
    3-saatlik blok veri kaynağı:
      1) Eğer mevcutsa: data/crime_forecast_7days_all_geoids_FRstyle.csv
      2) Değilse: artifact içindeki risk_3h_next7d_top3
    """
    # Önce yerel CSV'yi dene
    if os.path.exists(CSV_HOURLY_FRSTYLE):
        st.sidebar.success("3-saatlik veri kaynağı: 🔹 Yerel CSV (FRstyle)")
        raw = pd.read_csv(CSV_HOURLY_FRSTYLE)
        return normalize_hourly_schema(raw)

    # CSV yoksa eski davranış: artifact'ten oku
    st.sidebar.warning("3-saatlik veri kaynağı: 🪣 GitHub artifact (risk_3h_next7d_top3)")
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
    if mode.startswith("3 Saatlik"):
        src = load_hourly_dataframe()
        # 🔁 GEOID formatını harita için normalize et
        src = normalize_geoid_for_map(src)

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
        # 🔁 GEOID formatını harita için normalize et
        src = normalize_geoid_for_map(src)

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
            # Harita için GEOID bazlı risk ortalaması (sadece hücreler)
            # 1) Önce risk_prob'u dene
            metric_col = None
            use_prob = False

            if "risk_prob" in view_df_cells.columns:
                max_prob = pd.to_numeric(
                    view_df_cells["risk_prob"], errors="coerce"
                ).max()
                if pd.notna(max_prob) and max_prob > 0:
                    metric_col = "risk_prob"
                    use_prob = True

            # 2) Eğer risk_prob yoksa veya hep 0'sa risk_score'a dön
            if metric_col is None and "risk_score" in view_df_cells.columns:
                metric_col = "risk_score"
                use_prob = False

            if metric_col is None:
                agg = pd.DataFrame()
            else:
                tmp = view_df_cells.copy()
                tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")

                grp = tmp.groupby("geoid", as_index=False)[metric_col].mean()

                if use_prob:
                    grp = grp.rename(columns={metric_col: "risk_mean"})
                else:
                    # risk_score büyük ihtimalle yüzde → 0–100'ü 0–1'e çevir
                    max_val = grp[metric_col].max()
                    if pd.notna(max_val) and max_val > 1.0:
                        grp["risk_mean"] = grp[metric_col].clip(0, 100) / 100.0
                    else:
                        grp["risk_mean"] = grp[metric_col].clip(0.0, 1.0)

                agg = grp[["geoid", "risk_mean"]].copy()
                # GEOID'leri string olarak tut (GeoJSON'daki ile bire bir eşleşsin)
                agg["geoid"] = agg["geoid"].astype(str)
        else:
            view_df_cells = pd.DataFrame()
            agg = pd.DataFrame()

        # Opsiyonel kolonları GEOID bazında özetle (risk_prob, expected_crimes, top1_category vs.)
        def safe_mean(col_name: str):
            if len(view_df_cells) and col_name in view_df_cells.columns:
                out = view_df_cells.groupby("geoid", as_index=False)[col_name].mean()
                out["geoid"] = out["geoid"].astype(str)
                return out
            return None

        def safe_first(col_name: str):
            if len(view_df_cells) and col_name in view_df_cells.columns:
                tmp = (
                    view_df_cells.sort_values(time_col)
                    .groupby("geoid", as_index=False)[col_name]
                    .first()
                )
                tmp["geoid"] = tmp["geoid"].astype(str)
                return tmp
            return None

        if len(agg):
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
# 🗺️ HARİTA — EN ÜSTE (Folium + tıklama ile GEOID seçimi)
# ------------------------------------------------------------
if len(agg):
    agg["risk_bucket"] = agg["risk_mean"].map(bucket_of)
    agg_sorted = agg.sort_values("risk_mean", ascending=False).reset_index(drop=True)
else:
    agg_sorted = agg

st.subheader("🗺️ Harita — 5 seviye risk renklendirme")

geojson = load_geojson()

clicked_geoid = None  # haritada tıklanan GEOID

if not len(agg_sorted):
    if len(view_df_city):
        st.info(
            "Bu aralıkta sadece şehir geneli (GEOID=0) için risk üretilmiş; "
            "hücre (GEOID) bazlı risk olmadığı için harita devre dışı."
        )
    else:
        st.info("Seçilen aralıkta GEOID bazlı risk verisi bulunamadı.")
elif not geojson:
    st.info("GeoJSON (sf_cells.geojson) bulunamadı; harita devre dışı.")
else:
    gj_enriched = enrich_geojson_with_risk(geojson, agg_sorted)

    st.markdown(
        "**Lejand:** "
        "<span style='background:#ddd;padding:2px 6px;border-radius:4px;'>Çok Düşük</span> "
        "<span style='background:#b4d2ff;padding:2px 6px;border-radius:4px;'>Düşük</span> "
        "<span style='background:#ffdc82;padding:2px 6px;border-radius:4px;'>Orta</span> "
        "<span style='background:#ffaa6e;padding:2px 6px;border-radius:4px;'>Yüksek</span> "
        "<span style='background:#ff5a5a;padding:2px 6px;border-radius:4px;'>Çok Yüksek</span> ",
        unsafe_allow_html=True,
    )

    # --- Folium haritası
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=11,
        tiles="cartodbpositron",
        control_scale=True,
    )

    def style_fn(feature):
        props = feature.get("properties", {})
        rgba = props.get("fill_color", [220, 220, 220, 160])
        return {
            "fillColor": rgba_to_hex(rgba),
            "color": "#505050",
            "weight": 0.5,
            "fillOpacity": float(rgba[3]) / 255.0 if len(rgba) == 4 else 0.6,
        }

    def highlight_fn(feature):
        return {"weight": 2, "color": "#000000"}
    
    tooltip = folium.GeoJsonTooltip(
        fields=["display_id", "risk_bucket", "risk_mean_txt", "expected_count_txt", "top1_category"],
        aliases=[
            "GEOID:",
            "Risk seviyesi:",
            "Ortalama risk skoru (0–1):",
            "Beklenen toplam olay:",
            "En olası suç türü:",
        ],
        sticky=True,
    )

    folium.GeoJson(
        gj_enriched,
        name="Risk",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=tooltip,
    ).add_to(m)

    # Streamlit içinde haritayı render et ve tıklanan feature'ı yakala
    folium_ret = st_folium(
        m,
        width=None,
        height=520,
        returned_objects=["last_active_drawing"],
        key="sutam_fr_map",
    )

    if folium_ret and folium_ret.get("last_active_drawing"):
        props = folium_ret["last_active_drawing"].get("properties", {}) or {}
        clicked_geoid = str(
            props.get("geoid_norm")   # 👈 ÖNCE normalize edilmiş olan
            or props.get("display_id")
            or props.get("geoid")
            or props.get("GEOID")
            or ""
        ).strip()

        # tıklanan GEOID'i session_state'e yaz (diğer bileşenler kullanacak)
        if clicked_geoid:
            st.session_state["clicked_geoid_fr"] = clicked_geoid

# ------------------------------------------------------------
# 🧠 Özet kartlar
# ------------------------------------------------------------
st.title("🌀 Suç Tahmini — Haritalı GEOID görünüm")
st.caption(
    "3-saatlik bloklar (≤7 gün) veya günlük (≤365 gün) pencerede GEOID bazlı ortalama risk."
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
    options.extend(sorted(view_df_cells["geoid"].astype(str).unique().tolist()))

if options:
    # 1) Haritada tıklanan GEOID varsa onu al
    clicked_geoid = st.session_state.get("clicked_geoid_fr", clicked_geoid)

    # 2) Geçerli bir seçenek değilse 0. index
    default_index = 0
    if clicked_geoid and clicked_geoid in options:
        default_index = options.index(clicked_geoid)

    selected_geoid = st.selectbox(
        "Detay göstermek için GEOID seç:",
        options=options,
        index=default_index,
        format_func=geoid_label,
    )
else:
    selected_geoid = None
    
# Top-K her halükârda hesaplayalım (sadece hücreler üzerinden)
topk = agg_sorted.head(top_k).copy() if len(agg_sorted) else pd.DataFrame()

# ------------------------------------------------------------
# 🔍 Debug: Seçili GEOID için ham kayıtlar
# ------------------------------------------------------------
with st.expander("🔎 Debug: Seçili GEOID ham kayıtlar"):
    if selected_geoid is not None and len(view_df):
        df_dbg = (
            view_df[view_df["geoid"] == selected_geoid]
            [[time_col, "geoid", "risk_score"] + (
                ["risk_prob"] if "risk_prob" in view_df.columns else []
            )]
            .sort_values(time_col)
            .tail(10)
        )
        st.write(df_dbg)

        if "risk_prob" in view_df.columns:
            st.write(
                "risk_prob min/max:",
                float(pd.to_numeric(view_df["risk_prob"], errors="coerce").min()),
                float(pd.to_numeric(view_df["risk_prob"], errors="coerce").max()),
            )

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
                # 3-saatlik modda olduğu gibi son satırı kullan
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
            for k in range(1, 5+1):
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
        heat_index = "hour" if mode.startswith("3 Saatlik") else "date"
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
    "risk_3h_next7d_top3 / risk_daily_next365d_top5 (parquet veya csv); "
    "veya yerel CSV: 'data/crime_forecast_7days_all_geoids_FRstyle.csv'. "
    "Harita geometri kaynağı: repo içindeki 'data/sf_cells.geojson' dosyası."
)

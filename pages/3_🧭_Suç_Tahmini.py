# 3_🧭_Suç_Tahmini.py
# Tek sayfa: tarih/saat(veya gün/aralık) + kategori + GEOID filtreleriyle sonuçları listeler/grafikler.
# ZIP/URL/yerel CSV/Parquet okuma destekli. Actions artifact (token) > Release fallback.

import os
import io
import zipfile
import posixpath
from io import BytesIO
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------
# Sayfa ayarı
# ---------------------------
st.set_page_config(page_title="Suç Tahmini", page_icon="🌀", layout="wide")

# ---------------------------
# Artifact → Release fallback + token çözümleme
# ---------------------------
REPO_OWNER = "cem5113"
REPO_NAME  = "crime_prediction_data"
RELEASE_ASSET_ZIP = "fr-crime-outputs-parquet.zip"   # fallback

def _resolve_token() -> str | None:
    # env > secrets sırası
    if os.getenv("GITHUB_TOKEN"):
        return os.getenv("GITHUB_TOKEN")
    for k in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if k in st.secrets and st.secrets[k]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[k])
                return os.environ["GITHUB_TOKEN"]
        except Exception:
            pass
    return None

def _gh_headers() -> dict:
    hdrs = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    return hdrs

def _resolve_artifact_zip_url(owner: str, repo: str, name_contains: str):
    tok = _resolve_token()
    if not tok:
        return None, {}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(f"{base}/actions/artifacts?per_page=100", headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    arts = (r.json() or {}).get("artifacts", []) or []
    arts = [a for a in arts if (name_contains in a.get("name","")) and not a.get("expired")]
    if not arts:
        return None, {}
    arts.sort(key=lambda a: a.get("updated_at",""), reverse=True)
    url = f"{base}/actions/artifacts/{arts[0]['id']}/zip"
    return url, _gh_headers()

def _best_zip_url():
    # önce yeni artifact adı
    url, headers = _resolve_artifact_zip_url(REPO_OWNER, REPO_NAME, "fr-minimal-parquet")
    if url:
        return url, headers
    # eski ad(lar) yedek (istersen tut)
    url, headers = _resolve_artifact_zip_url(REPO_OWNER, REPO_NAME, "fr-crime-outputs-parquet")
    if url:
        return url, headers
    rel = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/{RELEASE_ASSET_ZIP}"
    return rel, {}

# ---------------------------
# ZIP/URL/yerel akıllı okuyucu (CSV + Parquet, iç ZIP dahil)
# ---------------------------
def _read_any_table_from_bytes(raw: bytes, name_hint: str = "") -> pd.DataFrame:
    bio = BytesIO(raw)
    # ipucu uzantısı varsa önce onu dene
    if name_hint.lower().endswith(".csv"):
        try:
            bio.seek(0); return pd.read_csv(bio)
        except Exception:
            pass
    if name_hint.lower().endswith(".parquet"):
        try:
            bio.seek(0); return pd.read_parquet(bio)
        except Exception:
            pass
    # ipucu yoksa parquet -> csv sırala
    try:
        bio.seek(0); return pd.read_parquet(bio)
    except Exception:
        bio.seek(0); return pd.read_csv(bio)

def _read_table_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    # ZIP değilse doğrudan dosya olarak dene
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as _z:
            pass
    except zipfile.BadZipFile:
        return _read_any_table_from_bytes(zip_bytes, name_hint=member_path)
        
    def _read(fp, name):
        if name.lower().endswith(".csv"):
            return pd.read_csv(fp)
        return pd.read_parquet(fp)

    target_base = posixpath.basename(member_path)

    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # 1) birebir
        if member_path in names:
            with z.open(member_path) as f:
                return _read(BytesIO(f.read()), member_path)

        # 1.b) basename ile
        cand = [n for n in names if n.endswith("/"+target_base) or n == target_base]
        if cand:
            with z.open(cand[0]) as f:
                return _read(BytesIO(f.read()), cand[0])

        # 2) iç ZIP
        for n in names:
            if not n.lower().endswith(".zip"):
                continue
            with z.open(n) as fz, zipfile.ZipFile(BytesIO(fz.read())) as z2:
                inner = z2.namelist()
                if member_path in inner:
                    with z2.open(member_path) as f2:
                        return _read(BytesIO(f2.read()), member_path)
                cand2 = [m for m in inner if m.endswith("/"+target_base) or m == target_base]
                if cand2:
                    with z2.open(cand2[0]) as f2:
                        return _read(BytesIO(f2.read()), cand2[0])

    raise FileNotFoundError(f"ZIP içinde bulunamadı: {member_path}")

def read_table_smart(spec: str) -> pd.DataFrame:
    """
    spec biçimleri:
      - Yerel dosya:         /path/to/file.parquet | .csv
      - Yerel zip içi:       zip::/path/file.zip::artifact/risk_hourly.parquet
      - URL zip içi:         urlzip::<URL veya AUTO>::artifact/risk_hourly.parquet
                             (Bulunamazsa otomatik .csv versiyonunu da dener)
    """
    def _try(url_or_bytes, member, from_zip: bool, headers=None):
        # Önce verilen üye, olmazsa .parquet<->.csv dönüşümü
        try_members = [member]
        if member.lower().endswith(".parquet"):
            try_members.append(member[:-8] + ".csv")
        elif member.lower().endswith(".csv"):
            try_members.append(member[:-4] + ".parquet")

        if from_zip:
            # url
            url = url_or_bytes
            r = requests.get(url, headers=headers or {}, timeout=120, allow_redirects=True)
            r.raise_for_status()
            for m in try_members:
                try:
                    return _read_table_from_zip_bytes(r.content, m)
                except FileNotFoundError:
                    continue
        else:
            # local zip bytes
            with open(url_or_bytes, "rb") as f:
                content = f.read()
            for m in try_members:
                try:
                    return _read_table_from_zip_bytes(content, m)
                except FileNotFoundError:
                    continue
        raise FileNotFoundError(f"ZIP içinde şu adaylar bulunamadı: {try_members}")

    if spec.startswith("urlzip::"):
        url, member = spec[len("urlzip::"):].split("::", 1)
        headers = {}
        if url == "AUTO":
            url, headers = _best_zip_url()
        return _try(url, member, from_zip=True, headers=headers)

    if spec.startswith("zip::"):
        zip_path, member = spec[len("zip::"):].split("::", 1)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP yok: {zip_path}")
        return _try(zip_path, member, from_zip=False)

    # düz dosya
    if not os.path.exists(spec):
        raise FileNotFoundError(f"Dosya yok: {spec}")
    if spec.lower().endswith(".csv"):
        return pd.read_csv(spec)
    return pd.read_parquet(spec)

# ---------------------------
# Normalize: CSV (date+hour_range) → timestamp, risk_score → p_stack
# ---------------------------
def _ensure_hourly_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}

    # --- GEOID ---
    geokey = None
    for k in ("geoid", "GEOID", "cell_id", "id"):
        if k in df.columns: geokey = k; break
        if k.lower() in cols_lower: geokey = cols_lower[k.lower()]; break
    if geokey is None:
        raise ValueError("GEOID/Cell ID kolonu bulunamadı.")
    df = df.rename(columns={geokey: "geoid"})
    df["geoid"] = df["geoid"].astype(str)

    # --- Zaman (3 yol: timestamp/datetime | date+hour | date+hour_range) ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], utc=False)
    else:
        date_col = None
        for k in ("date", "Date", "DATE"):
            if k in df.columns: date_col = k; break
        if date_col is None:
            raise ValueError("Saatlik veri için 'timestamp/datetime' ya da 'date' tabanlı kolonlar bekleniyor.")

        d0 = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")

        hour_col = None
        for k in ("hour", "Hour", "HOUR"):
            if k in df.columns: hour_col = k; break

        if hour_col is not None:
            h = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int).clip(0, 23)
            df["timestamp"] = d0 + pd.to_timedelta(h, unit="h")
        else:
            hr_col = None
            for k in ("hour_range","Hour_Range","HOUR_RANGE"):
                if k in df.columns: hr_col = k; break
            if hr_col is None:
                raise ValueError("Saatlik tabloda 'timestamp/datetime' ya da 'date+hour' / 'date+hour_range' kolonları yok.")
            start_h = (
                df[hr_col].astype(str)
                .str.extract(r"^(\d{1,2})")[0]
                .fillna("0").astype(int).clip(0, 23)
            )
            df["timestamp"] = d0 + pd.to_timedelta(start_h, unit="h")

    # --- category ---
    if "category" not in df.columns:
        df["category"] = "Genel"
    else:
        df["category"] = df["category"].astype(str)

    # --- p_stack (olasılık) ---
    if "p_stack" not in df.columns:
        score_col = None
        for c in ("risk_score", "risk", "prob", "probability", "score"):
            if c in df.columns: score_col = c; break
        if score_col is None:
            raise ValueError("Beklenen olasılık kolonu yok: p_stack / risk_score")
        df = df.rename(columns={score_col: "p_stack"})
    df["p_stack"] = pd.to_numeric(df["p_stack"], errors="coerce")

    return df[["timestamp", "geoid", "category", "p_stack"]].dropna(subset=["timestamp"])

# ---------------------------
# Cache'li yükleyiciler
# ---------------------------
@st.cache_data(show_spinner=False)
def load_hourly(path: str) -> pd.DataFrame:
    df = read_table_smart(path)
    return _ensure_hourly_schema(df)

@st.cache_data(show_spinner=False)
def load_daily(path: str) -> pd.DataFrame | None:
    try:
        df = read_table_smart(path)
    except Exception:
        return None

    # 'date' yoksa timestamp'tan üret
    if "date" not in df.columns:
        ts_col = None
        for c in ("timestamp", "datetime"):
            if c in df.columns:
                ts_col = c; break
        if ts_col is not None:
            dt = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
            if dt.notna().any():
                df["date"] = dt.dt.floor("D")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    return df

# ---------------------------
# Zaman filtresi
# ---------------------------
def filter_by_time(df: pd.DataFrame, mode: str,
                   ts: datetime | None,
                   day: datetime | None,
                   start: datetime | None,
                   end: datetime | None) -> pd.DataFrame:
    if mode == "Tek saat":
        if ts is None:
            return df.iloc[0:0]
        return df[df["timestamp"] == pd.to_datetime(ts)]
    elif mode == "Gün (24 saat)":
        if day is None:
            return df.iloc[0:0]
        t0 = pd.to_datetime(day).replace(hour=0, minute=0, second=0, microsecond=0)
        t1 = t0 + timedelta(days=1)
        return df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]
    else:  # Aralık
        if not (start and end):
            return df.iloc[0:0]
        t0 = pd.to_datetime(start); t1 = pd.to_datetime(end)
        return df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")

# ---------------------------
# Varsayılan yollar (AUTO: artifact -> release)
# ---------------------------
# risk_hourly çoğu zaman CSV; parquet varsa onu da okur
DEFAULT_HOURLY = "urlzip::AUTO::risk_hourly.parquet"
DEFAULT_DAILY  = "urlzip::AUTO::fr_crime_09.parquet"

# ---------------------------
# Sidebar — seçimler
# ---------------------------
st.sidebar.header("⚙️ Ayarlar")

hourly_path = st.sidebar.text_input(
    "Saatlik çıktı (ZIP/URL/yerel destekli)",
    value=DEFAULT_HOURLY,
    help=(
        "Örnekler:\n"
        "- URL ZIP (AUTO): urlzip::AUTO::risk_hourly.parquet (parquet de olabilir)\n"
        "- Yerel ZIP: zip::/path/to/fr-minimal-parquet.zip::risk_hourly.parquet
        "- Düz CSV/Parquet: /path/to/risk_hourly.csv | .parquet"
    ),
)
daily_path  = st.sidebar.text_input(
    "(Opsiyonel) Günlük/özet dosyası",
    value=DEFAULT_DAILY,
    help=(
        "Örnekler:\n"
        "- URL ZIP (AUTO): urlzip::AUTO::fr_crime_09.parquet\n"
        "- Yerel ZIP: zip::/path/to/fr-crime-outputs-parquet.zip::fr_crime_09.parquet\n"
        "- Düz parquet/csv: /path/to/risk_daily_by_category.parquet"
    ),
)

# Veri yükle
try:
    df = load_hourly(hourly_path)
except Exception as e:
    st.error(f"Saatlik dosya okunamadı:\n\n{e}")
    st.info("İpucu: Artifact için ortamda GITHUB_TOKEN/GH_TOKEN/github_token bulunmalı. "
            "Yoksa Release fallback denenir.")
    st.stop()

df_daily = load_daily(daily_path)

# ---------------------------
# Zaman modu & seçimler
# ---------------------------
time_mode = st.sidebar.radio("Zaman seçimi", ["Tek saat", "Gün (24 saat)", "Aralık"], horizontal=False)

ts_single = day_single = start_range = end_range = None
if time_mode == "Tek saat":
    ts_candidates = sorted(pd.to_datetime(df["timestamp"]).unique())
    ts_single = st.sidebar.selectbox("Zaman (timestamp)", options=ts_candidates, index=0) if ts_candidates else None
elif time_mode == "Gün (24 saat)":
    days = sorted(pd.to_datetime(df["timestamp"]).dt.date.unique())
    day_single = st.sidebar.selectbox("Gün", options=days, index=0) if days else None
else:
    ts_all = sorted(pd.to_datetime(df["timestamp"]).unique())
    if ts_all:
        start_range = st.sidebar.selectbox("Başlangıç", options=ts_all, index=0)
        end_range   = st.sidebar.selectbox("Bitiş", options=ts_all, index=len(ts_all)-1)
    else:
        st.sidebar.info("Uyarı: Saatlik veri boş görünüyor.")

# Kategori & GEOID seçimleri
cats = sorted(df["category"].astype(str).unique().tolist())
geoids = sorted(df["geoid"].astype(str).unique().tolist())

# Boşsa hata vermesin diye güvenli default
sel_cats = st.sidebar.multiselect(
    "Suç kategorileri",
    options=cats,
    default=cats if cats else [],
)

scope_choice = st.sidebar.radio("Alan", ["Tüm şehir", "GEOID seç"], horizontal=True)
if scope_choice == "GEOID seç":
    sel_geoids = st.sidebar.multiselect(
        "GEOID",
        options=geoids,
        default=geoids[:20] if geoids else [],
    )
else:
    sel_geoids = geoids if geoids else []

agg_daily_how = st.sidebar.selectbox("Günlük agregasyon (görünüm)", ["Ortalama", "Maksimum"], index=0)
top_k = st.sidebar.slider("Top-K sıralama (tablo)", min_value=10, max_value=200, value=50, step=10)
risk_cut = st.sidebar.slider("Risk eşiği (vurgulama)", 0.0, 1.0, 0.5, 0.05)

# ---------------------------
# İçerik — başlık
# ---------------------------
st.title("🌀 Suç Tahmini")
st.caption("Zaman–mekân–suç türü bazlı olasılık tahmini (stacking ensemble).")

# ---------------------------
# Filtrele
# ---------------------------
df_t = df.copy()
if time_mode:
    df_t = filter_by_time(df_t, time_mode, ts_single, day_single, start_range, end_range)
if sel_cats:
    df_t = df_t[df_t["category"].isin(sel_cats)]
if sel_geoids:
    df_t = df_t[df_t["geoid"].isin(sel_geoids)]

# ---------------------------
# Özet kartları
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kapsanan saat sayısı", f"{df_t['timestamp'].nunique():,}")
c2.metric("GEOID sayısı", f"{df_t['geoid'].nunique():,}")
c3.metric("Kategori sayısı", f"{df_t['category'].nunique():,}")
c4.metric("Ortalama risk", f"{df_t['p_stack'].mean():.3f}" if len(df_t) else "—")

# ---------------------------
# Görünümler
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🔝 Top-K tablo", "📈 Zaman serisi", "📊 Kategori/GEOID özet"])

with tab1:
    if time_mode == "Tek saat":
        show_df = df_t.sort_values("p_stack", ascending=False).head(top_k).reset_index(drop=True)
        st.subheader("Top-K (tek saat)")
        st.dataframe(
            show_df.style.highlight_between(subset="p_stack", left=risk_cut, right=1.0, color="#ffd6cc"),
            use_container_width=True, height=520
        )
    else:
        show_df = (
            df_t.groupby(["geoid", "category"], as_index=False)
                .agg(mean_risk=("p_stack", "mean"),
                     max_risk=("p_stack", "max"),
                     n_hours=("p_stack", "size"))
        )
        order_col = "mean_risk" if agg_daily_how == "Ortalama" else "max_risk"
        show_df = show_df.sort_values(order_col, ascending=False).head(top_k).reset_index(drop=True)
        st.subheader(f"Top-K (aralık/gün) — {order_col}")
        st.dataframe(
            show_df.style.highlight_between(subset=order_col, left=risk_cut, right=1.0, color="#ffd6cc"),
            use_container_width=True, height=520
        )

    st.download_button("⬇️ CSV indir (gösterilen)",
                       data=df_to_csv_bytes(show_df),
                       file_name="crime_forecast_topk.csv",
                       mime="text/csv")

with tab2:
    if len(df_t) == 0:
        st.info("Seçilen zaman/kapsam için veri yok.")
    else:
        geo_for_plot = st.multiselect("Grafik için GEOID seç",
                                      options=sorted(df_t["geoid"].unique().tolist()),
                                      default=sorted(df_t["geoid"].unique().tolist())[:3])
        df_plot = df_t[df_t["geoid"].isin(geo_for_plot)].copy()
        st.line_chart(
            df_plot.pivot_table(index="timestamp", columns="geoid", values="p_stack", aggfunc="mean").sort_index(),
            height=420
        )

with tab3:
    if len(df_t) == 0:
        st.info("Seçilen zaman/kapsam için veri yok.")
    else:
        cA, cB = st.columns(2)
        cat_summary = df_t.groupby("category", as_index=False)["p_stack"].mean().sort_values("p_stack", ascending=False)
        cA.subheader("Kategori ortalama risk")
        cA.bar_chart(cat_summary.set_index("category"), height=300)

        geo_summary = (
            df_t.groupby("geoid", as_index=False)["p_stack"].mean()
               .sort_values("p_stack", ascending=False)
               .head(20)
        )
        cB.subheader("GEOID ortalama risk (Top-20)")
        cB.bar_chart(geo_summary.set_index("geoid"), height=300)

# ---------------------------
# Günlük/özet (opsiyonel)
# ---------------------------
st.markdown("---")
st.subheader("📅 Günlük özet (opsiyonel)")

def _has_daily_view_cols(d: pd.DataFrame) -> bool:
    return ("date" in d.columns) and ("daily_score" in d.columns)

if df_daily is None:
    st.caption("`fr_crime_09.parquet` (veya `risk_daily_by_category.parquet`) bulunamadı/okunamadı — opsiyonel bölümdür.")
else:
    if _has_daily_view_cols(df_daily):
        days_all = sorted(pd.to_datetime(df_daily["date"]).unique())
        idx = len(days_all)-1 if len(days_all) else 0
        day_sel = st.selectbox("Gün seç", options=days_all, index=idx if len(days_all) else None)
        d1 = df_daily[pd.to_datetime(df_daily["date"]) == pd.to_datetime(day_sel)]
        st.dataframe(d1.sort_values("daily_score", ascending=False).head(top_k).reset_index(drop=True),
                     use_container_width=True, height=360)
    else:
        st.caption("Günlük özet için gerekli kolonlar bulunamadı (`date` + `daily_score`). Bu bölüm opsiyoneldir.")

# ---------------------------
# Dipnot
# ---------------------------
st.caption(
    "Model: üç motorlu stacking (short/mid) + mevsimsel baseline (long). "
    "Olasılıklar kalibre edilmiştir (isotonic). `confidence` ufka göre azaltılmış güven skorunu ifade eder."
)

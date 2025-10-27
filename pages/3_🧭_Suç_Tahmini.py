# 3_🧭_Suç_Tahmini.py
# Tek sayfa: tarih/saat(veya gün/aralık) + kategori + GEOID filtreleriyle sonuçları listeler/grafikler.
# ZIP/URL/yerel parquet okuma destekli. Actions artifact (token) > Release fallback.

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
st.set_page_config(
    page_title="Suç Tahmini",
    page_icon="🌀",
    layout="wide",
)

# ---------------------------
# Kaynak çözümleyici: Actions artifact -> Release fallback
# ---------------------------
REPO_OWNER = "cem5113"
REPO_NAME  = "crime_prediction_data"
RELEASE_ASSET_ZIP = "fr-crime-outputs-parquet.zip"  # varsa release asset adı

def _resolve_artifact_zip_url(owner: str, repo: str, name_contains: str, token: str | None):
    """Repo'daki en güncel, süresi dolmamış artifact ZIP linkini döndürür (URL, headers)."""
    if not token:
        return None, {}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(
        f"{base}/actions/artifacts?per_page=100",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=60,
    )
    r.raise_for_status()
    arts = (r.json() or {}).get("artifacts", []) or []
    arts = [a for a in arts if (name_contains in (a.get("name",""))) and not a.get("expired")]
    if not arts:
        return None, {}
    art = sorted(arts, key=lambda a: a.get("created_at",""), reverse=True)[0]
    zip_url = f"{base}/actions/artifacts/{art['id']}/zip"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    return zip_url, headers

def _best_zip_url():
    """
    1) Actions artifact (token varsa)
    2) Release fallback (yoksa)
    """
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("github_token")
    url, headers = _resolve_artifact_zip_url(REPO_OWNER, REPO_NAME, "fr-crime-outputs-parquet", token)
    if url:
        return url, headers
    rel = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/{RELEASE_ASSET_ZIP}"
    return rel, {}

# ---------------------------
# ZIP/URL/yerel akıllı okuyucu
# ---------------------------
def _read_parquet_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """
    Bir ZIP'in içindeki 'member_path'i Parquet olarak okur.
    - Doğrudan eşleşme
    - Basename ile klasör içinden bulma
    - İç ZIP'lerde (.zip) arama (ör. fr_parquet_outputs.zip)
    - artifact/ altını destekler
    """
    target_base = posixpath.basename(member_path)

    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # 1) Doğrudan arama
        if member_path in names:
            with z.open(member_path) as f:
                return pd.read_parquet(BytesIO(f.read()))

        # 1.b) Basename ile eşleşen ilk dosya (klasör farketmez)
        cand = [n for n in names if n.endswith("/"+target_base) or n == target_base]
        if cand:
            with z.open(cand[0]) as f:
                return pd.read_parquet(BytesIO(f.read()))

        # 2) İç ZIP'ler
        for nested in names:
            if not nested.lower().endswith(".zip"):
                continue
            with z.open(nested) as fz:
                with zipfile.ZipFile(BytesIO(fz.read())) as z2:
                    inner = z2.namelist()
                    if member_path in inner:
                        with z2.open(member_path) as f2:
                            return pd.read_parquet(BytesIO(f2.read()))
                    cand2 = [m for m in inner if m.endswith("/"+target_base) or m == target_base]
                    if cand2:
                        with z2.open(cand2[0]) as f2:
                            return pd.read_parquet(BytesIO(f2.read()))

    raise FileNotFoundError(f"ZIP içinde bulunamadı: {member_path}")

def read_parquet_smart(spec: str) -> pd.DataFrame:
    """
    spec biçimleri:
      - Yerel parquet:        /path/to/file.parquet
      - Yerel zip içi:        zip::/path/to/file.zip::artifact/risk_hourly.parquet
      - URL zip içi:          urlzip::<URL veya AUTO>::artifact/risk_hourly.parquet
      - URL iç ZIP’in içi:    urlzip::<URL veya AUTO>::fr_crime_09.parquet  (iç ZIP de taranır)
    """
    if spec.startswith("urlzip::"):
        url, member = spec[len("urlzip::"):].split("::", 1)
        headers = {}
        if url == "AUTO":  # artifact->release otomatik
            url, headers = _best_zip_url()
        r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
        r.raise_for_status()
        return _read_parquet_from_zip_bytes(r.content, member)

    elif spec.startswith("zip::"):
        zip_path, member = spec[len("zip::"):].split("::", 1)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP yok: {zip_path}")
        with open(zip_path, "rb") as f:
            return _read_parquet_from_zip_bytes(f.read(), member)

    else:
        # düz parquet
        if not os.path.exists(spec):
            raise FileNotFoundError(f"Dosya yok: {spec}")
        return pd.read_parquet(spec)

# ---------------------------
# Yardımcılar
# ---------------------------
@st.cache_data(show_spinner=False)
def load_hourly(path: str) -> pd.DataFrame:
    df = read_parquet_smart(path)
    # Beklenen kolonlar: timestamp, geoid, category, p_stack
    if "timestamp" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        else:
            raise ValueError("Saatlik tabloda 'timestamp' ya da 'datetime' kolonu yok.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)

    if "geoid" not in df.columns and "GEOID" in df.columns:
        df = df.rename(columns={"GEOID": "geoid"})
    df["geoid"] = df["geoid"].astype(str)

    if "category" not in df.columns:
        df["category"] = "Genel"
    else:
        df["category"] = df["category"].astype(str)

    if "p_stack" not in df.columns:
        raise ValueError("Beklenen kolon 'p_stack' (stacking olasılığı) bulunamadı.")
    return df

@st.cache_data(show_spinner=False)
def load_daily(path: str) -> pd.DataFrame | None:
    try:
        df = read_parquet_smart(path)
    except Exception:
        return None

    if "date" not in df.columns:
        ts_col = "timestamp" if "timestamp" in df.columns else ("datetime" if "datetime" in df.columns else None)
        if ts_col is not None:
            dt = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
            if dt.notna().any():
                df["date"] = dt.dt.floor("D")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)

    return df

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
    else:  # "Aralık"
        if not (start and end):
            return df.iloc[0:0]
        t0 = pd.to_datetime(start)
        t1 = pd.to_datetime(end)
        return df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ---------------------------
# Varsayılan yollar (AUTO: artifact -> release)
# ---------------------------
# Saatlik skorlar (artifact/risk_hourly.parquet)
DEFAULT_HOURLY = "urlzip::AUTO::artifact/risk_hourly.parquet"

# Opsiyonel günlük/özet – en zengin tablo (fr_crime_09.parquet)
DEFAULT_DAILY  = "urlzip::AUTO::fr_crime_09.parquet"

# ---------------------------
# Sidebar — seçimler
# ---------------------------
st.sidebar.header("⚙️ Ayarlar")

hourly_path = st.sidebar.text_input(
    "Saatlik çıktı dosyası (ZIP/URL/yerel destekli)",
    value=DEFAULT_HOURLY,
    help=(
        "Örnekler:\n"
        "- URL ZIP (AUTO): urlzip::AUTO::artifact/risk_hourly.parquet\n"
        "- Yerel ZIP: zip::/path/to/fr-crime-outputs-parquet.zip::artifact/risk_hourly.parquet\n"
        "- Düz parquet: /path/to/risk_hourly.parquet"
    ),
)
daily_path  = st.sidebar.text_input(
    "(Opsiyonel) Günlük/özet dosyası",
    value=DEFAULT_DAILY,
    help=(
        "Örnekler:\n"
        "- URL ZIP (AUTO): urlzip::AUTO::fr_crime_09.parquet\n"
        "- Yerel ZIP: zip::/path/to/fr-crime-outputs-parquet.zip::fr_crime_09.parquet\n"
        "- Düz parquet: /path/to/risk_daily_by_category.parquet"
    ),
)

# Veri yükle
try:
    df = load_hourly(hourly_path)
except Exception as e:
    st.error(f"Saatlik dosya okunamadı:\n\n{e}")
    # Token yoksa kısa ipucu:
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
    default_idx = 0 if len(ts_candidates) > 0 else None
    ts_single = st.sidebar.selectbox("Zaman (timestamp)", options=ts_candidates, index=default_idx) if len(ts_candidates) else None
elif time_mode == "Gün (24 saat)":
    days = sorted(pd.to_datetime(df["timestamp"]).dt.date.unique())
    default_idx = 0 if len(days) > 0 else None
    day_single = st.sidebar.selectbox("Gün", options=days, index=default_idx) if len(days) else None
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

sel_cats = st.sidebar.multiselect("Suç kategorileri", options=cats, default=cats)
scope_choice = st.sidebar.radio("Alan", ["Tüm şehir", "GEOID seç"], horizontal=True)
if scope_choice == "GEOID seç":
    sel_geoids = st.sidebar.multiselect("GEOID", options=geoids, default=geoids[:20])
else:
    sel_geoids = geoids  # hepsi

# Görünüm seçenekleri
agg_daily_how = st.sidebar.selectbox("Günlük agregasyon (görünüm)", ["Ortalama", "Maksimum"], index=0)
top_k = st.sidebar.slider("Top-K sıralama (tablo)", min_value=10, max_value=200, value=50, step=10)
risk_cut = st.sidebar.slider("Risk eşiği (vurgulama)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# ---------------------------
# İçerik — başlık
# ---------------------------
st.title("🌀 Suç Tahmini")
st.caption("Zaman–mekân–suç türü bazlı olasılık tahmini (stacking ensemble).")

# ---------------------------
# Filtrele
# ---------------------------
df_t = filter_by_time(df, time_mode, ts_single, day_single, start_range, end_range)
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
if len(df_t):
    c4.metric("Ortalama risk", f"{df_t['p_stack'].mean():.3f}")
else:
    c4.metric("Ortalama risk", "—")

# ---------------------------
# Görünümler
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🔝 Top-K tablo", "📈 Zaman serisi", "📊 Kategori/GEOID özet"])

# --- Top-K tablo ---
with tab1:
    if time_mode == "Tek saat":
        show_df = (
            df_t.sort_values("p_stack", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
        )
        st.subheader("Top-K (tek saat)")
        st.dataframe(
            show_df.style.highlight_between(subset="p_stack", left=risk_cut, right=1.0, color="#ffd6cc"),
            use_container_width=True, height=520
        )
    else:
        # Çok saat kapsıyorsa, geo×cat özete indir
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

    # İndir
    st.download_button(
        "⬇️ CSV indir (gösterilen)",
        data=df_to_csv_bytes(show_df),
        file_name="crime_forecast_topk.csv",
        mime="text/csv"
    )

# --- Zaman serisi ---
with tab2:
    if len(df_t) == 0:
        st.info("Seçilen zaman/kapsam için veri yok.")
    else:
        geo_for_plot = st.multiselect(
            "Grafik için GEOID seç",
            options=sorted(df_t["geoid"].unique().tolist()),
            default=sorted(df_t["geoid"].unique().tolist())[:3]
        )
        df_plot = df_t[df_t["geoid"].isin(geo_for_plot)].copy()
        st.line_chart(
            df_plot.pivot_table(index="timestamp", columns="geoid", values="p_stack", aggfunc="mean").sort_index(),
            height=420
        )

# --- Kategori/GEOID özet ---
with tab3:
    if len(df_t) == 0:
        st.info("Seçilen zaman/kapsam için veri yok.")
    else:
        cA, cB = st.columns(2)
        # Kategori ortalamaları
        cat_summary = (
            df_t.groupby("category", as_index=False)["p_stack"].mean()
                .sort_values("p_stack", ascending=False)
        )
        cA.subheader("Kategori ortalama risk")
        cA.bar_chart(cat_summary.set_index("category"), height=300)

        # GEOID ortalamaları (ilk 20)
        geo_summary = (
            df_t.groupby("geoid", as_index=False)["p_stack"].mean()
                .sort_values("p_stack", ascending=False)
                .head(20)
        )
        cB.subheader("GEOID ortalama risk (Top-20)")
        cB.bar_chart(geo_summary.set_index("geoid"), height=300)

# ---------------------------
# Günlük/özet görünüm (opsiyonel)
# ---------------------------
st.markdown("---")
st.subheader("📅 Günlük özet (opsiyonel)")

def _has_daily_view_cols(d: pd.DataFrame) -> bool:
    return ("date" in d.columns) and ("daily_score" in d.columns)

df_daily = df_daily  # zaten yüklendi
if df_daily is None:
    st.caption("`fr_crime_09.parquet` (veya `risk_daily_by_category.parquet`) bulunamadı/okunamadı — opsiyonel bölümdür.")
else:
    if _has_daily_view_cols(df_daily):
        days_all = sorted(pd.to_datetime(df_daily["date"]).unique())
        idx = len(days_all)-1 if len(days_all) else 0
        day_sel = st.selectbox("Gün seç", options=days_all, index=idx if len(days_all) else None)
        d1 = df_daily[pd.to_datetime(df_daily["date"]) == pd.to_datetime(day_sel)]
        if sel_cats:
            d1 = d1[d1["category"].astype(str).isin(sel_cats)]
        if sel_geoids:
            d1 = d1[d1["geoid"].astype(str).isin(sel_geoids)]
        st.dataframe(
            d1.sort_values("daily_score", ascending=False).head(top_k).reset_index(drop=True),
            use_container_width=True, height=360
        )
    else:
        st.caption("Günlük özet için gerekli kolonlar bulunamadı (`date` + `daily_score`). Bu bölüm bilgi amaçlıdır ve zorunlu değildir.")

# ---------------------------
# Dipnot
# ---------------------------
st.caption(
    "Model: üç motorlu stacking (short/mid) + mevsimsel baseline (long). "
    "Olasılıklar kalibre edilmiştir (isotonic). `confidence` ufka göre azaltılmış güven skorunu ifade eder."
)

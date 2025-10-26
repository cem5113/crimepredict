# 3_🧭_Suç_Tahmini.py
# app.py  —  Suç Tahmini (Streamlit)
# Tek sayfa: tarih/saat(veya gün/aralık) + kategori + GEOID filtreleriyle sonuçları listeler/grafikler.
# Bu dosyayı köke (crime_forecast/app.py) koyabilir veya pages/3_🌀_Suç_Tahmini.py olarak kaydedebilirsiniz.

import os
import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ---------------------------
# Sayfa ayarı
# ---------------------------
st.set_page_config(
    page_title="Suç Tahmini",
    page_icon="🌀",
    layout="wide",
)

# ---------------------------
# Yardımcılar (cache)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_hourly(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # tipler
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df["geoid"] = df["geoid"].astype(str)
    df["category"] = df["category"].astype(str)
    # güvenlik
    if "p_stack" not in df.columns:
        raise ValueError("Beklenen kolon 'p_stack' bulunamadı.")
    return df

@st.cache_data(show_spinner=False)
def load_daily(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=False)
    return df

def filter_by_time(df: pd.DataFrame, mode: str, ts: datetime | None,
                   day: datetime | None, start: datetime | None, end: datetime | None) -> pd.DataFrame:
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
# Dosya yolları (varsayılan)
# ---------------------------
DEFAULT_HOURLY = "data/outputs/risk_hourly_by_category.parquet"
DEFAULT_DAILY  = "data/outputs/risk_daily_by_category.parquet"

# ---------------------------
# Sidebar — seçimler
# ---------------------------
st.sidebar.header("⚙️ Ayarlar")

hourly_path = st.sidebar.text_input("Saatlik çıktı dosyası", value=DEFAULT_HOURLY)
daily_path  = st.sidebar.text_input("(Opsiyonel) Günlük çıktı dosyası", value=DEFAULT_DAILY)

# Veri yükle
if not os.path.exists(hourly_path):
    st.error(f"Saatlik dosya bulunamadı: {hourly_path}")
    st.stop()

df = load_hourly(hourly_path)
df_daily = load_daily(daily_path)

# Zaman modu
time_mode = st.sidebar.radio("Zaman seçimi", ["Tek saat", "Gün (24 saat)", "Aralık"], horizontal=False)

col_time = st.sidebar.container()
ts_single = day_single = start_range = end_range = None
if time_mode == "Tek saat":
    # Seçilebilir zamanları dosyadan çek
    ts_candidates = sorted(df["timestamp"].unique())
    default_ts = ts_candidates[0] if ts_candidates else None
    ts_single = st.sidebar.selectbox("Zaman (timestamp)", options=ts_candidates, index=0 if default_ts is not None else None)
elif time_mode == "Gün (24 saat)":
    days = sorted(pd.to_datetime(df["timestamp"]).date.unique())
    default_day = days[0] if days else None
    day_single = st.sidebar.selectbox("Gün", options=days, index=0 if default_day is not None else None)
else:
    ts_all = sorted(df["timestamp"].unique())
    if ts_all:
        start_range = st.sidebar.selectbox("Başlangıç", options=ts_all, index=0)
        end_range   = st.sidebar.selectbox("Bitiş", options=ts_all, index=len(ts_all)-1)
    else:
        st.sidebar.info("Uyarı: Saatlik veri boş görünüyor.")

# Kategori & GEOID seçimleri
cats = sorted(df["category"].unique().tolist())
geoids = sorted(df["geoid"].unique().tolist())

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
        show_df = (df_t
                   .sort_values("p_stack", ascending=False)
                   .head(top_k)
                   .reset_index(drop=True))
        st.subheader("Top-K (tek saat)")
        st.dataframe(
            show_df.style.highlight_between(subset="p_stack", left=risk_cut, right=1.0, color="#ffd6cc"),
            use_container_width=True, height=520
        )
    else:
        # Çok saat kapsıyorsa, saatlik satırları saklayıp geo×cat ortalamasıyla sıralayalım
        show_df = (df_t
                   .groupby(["geoid","category"], as_index=False)
                   .agg(mean_risk=("p_stack","mean"),
                        max_risk=("p_stack","max"),
                        n_hours=("p_stack","size")))
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
        # Seçilen GEOID'lerden birini (veya birkaçını) çizelim
        geo_for_plot = st.multiselect(
            "Grafik için GEOID seç",
            options=sorted(df_t["geoid"].unique().tolist()),
            default=sorted(df_t["geoid"].unique().tolist())[:3]
        )
        df_plot = df_t[df_t["geoid"].isin(geo_for_plot)].copy()
        # Streamlit native line_chart:
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
        cat_summary = (df_t.groupby("category", as_index=False)["p_stack"].mean()
                         .sort_values("p_stack", ascending=False))
        cA.subheader("Kategori ortalama risk")
        cA.bar_chart(cat_summary.set_index("category"), height=300)

        # GEOID ortalamaları (ilk 20)
        geo_summary = (df_t.groupby("geoid", as_index=False)["p_stack"].mean()
                         .sort_values("p_stack", ascending=False)
                         .head(20))
        cB.subheader("GEOID ortalama risk (Top-20)")
        cB.bar_chart(geo_summary.set_index("geoid"), height=300)

# ---------------------------
# Günlük görünüm (opsiyonel)
# ---------------------------
st.markdown("---")
st.subheader("📅 Günlük özet (opsiyonel)")
if df_daily is None:
    st.caption("`risk_daily_by_category.parquet` bulunamadı — 08_merge_outputs.py'yi günlük çıktı ile çalıştırırsanız burada görünür.")
else:
    # Gün seç ve listele
    days_all = sorted(df_daily["date"].unique())
    day_sel = st.selectbox("Gün seç", options=days_all, index=len(days_all)-1 if days_all else 0)
    d1 = df_daily[df_daily["date"] == pd.to_datetime(day_sel)]
    if sel_cats:
        d1 = d1[d1["category"].isin(sel_cats)]
    if sel_geoids:
        d1 = d1[d1["geoid"].isin(sel_geoids)]
    st.dataframe(
        d1.sort_values("daily_score", ascending=False).head(top_k).reset_index(drop=True),
        use_container_width=True, height=360
    )

# ---------------------------
# Dipnot
# ---------------------------
st.caption(
    "Model: üç motorlu stacking (short/mid) + mevsimsel baseline (long). "
    "Olasılıklar kalibre edilmiştir (isotonic). `confidence` ufka göre azaltılmış güven skorunu ifade eder."
)

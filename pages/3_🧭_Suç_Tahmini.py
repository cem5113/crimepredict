# pages/3_🔮_Suç_Tahmini_ve_Forecast.py
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from components.meta import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES

import io, zipfile
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

# ── constants & local path
from components.meta import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── local modules
from components.geo import load_geoid_layer, resolve_clicked_gid
from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from utils.ui import (
    SMALL_UI_CSS, render_result_card, build_map_fast,
    render_kpi_row, render_day_hour_heatmap as _fallback_heatmap
)

# utils/heatmap varsa onu kullan, yoksa ui.py'deki fallback'i kullan
try:
    from utils.heatmap import render_day_hour_heatmap  # type: ignore
except Exception:
    render_day_hour_heatmap = _fallback_heatmap  # type: ignore


# ───────────────────────────────── helpers ─────────────────────────────────
def ensure_keycol(df: pd.DataFrame, want: str = KEY_COL) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    alts = {want.upper(), want.lower(), "GEOID", "geoid", "GeoID"}
    hit = next((c for c in out.columns if c in alts), None)
    if hit:
        out = out.rename(columns={hit: want})
    if want in out.columns:
        out[want] = out[want].astype(str)
    return out

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """GEOID, datetime, category, crime_count ve popup için gerekli ana kolonları düzenler."""
    df = df.copy()

    # GEOID -> geoid (string)
    for cand in ["geoid", "GEOID", "GeoID"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "geoid"})
            break
    if "geoid" in df.columns:
        df["geoid"] = df["geoid"].astype(str)

    # datetime üret
    if "datetime" not in df.columns:
        if {"date", "time"}.issubset(df.columns):
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce", utc=True)
        else:
            for c in ["ts", "timestamp", "occurred_at", "reported_at", "received_time"]:
                if c in df.columns:
                    df["datetime"] = pd.to_datetime(df[c], errors="coerce", utc=True)
                    break
    if "datetime" in df.columns:
        df = df.dropna(subset=["datetime"])

    # kategori kolon adı
    if "category" not in df.columns:
        for c in ["type", "crime_type", "Category"]:
            if c in df.columns:
                df = df.rename(columns={c: "category"})
                break

    # hedef
    if "crime_count" not in df.columns:
        for c in ["911_request_count_hour_range", "911_request_count_daily(before_24_hours)"]:
            if c in df.columns:
                df["crime_count"] = pd.to_numeric(df[c], errors="coerce")
                break
        else:
            # en azından var/yok
            df["crime_count"] = 1.0

    keep = [c for c in [
        "geoid", "datetime", "category", "subcategory", "crime_count",
        "latitude", "longitude",
        # popup/özet için faydalı:
        "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d",
        "population", "bus_stop_count", "train_stop_count",
        "poi_total_count", "poi_risk_score", "poi_dominant_type",
        "Y_label"
    ] if c in df.columns]
    return df[keep].copy()

def load_artifacts() -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    fr_crime_10.parquet + artifacts/fr-crime-outputs-parquet.zip içindeki fr_crime_09.csv (opsiyonel)
    Dönüş: (df10_norm, df9_norm_or_None)
    """
    # 1) ana parquet
    p10 = os.path.join(ROOT, "fr_crime_10.parquet")
    if not os.path.exists(p10):
        st.warning("fr_crime_10.parquet bulunamadı. Sadece artifacts içeriğiyle devam edeceğim.")
        df10 = pd.DataFrame()
    else:
        df10 = pd.read_parquet(p10, engine="pyarrow")
        df10 = normalize_schema(df10)

    # 2) artifacts → iç zip
    df9 = None
    zpath = os.path.join(ROOT, "artifacts", "fr-crime-outputs-parquet.zip")
    if os.path.exists(zpath):
        try:
            z1 = zipfile.ZipFile(zpath, "r")
            inner = [n for n in z1.namelist() if n.endswith(".zip")]
            if inner:
                with z1.open(inner[0]) as inner_bytes:
                    z2 = zipfile.ZipFile(io.BytesIO(inner_bytes.read()), "r")
                    csv_name = [n for n in z2.namelist() if n.endswith("fr_crime_09.csv")]
                    if csv_name:
                        with z2.open(csv_name[0]) as f:
                            df9 = pd.read_csv(f)
                            df9 = normalize_schema(df9)
        except Exception as e:
            st.info(f"Artifacts içeriği okunurken bir uyarı oluştu: {e}")

    return df10, df9

def top3_by_geoid(df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """Her geoid için toplam crime_count'a göre en sık görülen ilk 3 kategoriyi döndürür."""
    if df is None or df.empty or not {"geoid","category","crime_count"}.issubset(df.columns):
        return {}
    tmp = (df.groupby(["geoid", "category"])["crime_count"]
           .sum()
           .reset_index()
           .sort_values(["geoid", "crime_count"], ascending=[True, False]))
    out: dict[str, list[tuple[str, int]]] = {}
    for gid, part in tmp.groupby("geoid"):
        out[str(gid)] = list(part.head(3)[["category", "crime_count"]].itertuples(index=False, name=None))
    return out

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")

def run_prediction(
    start_h: int,
    end_h: int,
    filters: dict,
    geo_df: pd.DataFrame,
    base_int: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str, int]:
    """Varolan aggregate_fast altyapınıza göre tahmin üretir; tier ataması utils tarafında yapılmış kabul edilir."""
    start_dt = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0, microsecond=0)
    horizon_h = max(1, end_h - start_h)
    start_iso = start_dt.isoformat()

    # events (opsiyonel) – yoksa boş geç
    try:
        from utils.reports import load_events  # type: ignore
        events_df = load_events(os.path.join(ROOT, "data", "events.csv"))
    except Exception:
        events_df = pd.DataFrame()

    agg = aggregate_fast(
        start_iso, horizon_h, geo_df, base_int,
        events=events_df, near_repeat_alpha=0.35,
        nr_lookback_h=24, nr_radius_m=400, nr_decay_h=12.0,
        filters=filters,
    )

    # 30 günlük referans (opsiyonel)
    try:
        long_start_iso = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET - 30 * 24)).replace(minute=0, second=0, microsecond=0).isoformat()
        agg_long = aggregate_fast(long_start_iso, 30 * 24, geo_df, base_int, events=events_df, near_repeat_alpha=0.0, filters=None)
    except Exception:
        agg_long = None

    agg = ensure_keycol(agg, KEY_COL)
    if agg_long is not None and not agg_long.empty:
        agg_long = ensure_keycol(agg_long, KEY_COL)

    return agg, agg_long, start_iso, horizon_h

def top_risky_table(
    df_agg: pd.DataFrame, n: int, show_ci: bool, start_iso: Optional[str], horizon_h: int
) -> pd.DataFrame:
    def poisson_ci(lam: float, z: float = 1.96) -> tuple[float, float]:
        s = float(np.sqrt(max(lam, 1e-9))); return max(0.0, lam - z * s), lam + z * s

    cols = [KEY_COL, "expected"] + (["nr_boost"] if "nr_boost" in df_agg.columns else [])
    df = ensure_keycol(df_agg, KEY_COL)[cols].sort_values("expected", ascending=False).head(n).reset_index(drop=True)
    lam = df["expected"].to_numpy()

    # Prob(≥1)
    # prob_ge_k utils.forecast içinden
    df["P(≥1)%"] = [round(prob_ge_k(float(l), 1) * 100, 1) for l in lam]

    # Saat etiketi
    try:
        if start_iso:
            _start = pd.to_datetime(start_iso)
            _end = _start + pd.to_timedelta(horizon_h, unit="h")
            df["Saat"] = f"{_start.strftime('%H:%M')}–{_end.strftime('%H:%M')} (SF)"
        else:
            df["Saat"] = "-"
    except Exception:
        df["Saat"] = "-"

    if show_ci:
        ci = [poisson_ci(float(l)) for l in lam]
        df["95% GA"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in ci]

    if "nr_boost" in df.columns:
        df["NR"] = df["nr_boost"].round(2)

    df["E[olay] (λ)"] = df["expected"].round(2)
    drop = ["expected"] + (["nr_boost"] if "nr_boost" in df.columns else [])
    return df.drop(columns=drop)


# ─────────────────────────── UI ───────────────────────────
st.set_page_config(page_title="🔮 Suç Tahmini & Forecast", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)
st.title("🔮 Suç Tahmini & Forecast")

# Üst bilgi
LAST_UPDATE_ISO_SF = now_sf_iso()
st.caption(f"Model: {MODEL_VERSION} • Son eğitim: {MODEL_LAST_TRAIN} • Son güncelleme (SF): {LAST_UPDATE_ISO_SF}")

# GEO katmanı
GEO_DF, GEO_FEATURES = load_geoid_layer(os.path.join(ROOT, "data", "sf_cells.geojson"))
GEO_DF = ensure_keycol(GEO_DF, KEY_COL)
if GEO_DF.empty:
    st.error("GEOJSON yüklenemedi veya satır yok.")
    st.stop()

# Base intensity
BASE_INT = precompute_base_intensity(GEO_DF)

# Artifacts (popup için en sık 3 suç)
with st.spinner("Geçmiş veriler okunuyor..."):
    DF10, DF9 = load_artifacts()
    HIST = DF10 if DF10 is not None and not DF10.empty else (DF9 if DF9 is not None and not DF9.empty else pd.DataFrame())
TOP3 = top3_by_geoid(HIST)

# Sidebar (yalnızca tahmin sayfasına özel)
with st.sidebar:
    st.subheader("Tahmin Kontrolleri", anchor=False)
    current_time = datetime.now().strftime('%H:%M')
    current_date = datetime.now().strftime('%Y-%m-%d')
    ufuk_label = f"Zaman Aralığı (from {current_time}, today, {current_date})"
    ufuk = st.radio(ufuk_label, ["24s", "48s", "7g"], index=0, horizontal=True)
    max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7 * 24, 24)
    start_h, end_h = st.slider("Saat filtresi", 0, max_h, (0, max_h), step=step)

    sel_categories = st.multiselect("Kategori", ["(Hepsi)"] + CATEGORIES, default=[])
    filters = {"cats": CATEGORIES if sel_categories and "(Hepsi)" in sel_categories else (sel_categories or None)}

    st.markdown("—")
    show_popups = st.checkbox("Hücre pop-up’larında en çok 3 suç türünü göster", value=True)
    use_hot_hours = st.checkbox("Geçici hotspot için gün içi saat filtresi", value=False)
    if use_hot_hours:
        hot_hours_rng = st.slider("Saat aralığı (hotspot)", 0, 24, (0, 24))
    else:
        hot_hours_rng = (0, 24)

# State
st.session_state.setdefault("agg", None)
st.session_state.setdefault("agg_long", None)
st.session_state.setdefault("start_iso", None)
st.session_state.setdefault("horizon_h", None)
st.session_state.setdefault("explain", {})  # {"geoid": ...}

# Tahmin butonu ve çalıştırma
colA, colB = st.columns([1, 3])
with colA:
    btn_predict = st.button("Tahmin et", type="primary", use_container_width=True)
with colB:
    st.caption("Not: Ufuk ve kategori filtresine göre hızlandırılmış agregasyon yapılır.")

if btn_predict or st.session_state["agg"] is None:
    with st.spinner("Tahmin hesaplanıyor..."):
        agg, agg_long, start_iso, horizon_h = run_prediction(start_h, end_h, filters, GEO_DF, BASE_INT)
        st.session_state.update({
            "agg": agg,
            "agg_long": agg_long,
            "start_iso": start_iso,
            "horizon_h": horizon_h
        })

agg = st.session_state.get("agg")
start_iso = st.session_state.get("start_iso")
horizon_h = int(st.session_state.get("horizon_h") or 0)

# Harita
st.subheader("Harita", anchor=False)
if isinstance(agg, pd.DataFrame) and not agg.empty:
    # build_map_fast: top_counts parametresini destekliyorsa popup gösterir
    try:
        m = build_map_fast(
            df_agg=agg,
            geo_features=GEO_FEATURES,
            geo_df=GEO_DF,
            show_popups=show_popups,
            top_counts=TOP3,                 # <— en çok 3 suç için
            show_hotspot=True,
            perm_hotspot_mode="heat",
            show_temp_hotspot=True,
            temp_hotspot_points=pd.DataFrame(columns=["latitude","longitude","weight"]),
            add_layer_control=False,
            risk_layer_show=True,
            perm_hotspot_show=True,
            temp_hotspot_show=True,
            risk_layer_name="Tahmin (risk)",
            perm_hotspot_layer_name="Hotspot (kalıcı)",
            temp_hotspot_layer_name="Hotspot (geçici)",
        )
    except TypeError:
        # eski imza
        m = build_map_fast(
            df_agg=agg,
            geo_features=GEO_FEATURES,
            geo_df=GEO_DF,
            show_popups=show_popups,
            show_hotspot=True,
            perm_hotspot_mode="heat",
            show_temp_hotspot=True,
            temp_hotspot_points=pd.DataFrame(columns=["latitude","longitude","weight"]),
        )

    # Varsayılan layer control’ü kaldırıp sade bir zemin ekleyelim
    for k, ch in list(m._children.items()):
        if isinstance(ch, folium.map.LayerControl):
            del m._children[k]
    folium.TileLayer(
        tiles="CartoDB positron", name="cartodbpositron", control=True,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
             'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    ).add_to(m)

    # Harita render + tıklama ile geoid yakalama
    ret = st_folium(m, key="riskmap", height=560, width=None, returned_objects=["last_object_clicked", "last_clicked"])
    if ret:
        gid, _ = resolve_clicked_gid(GEO_DF, ret)
        if gid:
            st.session_state["explain"] = {"geoid": gid}

else:
    st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

# Sonuç kartı
info = st.session_state.get("explain")
if info and info.get("geoid"):
    render_result_card(agg, info["geoid"], start_iso, horizon_h)
else:
    st.caption("Haritada bir hücreye tıklayın; kart burada görünecek.")

# KPI ve Top-5 tablo
st.subheader("Risk Özeti", anchor=False)
a = st.session_state.get("agg")
if isinstance(a, pd.DataFrame) and not a.empty:
    kpi_expected = round(float(a["expected"].sum()), 2)
    tier = a.get("tier", pd.Series(dtype=str))
    cnts = {
        "Çok Yüksek": int((tier == "Çok Yüksek").sum()),
        "Yüksek": int((tier == "Yüksek").sum()),
        "Orta": int((tier == "Orta").sum()),
        "Düşük": int((tier == "Düşük").sum()),
        "Çok Düşük": int((tier == "Çok Düşük").sum()),
    }
    render_kpi_row([
        ("Beklenen olay (ufuk)", kpi_expected, "Seçili zaman ufkunda toplam beklenen olay sayısı"),
        ("Çok Yüksek", cnts["Çok Yüksek"], "En yüksek riskli hücre sayısı (üst %20)"),
        ("Yüksek", cnts["Yüksek"], "Yüksek kademe riskli hücre sayısı"),
        ("Orta", cnts["Orta"], "Orta kademe riskli hücre sayısı"),
        ("Düşük", cnts["Düşük"], "Düşük kademe riskli hücre sayısı"),
        ("Çok Düşük", cnts["Çok Düşük"], "En düşük riskli hücre sayısı (alt %20)"),
    ])
else:
    st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

st.subheader("Top-5 kritik GEOID", anchor=False)
if isinstance(a, pd.DataFrame) and not a.empty:
    tab = top_risky_table(
        a, n=5, show_ci=True,
        start_iso=st.session_state.get("start_iso"),
        horizon_h=int(st.session_state.get("horizon_h") or 0)
    )
    st.dataframe(tab, use_container_width=True)
    st.markdown("Seç / odağı haritada göster:")
    cols = st.columns(len(tab))
    for i, row in enumerate(tab.itertuples()):
        with cols[i]:
            if st.button(str(row.geoid)):
                st.session_state["explain"] = {"geoid": str(row.geoid)}
                st.experimental_rerun()
    st.caption("Butona tıklayınca haritada centroid işaretlenir ve açıklama kartı güncellenir.")
else:
    st.caption("Tablo, bir tahmin üretildiğinde gösterilir.")

# Isı matrisi
st.subheader("Gün × Saat Isı Matrisi", anchor=False)
if isinstance(a, pd.DataFrame) and not a.empty and st.session_state.get("start_iso"):
    H = int(st.session_state.get("horizon_h") or 24)
    # events_df opsiyonel – utils.heatmap, agg ve base_int ile çalışır
    try:
        from utils.reports import load_events  # type: ignore
        events_src = load_events(os.path.join(ROOT, "data", "events.csv"))
    except Exception:
        events_src = pd.DataFrame()
    render_day_hour_heatmap(
        agg=st.session_state["agg"],
        start_iso=st.session_state["start_iso"],
        horizon_h=H,
        geo_df=GEO_DF,
        base_int=BASE_INT,
        filters={"cats": CATEGORIES},  # burada geniş gösteriyoruz; istersen sidebar filtresini verebilirsin
        events_df=events_src,
    )
else:
    st.caption("Isı matrisi, bir tahmin üretildiğinde gösterilir.")

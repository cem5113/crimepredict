# pages/3_🧭_Suç_Tahmini.py
from __future__ import annotations
import io, os, zipfile
from datetime import datetime, timedelta, date, time as dt_time
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Sayfa başlığı / layout
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧭 Suç Tahmini", layout="wide")
st.title("🧭 Suç Tahmini — Seçime Göre Olasılık & Peak Saatler")

# ────────────────────────────────────────────────────────────────────────────────
# Opsiyonel kategori listesi (varsa kullan)
# ────────────────────────────────────────────────────────────────────────────────
try:
    from components.meta import CATEGORIES  # optional
except Exception:
    CATEGORIES = []

# ────────────────────────────────────────────────────────────────────────────────
# Kullanıcı ayarları (secrets ile override edilebilir)
# ────────────────────────────────────────────────────────────────────────────────
ARTIFACT_ZIP_PATH = st.secrets.get("ARTIFACT_ZIP_PATH", "fr-crime-outputs-parquet.zip")
ARTIFACT_INNER_ZIP = st.secrets.get("ARTIFACT_INNER_ZIP", "fr_parquet_outputs.zip")

# Risk / geçmiş / metrik dosya adayları (İÇ ZIP içinde aranır)
RISK_FILE_CANDIDATES = [
    "artifact/risk_hourly.parquet",  # önce artifact/ altında ara
    "risk_hourly.parquet",
    "risk_hourly.csv",
]
HIST_FILE_CANDIDATES = [
    "fr_crime_09.parquet",
    "fr_crime_09.csv",
    "fr_crime_10.parquet",
]
METRICS_FILE_CANDIDATES = [
    "artifact/metrics_base_ohe.parquet",
    "artifact/metrics_stacking_ohe.parquet",
    "metrics_base_ohe.parquet",
    "metrics_stacking_ohe.parquet",
]

# ────────────────────────────────────────────────────────────────────────────────
# ZIP okuma yardımcıları
# ────────────────────────────────────────────────────────────────────────────────
def _open_inner_zip(bin_bytes: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(bin_bytes))

def _find_file_in_zip(zf: zipfile.ZipFile, candidates: List[str]) -> Optional[str]:
    names = zf.namelist()
    low = {n.lower(): n for n in names}
    for cand in candidates:
        c = cand.lower()
        for k, real in low.items():
            if k.endswith(c):
                return real
    return None

def _read_any_df_from_zip(zf: zipfile.ZipFile, path: str) -> pd.DataFrame:
    with zf.open(path) as f:
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(f.read()))
        if path.lower().endswith(".csv"):
            return pd.read_csv(f)
        raise ValueError(f"Desteklenmeyen format: {path}")

def load_artifact_frames():
    """Dış ZIP → (varsa) iç ZIP → risk + history + metrics DataFrame'leri döndürür."""
    if not os.path.exists(ARTIFACT_ZIP_PATH):
        st.error(f"ZIP bulunamadı: {ARTIFACT_ZIP_PATH}")
        return None, None, None

    with zipfile.ZipFile(ARTIFACT_ZIP_PATH) as outer:
        # İç ZIP'i secrets'taki ad ile bul; yoksa ilk .zip'i al
        inner_name = None
        for n in outer.namelist():
            if n.lower().endswith(ARTIFACT_INNER_ZIP.lower()):
                inner_name = n
                break
        if inner_name is None:
            for n in outer.namelist():
                if n.lower().endswith(".zip"):
                    inner_name = n
                    break

        inner = outer if inner_name is None else _open_inner_zip(outer.read(inner_name))

        risk_name   = _find_file_in_zip(inner, RISK_FILE_CANDIDATES)
        hist_name   = _find_file_in_zip(inner, HIST_FILE_CANDIDATES)
        metric_name = _find_file_in_zip(inner, METRICS_FILE_CANDIDATES)

        risk_df = _read_any_df_from_zip(inner, risk_name) if risk_name else None
        hist_df = _read_any_df_from_zip(inner, hist_name) if hist_name else None
        metrics = _read_any_df_from_zip(inner, metric_name) if metric_name else None

        # küçük log
        if risk_name:   st.caption(f"Risk: `{risk_name}`")
        if hist_name:   st.caption(f"Geçmiş: `{hist_name}`")
        if metric_name: st.caption(f"Metrikler: `{metric_name}`")

        return risk_df, hist_df, metrics

# ────────────────────────────────────────────────────────────────────────────────
# Normalizasyon & yardımcılar
# ────────────────────────────────────────────────────────────────────────────────
def _ensure_dt(df: pd.DataFrame, col: str):
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # GEOID → geoid
    if "geoid" not in out.columns:
        if "GEOID" in out.columns:
            out.rename(columns={"GEOID": "geoid"}, inplace=True)
    # category / subcategory
    if "category" not in out.columns and "subcategory" in out.columns:
        out.rename(columns={"subcategory": "category"}, inplace=True)
    # hour üret
    if "hour" not in out.columns:
        for c in ["event_hour", "event_hour_x", "event_hour_y"]:
            if c in out.columns:
                out.rename(columns={c: "hour"}, inplace=True)
                break
    if "datetime" in out.columns:
        _ensure_dt(out, "datetime")
        if "hour" not in out.columns:
            out["hour"] = out["datetime"].dt.hour
        if "date" not in out.columns:
            out["date"] = out["datetime"].dt.date.astype(str)
    return out

def _combined_prob(ps: np.ndarray) -> float:
    one_minus = np.clip(1.0 - ps.astype(float), 1e-12, 1.0)
    return float(1.0 - np.prod(one_minus))

def _peak_hours_from_history(hist_df: pd.DataFrame, geoid: str, lookback_days: int = 365):
    if hist_df is None or hist_df.empty:
        return {}
    h = _normalize_cols(hist_df)
    if "datetime" not in h.columns:
        return {}
    _ensure_dt(h, "datetime")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    h = h[(h["geoid"].astype(str) == str(geoid)) & (h["datetime"] >= cutoff)].copy()
    if h.empty:
        return {}
    h["hod"] = h["datetime"].dt.hour
    grp = h.groupby(["category", "hod"]).size().reset_index(name="cnt")
    peaks = {}
    for cat, g in grp.groupby("category"):
        top = g.sort_values("cnt", ascending=False).head(3)
        peaks[cat] = [(int(r.hod), int(r.cnt)) for _, r in top.iterrows()]
    return peaks

def summarize_window_with_risk(risk_df: pd.DataFrame, geoid: str, start_dt, end_dt,
                               categories: Optional[List[str]] = None) -> pd.DataFrame:
    """risk_hourly tablosundan (date+hour bazlı) pencere özetini üretir."""
    r = _normalize_cols(risk_df)

    # row_dt oluştur
    if "date" in r.columns and "hour" in r.columns:
        r["row_dt"] = pd.to_datetime(r["date"]) + pd.to_timedelta(r["hour"].astype(int), unit="h")
    elif "datetime" in r.columns:
        r["row_dt"] = pd.to_datetime(r["datetime"])
    else:
        raise ValueError("risk_hourly içinde date+hour veya datetime beklenir.")

    # filtre
    m = (
        (r["row_dt"] >= start_dt) &
        (r["row_dt"] < end_dt) &
        (r["geoid"].astype(str) == str(geoid))
    )
    r = r.loc[m].copy()
    if r.empty:
        return pd.DataFrame()

    # kategori sütunu yoksa tek kategori kabul
    if "category" not in r.columns:
        r["category"] = "(Tümü)"

    # prob sütunu farklı isimlerde olabilir
    if "prob" not in r.columns:
        for alt in ["probability", "risk", "p"]:
            if alt in r.columns:
                r.rename(columns={alt: "prob"}, inplace=True)
                break
    if "prob" not in r.columns:
        raise ValueError("risk_hourly içinde prob/probability/risk kolonu bulunamadı.")

    r["prob"] = r["prob"].astype(float).clip(0.0, 0.95)

    # kategori filtresi
    if categories:
        r = r[r["category"].isin(categories)]
        if r.empty:
            return pd.DataFrame()

    agg = (
        r.groupby("category")
         .agg(
             hours_in_window=("prob", "size"),
             expected_count=("prob", "sum"),
             combined_prob=("prob", lambda s: _combined_prob(s.values)),
         )
         .reset_index()
         .sort_values("combined_prob", ascending=False)
    )
    return agg

# ────────────────────────────────────────────────────────────────────────────────
# Veri yükleme
# ────────────────────────────────────────────────────────────────────────────────
risk_df, hist_df, metrics_df = load_artifact_frames()
if risk_df is None and hist_df is None:
    st.error("ZIP içinde risk_hourly veya fr_crime_09 bulunamadı.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# UI — kullanıcı seçimleri
# ────────────────────────────────────────────────────────────────────────────────
# GEOID listesi
src = risk_df if risk_df is not None else hist_df
try:
    gtmp = _normalize_cols(src)
    geoid_list = sorted(map(str, pd.unique(gtmp["geoid"])))
except Exception:
    geoid_list = ["06075010100"]

with st.sidebar:
    st.subheader("⚙️ Seçimler")
    geoid = st.selectbox("GEOID", geoid_list)
    start_date = st.date_input("Başlangıç tarihi", date.today())

    horizon = st.select_slider(
        "Tahmin ufku (saat)", options=[24, 48, 72, 168], value=48,
        help="24 saate kadar saatlik, 72 saate kadar ~3s blok, haftalığa kadar günlük özet önerilir. (Birleşik olasılık tüm pencereyi kapsar.)"
    )

    # Kategori seçenekleri (öncelik: meta → veri)
    if CATEGORIES:
        all_cats = CATEGORIES
    else:
        try:
            all_cats = sorted(map(str, pd.unique(gtmp.get("category", pd.Series(["(Tümü)"])))))
        except Exception:
            all_cats = []
    cats = st.multiselect("Suç türleri (boş = tümü)", all_cats, default=[])

    run = st.button("📊 Hesapla")

start_dt = datetime.combine(start_date, dt_time(0, 0))
end_dt   = start_dt + timedelta(hours=int(horizon))

# ────────────────────────────────────────────────────────────────────────────────
# Çalıştır — özet + peak saatler + metrikler
# ────────────────────────────────────────────────────────────────────────────────
if run:
    # 1) risk_hourly ile özet
    agg = pd.DataFrame()
    try:
        if risk_df is not None:
            agg = summarize_window_with_risk(
                risk_df, geoid, start_dt, end_dt, categories=cats or None
            )
    except Exception as e:
        st.warning(f"Risk verisi özetlenemedi: {e}")

    if agg.empty:
        st.warning("Seçili aralık için risk satırı bulunamadı. (Gerekirse history fallback eklenebilir.)")
    else:
        st.markdown(
            f"**{geoid}** için **{start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M}** aralığı:"
        )

        # 2) Peak saatler (geçmiş veriden)
        peak_map = _peak_hours_from_history(hist_df, geoid, lookback_days=365) if hist_df is not None else {}

        # 3) Tablo
        show = agg.copy()
        show["combined_prob_pct"] = (show["combined_prob"] * 100).round(1).astype(str) + "%"
        show = show.rename(
            columns={
                "category": "Suç",
                "hours_in_window": "Saat sayısı",
                "expected_count": "Beklenen adet",
                "combined_prob": "Birleşik olasılık",
                "combined_prob_pct": "Olasılık (%)",
            }
        )
        st.dataframe(show[["Suç", "Olasılık (%)", "Beklenen adet", "Saat sayısı"]], use_container_width=True)

        # 4) Liste formatında, peak saatlerle
        st.markdown("**Detaylı liste (peak saatlerle):**")
        for _, r in agg.iterrows():
            cat = r["category"]
            peaks = ", ".join([f"{h:02d}:00 ({c})" for h, c in peak_map.get(cat, [])]) if peak_map else "—"
            st.write(
                f"- **{cat}**: {r['combined_prob']:.3f} ({r['combined_prob']*100:.1f}%) "
                f"| Expected={r['expected_count']:.2f} | Hours={int(r['hours_in_window'])} "
                f"| Peak: {peaks}"
            )

    # 5) (Opsiyonel) Model metrikleri
    if metrics_df is not None:
        st.divider()
        st.subheader("📈 Model Metrikleri")
        try:
            cols_lower = {c.lower(): c for c in metrics_df.columns}
            pick = [
                cols_lower.get(k) for k in ["fold", "auc", "roc_auc", "pr_auc", "f1", "accuracy", "logloss"]
                if cols_lower.get(k) is not None
            ]
            if pick:
                st.dataframe(metrics_df[pick], use_container_width=True)
            else:
                st.dataframe(metrics_df.head(50), use_container_width=True)
        except Exception:
            st.dataframe(metrics_df.head(50), use_container_width=True)

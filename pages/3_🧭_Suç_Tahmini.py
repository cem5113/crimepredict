# pages/3_🧭_Suç_Tahmini.py
from __future__ import annotations
import io, os, zipfile
from datetime import datetime, timedelta, date, time as dt_time
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ── Projedeki yardımcılar/constant'lar
try:
    from components.meta import CATEGORIES
except Exception:
    CATEGORIES = []

# ========== KULLANICI AYARLARI ==========
# Artifact ZIP dosyasına giden yol (ör: local path ya da mount edilen klasör)
# Örn: "/mount/src/artifacts/fr-crime-outputs-parquet.zip"
ARTIFACT_ZIP_PATH = st.secrets.get("ARTIFACT_ZIP_PATH", "fr-crime-outputs-parquet.zip")

# Risk veri dosya ad adayları (ZIP içinde arayacağımız)
RISK_FILE_CANDIDATES = [
    "risk_hourly.parquet",          # ideal
    "risk_next_24h.parquet",        # başka bir ad
    "risk_hourly.csv",
]
# Geçmiş olay dosya ad adayları (ZIP içinde arayacağımız)
HIST_FILE_CANDIDATES = [
    "fr_crime_09.parquet",
    "fr_crime_10.parquet",
    "fr_crime_09.csv",
]
# Stacking metrikleri (opsiyonel)
METRICS_FILE_CANDIDATES = [
    "artifact/metrics_stacking_ohe.parquet",
    "metrics_stacking_ohe.parquet",
]

# ========================================

st.set_page_config(page_title="🧭 Suç Tahmini", layout="wide")
st.title("🧭 Suç Tahmini — Top Suçlar & Peak Saatler")

# ---------------- ZIP OKUMA YARDIMCILAR ----------------
def _open_inner_zip(bin_bytes: bytes) -> zipfile.ZipFile:
    """ZIP içindeki ZIP'i açmak için yardımcı."""
    return zipfile.ZipFile(io.BytesIO(bin_bytes))

def _find_file_in_zip(zf: zipfile.ZipFile, candidates: List[str]) -> Optional[str]:
    names = zf.namelist()
    lower_map = {n.lower(): n for n in names}
    for cand in candidates:
        for key, real in lower_map.items():
            if key.endswith(cand.lower()):
                return real
    return None

def _read_any_df_from_zip(zf: zipfile.ZipFile, path: str) -> pd.DataFrame:
    with zf.open(path) as f:
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(f.read()))
        elif path.lower().endswith(".csv"):
            return pd.read_csv(f)
        else:
            # bazen parquet klasör/dataset olabilir; folder ise olmaz.
            raise ValueError(f"Desteklenmeyen format: {path}")

def load_artifact_frames(artifact_zip_path: str):
    """Dış ZIP → (gerekirse) iç ZIP → hedef dosyaları DataFrame olarak döndürür."""
    if not os.path.exists(artifact_zip_path):
        st.warning(f"Artifact ZIP bulunamadı: {artifact_zip_path}")
        return None, None, None

    with zipfile.ZipFile(artifact_zip_path) as outer:
        # İçte bir ZIP daha varsa onu bul
        inner_zip_name = None
        for n in outer.namelist():
            if n.lower().endswith(".zip"):
                inner_zip_name = n
                break

        if inner_zip_name is None:
            # doğrudan dış ZIP içinde ara
            inner = outer
        else:
            # iç ZIP'i aç
            inner_bytes = outer.read(inner_zip_name)
            inner = _open_inner_zip(inner_bytes)

        # Risk & tarihsel & metrik dosyalarını bul
        risk_name   = _find_file_in_zip(inner, RISK_FILE_CANDIDATES)
        hist_name   = _find_file_in_zip(inner, HIST_FILE_CANDIDATES)
        metric_name = _find_file_in_zip(inner, METRICS_FILE_CANDIDATES)

        risk_df = _read_any_df_from_zip(inner, risk_name) if risk_name else None
        hist_df = _read_any_df_from_zip(inner, hist_name) if hist_name else None
        metrics = _read_any_df_from_zip(inner, metric_name) if metric_name else None

        return risk_df, hist_df, metrics

# ------------- NORMALİZE EDİCİLER -------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # geoid
    if "geoid" in cols:
        gcol = cols["geoid"]
    elif "geoid" not in cols and "geoid" not in df.columns and "GEOID" in df.columns:
        gcol = "GEOID"
    elif "geoid" not in cols and "geoid" in df.columns:
        gcol = "geoid"
    else:
        gcol = cols.get("geoid", "geoid")

    # category
    if "category" in cols:
        catcol = cols["category"]
    elif "subcategory" in cols:
        catcol = cols["subcategory"]
    else:
        # yoksa tek bir kategoriye indirgenmiş olabilir
        catcol = None

    # hour
    hourcol = None
    for cand in ["hour", "event_hour", "event_hour_x", "event_hour_y"]:
        if cand in df.columns:
            hourcol = cand; break

    # date/datetime/timestamp
    tscol = None
    for cand in ["datetime", "timestamp"]:
        if cand in df.columns:
            tscol = cand; break

    # prob (risk tablosu için)
    probcol = None
    for cand in ["prob", "probability", "risk", "p"]:
        if cand in df.columns:
            probcol = cand; break

    out = df.copy()
    # geoid birleşik
    if gcol not in out.columns:
        raise ValueError("Veride 'geoid' veya 'GEOID' sütunu bulunamadı.")
    out.rename(columns={gcol: "geoid"}, inplace=True)

    # kategori birleşik
    if catcol is not None and catcol in out.columns and catcol != "category":
        out.rename(columns={catcol: "category"}, inplace=True)

    # saat sütunu yoksa datetime'tan türet
    if hourcol is None and tscol and tscol in out.columns:
        out[tscol] = pd.to_datetime(out[tscol])
        out["hour"] = out[tscol].dt.hour
    elif hourcol:
        if hourcol != "hour":
            out.rename(columns={hourcol: "hour"}, inplace=True)
        out["hour"] = out["hour"].astype(int)

    # tarih sütunu yoksa datetime'tan türet
    if "date" not in out.columns:
        if tscol and tscol in out.columns:
            out["date"] = pd.to_datetime(out[tscol]).dt.date.astype(str)

    # prob yoksa (risk dataframe’i değilse) None olarak bırak
    if probcol and probcol != "prob":
        out.rename(columns={probcol: "prob"}, inplace=True)

    return out

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

# ------------- ÖZETLEME (window) -------------
def combined_prob(ps: np.ndarray) -> float:
    one_minus = np.clip(1.0 - ps.astype(float), 1e-12, 1.0)
    return float(1.0 - np.prod(one_minus))

def summarize_for_window(risk_df: pd.DataFrame,
                         hist_df: Optional[pd.DataFrame],
                         geoid: str,
                         start_dt: datetime,
                         end_dt: datetime,
                         top_k: int = 5,
                         hist_days_for_peaks: int = 365):
    # Risk DF normalize
    rsk = normalize_columns(risk_df)
    # saatlik zaman damgası üret
    if "date" in rsk.columns and "hour" in rsk.columns:
        rsk["row_dt"] = pd.to_datetime(rsk["date"]) + pd.to_timedelta(rsk["hour"].astype(int), unit="h")
    elif "datetime" in rsk.columns:
        rsk["row_dt"] = pd.to_datetime(rsk["datetime"])
    else:
        raise ValueError("Risk tablosunda 'date+hour' ya da 'datetime' bekleniyor.")

    # Seçili pencere + geoid
    win = rsk[(rsk["geoid"] == geoid) & (rsk["row_dt"] >= start_dt) & (rsk["row_dt"] < end_dt)].copy()
    if win.empty:
        return None

    # kategori kolonu yoksa tek kategorili varsayalım
    if "category" not in win.columns:
        win["category"] = "(Tümü)"

    # olasılık kolonu yoksa basit baseline (geçmiş sıklık) ile doldurmaya çalış
    if "prob" not in win.columns:
        # Eğer risk yoksa, historical df'ten saat-of-day oranına göre kaba p üretilebilir.
        if hist_df is None:
            st.warning("Risk olasılığı yok ve geçmiş veri yok; özet üretilemiyor.")
            return None
        hist = normalize_columns(hist_df)
        hist = ensure_datetime(hist, "datetime")
        hist = hist[hist["geoid"] == geoid].copy()
        hist["hod"] = hist["datetime"].dt.hour
        base = hist.groupby(["category","hod"]).size().reset_index(name="cnt")
        base["p"] = base["cnt"] / base["cnt"].max()
        win["prob"] = 0.01  # çok kaba varsayılan
        # istenirse hod eşleşmesi ile p atanabilir

    win["prob"] = win["prob"].astype(float)

    agg = win.groupby("category").agg(
        hours_in_window=("prob","size"),
        expected_count=("prob","sum"),
        combined_prob=("prob", lambda s: combined_prob(s.values)),
    ).reset_index()

    agg["rank_by_prob"] = agg["combined_prob"].rank(method="first", ascending=False).astype(int)
    agg = agg.sort_values("combined_prob", ascending=False)

    # Peak saatleri (historical)
    peak_map = {}
    if hist_df is not None:
        h = normalize_columns(hist_df)
        if "datetime" in h.columns:
            h = ensure_datetime(h, "datetime")
            cutoff = pd.to_datetime(datetime.utcnow() - timedelta(days=hist_days_for_peaks))
            h = h[(h["geoid"] == geoid) & (h["datetime"] >= cutoff)].copy()
            if not h.empty:
                h["hour_of_day"] = h["datetime"].dt.hour
                grp = h.groupby(["category","hour_of_day"]).size().reset_index(name="cnt")
                for cat, g in grp.groupby("category"):
                    pk = g.sort_values("cnt", ascending=False).head(3)
                    peak_map[cat] = [(int(r.hour_of_day), int(r.cnt)) for _, r in pk.iterrows()]

    # Recent true count (pencere içi)
    recent_counts = {}
    if hist_df is not None:
        h2 = normalize_columns(hist_df)
        if "datetime" in h2.columns:
            h2 = ensure_datetime(h2, "datetime")
            mask = (h2["geoid"] == geoid) & (h2["datetime"] >= start_dt) & (h2["datetime"] < end_dt)
            rc = h2.loc[mask].groupby("category").size()
            recent_counts = rc.to_dict()

    # Nihai liste
    rows = []
    for _, r in agg.head(5).iterrows():
        cat = r["category"]
        rows.append({
            "category": cat,
            "combined_prob": float(r["combined_prob"]),
            "combined_prob_pct": f"{r['combined_prob']*100:.1f}%",
            "expected_count": float(r["expected_count"]),
            "hours_covered": int(r["hours_in_window"]),
            "recent_true_count_in_window": int(recent_counts.get(cat, 0)),
            "peak_hours": peak_map.get(cat, []),
        })
    return rows

# ------------- DATA YÜKLE -------------
risk_df, hist_df, metrics_df = load_artifact_frames(ARTIFACT_ZIP_PATH)

if risk_df is None and hist_df is None:
    st.error("Artifact içinden risk ya da geçmiş veri bulunamadı. ZIP yolunu kontrol edin.")
    st.stop()

# ------------- UI -------------
with st.sidebar:
    st.subheader("⚙️ Seçimler")
    # GEOID listesi (hist ya da risk'ten)
    _src = risk_df if risk_df is not None else hist_df
    try:
        gtmp = normalize_columns(_src)
        geoids = sorted(map(str, pd.unique(gtmp["geoid"])))
    except Exception:
        geoids = []

    geoid = st.selectbox("GEOID", geoids or ["06075010100"])

    # Başlangıç tarihi (varsayılan bugün)
    start_date = st.date_input("Başlangıç tarihi", date.today())
    # Ufuk & çözünürlük
    horizon = st.select_slider("Tahmin ufku",
                               options=[24, 48, 72, 168],
                               value=48,
                               help="24 saate kadar saatlik, 72 saate kadar 3 saatlik, haftalığa kadar günlük gösterebiliriz.")
    # Kategori filtresi
    if CATEGORIES:
        cats = st.multiselect("Suç türleri (boş = tümü)", CATEGORIES, default=[])
    else:
        # veri setinden çıkar
        try:
            cats_all = sorted(map(str, pd.unique(gtmp.get("category", pd.Series(["(Tümü)"])))))
        except Exception:
            cats_all = []
        cats = st.multiselect("Suç türleri (boş = tümü)", cats_all, default=[])

    run = st.button("📊 Hesapla")

# pencere
start_dt = datetime.combine(start_date, dt_time(0, 0))
end_dt   = start_dt + timedelta(hours=int(horizon))

# ------------- ÇALIŞTIR -------------
if run:
    rows = summarize_for_window(risk_df, hist_df, geoid, start_dt, end_dt, top_k=5)

    if not rows:
        st.warning("Seçilen pencere için veri bulunamadı.")
    else:
        st.markdown(f"**{geoid}** için **{start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M}** aralığında en olası suçlar:")
        for r in rows:
            if cats and r["category"] not in cats:
                continue
            peaks = ", ".join([f"{h:02d}:00 ({c})" for h, c in r["peak_hours"]]) if r["peak_hours"] else "—"
            st.write(
                f"- **{r['category']}**: {r['combined_prob_pct']}  "
                f"(expected={r['expected_count']:.2f}, hours={r['hours_covered']}, recent={r['recent_true_count_in_window']})  "
                f"→ Peak: {peaks}"
            )

    # (Opsiyonel) Model metrikleri bölümü
    if metrics_df is not None:
        st.divider()
        st.subheader("📈 Model Metrikleri (Stacking)")
        try:
            # kısa özet göster
            cols = [c for c in metrics_df.columns if c.lower() in {"fold","auc","roc_auc","pr_auc","f1","accuracy","logloss"}]
            if cols:
                st.dataframe(metrics_df[cols])
            else:
                st.dataframe(metrics_df.head(50))
        except Exception:
            st.dataframe(metrics_df.head(50))


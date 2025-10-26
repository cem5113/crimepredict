# pages/3_🧭_Suç_Tahmini.py
from __future__ import annotations
import sys, pathlib

# --- bootstrap ---
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- güvenli constants import ---
try:
    from utils.constants import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES
except Exception:
    SF_TZ_OFFSET  = -7
    KEY_COL       = "geoid"
    MODEL_VERSION = "v0"
    MODEL_LAST_TRAIN = "-"
    CATEGORIES    = ["Assault","Burglary","Robbery","Theft","Vandalism","Vehicle Theft"]

import os, io, zipfile, requests
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from components.last_update import show_last_update_badge


st.set_page_config(page_title="🔮 Suç Tahmini (Stacking Model)", layout="wide")
st.title("🔮 Suç Tahmini ve Model Performansı")

# ───────────────────────────────
# 📦 GitHub artifact'tan veri yükleme
# ───────────────────────────────
# --- ayarlar ---
OWNER = "cem5113"
REPO  = "crime_prediction_data"
TARGET_ARTIFACT = "fr-crime-outputs-parquet"

def resolve_github_token() -> str | None:
    import os
    import streamlit as st
    for k in ("GITHUB_TOKEN","GH_TOKEN","github_token"):
        v = os.getenv(k) or (getattr(st, "secrets", {}).get(k) if hasattr(st, "secrets") else None)
        if v:
            os.environ["GITHUB_TOKEN"] = v
            return v
    return None

@st.cache_data(show_spinner=True)
def load_artifact_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Önce artifact/risk_hourly.parquet (küçük ve hazır) -> df_risk
    2) Yoksa fr_crime_09.parquet'ten sadece gerekli kolonları okuyup GEOID bazında risk hesapla
    3) Metrikleri artifact/metrics_stacking_ohe.parquet'ten yükle (varsa)
    """
    token = resolve_github_token()
    if not token:
        st.error("GitHub Token bulunamadı (GH_TOKEN / GITHUB_TOKEN).")
        return pd.DataFrame(), pd.DataFrame()

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    try:
        # 1) Son başarılı run → artifact listesi
        runs = requests.get(
            f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs?per_page=10",
            headers=headers, timeout=30
        ).json()
        run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]
        for rid in run_ids:
            arts = requests.get(
                f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{rid}/artifacts",
                headers=headers, timeout=30
            ).json().get("artifacts", [])

            target = next((a for a in arts if a["name"] == TARGET_ARTIFACT and not a.get("expired")), None)
            if not target:
                continue

            zdata = requests.get(target["archive_download_url"], headers=headers, timeout=60).content
            outer = zipfile.ZipFile(io.BytesIO(zdata))

            # iç zip’i bul ve aç
            inner_zip_name = next((n for n in outer.namelist() if n.lower().endswith(".zip")), None)
            if not inner_zip_name:
                st.error("İç zip (ör. fr_parquet_outputs.zip) bulunamadı.")
                return pd.DataFrame(), pd.DataFrame()
            nested = zipfile.ZipFile(io.BytesIO(outer.read(inner_zip_name)))

            names = nested.namelist()

            # ---- A) Öncelik: küçük dosya (hazır risk) ----
            risk_ready = next((n for n in names if n.endswith("artifact/risk_hourly.parquet")), None)

            # ---- B) Metrikler ----
            metrics_name = next((n for n in names if n.endswith("artifact/metrics_stacking_ohe.parquet")), None)
            metrics = pd.DataFrame()
            if metrics_name:
                try:
                    metrics = pd.read_parquet(io.BytesIO(nested.read(metrics_name)))
                except Exception as e:
                    st.warning(f"metrics_stacking_ohe.parquet okunamadı: {e}")

            # ---- A yolu: risk_hourly varsa direkt onu kullan ----
            if risk_ready:
                df = pd.read_parquet(io.BytesIO(nested.read(risk_ready)))
                # Beklenen kolonlar: geoid, hour, risk (ya da score). Esnek adlandırma:
                low = {c.lower(): c for c in df.columns}
                gcol = low.get("geoid") or low.get("geoid_x") or low.get("id") or list(df.columns)[0]
                # hour kolonunu şart koşma; sadece GEOID bazında ortalama risk göster
                scol = low.get("risk") or low.get("score") or low.get("prob") or list(df.columns)[-1]
                df = (df[[gcol, scol]]
                      .rename(columns={gcol: "geoid", scol: "risk_score"})
                      .groupby("geoid", as_index=False)["risk_score"].mean())
                st.success("✅ risk_hourly.parquet yüklendi (hafif).")
                return df, metrics

            # ---- B yolu: büyük dosyadan minimal okuma ----
            crime_name = next((n for n in names if n.endswith("fr_crime_09.parquet")), None)
            if not crime_name:
                st.error("fr_crime_09.parquet bulunamadı.")
                return pd.DataFrame(), metrics

            # Sadece gerekli kolonlar
            candidate_cols = [
                "geoid","GEOID","id",
                "latitude","lat",
                "longitude","lon",
                "category",
                "event_hour","event_hour_x",
                "Y_label"
            ]
            # Parquet'te kolonları görmeden ‘columns=’ veremeyiz; önce schema alalım
            tmp = pd.read_parquet(io.BytesIO(nested.read(crime_name)), columns=None)
            cols_exist = [c for c in candidate_cols if c in tmp.columns]

            # hafızayı koru: yalnız bu kolonları oku
            df_big = tmp[cols_exist].copy()
            del tmp  # memory

            # kolon adlarını normalize et
            low = {c.lower(): c for c in df_big.columns}
            gcol = low.get("geoid") or low.get("id")
            latc = low.get("latitude") or low.get("lat")
            lonc = low.get("longitude") or low.get("lon")
            ycol = low.get("y_label")
            hcol = low.get("event_hour") or low.get("event_hour_x")
            ccol = low.get("category")

            # çok büyükse örnekle (örn. 300k satır)
            N_MAX = 300_000
            if len(df_big) > N_MAX:
                df_big = df_big.sample(N_MAX, random_state=7)

            # risk: GEOID bazında ortalama Y_label (yoksa olay sayısı)
            if ycol and ycol in df_big.columns:
                risk = df_big.groupby(gcol)[ycol].mean().reset_index(name="risk_score")
            else:
                risk = df_big.groupby(gcol).size().reset_index(name="risk_score")

            # merkezler: lat/lon ortalaması
            if latc in df_big.columns and lonc in df_big.columns:
                centers = (df_big.groupby(gcol)[[latc, lonc]].mean()
                           .reset_index().rename(columns={gcol: "geoid", latc: "latitude", lonc: "longitude"}))
                risk = risk.rename(columns={gcol: "geoid"}).merge(centers, on="geoid", how="left")
            else:
                risk = risk.rename(columns={gcol: "geoid"})

            st.success("✅ fr_crime_09.parquet minimal kolonlarla yüklendi.")
            return risk, metrics

        st.error(f"Artifact bulunamadı ({TARGET_ARTIFACT}).")
        return pd.DataFrame(), pd.DataFrame()

    except Exception as e:
        st.error(f"Artifact indirilemedi/açılırken hata: {e}")
        return pd.DataFrame(), pd.DataFrame()

df, metrics = load_artifact_data()

# ───────────────────────────────
# 🕒 Güncel durum ve başlık
# ───────────────────────────────
show_last_update_badge(
    data_upto=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)
# ───────────────────────────────
# 📊 Model Metrikleri
# ───────────────────────────────
st.subheader("📈 Model Performans Özeti")

if not metrics.empty:
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("Model metrikleri bulunamadı (metrics_stacking_ohe.parquet).")

# ───────────────────────────────
# 🔮 Tahmin Haritası
# ───────────────────────────────
st.subheader("📍 Suç Olasılığı Haritası")

if not df.empty:
    df = df.rename(columns=lambda c: c.strip())
    # GEOID ve koordinatları kontrol et
    lat_col = next((c for c in df.columns if "lat" in c.lower()), "latitude")
    lon_col = next((c for c in df.columns if "lon" in c.lower()), "longitude")
    geoid_col = next((c for c in df.columns if "geoid" in c.lower()), "GEOID")

    # Filtreler
    st.sidebar.markdown("### 🔎 Filtreler")
    cats = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []
    sel_cat = st.sidebar.selectbox("Suç Kategorisi", ["(Tümü)"] + cats)
    hour_range = st.sidebar.slider("Saat aralığı", 0, 24, (0, 24))

    # Zaman filtresi
    if "event_hour_x" in df.columns:
        df = df[df["event_hour_x"].between(hour_range[0], hour_range[1])]
    if sel_cat != "(Tümü)" and "category" in df.columns:
        df = df[df["category"] == sel_cat]

    # Risk skorlarını oluştur (örnek)
    if "Y_label" in df.columns:
        risk = df.groupby(geoid_col)["Y_label"].mean().reset_index(name="risk_score")
    else:
        risk = df.groupby(geoid_col).size().reset_index(name="risk_score")

    # Coğrafi merkez hesaplama
    if lat_col in df.columns and lon_col in df.columns:
        geo = df.groupby(geoid_col)[[lat_col, lon_col]].mean().reset_index()
        risk = risk.merge(geo, on=geoid_col, how="left")

    # PyDeck haritası
    layer = pdk.Layer(
        "HeatmapLayer",
        data=risk,
        get_position=[lon_col, lat_col],
        get_weight="risk_score",
        radiusPixels=40,
        opacity=0.6,
    )

    view = pdk.ViewState(
        latitude=risk[lat_col].mean(),
        longitude=risk[lon_col].mean(),
        zoom=11.2,
        pitch=30,
    )

    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v11", initial_view_state=view, layers=[layer]))

    # En riskli GEOID'ler
    st.subheader("🚨 En Riskli 10 GEOID")
    top10 = risk.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(top10, use_container_width=True)

    st.caption("Not: Risk skoru, GEOID bazında suç gerçekleşme olasılığının normalize edilmiş değeridir.")
else:
    st.warning("Veri yüklenemedi veya boş. Artifact bağlantısını kontrol edin.")

# ───────────────────────────────
# 🧠 Sonuç Özeti
# ───────────────────────────────
st.markdown("---")
st.markdown(
    """
    **🧠 Özet:**  
    Bu sayfa, stacking tabanlı modelin son tahmin çıktısını `fr_crime_09.csv` üzerinden yükler ve  
    model metriklerini `metrics_stacking_ohe.parquet` dosyasından alır.  
    Harita, her GEOID için normalize edilmiş suç riski yoğunluğunu gösterir.  
    """
)

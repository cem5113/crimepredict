# pages/3_🧭_Suç_Tahmini.py
# -----------------------------------------------------------
# Basit: Artifact ZIP -> Parquet oku -> Seçime göre anlık tahmin
# -----------------------------------------------------------

import io, zipfile, requests, math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# ML
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="🧭 Suç Tahmini (Demo)", layout="wide")
st.title("🧭 Suç Tahmini (Demo)")
st.caption("GitHub artifact'tan veri okunur, kullanıcı seçimine göre anlık tahmin yapılır.")

# --- Kullanıcı ayarları / sabitler ---
ZIP_URL_DEFAULT = "https://github.com/cem5113/crime_prediction_data/releases/latest/download/fr-crime-outputs-parquet.zip"
MAX_ROWS_PREVIEW = 1000

# -----------------------------------------------------------
# Yardımcılar
# -----------------------------------------------------------
def _find_col(df, candidates, default=None):
    """Aday isimler içinden ilk mevcut olan sütunu döndür."""
    for c in candidates:
        if c in df.columns:
            return c
    return default

def _to_datetime_safe(s, fallback_format=None):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        if fallback_format:
            return pd.to_datetime(s, format=fallback_format, errors="coerce")
        return pd.to_datetime(s, errors="coerce")

@st.cache_data(show_spinner=True, ttl=60*30)  # 30 dk cache
def load_from_artifact(zip_url: str):
    """Artifact ZIP indir, gerekli parquet'leri oku."""
    resp = requests.get(zip_url, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()

    # Dosya yolları esnek olsun
    def pick(name_part):
        for n in names:
            if name_part in n and n.endswith(".parquet"):
                return n
        return None

    data_fp = pick("fr_crime_09") or pick("fr_crime_10") or pick("fr_crime")  # en zengin ilk
    metrics_fp = pick("metrics_stacking_ohe")

    if not data_fp:
        raise FileNotFoundError("ZIP içinde fr_crime_09/10/parquet bulunamadı.")

    df = pd.read_parquet(zf.open(data_fp))
    metrics = None
    if metrics_fp:
        try:
            metrics = pd.read_parquet(zf.open(metrics_fp))
        except Exception:
            metrics = None
    return df, metrics, names

def pick_last_12m(df, date_col):
    """Son 12 aya indirger (varsa)."""
    if date_col is None:
        return df
    dts = _to_datetime_safe(df[date_col])
    max_dt = dts.max()
    if pd.isna(max_dt):
        return df
    start_dt = max_dt - pd.Timedelta(days=365)
    return df.loc[(dts >= start_dt) & (dts <= max_dt)].copy()

def build_model(train_df, target_col="Y_label", exclude_cols=None, seed=42):
    """Hızlı bir XGBClassifier kur."""
    if exclude_cols is None:
        exclude_cols = []
    cols_num = train_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    cols_num = [c for c in cols_num if c != target_col and c not in exclude_cols]

    X = train_df[cols_num]
    y = train_df[target_col].astype(int)

    # Basit train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y if y.nunique()==2 else None
    )

    # Hafif model (hızlı)
    model = XGBClassifier(
        n_estimators=220,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=0,
        random_state=seed,
        tree_method="hist"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test))==2 else np.nan

    return model, cols_num, float(auc)

@st.cache_resource(show_spinner=True)
def train_cached(train_df, target_col, exclude_cols):
    return build_model(train_df, target_col=target_col, exclude_cols=exclude_cols)

# -----------------------------------------------------------
# 1) Artifact'tan veri yükle
# -----------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Veri Kaynağı")
    zip_url = st.text_input("Artifact ZIP URL", ZIP_URL_DEFAULT)
    if st.button("🔄 Veriyi İndir & Yükle"):
        st.session_state["_reload"] = True

# İlk yükleme
if "_reload" not in st.session_state:
    st.session_state["_reload"] = True

if st.session_state["_reload"]:
    try:
        df_raw, metrics_df, zip_names = load_from_artifact(zip_url)
        st.session_state["df_raw"] = df_raw
        st.session_state["metrics_df"] = metrics_df
        st.session_state["zip_names"] = zip_names
        st.session_state["_reload"] = False
    except Exception as e:
        st.error(f"Veri indirilemedi: {e}")
        st.stop()

df_raw = st.session_state["df_raw"].copy()
metrics_df = st.session_state.get("metrics_df")
zip_names = st.session_state.get("zip_names", [])

with st.expander("ZIP içeriği"):
    st.code("\n".join(zip_names) or "(boş)")

st.success(f"✅ Veri yüklendi: {df_raw.shape[0]:,} satır × {df_raw.shape[1]:,} sütun")

# -----------------------------------------------------------
# 2) Sütun tespiti ve son 12 ay filtresi
# -----------------------------------------------------------
# Temel kolonlar
geoid_col = _find_col(df_raw, ["GEOID", "geoid", "Key", "key"], default=None)
date_col  = _find_col(df_raw, ["date", "Date", "dt", "event_date"], default=None)
hour_col  = _find_col(df_raw, ["event_hour_x", "event_hour", "hour", "event_hour_y"], default=None)
cat_col   = _find_col(df_raw, ["category", "Category", "cat"], default=None)
y_col     = _find_col(df_raw, ["Y_label", "label", "y"], default="Y_label")

if geoid_col is None or y_col not in df_raw.columns:
    st.error("Gerekli sütunlar yok: en azından GEOID ve Y_label bulunmalı.")
    st.stop()

df = df_raw.copy()
df[y_col] = df[y_col].astype(int)

# Son 12 ay (varsa date)
df_12m = pick_last_12m(df, date_col)
if len(df_12m) < len(df):
    st.info(f"⏱️ Model eğitimi için son 12 ay seçildi: {len(df_12m):,} satır.")
else:
    st.info("⏱️ Tarih sütunu tespit edilemedi veya uygun değil; tüm veri kullanılacak.")
df_train = df_12m

# -----------------------------------------------------------
# 3) Kenar panel: seçimler
# -----------------------------------------------------------
with st.sidebar:
    st.subheader("🔎 Seçimler")
    # Kategori seçimi (opsiyonel)
    cats = sorted(df[cat_col].dropna().unique().tolist()) if cat_col else []
    cat_sel = st.selectbox("Suç kategorisi", ["(tümü)"] + cats, index=0)

    # Saat seçimi
    if hour_col and df[hour_col].notna().any():
        min_h = int(pd.Series(df[hour_col]).min())
        max_h = int(pd.Series(df[hour_col]).max())
        hour_sel = st.slider("Saat (event_hour)", min_value=min(0, min_h), max_value=max(23, max_h), value=12, step=1)
    else:
        hour_sel = st.slider("Saat (kolon bulunamadı, filtre görsel amaçlı)", 0, 23, 12, 1)

    # GEOID seçimi
    geoids = df[geoid_col].dropna().astype(str).unique().tolist()
    default_geoid = geoids[0] if geoids else ""
    geoid_sel = st.text_input("GEOID", value=default_geoid)

    # Eğitim tetikleyici
    go_train = st.button("🚀 Modeli Eğit ve Tahmin Yap")

# -----------------------------------------------------------
# 4) Veri altkümeleri ve model
# -----------------------------------------------------------
# Gösterim filtresi (sadece önizleme ve kullanıcı tahmini için)
show_df = df.copy()
if cat_col and cat_sel != "(tümü)":
    show_df = show_df.loc[show_df[cat_col] == cat_sel]
if hour_col:
    show_df = show_df.loc[show_df[hour_col] == hour_sel]

st.markdown("### 📋 Veri Önizleme")
st.dataframe(show_df.head(MAX_ROWS_PREVIEW))

# Eğitim setinden bazı kolonları hariç tut (ID/kimliksel metinler)
exclude_cols = [c for c in ["id", "ID", "datetime", "received_time", geoid_col] if c in df_train.columns]

if go_train:
    with st.spinner("Model eğitiliyor..."):
        model, used_cols, auc = train_cached(df_train, target_col=y_col, exclude_cols=exclude_cols)
    st.success(f"🎯 ROC-AUC: {auc:.3f}" if not math.isnan(auc) else "🎯 ROC-AUC hesaplanamadı")

    # Kullanıcı kombinasyonu için örnek satırlar
    q = df.copy()
    if cat_col and cat_sel != "(tümü)":
        q = q.loc[q[cat_col] == cat_sel]
    if hour_col:
        q = q.loc[q[hour_col] == hour_sel]
    if geoid_sel:
        q = q.loc[q[geoid_col].astype(str) == str(geoid_sel)]

    if len(q) == 0:
        st.warning("Bu seçimlerle satır bulunamadı. Filteleri genişletmeyi deneyin.")
    else:
        # Tahmin
        # Not: Eğitimde kullandığımız sayısal sütunları alalım; eksikler olursa dolduralım.
        Xq = q.reindex(columns=used_cols, fill_value=0)
        try:
            q_pred = model.predict_proba(Xq)[:, 1]
        except Exception:
            # Tür uyuşmazlıkları için emniyet
            Xq = Xq.apply(pd.to_numeric, errors="coerce").fillna(0)
            q_pred = model.predict_proba(Xq)[:, 1]

        q = q.iloc[:len(q_pred)].copy()
        q["risk_score"] = q_pred

        # GEOID bazında özet + metrik (varsa)
        geo_mean = float(q["risk_score"].mean())
        st.markdown("### 🔮 Tahmin Sonucu")
        cols = st.columns(3)
        cols[0].metric("Seçime göre ortalama risk", f"{geo_mean*100:.1f}%")
        if metrics_df is not None and geoid_col in metrics_df.columns:
            m_sub = metrics_df.loc[metrics_df[geoid_col].astype(str) == str(geoid_sel)]
            # Aday metrik isimleri
            met_name = _find_col(m_sub, ["f1_score","roc_auc","auc","accuracy"], default=None)
            if met_name and len(m_sub):
                mval = float(pd.to_numeric(m_sub[met_name], errors="coerce").dropna().mean())
                cols[1].metric(f"GEOID başarı ({met_name})", f"{mval:.3f}")
        cols[2].metric("Satır sayısı (tahmin)", f"{len(q):,}")

        with st.expander("🔎 Detaylı Tahmin Satırları"):
            st.dataframe(q[[geoid_col] + ([cat_col] if cat_col else []) + ([hour_col] if hour_col else []) + ["risk_score"]].head(1000))

        # Top-N riskli GEOID (seçime göre)
        topk = (q.groupby(geoid_col)["risk_score"].mean().sort_values(ascending=False).head(10))
        st.markdown("### 🚨 En Riskli 10 GEOID (mevcut filtrelerle)")
        st.dataframe(topk.reset_index(names=[geoid_col]).rename(columns={"risk_score":"avg_risk"}))

else:
    st.info("Sol panelden seçimleri yapıp **'🚀 Modeli Eğit ve Tahmin Yap'** butonuna basın.")

# -----------------------------------------------------------
# Bitiş
# -----------------------------------------------------------
st.caption("Not: Bu sayfa demo amaçlıdır. Özellik mühendisliği, sınıf dengesi ve zaman bağımlılığı üretim seviyesinde optimize edilmelidir.")

# app.py — SUTAM (revize tam sürüm)

from __future__ import annotations
import os
import traceback
import streamlit as st

# ── Proje bileşenleri
from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN
from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH
from components.gh_data import raw_url, download_actions_artifact_zip, unzip
from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH

# ── Sayfa ayarları
st.set_page_config(page_title=APP_NAME, layout="wide")
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption(f"Rol: {APP_ROLE}")

for name, stmt in [
    ("streamlit", "import streamlit as st"),
    ("pandas", "import pandas as pd"),
    ("numpy", "import numpy as np"),
    ("requests", "import requests"),
    ("folium", "import folium"),
    ("streamlit_folium", "from streamlit_folium import st_folium"),
    ("components.config", "from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH, GH_TOKEN"),
    ("components.gh_data", "from components.gh_data import raw_url, download_actions_artifact_zip, unzip"),
]:
    try:
        exec(stmt, {})
    except Exception as e:
        import streamlit as st
        st.error(f"Import hatası: {name} → {type(e).__name__}: {e}")
        st.stop()

# ── Yardımcı: GitHub token çözümleyici
def resolve_and_set_github_token() -> str | None:
    """
    Aşağıdaki öncelikle token'ı bulur:
      1) os.environ["GITHUB_TOKEN"]
      2) st.secrets["github_token"]
      3) st.secrets["GH_TOKEN"]
      4) st.secrets["GITHUB_TOKEN"]
    Bulursa os.environ["GITHUB_TOKEN"] içine yazar (modüller için tek doğruluk kaynağı).
    """
    # 1) Halihazırda env'de varsa aynen kullan
    env_tok = os.getenv("GITHUB_TOKEN")
    if env_tok:
        return env_tok

    # 2-4) Secrets içinden olası anahtar adlarıyla ara
    cand = None
    try:
        if "github_token" in st.secrets:
            cand = st.secrets["github_token"]
        elif "GH_TOKEN" in st.secrets:
            cand = st.secrets["GH_TOKEN"]
        elif "GITHUB_TOKEN" in st.secrets:
            cand = st.secrets["GITHUB_TOKEN"]
    except Exception:
        # Streamlit dışı koşullarda st.secrets erişimi hata verebilir; sessiz geç
        cand = None

    if cand:
        os.environ["GITHUB_TOKEN"] = str(cand)
        return str(cand)

    return None

# ── Başlık
st.title(APP_NAME)
st.caption(f"Rol: {APP_ROLE}")

# ── GitHub Token senkronu
token = resolve_and_set_github_token()

with st.expander("🔐 Bağlantı & Token Tanılama", expanded=token is None):
    st.write("**Veri deposu:**", DATA_REPO, "—", DATA_BRANCH)
    st.code(raw_url("sf-crime-parquet/risk_hourly.parquet"), language="text")

    cols = st.columns(3)
    cols[0].metric("Token bulundu mu?", "Evet" if bool(token) else "Hayır")
    # Güvenlik: token'ı asla düz metin olarak göstermiyoruz.
    cols[1].metric("Env'de GITHUB_TOKEN", "Evet" if bool(os.getenv("GITHUB_TOKEN")) else "Hayır")
    cols[2].metric("Secrets erişimi", "Evet" if "secrets" in dir(st) else "Bilinmiyor")

    if not token:
        st.warning(
            "GitHub token bulunamadı. `st.secrets['github_token']` **veya** "
            "`GITHUB_TOKEN` ortam değişkenini ayarlayın.\n\n"
            "Secrets örneği (`.streamlit/secrets.toml`):\n"
            '```toml\ngithub_token = "github_pat_xxx..."\n```'
        )
    else:
        st.success("Token başarıyla ayarlandı. (modüller `os.getenv('GITHUB_TOKEN')` üzerinden okuyabilir)")

# ── Yönlendirme / Pages kısa yolları
st.info("🗺️ Haritaya gitmek için aşağıdaki sayfayı açabilirsin.")
links = st.columns([1,1,2])
with links[0]:
    try:
        # Streamlit 1.30+ için
        st.page_link("pages/3_🔮_Suç_Tahmini.py",   label="🔮 Suç Tahmini", icon="🔮")
    except Exception:
        st.write("`pages/01_🧭_Suç_Tahmini.py` hazırsa soldaki **Pages** menüsünden ulaşabilirsin.")

with links[1]:
    try:
        st.page_link("pages/2_🗺️_Risk_Haritası.py", label="🗺️ Risk Haritası", icon="🗺️")
    except Exception:
        pass

# ── (Opsiyonel) GitHub Actions artifact indirme & açma
with st.expander("📦 Actions artifact indir (opsiyonel)"):
    artifact_name_input = st.text_input("Artifact adı", value="sf-crime-parquet")
    download_dir = st.text_input("İndirme klasörü", value="downloads")
    extract_dir  = st.text_input("Çıkarma klasörü", value="downloads/extracted")
    run_btn = st.button("Artifact'ı indir ve çıkar")

    if run_btn:
        try:
            if not os.getenv("GITHUB_TOKEN"):
                raise RuntimeError(
                    "Artifact indirilemedi/okunamadı: GitHub token yok. "
                    "st.secrets['github_token'] ekleyin veya GITHUB_TOKEN ortam değişkenini ayarlayın."
                )
            zip_path = download_actions_artifact_zip(artifact_name_input, download_dir)
            out_dir = unzip(zip_path, extract_dir)
            st.success(f"✅ İndirildi ve açıldı: {out_dir}")
        except Exception as e:
            st.error("❌ Artifact indirilemedi.")
            with st.expander("Hata ayrıntıları"):
                st.code("".join(traceback.format_exception_only(type(e), e)).strip())
            st.stop()

# ── Model / Versiyon rozeti
show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

# ── Bilgi kutusu (kısa açıklama)
st.markdown(
    """
**Notlar**
- Public raw okuma için yukarıdaki `risk_hourly.parquet` bağlantısı örneği gösterildi.
- Artifact indirmek için GitHub Actions çalıştırdığınız repoda üretilen artifact adını doğru giriniz.
- Token güvenlik nedeniyle asla ekranda gösterilmez; yalnızca var/yok statüsü paylaşılır.
"""
)

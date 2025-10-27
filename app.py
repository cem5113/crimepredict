# app.py — SUTAM (düzeltilmiş)

from __future__ import annotations

# 1) Streamlit'i import et ve İLK iş olarak sayfa config'ini ayarla
import streamlit as st
st.set_page_config(page_title="crimepredict", layout="wide", page_icon="🛰️")

# 2) Diğer importlar (import aşamasında st.* çağrısı OLMAMALI)
import os
import traceback

# 3) Config'i denerken UI çağrısı yapma — bayrakla sonra uyar
_config_missing = False
try:
    from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH
except Exception:
    _config_missing = True
    APP_NAME    = "crimepredict"
    APP_ROLE    = "Kullanıcı"
    DATA_REPO   = "cem5113/crime_prediction_data"
    DATA_BRANCH = "main"

# 4) Opsiyonel modüller — artık UI çağrısı yapılabilir (set_page_config sonrası)
def _try_import(name: str, default=None):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        st.info(f"Opsiyonel '{name}' modülü bulunamadı.")
        return default

_last_update = _try_import("components.last_update")
_meta        = _try_import("components.meta")
_gh          = _try_import("components.gh_data")

# 5) Güvenli erişimler (yoksa no-op / varsayılanlar)
show_last_update_badge          = getattr(_last_update, "show_last_update_badge", lambda *a, **k: None)
MODEL_VERSION: str              = getattr(_meta, "MODEL_VERSION", "v0")
MODEL_LAST_TRAIN: str           = getattr(_meta, "MODEL_LAST_TRAIN", "-")
raw_url                         = getattr(_gh, "raw_url", lambda *a, **k: "")
download_actions_artifact_zip   = getattr(_gh, "download_actions_artifact_zip", lambda *a, **k: ("", {}))
best_artifact_url               = getattr(_gh, "best_artifact_url", lambda *a, **k: ("", {}))

# 6) Yardımcı: unzip (eski kodda tanımlı değildi → NameError oluyordu)
def unzip(zip_path: str, out_dir: str) -> str:
    import zipfile
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir

# 7) Token çözümleyici
def resolve_and_set_github_token() -> str | None:
    env_tok = os.getenv("GITHUB_TOKEN")
    if env_tok:
        return env_tok
    cand = None
    try:
        if "github_token" in st.secrets:
            cand = st.secrets["github_token"]
        elif "GH_TOKEN" in st.secrets:
            cand = st.secrets["GH_TOKEN"]
        elif "GITHUB_TOKEN" in st.secrets:
            cand = st.secrets["GITHUB_TOKEN"]
    except Exception:
        cand = None
    if cand:
        os.environ["GITHUB_TOKEN"] = str(cand)
        return str(cand)
    return None

# 8) UI
st.title(APP_NAME)
st.caption(f"Rol: {APP_ROLE}")

if _config_missing:
    st.warning("Config modülü bulunamadı, varsayılan ayarlar kullanılıyor.")

token = resolve_and_set_github_token()

with st.expander("🔐 Bağlantı & Token Tanılama", expanded=token is None):
    st.write("**Veri deposu:**", DATA_REPO, "—", DATA_BRANCH)
    try:
        st.code(raw_url("sf-crime-parquet/risk_hourly.parquet"), language="text")
    except Exception:
        st.code("raw_url(...) kullanılamıyor", language="text")

    c1, c2, c3 = st.columns(3)
    c1.metric("Token bulundu mu?", "Evet" if bool(token) else "Hayır")
    c2.metric("Env'de GITHUB_TOKEN", "Evet" if bool(os.getenv("GITHUB_TOKEN")) else "Hayır")
    c3.metric("Secrets erişimi", "Evet" if "secrets" in dir(st) else "Bilinmiyor")

    if not token:
        st.warning(
            "GitHub token bulunamadı. `st.secrets['github_token']` **veya** "
            "`GITHUB_TOKEN` ortam değişkenini ayarlayın.\n\n"
            "Secrets örneği (`.streamlit/secrets.toml`):\n"
            '```toml\ngithub_token = "github_pat_xxx..."\n```'
        )
    else:
        st.success("Token ayarlandı. (Modüller `os.getenv('GITHUB_TOKEN')` üzerinden okuyabilir)")

st.info("🗺️ Haritaya gitmek için aşağıdaki sayfayı açabilirsin.")
col1, col2, _ = st.columns([1,1,2])
with col1:
    try:
        st.page_link("pages/3_🔮_Suç_Tahmini.py", label="🔮 Suç Tahmini", icon="🔮")
    except Exception:
        st.write("`Pages` menüsünden erişebilirsin.")
with col2:
    try:
        st.page_link("pages/2_🗺️_Risk_Haritası.py", label="🗺️ Risk Haritası", icon="🗺️")
    except Exception:
        pass

# ... önceki kod ...

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

            # 🔧 ÖNEMLİ: iki değeri ayır
            zip_path, _meta_info = download_actions_artifact_zip(artifact_name_input, download_dir)
            if not isinstance(zip_path, str) or not zip_path:
                raise RuntimeError(f"Geçersiz zip_path döndü: {zip_path!r}")

            out_dir = unzip(zip_path, extract_dir)
            st.success(f"✅ İndirildi ve açıldı: {out_dir}")
        except Exception as e:
            st.error("❌ Artifact indirilemedi.")
            with st.expander("Hata ayrıntıları"):
                st.code("".join(traceback.format_exception_only(type(e), e)).strip())
            st.stop()

show_last_update_badge(
    data_upto=None,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

st.markdown(
    """
**Notlar**
- Public raw okuma için yukarıdaki `risk_hourly.parquet` bağlantısı örneği gösterildi.
- Artifact indirmek için GitHub Actions çalıştırdığınız repoda üretilen artifact adını doğru giriniz.
- Token güvenlik nedeniyle asla ekranda gösterilmez; yalnızca var/yok statüsü paylaşılır.
"""
)

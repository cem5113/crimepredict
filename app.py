# app.py — SUTAM (revize tam sürüm)

from __future__ import annotations
import os
import traceback
import streamlit as st

# ── Proje bileşenleri
from components.config import APP_NAME, APP_ROLE, DATA_REPO, DATA_BRANCH
from components.last_update import show_last_update_badge
from components.meta import MODEL_VERSION, MODEL_LAST_TRAIN
from components.gh_data import raw_url, download_actions_artifact_zip, unzip

# ── Sayfa ayarları
st.set_page_config(page_title=APP_NAME, layout="wide")

# --- Veri Denetim: artifact/release içindeki 3 dosyayı bul ve önizle ---
import io, os, zipfile, posixpath, requests
from io import BytesIO
import pandas as pd
import streamlit as st

REPO_OWNER = "cem5113"
REPO_NAME  = "crime_prediction_data"
RELEASE_ASSET_ZIP = "fr-crime-outputs-parquet.zip"

# Aranan üyeler (ilk üç “hedef”; risk_hourly için csv fallback de listede)
TARGETS = [
    "artifact/risk_hourly.parquet",
    "artifact/risk_hourly.csv",              # fallback
    "artifact/metrics_stacking_ohe.parquet",
    "fr_crime_09.parquet",
]

def _resolve_and_export_token() -> str | None:
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        for k in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
            if k in st.secrets and st.secrets[k]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[k])
                return os.environ["GITHUB_TOKEN"]
    except Exception:
        pass
    return None

def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if os.getenv("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"
    return h

def _resolve_artifact_zip_url(owner: str, repo: str, name_contains: str):
    tok = _resolve_and_export_token()
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
    url, headers = _resolve_artifact_zip_url(REPO_OWNER, REPO_NAME, "fr-crime-outputs-parquet")
    if url:
        return url, headers
    # Release fallback
    rel = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/{RELEASE_ASSET_ZIP}"
    return rel, {}

def _read_table_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """ZIP/inner-ZIP içinde member_path'i CSV/Parquet olarak okur."""
    def _read(fp, name):
        return pd.read_csv(fp) if name.lower().endswith(".csv") else pd.read_parquet(fp)

    target_base = posixpath.basename(member_path)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()

        # 1) birebir
        if member_path in names:
            with z.open(member_path) as f:
                return _read(BytesIO(f.read()), member_path)

        # 1.b) sadece dosya adıyla
        cand = [n for n in names if n.endswith("/"+target_base) or n == target_base]
        if cand:
            with z.open(cand[0]) as f:
                return _read(BytesIO(f.read()), cand[0])

        # 2) iç ZIP'lerde ara
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

def check_three_sources():
    url, headers = _best_zip_url()
    r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    r.raise_for_status()
    content = r.content

    found = {}
    previews = {}

    for member in TARGETS:
        try:
            df = _read_table_from_zip_bytes(content, member)
            found[member] = True
            previews[member] = df.head(5)
        except Exception:
            found[member] = False

    return found, previews, url

with st.expander("🧪 Veri Denetim (artifact/release → 3 dosya kontrolü)", expanded=False):
    run = st.button("Dosyaları tara")
    if run:
        try:
            found, previews, url = check_three_sources()
            st.write("Kaynak ZIP:", url)
            cols = st.columns(2)
            for i, member in enumerate(TARGETS):
                ok = found.get(member, False)
                (cols[i % 2]).markdown(f"**{member}** → {'✅ bulundu' if ok else '❌ yok'}")
                if ok:
                    (cols[i % 2]).dataframe(previews[member], use_container_width=True, height=220)
            if not any(found.values()):
                st.error("Hiçbiri bulunamadı. Token/asset adı ya da release içeriğini kontrol et.")
        except Exception as e:
            st.error(f"Denetim hatası: {e}")

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

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

# ── EKLE: Gerekli dosyaları tanıla (Actions artifact → Release fallback)
import zipfile, posixpath
from io import BytesIO

REQUIRED = {
    "risk_hourly": ["artifact/risk_hourly.parquet", "artifact/risk_hourly.csv"],
    "metrics_base_ohe": ["artifact/metrics_base_ohe.parquet"],
    "fr_crime_09": ["fr_crime_09.parquet"],
}
RELEASE_ASSET_ZIP = "fr-crime-outputs-parquet.zip"  # fallback asset adı

def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h

def _actions_artifact_zip_url(owner: str, repo: str, name_contains: str):
    tok = os.getenv("GITHUB_TOKEN")
    if not tok:
        return None, {}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(f"{base}/actions/artifacts?per_page=100", headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    arts = (r.json() or {}).get("artifacts", []) or []
    arts = [a for a in arts if (name_contains in a.get("name", "")) and not a.get("expired")]
    if not arts:
        return None, {}
    arts.sort(key=lambda a: a.get("updated_at", ""), reverse=True)
    return f"{base}/actions/artifacts/{arts[0]['id']}/zip", _gh_headers()

def _best_zip_url(owner: str, repo: str):
    url, headers = _actions_artifact_zip_url(owner, repo, "fr-crime-outputs-parquet")
    if url:
        return url, headers
    # Release fallback
    rel = f"https://github.com/{owner}/{repo}/releases/latest/download/{RELEASE_ASSET_ZIP}"
    return rel, {}

def _read_bytes(url: str, headers: dict) -> bytes:
    r = requests.get(url, headers=headers or {}, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return r.content

def _search_in_zip_bytes(zip_bytes: bytes, wanted: list[str]) -> str | None:
    """
    ZIP ve varsa iç ZIP'lerde 'wanted' listesindeki ilk eşleşmeyi arar.
    Eşleşme bulunursa ZIP içindeki tam yolu döndürür.
    """
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()
        # 1) birebir ya da basename ile
        for m in wanted:
            base = posixpath.basename(m)
            # birebir
            if m in names:
                return m
            # basename ile
            for n in names:
                if n.endswith("/" + base) or n == base:
                    return n
        # 2) iç ZIP’leri tara
        for n in names:
            if not n.lower().endswith(".zip"):
                continue
            with z.open(n) as fz, zipfile.ZipFile(BytesIO(fz.read())) as z2:
                inner = z2.namelist()
                for m in wanted:
                    base = posixpath.basename(m)
                    if m in inner:
                        return f"{n} :: {m}"
                    for inn in inner:
                        if inn.endswith("/" + base) or inn == base:
                            return f"{n} :: {inn}"
    return None

with st.expander("📂 Veri tanılama: artifact/release ZIP içinde aranan dosyalar", expanded=True):
    try:
        url, headers = _best_zip_url(REPO_OWNER, REPO_NAME)
        src = "Actions artifact" if "api.github.com" in url else "Release asset"
        b = _read_bytes(url, headers)
        st.write(f"Kaynak: **{src}**")
        rows = []
        for key, candidates in REQUIRED.items():
            found = _search_in_zip_bytes(b, candidates)
            rows.append({
                "dosya_grubu": key,
                "aranan": ", ".join(candidates),
                "durum": "✅ Bulundu" if found else "❌ Yok",
                "zip_içi_yol": found or "—"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Kullanım ipuçları
        tips = []
        if rows[0]["durum"].startswith("✅"):
            # risk_hourly bulunduysa, kullanıcıya doğrudan spec örnekleri
            if "::" in rows[0]["zip_içi_yol"]:
                outer, inner = rows[0]["zip_içi_yol"].split("::", 1)
                inner = inner.strip()
            else:
                inner = rows[0]["zip_içi_yol"]
            tips.append(f"**Saatlik spec** (AUTO): `urlzip::AUTO::{inner}`")
        if rows[2]["durum"].startswith("✅"):
            p = rows[2]["zip_içi_yol"]
            if "::" in p:
                outer, inner = p.split("::", 1); inner = inner.strip()
            else:
                inner = p
            tips.append(f"**Günlük/özet spec** (AUTO): `urlzip::AUTO::{inner}`")
        if tips:
            st.info("Kullanım önerileri:\n\n- " + "\n- ".join(tips))
    except Exception as e:
        st.error(f"Tanılama çalıştırılamadı: {e}")

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

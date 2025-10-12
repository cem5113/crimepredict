# core/data_boot.py
import os
from pathlib import Path
import streamlit as st
from .artifacts import ensure_latest_zip

def configure_artifact_env() -> Path:
    """
    1) Eğer CRIME_ARTIFACT_ZIP env ile manuel yol gelmiş ve dosya varsa onu kullan.
    2) Aksi halde latest release'ten indirmeyi dene.
    3) İndirme başarısızsa: proje kökünde sf-crime-parquet.zip varsa ona düş.
       Hiçbiri yoksa anlaşılır hata mesajıyla kontrollü dur.
    """
    manual = os.getenv("CRIME_ARTIFACT_ZIP")
    if manual and Path(manual).exists():
        path = Path(manual).resolve()
        os.environ["CRIME_ARTIFACT_ZIP"] = str(path)
        st.sidebar.success(f"Veri (env): {path.name}")
        return path

    try:
        path = ensure_latest_zip()
        os.environ["CRIME_ARTIFACT_ZIP"] = str(path.resolve())
        st.sidebar.success(f"Veri indirildi: {path.name}")
        return path
    except Exception as e:
        st.sidebar.warning(f"Artefact indirilemedi: {e}")
        fallback = Path("sf-crime-parquet.zip")
        if fallback.exists():
            os.environ["CRIME_ARTIFACT_ZIP"] = str(fallback.resolve())
            st.sidebar.info("Yerel sf-crime-parquet.zip kullanılıyor.")
            return fallback

        # Kullanıcıya anlaşılır mesaj verip kontrollü durdur
        st.error(
            "Veri indirilemedi ve yerelde ZIP bulunamadı.\n\n"
            "Çözüm: (A) Actions'da 'latest' release oluştur veya (B) aynı klasöre sf-crime-parquet.zip koy "
            "veya (C) CRIME_ARTIFACT_ZIP ortam değişkeniyle tam yolu ver."
        )
        raise

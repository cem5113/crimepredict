# core/artifacts.py
from __future__ import annotations
import os, time, hashlib, requests
from pathlib import Path

# Release asset (public, tokensız)
RELEASE_URL = "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf-crime-parquet.zip"

# İndirilecek yerel yol (proje kökü)
LOCAL_ZIP = Path(os.getenv("CRIME_LOCAL_ZIP", "sf-crime-parquet.zip"))

# Kaç saniyede bir tazeleme? (günde 1 run varsa 3 saat yeterli)
REFRESH_SEC = int(os.getenv("CRIME_REFRESH_SEC", "10800"))  # 3 saat

def _etag_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".etag")

def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None

def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def ensure_latest_zip(url: str = RELEASE_URL, target: Path = LOCAL_ZIP) -> Path:
    """
    Yerelde zip yoksa indirir; varsa ETag ile koşullu GET yapar.
    Değişmemişse indirmez. Başarılıysa zip yolunu döndürür.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    etag_file = _etag_path(target)
    last_fetch = target.stat().st_mtime if target.exists() else 0
    if time.time() - last_fetch < REFRESH_SEC and target.exists():
        return target  # taze kabul et

    headers = {"Accept": "application/octet-stream"}
    etag = _read_text(etag_file)
    if etag:
        headers["If-None-Match"] = etag

    r = requests.get(url, headers=headers, allow_redirects=True, timeout=60)
    if r.status_code == 304 and target.exists():
        # Değişmemiş
        return target
    r.raise_for_status()

    # ETag’i sakla
    if "ETag" in r.headers:
        _write_text(etag_file, r.headers["ETag"])

    # Atomik yazım
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(target)
    return target

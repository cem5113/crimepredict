# core/artifacts.py
from __future__ import annotations
import os, time, requests
from pathlib import Path

RELEASE_URL = "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf-crime-parquet.zip"
LOCAL_ZIP = Path(os.getenv("CRIME_LOCAL_ZIP", "sf-crime-parquet.zip"))
REFRESH_SEC = int(os.getenv("CRIME_REFRESH_SEC", "10800"))  # 3 saat

class ArtifactDownloadError(RuntimeError):
    pass

def _etag_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".etag")

def _read_text(p: Path) -> str | None:
    try: return p.read_text(encoding="utf-8").strip()
    except Exception: return None

def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def ensure_latest_zip(url: str = RELEASE_URL, target: Path = LOCAL_ZIP) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (time.time() - target.stat().st_mtime) < REFRESH_SEC:
        return target  # taze

    headers = {"Accept": "application/octet-stream"}
    etag_file = _etag_path(target)
    etag = _read_text(etag_file)
    if etag:
        headers["If-None-Match"] = etag

    r = requests.get(url, headers=headers, allow_redirects=True, timeout=60)
    if r.status_code == 304 and target.exists():
        return target
    if r.status_code == 404:
        raise ArtifactDownloadError(
            "Latest release bulunamadı (404). Releases sekmesinde 'latest' tag'li "
            "bir yayın ve içinde sf-crime-parquet.zip olduğundan emin olun."
        )
    if r.status_code in (401, 403):
        raise ArtifactDownloadError(
            f"Erişim hatası ({r.status_code}). Repo private ise public yapın ya da "
            "token ile erişilebilen bir URL kullanın."
        )
    r.raise_for_status()

    if "ETag" in r.headers:
        _write_text(etag_file, r.headers["ETag"])

    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(target)
    return target

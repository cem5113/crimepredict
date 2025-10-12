from __future__ import annotations
import os, time, requests
from pathlib import Path
import streamlit as st

# ---- Release asset (tokensız) fallback ----
RELEASE_URL = "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf-crime-parquet.zip"
LOCAL_ZIP = Path(os.getenv("CRIME_LOCAL_ZIP", "sf-crime-parquet.zip"))
REFRESH_SEC = int(os.getenv("CRIME_REFRESH_SEC", "10800"))  # 3 saat
API_ROOT = "https://api.github.com"

class ArtifactDownloadError(RuntimeError):
    pass

def _etag_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".etag")

def _read_text(p: Path) -> str | None:
    try: return p.read_text(encoding="utf-8").strip()
    except Exception: return None

def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def _is_fresh(p: Path) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < REFRESH_SEC

# ---- helpers: secrets/headers ----
def _to_bool(x) -> bool:
    if isinstance(x, bool): return x
    if x is None: return False
    if isinstance(x, str): return x.strip().lower() in ("1","true","yes","on")
    return False

def _gh_token() -> str | None:
    # sırayla dene: secrets.GH_TOKEN, secrets.GITHUB_TOKEN, env
    return (
        (st.secrets.get("GH_TOKEN") if "GH_TOKEN" in st.secrets else None)
        or (st.secrets.get("GITHUB_TOKEN") if "GITHUB_TOKEN" in st.secrets else None)
        or os.getenv("GH_TOKEN")
        or os.getenv("GITHUB_TOKEN")
    )

def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

# ---------- OPTION A: GitHub Releases (no token) ----------
def _download_from_release(url: str, target: Path) -> Path:
    headers = {"Accept": "application/octet-stream"}
    etag_file = _etag_path(target)
    etag = _read_text(etag_file)
    if etag: headers["If-None-Match"] = etag

    r = requests.get(url, headers=headers, allow_redirects=True, timeout=60)
    if r.status_code == 304 and target.exists():
        return target
    if r.status_code == 404:
        raise ArtifactDownloadError(
            "Latest release bulunamadı (404). Releases > 'latest' altında sf-crime-parquet.zip yayınlayın "
            "ya da USE_ARTIFACT=true + GH_TOKEN/GITHUB_TOKEN ile Actions artifact indirmeyi kullanın."
        )
    if r.status_code in (401, 403):
        raise ArtifactDownloadError(f"Release erişim hatası ({r.status_code}).")
    r.raise_for_status()

    if "ETag" in r.headers: _write_text(etag_file, r.headers["ETag"])
    tmp = target.with_suffix(".tmp"); tmp.write_bytes(r.content); tmp.replace(target)
    return target

# ---------- OPTION B: GitHub Actions artifact (needs token) ----------
def _download_from_actions(repo: str, artifact_name: str, token: str, target: Path) -> Path:
    """
    Repo genelindeki artefact listesinde (son 100) adı eşleşen EN YENİ ve expired olmayan artefact'ı bulur,
    archive_download_url'den ZIP'i indirir ve target'a yazar.
    """
    # 1) artefact listesi (repo genelinde)
    arts = requests.get(
        f"{API_ROOT}/repos/{repo}/actions/artifacts",
        params={"per_page": 100},
        headers=_gh_headers(token), timeout=60
    )
    arts.raise_for_status()
    items = arts.json().get("artifacts", []) or []
    # isme göre eşleşenleri filtrele, created_at'e göre sırala (yeni -> eski)
    matches = [a for a in items if a.get("name") == artifact_name and not a.get("expired")]
    if not matches:
        raise ArtifactDownloadError(f"Actions: '{artifact_name}' adlı artefact bulunamadı (ya da süresi dolmuş).")
    matches.sort(key=lambda a: a.get("created_at",""), reverse=True)
    chosen = matches[0]

    # 2) indir
    dl = requests.get(chosen["archive_download_url"], headers=_gh_headers(token), timeout=120)
    if dl.status_code in (401,403):
        raise ArtifactDownloadError("Actions: Token yetkisi yetersiz (401/403). PAT'e 'actions:read' izni verin.")
    dl.raise_for_status()

    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(dl.content)
    tmp.replace(target)
    return target

def ensure_latest_zip(url: str = RELEASE_URL, target: Path = LOCAL_ZIP) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_fresh(target):
        return target

    # secrets oku
    use_artifact = _to_bool(st.secrets.get("USE_ARTIFACT")) if hasattr(st, "secrets") else False
    repo  = st.secrets.get("REPO") if hasattr(st, "secrets") else None
    artifact_name = st.secrets.get("ARTIFACT_NAME") if hasattr(st, "secrets") else None
    token = _gh_token()

    if use_artifact:
        if not repo or not artifact_name or not token:
            raise ArtifactDownloadError("USE_ARTIFACT=true fakat REPO / ARTIFACT_NAME / GH_TOKEN(GITHUB_TOKEN) eksik.")
        return _download_from_actions(repo=repo, artifact_name=artifact_name, token=token, target=target)

    # Fallback: releases (tokensız)
    return _download_from_release(url, target)

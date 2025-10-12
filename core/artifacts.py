# core/artifacts.py
from __future__ import annotations
import os, time, requests, zipfile, io
from pathlib import Path
import streamlit as st

# ---- Defaults for release-asset (tokensız) ----
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

def _is_fresh(p: Path) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < REFRESH_SEC

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
            "ya da USE_ARTIFACT=true + GH_TOKEN ile Actions artifact indirmeyi kullanın."
        )
    if r.status_code in (401, 403):
        raise ArtifactDownloadError(f"Release erişim hatası ({r.status_code}).")
    r.raise_for_status()

    if "ETag" in r.headers: _write_text(etag_file, r.headers["ETag"])
    tmp = target.with_suffix(".tmp"); tmp.write_bytes(r.content); tmp.replace(target)
    return target

# ---------- OPTION B: GitHub Actions artifact (needs token) ----------
API_ROOT = "https://api.github.com"

def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def _download_from_actions(repo: str, branch: str, artifact_name: str, token: str, target: Path) -> Path:
    """
    1) Branch'taki son successful workflow run'ı bul
    2) O run'ın artifacts listesinden artifact_name'i yakala
    3) download_url'den ZIP'i indir → içindeki tek/çok dosyayı sf-crime-parquet.zip olarak kaydet
    """
    # 1) latest successful run id
    runs = requests.get(
        f"{API_ROOT}/repos/{repo}/actions/runs",
        params={"branch": branch, "status": "success", "per_page": 10},
        headers=_gh_headers(token), timeout=60
    )
    runs.raise_for_status()
    runs_json = runs.json().get("workflow_runs", [])
    if not runs_json:
        raise ArtifactDownloadError("Actions: başarılı run bulunamadı.")

    run_id = runs_json[0]["id"]

    # 2) artifacts of run
    arts = requests.get(
        f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/artifacts",
        headers=_gh_headers(token), timeout=60
    )
    arts.raise_for_status()
    items = arts.json().get("artifacts", [])
    match = next((a for a in items if a["name"] == artifact_name and not a["expired"]), None)
    if not match:
        raise ArtifactDownloadError(f"Actions: '{artifact_name}' adlı artifact bulunamadı.")

    # 3) download and write
    dl = requests.get(match["archive_download_url"], headers=_gh_headers(token), timeout=120)
    dl.raise_for_status()

    # Artifact download endpoint ZIP döndürür (içinde senin dosyaların)
    # Biz komple ZIP'i tek dosya olarak saklıyoruz (uygulama zaten zip bekliyor)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(dl.content)
    tmp.replace(target)
    return target

def ensure_latest_zip(url: str = RELEASE_URL, target: Path = LOCAL_ZIP) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_fresh(target):
        return target

    use_artifact = False
    repo = branch = name = token = None
    try:
        use_artifact = bool(st.secrets.get("USE_ARTIFACT", False))
        repo  = st.secrets.get("REPO")
        branch = st.secrets.get("BRANCH", "main")
        name  = st.secrets.get("ARTIFACT_NAME", "sf-crime-parquet")
        token = st.secrets.get("GH_TOKEN")
    except Exception:
        pass

    if use_artifact:
        if not (repo and name and token):
            raise ArtifactDownloadError("USE_ARTIFACT=true fakat REPO / ARTIFACT_NAME / GH_TOKEN eksik.")
        return _download_from_actions(repo, branch, name, token, target)

    # fallback: releases (no token)
    return _download_from_release(url, target)

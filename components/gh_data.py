# components/gh_data.py
from __future__ import annotations
import io, os, zipfile, json, time
from pathlib import Path
from typing import Optional
from components.utils.config import DATA_REPO, DATA_BRANCH, RAW_BASE
import os
import requests
import streamlit as st

# ---- Basit RAW okuma (public dosyalar) ----
def raw_url(path: str) -> str:
    path = path.lstrip("/")
    return f"{RAW_BASE}/{path}"

# ---- GitHub REST helper ----
def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN") or st.secrets.get("GH_TOKEN", None)
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h

# ---- Releases -> asset indirme (isme göre, en yeni) ----
def download_release_asset_by_name(asset_name: str, out_dir: str | Path) -> Path:
    """
    En yeni release içinden ismi asset_name olan dosyayı indirir.
    Private repo ya da rate-limit için GH_TOKEN önerilir.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # latest release
    owner_repo = DATA_REPO
    url = f"https://api.github.com/repos/{owner_repo}/releases/latest"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    rel = r.json()
    assets = rel.get("assets", [])
    match = next((a for a in assets if a["name"] == asset_name), None)
    if not match:
        raise FileNotFoundError(f"Release'ta '{asset_name}' bulunamadı.")
    dl_url = match["browser_download_url"]
    bin = requests.get(dl_url, headers=_gh_headers(), timeout=600)
    bin.raise_for_status()
    out_path = out_dir / asset_name
    out_path.write_bytes(bin.content)
    return out_path

# ---- GitHub Actions artifacts (en güncel, isme göre) ----
def download_actions_artifact_zip(artifact_name: str, out_dir: str | Path) -> Path:
    """
    Repo'daki son başarılı workflow artifact'ini indirir (zip).
    GH_TOKEN gerekir (Actions artifact indirme yetkisi).
    """
    if not GH_TOKEN:
        raise RuntimeError("GH_TOKEN gerekiyor (actions artifact indirmek için).")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://api.github.com/repos/{DATA_REPO}"
    # listele (en yeniler üstte)
    r = requests.get(f"{base}/actions/artifacts?per_page=100", headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    item = next((a for a in items if a["name"] == artifact_name and not a["expired"]), None)
    if not item:
        raise FileNotFoundError(f"Artifact '{artifact_name}' bulunamadı ya da süresi dolmuş.")
    # download url
    dl = requests.get(f"{base}/actions/artifacts/{item['id']}/zip", headers=_gh_headers(), timeout=600)
    dl.raise_for_status()
    out_path = out_dir / f"{artifact_name}.zip"
    out_path.write_bytes(dl.content)
    return out_path

# ---- Zip çıkar ----
def unzip(zip_path: str | Path, to_dir: str | Path) -> Path:
    to = Path(to_dir); to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(to)
    return to

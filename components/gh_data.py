# components/gh_data.py
from __future__ import annotations
import io, os, zipfile, json, time
from pathlib import Path
from typing import Optional

from components.config import DATA_REPO, DATA_BRANCH, RAW_BASE, GH_TOKEN
import requests
import streamlit as st

def raw_url(path: str) -> str:
    path = path.lstrip("/")
    return f"{RAW_BASE}/{path}"

def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN") or GH_TOKEN
    if not tok:
        try:
            tok = st.secrets.get("GH_TOKEN", None)
        except Exception:
            tok = None
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h

def download_release_asset_by_name(asset_name: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://api.github.com/repos/{DATA_REPO}/releases/latest"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    assets = r.json().get("assets", [])
    match = next((a for a in assets if a["name"] == asset_name), None)
    if not match:
        raise FileNotFoundError(f"Release'ta '{asset_name}' bulunamadı.")
    dl_url = match["browser_download_url"]
    bin = requests.get(dl_url, headers=_gh_headers(), timeout=600)
    bin.raise_for_status()
    out_path = Path(out_dir) / asset_name
    out_path.write_bytes(bin.content)
    return out_path

def download_actions_artifact_zip(artifact_name: str, out_dir: str | Path) -> Path:
    has_token = os.getenv("GITHUB_TOKEN") or GH_TOKEN
    if not has_token:
        try:
            has_token = st.secrets.get("GH_TOKEN", None)
        except Exception:
            has_token = None
    if not has_token:
        raise RuntimeError("GH_TOKEN gerekiyor (actions artifact indirmek için).")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://api.github.com/repos/{DATA_REPO}"
    r = requests.get(f"{base}/actions/artifacts?per_page=100", headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    items = r.json().get("artifacts", [])
    item = next((a for a in items if a["name"] == artifact_name and not a["expired"]), None)
    if not item:
        raise FileNotFoundError(f"Artifact '{artifact_name}' bulunamadı ya da süresi dolmuş.")
    dl = requests.get(f"{base}/actions/artifacts/{item['id']}/zip", headers=_gh_headers(), timeout=600)
    dl.raise_for_status()
    out_path = Path(out_dir) / f"{artifact_name}.zip"
    out_path.write_bytes(dl.content)
    return out_path

def unzip(zip_path: str | Path, to_dir: str | Path) -> Path:
    to = Path(to_dir); to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(to)
    return to

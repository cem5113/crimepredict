check_data_sources.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, posixpath, zipfile, requests
from io import BytesIO
import pandas as pd

REPO_OWNER = "cem5113"
REPO_NAME  = "crime_prediction_data"
RELEASE_ASSET_ZIP = "fr-crime-outputs-parquet.zip"

TARGETS = [
    "artifact/risk_hourly.parquet",
    "artifact/risk_hourly.csv",
    "artifact/metrics_stacking_ohe.parquet",
    "fr_crime_09.parquet",
]

def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("github_token")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h

def _artifact_url():
    base = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
    r = requests.get(f"{base}/actions/artifacts?per_page=100", headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    arts = (r.json() or {}).get("artifacts", []) or []
    arts = [a for a in arts if ("fr-crime-outputs-parquet" in a.get("name","")) and not a.get("expired")]
    if not arts:
        return None, {}
    arts.sort(key=lambda a: a.get("updated_at",""), reverse=True)
    return f"{base}/actions/artifacts/{arts[0]['id']}/zip", _gh_headers()

def _best_zip_url():
    u, h = _artifact_url()
    if u:
        return u, h
    return f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest/download/{RELEASE_ASSET_ZIP}", {}

def _read_table_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    def _read(fp, name): return pd.read_csv(fp) if name.lower().endswith(".csv") else pd.read_parquet(fp)
    target_base = posixpath.basename(member_path)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        names = z.namelist()
        if member_path in names:
            with z.open(member_path) as f: return _read(BytesIO(f.read()), member_path)
        cand = [n for n in names if n.endswith("/"+target_base) or n == target_base]
        if cand:
            with z.open(cand[0]) as f: return _read(BytesIO(f.read()), cand[0])
        for n in names:
            if not n.lower().endswith(".zip"): continue
            with z.open(n) as fz, zipfile.ZipFile(BytesIO(fz.read())) as z2:
                inner = z2.namelist()
                if member_path in inner:
                    with z2.open(member_path) as f2: return _read(BytesIO(f2.read()), member_path)
                cand2 = [m for m in inner if m.endswith("/"+target_base) or m == target_base]
                if cand2:
                    with z2.open(cand2[0]) as f2: return _read(BytesIO(f2.read()), cand2[0])
    raise FileNotFoundError(member_path)

def main():
    url, headers = _best_zip_url()
    print("Kaynak:", url)
    r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    r.raise_for_status()
    blob = r.content

    any_found = False
    for member in TARGETS:
        try:
            df = _read_table_from_zip_bytes(blob, member)
            any_found = True
            print(f"[OK] {member}  (satır={len(df):,}, kolonlar={list(df.columns)[:6]}...)")
            print(df.head(3))
        except Exception as e:
            print(f"[NO] {member}  -> {e}")

    if not any_found:
        raise SystemExit("Hiçbir hedef dosya bulunamadı.")

if __name__ == "__main__":
    main()

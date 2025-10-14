# crimepredict/dataio/loaders.py
from __future__ import annotations

# --- stdlib ---
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib

# --- third-party ---
import pandas as pd
import requests

# ===================== paths & settings (robust) =====================

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[1]            # repo-root/crimepredict
_REPO_ROOT = _PKG_ROOT.parent           # repo-root

# settings modülünü bul: (1) crimepredict.config.settings → (2) config.settings → (3) yoksa None
_settings = None
for mod in ("crimepredict.config.settings", "config.settings"):
    try:
        _settings = importlib.import_module(mod)
        break
    except Exception:
        _settings = None

# ENV → settings → defaults
DATA_DIR = Path(
    os.getenv("CRIME_DATA_DIR")
    or (getattr(_settings, "DATA_DIR", None) if _settings else None)
    or (_REPO_ROOT / "data")
).resolve()

RESULTS_DIR = Path(
    os.getenv("CRIME_RESULTS_DIR")
    or (getattr(_settings, "RESULTS_DIR", None) if _settings else None)
    or (_REPO_ROOT / "results")
).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ===================== env & constants =====================

GITHUB_REPO          = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")
GITHUB_ARTIFACT_NAME = os.getenv("GITHUB_ARTIFACT_NAME", "sutam-results")
GH_TOKEN             = os.getenv("GH_TOKEN", "")

CRIME_CSV_URL = os.getenv(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv",
)

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

REQUIRED_COLS: List[str] = ["GEOID", "date", "event_hour"]

# ===================== helpers =====================

def _headers(require_auth: bool = False) -> Optional[Dict[str, str]]:
    base = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        base["Authorization"] = f"Bearer {GH_TOKEN}"
    if require_auth and not GH_TOKEN:
        return None
    return base

def _read_anyframe_from_bytes(blob: bytes) -> pd.DataFrame:
    # 1) CSV dene
    try:
        return pd.read_csv(io.BytesIO(blob), low_memory=False)
    except Exception:
        pass
    # 2) Parquet dene
    try:
        return pd.read_parquet(io.BytesIO(blob))  # type: ignore
    except Exception:
        pass
    # 3) Disk üstünden son çare
    tmp = DATA_DIR / "_artifact_tmp_blob"
    tmp.write_bytes(blob)
    try:
        try:
            return pd.read_parquet(tmp)  # type: ignore
        except Exception:
            return pd.read_csv(tmp, low_memory=False)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

def _artifact_bytes(picks: List[str], artifact_name: Optional[str] = None) -> Optional[bytes]:
    """
    GitHub Actions artifact’larından ilk eşleşeni bytes döndürür.
    GH_TOKEN yoksa None döner.
    'picks' içindeki dosya isimlerini tam ya da sonda eşleşerek arar.
    """
    headers = _headers(require_auth=True)
    if headers is None:
        return None

    artifact_name = artifact_name or GITHUB_ARTIFACT_NAME
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"

    try:
        runs = requests.get(runs_url, headers=headers, timeout=30).json()
        run_ids = [
            r["id"]
            for r in runs.get("workflow_runs", [])
            if r.get("conclusion") == "success"
        ]
    except Exception:
        return None

    for rid in run_ids:
        try:
            arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
            arts = requests.get(arts_url, headers=headers, timeout=30).json().get("artifacts", [])
        except Exception:
            continue

        ordered = (
            [a for a in arts if a.get("name") == artifact_name and not a.get("expired", False)]
            or [a for a in arts if not a.get("expired", False)]
        )

        for a in ordered:
            try:
                z_content = requests.get(a["archive_download_url"], headers=headers, timeout=60).content
                zf = zipfile.ZipFile(io.BytesIO(z_content))
                names = zf.namelist()

                # Doğrudan aday yollar
                for p in picks:
                    for cand in (p, f"results/{p}", f"out/{p}", f"crime_prediction_data/{p}"):
                        if cand in names:
                            return zf.read(cand)

                # Sonda eşleşme
                for n in names:
                    if any(n.endswith(p) for p in picks):
                        return zf.read(n)
            except Exception:
                continue
    return None

def _normalize_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L)

def _to_naive_datetime(s: pd.Series) -> pd.Series:
    """Seri’yi datetime’a çevirir; tz varsa tz’siz (naive) yapar."""
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    # tz-aware ise tz_convert(None) çalışır; değilse dokunma
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        # Bazı durumlarda tz_localize(None) gerekebilir
        try:
            dt = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)
        except Exception:
            pass
    return dt

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = _to_naive_datetime(out["date"])
    elif "datetime" in out.columns:
        out["date"] = _to_naive_datetime(out["datetime"])
    else:
        out["date"] = pd.NaT

    if "event_hour" not in out.columns:
        hours = pd.to_datetime(out["date"], errors="coerce").dt.hour
        out["event_hour"] = hours.fillna(0).astype(int)
    return out

def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # En yaygın isimlerden türet
    lat_cands = [c for c in out.columns if c.lower() in ["lat", "latitude", "y", "lat_dd"]]
    lon_cands = [c for c in out.columns if c.lower() in ["lon", "longitude", "x", "lng", "long", "lon_dd"]]
    if lat_cands and "lat" not in out.columns:
        out["lat"] = pd.to_numeric(out[lat_cands[0]], errors="coerce")
    if lon_cands and "lon" not in out.columns:
        out["lon"] = pd.to_numeric(out[lon_cands[0]], errors="coerce")
    return out

def _parse_and_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_latlon(_ensure_time_cols(df))
    if "GEOID" in df.columns:
        try:
            df["GEOID"] = _normalize_geoid(df["GEOID"])
        except Exception:
            df["GEOID"] = df["GEOID"].astype(str)
    return df

def _validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return len(missing) == 0, missing

def _cache_latest(df: pd.DataFrame) -> None:
    try:
        (DATA_DIR / "sf_crime_latest.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    except Exception:
        pass

# ===================== metadata =====================

def load_metadata() -> Dict[str, Any]:
    """
    Öncelik: results/metadata.json → artifact → {}
    """
    p = RESULTS_DIR / "metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        blob = _artifact_bytes(
            picks=["metadata.json", "results/metadata.json", "out/metadata.json"],
            artifact_name=GITHUB_ARTIFACT_NAME,
        )
        if blob:
            return json.loads(blob.decode("utf-8"))
    except Exception:
        pass
    return {}

def load_metadata_or_default() -> Dict[str, Any]:
    m = load_metadata()
    if m:
        return m
    return {
        "source": "unknown",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "rows": 0,
        "cols": 0,
        "columns": [],
        "date_min": None,
        "date_max": None,
        "has_latlon": False,
    }

# ===================== public API =====================

def load_sf_crime_latest() -> Tuple[pd.DataFrame, str]:
    """
    Kaynak sırası:
      1) GitHub Actions artifact (GH_TOKEN gerekli)
      2) Release (latest): sf_crime.csv (CRIME_CSV_URL)
      3) RESULTS_DIR: sf_crime_latest.parquet|csv
      4) DATA_DIR lokal cache
      5) boş DataFrame

    Dönüş: (df, src_tag)  → src_tag ∈ {"artifact","release","results","local:<ad>","empty"}
    """
    # --- 1) Artifact ---
    try:
        picks = [
            "sf_crime_latest.parquet",
            "sf_crime_latest.csv",
            "sf_crime.csv",
            "sf_crime_09.csv",
            "metrics_all.csv",
        ]
        blob = _artifact_bytes(picks=picks, artifact_name=GITHUB_ARTIFACT_NAME)
        if blob:
            df = _read_anyframe_from_bytes(blob)
            df = _parse_and_cleanup(df)
            _cache_latest(df)
            return df, "artifact"
    except Exception as e:
        print("artifact erişimi başarısız:", e)

    # --- 2) Release (latest) ---
    try:
        r = requests.get(CRIME_CSV_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        df = _parse_and_cleanup(df)
        (DATA_DIR / "sf_crime_release.csv").write_bytes(r.content)
        _cache_latest(df)
        return df, "release"
    except Exception as e:
        print("release fallback başarısız:", e)

    # --- 3) RESULTS_DIR ---
    for cand, tag in [
        (RESULTS_DIR / "sf_crime_latest.parquet", "results"),
        (RESULTS_DIR / "sf_crime_latest.csv", "results"),
    ]:
        if cand.exists():
            try:
                if cand.suffix.lower() == ".parquet":
                    df = pd.read_parquet(cand)  # type: ignore
                else:
                    df = pd.read_csv(cand, low_memory=False)
                df = _parse_and_cleanup(df)
                _cache_latest(df)
                return df, tag
            except Exception as e:
                print("RESULTS okumada hata:", e)

    # --- 4) DATA_DIR cache ---
    for name in ["sf_crime_latest.csv", "sf_crime_artifact_cache.csv", "sf_crime_09.csv", "metrics_all.csv", "sf_crime.csv"]:
        p = DATA_DIR / name
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                df = _parse_and_cleanup(df)
                _cache_latest(df)
                return df, f"local:{name}"
            except Exception:
                continue

    # --- 5) boş ---
    df = pd.DataFrame({"GEOID": [], "date": [], "event_hour": [], "crime_count": [], "lat": [], "lon": []})
    return df, "empty"

__all__ = [
    "load_sf_crime_latest",
    "load_metadata",
    "load_metadata_or_default",
    "_validate_schema",
]

# core/data.py
from __future__ import annotations
import zipfile, tempfile, pathlib, hashlib
from typing import Iterable, Optional, Dict, List
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import streamlit as st
from config.settings import ARTIFACT_ZIP, DEFAULT_PARQUET_MEMBER

LEVEL_MAP = {
    "very high": 3, "vh": 3,
    "high": 2, "h": 2,
    "medium": 1, "med": 1, "m": 1,
    "low": 0, "l": 0,
}

def _coerce_risk_level(s: pd.Series) -> pd.Series:
    """risk_level'i güvenle numerik'e çevirir; metinleri LEVEL_MAP ile mapler."""
    if s.dtype != "O" and pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s_str = s.astype(str).str.strip().str.lower()
    num = pd.to_numeric(s_str, errors="coerce")
    if num.notna().any():
        return num
    return s_str.map(LEVEL_MAP).astype("float32")


# UI tarafında beklediğimiz ana sütunlar
REQUIRED_COLS = ["geoid","lat","lon","timestamp","risk_score","risk_level","pred_expected"]

def _md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def list_members(prefix: str = "", suffix: str = ".parquet") -> list[str]:
    if not ARTIFACT_ZIP.exists():
        return []
    with zipfile.ZipFile(ARTIFACT_ZIP) as zf:
        return sorted([n for n in zf.namelist() if n.endswith(suffix) and n.startswith(prefix)])

def _extract_member_to_tmp(member_name: str) -> pathlib.Path:
    with zipfile.ZipFile(ARTIFACT_ZIP) as zf, zf.open(member_name) as fh, \
         tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp.write(fh.read())
        return pathlib.Path(tmp.name)

def _read_schema(member_name: str) -> List[str]:
    tmp_path = _extract_member_to_tmp(member_name)
    try:
        md = pq.read_metadata(tmp_path)
        return [md.schema.column(i).name for i in range(md.num_columns)]
    except Exception:
        # fallback: dataset schema
        try:
            return ds.dataset(tmp_path, format="parquet").schema.names
        finally:
            pass
    finally:
        try: tmp_path.unlink()
        except Exception: pass

# Kolon eşleme tablosu: bulamazsa sırayla dener
COL_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "ts", "time", "datetime", "event_ts"],
    "risk_score": ["risk_score", "risk", "score", "risk_prob", "prob", "riskvalue"],
    "risk_level": ["risk_level", "level", "risk_bin", "riskclass"],
    "pred_expected": ["pred_expected", "expected_events", "expected", "lambda", "intensity"],
    "geoid": ["geoid", "GEOID", "geo_id", "tract_id", "grid_id", "cell_id"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "lng", "longitude", "x"],
    "neighborhood": ["neighborhood", "nbhd", "district", "area_name"],
}

def _resolve_columns(available: List[str], wanted: Iterable[str]) -> Dict[str, Optional[str]]:
    avail_lower = {c.lower(): c for c in available}
    mapping: Dict[str, Optional[str]] = {}
    for target in wanted:
        chosen = None
        for cand in COL_ALIASES.get(target, [target]):
            if cand.lower() in avail_lower:
                chosen = avail_lower[cand.lower()]
                break
        mapping[target] = chosen  # yoksa None kalır
    return mapping

@st.cache_data(show_spinner=False)
def load_parquet(member_name: str,
                 columns: Optional[Iterable[str]] = None,
                 row_filter: Optional[ds.Expression] = None) -> pd.DataFrame:
    """
    ZIP içindeki tek bir parquet dosyasını okur (sütun seçimi + predicate pushdown).
    Şema güvenli: istenen kolonlar dosyada yoksa mevcut olanlarla kesişimi alır.
    """
    zip_md5 = _md5(ARTIFACT_ZIP)
    _ = (zip_md5, member_name, tuple(columns or []), str(row_filter))  # cache key için

    # Şema oku ve kolonları eşle
    available = _read_schema(member_name)
    if not available:
        return pd.DataFrame()

    if columns:
        mapping = _resolve_columns(available, columns)
        chosen_cols = [c for c in mapping.values() if c is not None]
    else:
        chosen_cols = available

    tmp_path = _extract_member_to_tmp(member_name)
    try:
        if row_filter is None and chosen_cols:
            table = pq.read_table(tmp_path, columns=chosen_cols)
        else:
            dataset = ds.dataset(tmp_path, format="parquet")
            table = dataset.to_table(filter=row_filter, columns=chosen_cols if chosen_cols else None)
        df = table.to_pandas()
        # Eşlenmiş adları UI'nin beklediği adlara normalleştir
        if columns:
            mapping = _resolve_columns(available, columns)
            rename_map = {v: k for k, v in mapping.items() if v and k != v}
            if rename_map:
                df = df.rename(columns=rename_map)
        return df
    finally:
        try: tmp_path.unlink()
        except Exception: pass

def _pick_best_member() -> Optional[str]:
    members = list_members()
    for cand in (DEFAULT_PARQUET_MEMBER, "risk_hourly.parquet", "sf_crime_09.parquet", "sf_crime_08.parquet", "sf_crime_01.parquet"):
        if cand in members:
            return cand
    return members[0] if members else None

def get_latest_kpis() -> dict:
    """
    Esnek KPI: Kolon isimleri farklıysa da hesaplar.
    - last_update: timestamp/ts/datetime'dan
    - avg_risk: risk_score/risk/score/prob'dan
    - high_rate: risk_level yoksa risk_score > eşik (0.7) varsayılanı ile
    """
    member = _pick_best_member()
    if not member:
        return {"rows": 0, "last_update": "—", "avg_risk": 0.0, "high_rate": 0.0, "member": "N/A"}

    # Şema esnekliği için tam listeyi isteyelim; sonradan seçeriz
    df = load_parquet(member, columns=[
        "timestamp","risk_score","risk_level","pred_expected","geoid","lat","lon"
    ])

    if df.empty:
        return {"rows": 0, "last_update": "—", "avg_risk": 0.0, "high_rate": 0.0, "member": member}

    # Zaman damgasını normalize et (saniye/ms)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        if ts.isna().all():  # belki ms olabilir
            ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        last_ts = ts.max()
    else:
        last_ts = pd.NaT

    # risk_score eşleme
    risk_col = next((c for c in ["risk_score","risk","score","prob"] if c in df.columns), None)
    avg_risk = float(df[risk_col].mean()) if risk_col else 0.0

    # yüksek risk oranı: risk_level varsa ondan; yoksa eşikten
    if "risk_level" in df.columns:
        high_rate = float((df["risk_level"] >= 2).mean())
    elif risk_col:
        high_rate = float((df[risk_col] >= 0.7).mean())
    else:
        high_rate = 0.0

    return {
        "rows": int(len(df)),
        "last_update": (last_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(last_ts) else "—"),
        "avg_risk": round(avg_risk, 3),
        "high_rate": round(high_rate, 3),
        "member": member
    }

def sample_for_map(limit: int = 50000) -> pd.DataFrame:
    """Home mini-harita için hafif örnek (eşdeğer kolon adlarını destekler)."""
    member = _pick_best_member()
    if not member:
        return pd.DataFrame(columns=REQUIRED_COLS)
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df = load_parquet(member, columns=["geoid","lat","lon","timestamp","risk_score","risk_level","pred_expected"])
    if df.empty:
        return df

    # timestamp normalizasyonu
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df["timestamp"] = ts

    # downsample: yüksek risk öncelikli
    if len(df) > limit:
        if "risk_score" in df.columns:
            top = df.nlargest(limit//2, "risk_score")
            rest = df.drop(top.index, errors="ignore")
            if len(rest) > (limit - len(top)):
                rest = rest.sample(limit - len(top), random_state=42)
            df = pd.concat([top, rest], ignore_index=True)
        else:
            df = df.sample(limit, random_state=42)
    return df

# core/data.py
from __future__ import annotations
import zipfile, tempfile, pathlib, hashlib
from typing import Iterable, Optional
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import streamlit as st
from config.settings import ARTIFACT_ZIP, DEFAULT_PARQUET_MEMBER

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

@st.cache_data(show_spinner=False)
def load_parquet(member_name: str,
                 columns: Optional[Iterable[str]] = None,
                 row_filter: Optional[ds.Expression] = None) -> pd.DataFrame:
    """
    ZIP içindeki tek bir parquet dosyasını okur (sütun seçimi + predicate pushdown destekli).
    Cache key: md5(zip) + member + columns + filter
    """
    zip_md5 = _md5(ARTIFACT_ZIP)
    cache_key = (zip_md5, member_name, tuple(columns or []), str(row_filter))

    with zipfile.ZipFile(ARTIFACT_ZIP) as zf, zf.open(member_name) as fh, \
         tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp.write(fh.read())
        tmp_path = pathlib.Path(tmp.name)

    try:
        if row_filter is None and columns:
            table = pq.read_table(tmp_path, columns=list(columns))
        else:
            dataset = ds.dataset(tmp_path, format="parquet")
            table = dataset.to_table(filter=row_filter, columns=list(columns) if columns else None)
        df = table.to_pandas()
        return df
    finally:
        try: tmp_path.unlink()
        except Exception: pass

def get_latest_kpis() -> dict:
    """
    Basit ve hızlı KPI: satır sayısı, son timestamp, ortalama risk, yüksek risk oranı.
    risk_hourly/parquet veya sf_crime_0X.parquet’tan okur.
    """
    member = None
    members = list_members()
    # tercih sırası
    for cand in (DEFAULT_PARQUET_MEMBER, "sf_crime_09.parquet", "sf_crime_08.parquet", "sf_crime_01.parquet"):
        if cand in members:
            member = cand; break
    if not member:
        return {"rows": 0, "last_update": "—", "avg_risk": 0.0, "high_rate": 0.0, "member": "N/A"}

    cols = ["timestamp","risk_score","risk_level"]
    df = load_parquet(member, columns=cols)
    if df.empty:
        return {"rows": 0, "last_update": "—", "avg_risk": 0.0, "high_rate": 0.0, "member": member}
    last_ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").max()
    avg_risk = float(df["risk_score"].mean())
    high_rate = float((df["risk_level"] >= 2).mean())
    return {
        "rows": int(len(df)),
        "last_update": last_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(last_ts) else "—",
        "avg_risk": round(avg_risk, 3),
        "high_rate": round(high_rate, 3),
        "member": member
    }

def sample_for_map(limit: int = 50000) -> pd.DataFrame:
    """Home mini-harita için hafif örnek (gereken kolonlarla)."""
    member = None
    members = list_members()
    for cand in (DEFAULT_PARQUET_MEMBER, "sf_crime_09.parquet", "sf_crime_08.parquet"):
        if cand in members:
            member = cand; break
    if not member:
        return pd.DataFrame(columns=REQUIRED_COLS)
    df = load_parquet(member, columns=["geoid","lat","lon","timestamp","risk_score","risk_level","pred_expected"])
    if len(df) > limit:
        # basit downsample: en yüksek riskten öncelikli + uniform örnek karışık
        top = df.nlargest(limit//2, "risk_score")
        rest = df.drop(top.index, errors="ignore").sample(limit - len(top), random_state=42) if len(df) > limit//2 else df
        df = pd.concat([top, rest], ignore_index=True)
    return df

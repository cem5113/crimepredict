# utils/heatmap.py
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import pandas as pd
import streamlit as st

from utils.constants import SF_TZ_OFFSET
from utils.forecast import aggregate_fast

def _safe_float_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce")
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0).clip(lower=0).sum())

@st.cache_data(show_spinner=False)
def _cached_aggregate_fast_once(si: str, geo_df: pd.DataFrame, base_int: pd.DataFrame,
                                events_df: Optional[pd.DataFrame], filters: Optional[Dict]) -> float:
    """Tek saatlik toplam expected’ı cache’ler."""
    agg_i = aggregate_fast(
        si, 1, geo_df, base_int,
        events=events_df,
        near_repeat_alpha=0.35,
        nr_lookback_h=24,
        nr_radius_m=400,
        nr_decay_h=12.0,
        filters=filters,
    )
    return _safe_float_sum(agg_i.get("expected", pd.Series(dtype=float)))

def render_day_hour_heatmap(
    agg: pd.DataFrame,
    start_iso: str,
    horizon_h: int,
    *,
    geo_df: pd.DataFrame,
    base_int: pd.DataFrame,
    filters: Optional[Dict] = None,
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    if not start_iso or not isinstance(horizon_h, (int, np.integer)) or horizon_h <= 0:
        st.caption("Isı matrisi için geçersiz başlangıç/ufuk.")
        return

    start = pd.to_datetime(start_iso)
    H = int(horizon_h)

    rows = []
    prog = st.progress(0, text="Saatlik tahminler hesaplanıyor…")
    for i in range(H):
        si = (start + pd.to_timedelta(i, unit="h")).isoformat()
        tot = _cached_aggregate_fast_once(si, geo_df, base_int, events_df, filters)
        t_sf = (start + pd.to_timedelta(i, unit="h")) + pd.Timedelta(hours=SF_TZ_OFFSET)
        rows.append({"dow": t_sf.weekday(), "hour": t_sf.hour, "E_city": tot})
        prog.progress((i + 1) / H)

    if not rows:
        st.caption("Isı matrisi için satır üretilemedi.")
        return

    mat = (pd.DataFrame(rows)
           .groupby(["dow", "hour"], as_index=False)["E_city"].sum()
           .pivot(index="dow", columns="hour", values="E_city")
           .reindex(range(7), fill_value=0.0)
           .reindex(columns=range(24), fill_value=0.0)
           .fillna(0.0))
    mat.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    mat.columns = [f"{h:02d}" for h in mat.columns]

    st.dataframe(mat.round(2), use_container_width=True)
    arr = mat.to_numpy()
    i, j = np.unravel_index(np.argmax(arr), arr.shape)
    st.caption(f"Toplam beklenen: {mat.values.sum():.2f} • En yoğun: {mat.index[i]} {mat.columns[j]}")

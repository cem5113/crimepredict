from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Iterable, Optional
import json
import numpy as np
import pandas as pd

# Project constants
try:
    # Prefer absolute import path in this repo structure
    from crimepredict.utils.constants import KEY_COL, KEY_COL_ALIASES  # type: ignore
except Exception:  # fallback for older layouts/tests
    from utils.constants import KEY_COL  # type: ignore
    KEY_COL_ALIASES = ["geoid", "GEOID", "id"]

__all__ = [
    "SF_CENTER",
    "SF_ZOOM",
    "ensure_key_col",
    "join_neighborhood",
    "polygon_centroid",
    "load_geoid_layer",
    "nearest_geoid",
    "resolve_clicked_gid",
    "get_map_init",
]

# ── Default map init (San Francisco) ─────────────────────────────────────────
SF_CENTER: Tuple[float, float] = (37.7749, -122.4194)  # (lat, lon)
SF_ZOOM: int = 12


# ── Utilities ────────────────────────────────────────────────────────────────
def _as_str_series(s: pd.Series) -> pd.Series:
    """Return a trimmed string Series (safe for merging keys)."""
    return s.astype(str).str.strip()


def ensure_key_col(df: pd.DataFrame, *, key: str = KEY_COL, aliases: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Ensure dataframe has canonical key column `key` by copying from aliases if needed.

    - Does not mutate the input; returns a new DataFrame.
    - If multiple aliases are present, first non-empty column is used.
    """
    if df is None or df.empty:
        return df

    aliases = list(aliases or KEY_COL_ALIASES)
    out = df.copy()

    if key in out.columns:
        out[key] = _as_str_series(out[key])
        return out

    for cand in aliases:
        if cand in out.columns:
            col = _as_str_series(out[cand])
            if col.notna().any():
                out[key] = col
                return out

    # If we reach here, create an empty key to avoid KeyError downstream
    out[key] = ""
    return out


# ── Neighborhood join ────────────────────────────────────────────────────────
def join_neighborhood(df_agg: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """df_agg + geo_df → neighborhood ekler (varsa). KEY_COL ile eşler.

    This is robust to different key column names via `ensure_key_col`.
    """
    if df_agg is None or df_agg.empty:
        return df_agg
    if "neighborhood" in df_agg.columns:
        return df_agg
    if geo_df is None or geo_df.empty or "neighborhood" not in geo_df.columns:
        return df_agg

    a = ensure_key_col(df_agg)
    g = ensure_key_col(geo_df)[[KEY_COL, "neighborhood"]].copy()
    return a.merge(g, on=KEY_COL, how="left")


# ── Geometry helpers ─────────────────────────────────────────────────────────
def polygon_centroid(lonlat_loop: List[List[float]] | List[Tuple[float, float]]):
    """Compute centroid for a flat polygon ring of (lon, lat) points.

    If ring is explicitly closed (first==last), last point is ignored.
    Falls back to arithmetic mean if area ~ 0.
    Returns (Cx, Cy) in (lon, lat).
    """
    if not lonlat_loop:
        return 0.0, 0.0

    x, y = zip(*lonlat_loop)
    A = Cx = Cy = 0.0
    closed = len(lonlat_loop) >= 2 and lonlat_loop[0] == lonlat_loop[-1]
    rng = range(len(lonlat_loop) - 1) if closed else range(len(lonlat_loop))

    for i in rng:
        j = (i + 1) % len(lonlat_loop)
        cross = x[i] * y[j] - x[j] * y[i]
        A += cross
        Cx += (x[i] + x[j]) * cross
        Cy += (y[i] + y[j]) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return float(sum(x) / len(x)), float(sum(y) / len(y))
    return float(Cx / (6 * A)), float(Cy / (6 * A))


# ── GeoJSON loading ──────────────────────────────────────────────────────────
def load_geoid_layer(path: str = "data/sf_cells.geojson", key_field: str = KEY_COL):
    """
    Read a GeoJSON grid/layer and return (DataFrame, FeaturesList).

    - DataFrame columns: [key_field, centroid_lon, centroid_lat]
    - Each returned feature gets `properties.id` and centroid properties guaranteed.
    - Works even if the GeoJSON uses alias key names (GEOID, id, ...).
    - If file missing, returns (empty DataFrame with expected columns, []).
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[key_field, "centroid_lon", "centroid_lat"]), []

    gj = json.loads(p.read_text(encoding="utf-8"))
    rows, feats_out = [], []

    for feat in gj.get("features", []):
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}

        # Robust key extraction
        geoid = str(
            props.get(key_field)
            or props.get(key_field.upper())
            or props.get("GEOID")
            or props.get("geoid")
            or props.get("id")
            or ""
        ).strip()
        if not geoid:
            continue

        # Centroid: prefer provided, else compute from geometry
        lon = props.get("centroid_lon")
        lat = props.get("centroid_lat")
        if lon is None or lat is None:
            gtype = geom.get("type")
            if gtype == "Polygon":
                ring = geom.get("coordinates", [[]])[0]
                lon, lat = polygon_centroid(ring)
            elif gtype == "MultiPolygon":
                # Use the first polygon's outer ring as representative
                ring = geom.get("coordinates", [[[]]])[0][0]
                lon, lat = polygon_centroid(ring)
            elif gtype == "Point":
                coords = geom.get("coordinates", [None, None])
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    lon, lat = float(coords[0]), float(coords[1])
                else:
                    continue
            else:
                # Unsupported geometry type
                continue

        row = {key_field: geoid, "centroid_lon": float(lon), "centroid_lat": float(lat)}
        rows.append(row)

        # Normalize properties for UI tooltips/popups
        props = dict(props)  # shallow copy
        props["id"] = geoid
        props.setdefault("centroid_lon", float(lon))
        props.setdefault("centroid_lat", float(lat))
        feat = dict(feat)
        feat["properties"] = props
        feats_out.append(feat)

    df = pd.DataFrame(rows)
    # Ensure canonical key column name for downstream joins
    if key_field != KEY_COL and not df.empty:
        df = df.rename(columns={key_field: KEY_COL})
    return df, feats_out


# ── Nearest GEOID by lat/lon ─────────────────────────────────────────────────
def nearest_geoid(geo_df: pd.DataFrame, lat: float, lon: float) -> str | None:
    """Return the nearest GEOID (by centroid) for a given (lat, lon)."""
    if geo_df is None or geo_df.empty or KEY_COL not in geo_df.columns:
        return None
    la = geo_df["centroid_lat"].to_numpy()
    lo = geo_df["centroid_lon"].to_numpy()
    d2 = (la - float(lat)) ** 2 + (lo - float(lon)) ** 2
    i = int(np.argmin(d2))
    return str(geo_df.iloc[i][KEY_COL])


# ── Click resolver helpers (Streamlit Folium / deck.gl interoperability) ─────
def _extract_latlon_from_ret(ret) -> Tuple[float, float] | None:
    """Extract (lat, lon) robustly from a variety of st_folium return shapes."""
    if not ret:
        return None
    lc = ret.get("last_clicked") if isinstance(ret, dict) else None
    if lc is None:
        return None

    # 1) [lat, lon]
    if isinstance(lc, (list, tuple)) and len(lc) >= 2:
        return float(lc[0]), float(lc[1])

    # 2) {"lat":..., "lng"|"lon":...}
    if isinstance(lc, dict):
        if "lat" in lc and ("lng" in lc or "lon" in lc):
            return float(lc["lat"]), float(lc.get("lng", lc.get("lon")))
        # 3) {"latlng": {"lat":..., "lng":...}} or [lat, lon]
        ll = lc.get("latlng")
        if isinstance(ll, (list, tuple)) and len(ll) >= 2:
            return float(ll[0]), float(ll[1])
        if isinstance(ll, dict) and "lat" in ll and ("lng" in ll or "lon" in ll):
            return float(ll["lat"]), float(ll.get("lng", ll.get("lon")))
    return None


def resolve_clicked_gid(geo_df: pd.DataFrame, ret: dict) -> tuple[str | None, tuple[float, float] | None]:
    """
    Given st_folium's result dict (or a similar structure), resolve the clicked
    GEOID and lat/lon. Returns: (geoid | None, (lat, lon) | None)
    """
    gid, latlon = None, None
    obj = ret.get("last_object_clicked") if isinstance(ret, dict) else None

    if isinstance(obj, dict):
        props = obj.get("properties", {}) or obj.get("feature", {}).get("properties", {}) or {}
        gid = str(
            obj.get("id")
            or props.get("id")
            or props.get(KEY_COL)
            or props.get(KEY_COL.upper(), props.get("GEOID"))
            or ""
        ).strip() or None

        # Try geometry-based coordinates first
        geom = obj.get("geometry") or obj.get("feature", {}).get("geometry")
        if isinstance(geom, dict) and geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                latlon = (float(coords[1]), float(coords[0]))
        if not latlon:
            lat = (
                obj.get("lat")
                or obj.get("latlng", {}).get("lat")
                or obj.get("location", {}).get("lat")
            )
            lng = (
                obj.get("lng")
                or obj.get("latlng", {}).get("lng")
                or obj.get("location", {}).get("lng")
                or obj.get("lon")
            )
            if lat is not None and lng is not None:
                latlon = (float(lat), float(lng))

    if not gid:
        if not latlon:
            latlon = _extract_latlon_from_ret(ret)
        if latlon:
            gid = nearest_geoid(geo_df, latlon[0], latlon[1])

    return gid, latlon


# ── Initial map view resolver ────────────────────────────────────────────────
def get_map_init(geo_df: pd.DataFrame | None = None) -> tuple[float, float, int]:
    """Return initial (lat, lon, zoom) for the map.

    - If geo_df has centroid columns, use their mean as center.
    - Else fall back to San Francisco defaults.
    """
    if isinstance(geo_df, pd.DataFrame) and not geo_df.empty:
        try:
            lat = float(geo_df["centroid_lat"].mean())
            lon = float(geo_df["centroid_lon"].mean())
            if np.isfinite(lat) and np.isfinite(lon):
                return lat, lon, SF_ZOOM
        except Exception:
            pass
    return SF_CENTER[0], SF_CENTER[1], SF_ZOOM

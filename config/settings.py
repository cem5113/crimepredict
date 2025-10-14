from __future__ import annotations
import os
from pathlib import Path

# ---- zorunlu/varsayılan yollar ----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))

# parquet veya csv kaynak yolu (yerelde tutuyorsun)
DEFAULT_PARQUET = os.getenv("DEFAULT_PARQUET", str(DATA_DIR / "risk_hourly.parquet"))
DEFAULT_CSV = os.getenv("DEFAULT_CSV", str(DATA_DIR / "risk_hourly.csv"))

# sf hücreleri geojson (GEOID → centroid eşleştirme için)
SF_CELLS_GEOJSON = os.getenv("SF_CELLS_GEOJSON", str(ROOT / "core" / "data" / "sf_cells.geojson"))

# klasörleri oluştur
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

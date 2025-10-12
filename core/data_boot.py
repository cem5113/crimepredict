# core/data_boot.py
import os
from pathlib import Path
from .artifacts import ensure_latest_zip

def configure_artifact_env():
    path: Path = ensure_latest_zip()  # indir veya sıcaksa geç
    os.environ["CRIME_ARTIFACT_ZIP"] = str(path.resolve())
    return path

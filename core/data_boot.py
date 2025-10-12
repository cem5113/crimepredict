# core/data_boot.py
import os
from pathlib import Path
from .artifacts import ensure_latest_zip

def configure_artifact_env() -> Path:
    zip_path: Path = ensure_latest_zip()
    os.environ["CRIME_ARTIFACT_ZIP"] = str(zip_path.resolve())
    return zip_path


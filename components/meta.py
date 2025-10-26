# components/meta.py  (OPSİYONEL)
from __future__ import annotations
import json
from pathlib import Path

def load_local_metadata(path: str | Path = "data/_metadata.json") -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

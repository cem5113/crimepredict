# ui/map_layers.py
from typing import Dict, Any
import numpy as np


# Projenizdeki gerçek harita katmanı üretimiyle değiştirin.


def grid_to_heatmap_layer(grid: np.ndarray) -> Dict[str, Any]:
# UI tarafında kullanılacak yalın bir payload (örnek)
return {"type": "heatmap", "shape": grid.shape, "min": float(grid.min()), "max": float(grid.max())}




def hotspot_layer(grid: np.ndarray, threshold: float = 0.9) -> Dict[str, Any]:
mask = (grid >= threshold).astype(int).sum()
return {"type": "hotspot", "count": int(mask)}




def history_layer() -> Dict[str, Any]:
return {"type": "history", "enabled": True}

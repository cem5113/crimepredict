# patrol/metrics.py

from typing import List, Set, Dict
import numpy as np
from config import settings


# Kapsama/çeşitlilik/örtüşme örnek hesapları (basit sürüm)


def coverage(grid: np.ndarray, threshold: float = 0.7) -> float:
total = grid.size
covered = (grid >= threshold).sum()
return float(covered) / float(total)




def diversity(route_cells: List[Set[int]]) -> float:
# Basit çeşitlilik: her rotanın kapsadığı hücre sayısı varyansı düşükse çeşitlilik düşük
sizes = [len(s) for s in route_cells]
if not sizes:
return 0.0
mean = sum(sizes) / len(sizes)
var = sum((s - mean) ** 2 for s in sizes) / max(len(sizes), 1)
# normalize: daha az varyans ⇒ daha düşük çeşitlilik
return 1.0 / (1.0 + var)




def overlap_jaccard(route_cells: List[Set[int]]) -> float:
pairs = []
n = len(route_cells)
for i in range(n):
for j in range(i + 1, n):
a, b = route_cells[i], route_cells[j]
u = len(a | b)
inter = len(a & b)
pairs.append(inter / u if u else 0.0)
return sum(pairs) / len(pairs) if pairs else 0.0




def combined_score(cov: float, div: float, ovl: float) -> float:
return (settings.W_COVERAGE * cov) + (settings.W_DIVERSITY * div) - (settings.W_OVERLAP * ovl)

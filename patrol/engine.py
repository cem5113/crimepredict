# patrol/engine.py
from typing import List, Dict, Any, Set
import numpy as np


from services.risk_service import get_risk_grid
from ui.state import TimeQuery
from config import settings
from patrol.metrics import coverage, diversity, overlap_jaccard, combined_score


# Bu engine, grid üzerinde basit bir durak seçimi yaparak örnek rotalar üretir.
# Projenizdeki gerçek rotalama/heuristic ile değiştirin.




def _pick_cells(grid: np.ndarray, k: int = 10, min_gap: int = 5) -> List[tuple]:
# En yüksek riskli k hücreyi al (örnek)
H, W = grid.shape
flat_idx = np.argsort(grid, axis=None)[-k:][::-1]
cells = [(i // W, i % W) for i in flat_idx]
return cells




def _cells_to_route(cells: List[tuple]) -> List[Dict[str, Any]]:
# Hücre -> pseudo koordinat (lat/lon) dönüşümü (örnek)
# Projede elinizde grid->coor dönüşümü varsa onu kullanın.
route = []
for (r, c) in cells:
route.append({"lat": 37.70 + r * 0.001, "lon": -122.52 + c * 0.001})
return route




def _route_to_cellset(route: List[Dict[str, float]]) -> Set[int]:
# Basit hücre kimliği: lat/lon'dan türetilmiş indeks (örnek)
# Gerçekte route polyline + buffer ile grid kesişimi kullanmalısınız.
ids = set()
for p in route:
r = int((p["lat"] - 37.70) / 0.001)
c = int((p["lon"] + 122.52) / 0.001)
ids.add(r * 1000 + c)
return ids




def _estimate_distance_km(route: List[Dict[str, float]], routing_factor: float, avg_speed_kmh: float, dwell_min: float):
import math
def haversine(lat1, lon1, lat2, lon2):
R = 6371.0
dlat = math.radians(lat2 - lat1)
dlon = math.radians(lon2 - lon1)
a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
return 2 * R * math.asin(math.sqrt(a))
dist = 0.0
for i in range(len(route)-1):
a, b = route[i], route[i+1]
dist += haversine(a['lat'], a['lon'], b['lat'], b['lon'])
dist *= routing_factor
travel_min = 60.0 * dist / max(avg_speed_kmh, 1e-3)
dwell_total = dwell_min * len(route)
return dist, travel_min + dwell_total




def generate_recommendations(now_utc, tq: TimeQuery, n_alternatives: int = 3, k_stops: int = 10):
rr = get_risk_grid(now_utc, tq)
grid = rr["grid"]
start, end = rr["start"], rr["end"]


recs = []
route_cells = []
for i in range(n_alternatives):
cells = _pick_cells(grid, k=k_stops)
route = _cells_to_route(cells)
cellset = _route_to_cellset(route)
dist_km, duration_min = _estimate_distance_km(
route,
settings.ROUTING_FACTOR,
settings.AVG_SPEED_KMH,
settings.DEFAULT_DWELL_MIN,
)
route_cells.append(cellset)
recs.append({
"id": f"rec_{i+1}",
"route": route,
"distance_km": dist_km,
"duration_min": duration_min,
})


}

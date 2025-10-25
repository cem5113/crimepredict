# services/risk_service.py
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
from hashlib import md5


from config import settings
from ui.state import TimeQuery, TimeMode
from services.time_utils import compute_time_window
from services.cache import CACHE


# --- Veri Erişim Katmanı (Sizin kaynaklarınıza bağlayın) ---
# Aşağıdaki stub fonksiyonları proje veri erişiminize göre gerçekleyin.


def fetch_hourly_predictions(start, end) -> List[np.ndarray]:
"""[start,end) aralığındaki saatler için grid listesi döner (1..24 adet).
Her grid: (H, W) numpy array risk skorları.
Burada örnek olarak tek bir 100x100 rasgele grid üretiyoruz.
"""
hours = int((end - start).total_seconds() // 3600) or 1
return [np.random.rand(100, 100).astype("float32") for _ in range(hours)]




def fetch_daily_predictions(day_start) -> np.ndarray | None:
"""Günlük grid varsa döner; yoksa None (opsiyonel)."""
return None


# --- Agregasyon yardımcıları ---


def _weighted_mean(grids: List[np.ndarray], weights: List[float]) -> np.ndarray:
w = np.array(weights[: len(grids)], dtype="float32")
w = w / max(w.sum(), 1e-6)
stack = np.stack(grids, axis=0)
return (stack * w[:, None, None]).sum(axis=0)




def _reduce(grids: List[np.ndarray], how: str) -> np.ndarray:
stack = np.stack(grids, axis=0)
if how == "max":
return stack.max(axis=0)
if how == "mean":
return stack.mean(axis=0)
if how == "p95":
return np.percentile(stack, 95, axis=0)
# default
return stack.mean(axis=0)


# --- Ana arayüz ---


def get_risk_grid(now_utc: datetime, tq: TimeQuery) -> Dict[str, Any]:
tq = tq.clamp()
start, end = compute_time_window(now_utc, tq)


cache_key = md5(f"risk:{tq.mode}:{tq.offset}:{tq.tz}:{settings.RISK_AGG}".encode()).hexdigest()
ttl = settings.CACHE_TTL_HOURLY_SEC if tq.mode in (TimeMode.NOW, TimeMode.HOURLY) else settings.CACHE_TTL_DAILY_SEC
cached = CACHE.get(cache_key)
if cached is not None:
return {"grid": cached, "start": start, "end": end}


if tq.mode in (TimeMode.NOW, TimeMode.HOURLY):
hourly = fetch_hourly_predictions(start, end)
grid = hourly[0]
else: # DAILY
# Varsa direkt günlük grid al
day_grid = fetch_daily_predictions(start)
if day_grid is not None:
grid = day_grid
else:
# Saatlikleri toplayıp günlük yap
day_end = end
hourly = fetch_hourly_predictions(start, day_end) # 24'e kadar
if settings.RISK_AGG == "weighted_mean":
grid = _weighted_mean(hourly, settings.BUSINESS_PROFILE)
else:
grid = _reduce(hourly, settings.RISK_AGG)


# Risk multiplier
if settings.RISK_MULTIPLIER_ENABLED:
grid = grid * float(settings.RISK_MULTIPLIER)


CACHE.set(cache_key, grid, ttl)
return {"grid": grid, "start": start, "end": end}

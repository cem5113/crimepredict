# config/settings.py
from dataclasses import dataclass
from typing import List


# === Zaman Modu & Agregasyon Ayarları ===
RISK_AGG: str = "weighted_mean" # "max" | "mean" | "p95"
# 24 saatlik profil (0..23). Gece/akşam daha yüksek ağırlık örneği
BUSINESS_PROFILE: List[float] = [
0.6, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9,
1.0, 1.0, 0.9, 0.8, 0.8, 0.9, 1.1, 1.2,
1.3, 1.4, 1.4, 1.3, 1.2, 1.1, 1.0, 0.8
]


# Günlükten saatliğe indirgemede (gerekirse) diurnal çarpan profili
DIURNAL_WEIGHTS: List[float] = [
0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.9,
1.0, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1,
1.2, 1.3, 1.3, 1.2, 1.1, 1.0, 0.9, 0.7
]


# === Risk Ağırlık Çarpanı (opsiyonel) ===
RISK_MULTIPLIER_ENABLED: bool = True
RISK_MULTIPLIER: float = 1.0


# === Rota Süre/Mesafe Tahmin Ayarları ===
AVG_SPEED_KMH: float = 25.0 # şehir içi ortalama
ROUTING_FACTOR: float = 1.25 # sokak dolanması payı
DEFAULT_DWELL_MIN: float = 3.0 # nokta başı bekleme


# === Cache Ayarları ===
CACHE_TTL_HOURLY_SEC: int = 1800 # 30dk
CACHE_TTL_DAILY_SEC: int = 7200 # 2saat


# === Harita ===
DEFAULT_TZ: str = "America/Los_Angeles"


# === Skor ağırlıkları ===
W_COVERAGE: float = 1.0
W_DIVERSITY: float = 0.7
W_OVERLAP: float = 0.8 # çıkarılacak (ceza)


@dataclass
class UIFlags:
show_history: bool = False
show_hotspot: bool = True
show_risk: bool = True


UI_FLAGS = UIFlags()

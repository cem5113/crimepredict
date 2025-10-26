# utils/constants.py  (REVİZYON — tek kaynak)
from __future__ import annotations

# Zaman dilimi (SF ~ UTC-7 yaz, UTC-8 kış; burada basit sabit)
SF_TZ_OFFSET     = -7

# GEO anahtar sütunu
KEY_COL = "geoid"
KEY_COL_ALIASES = ["geoid", "GEOID", "id"]

# Model meta
MODEL_VERSION    = "v1.0.0"
MODEL_LAST_TRAIN = "2025-10-26"   # dilersen datetime.utcnow().strftime("%Y-%m-%d")

# UI listeleri
CATEGORIES = ["Assault", "Burglary", "Robbery", "Theft", "Vandalism", "Vehicle Theft"]
DAYS       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASONS    = ["Winter", "Spring", "Summer", "Autumn"]

# Suç türü kolonları (model çıktısı sütun adları)
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]

# Kategori → kolon eşleştirmesi (filtre için)
CATEGORY_TO_KEYS = {
    "Assault"       : ["assault"],
    "Burglary"      : ["burglary"],
    "Robbery"       : ["robbery"],
    "Theft"         : ["theft", "larceny"],
    "Vandalism"     : ["vandalism"],
    "Vehicle Theft" : ["vehicle_theft", "auto_theft", "motor_vehicle_theft"],
}

# Tier etiketleri
TIER_LEVELS_5 = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]

# Harita katman adları
RISK_LAYER_NAME          = "Tahmin (risk)"
PERM_HOTSPOT_LAYER_NAME  = "Hotspot (kalıcı)"
TEMP_HOTSPOT_LAYER_NAME  = "Hotspot (geçici)"

# Varsayılan harita görünümü
MAP_CENTER = [37.7749, -122.4194]
MAP_ZOOM_START = 12
MAP_TILE_STYLE = "cartodbpositron"

# Renk paleti (opsiyonel)
RISK_COLORS = {
    "Çok Düşük": "#c7e9b4",
    "Düşük": "#7fcdbb",
    "Orta": "#41b6c4",
    "Yüksek": "#1d91c0",
    "Çok Yüksek": "#225ea8",
}

DEFAULT_LAYER_VISIBILITY = {
    "risk_layer_show": True,
    "perm_hotspot_show": True,
    "temp_hotspot_show": True,
}

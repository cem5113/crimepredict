# components/meta.py
from datetime import datetime, timezone, timedelta

# San Francisco saat farkı (UTC-8 kışın, UTC-7 yazın)
# Burada basitçe UTC-8 olarak tutabiliriz:
SF_TZ_OFFSET = -8  # saat cinsinden

# Harita / hücre anahtarı
KEY_COL = "geoid"

# Model versiyonu bilgileri
MODEL_VERSION = "v1.1.0"

# Son eğitim tarihi – otomatik olarak UTC anını biçimlendirir
MODEL_LAST_TRAIN = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# Kategoriler – modelin desteklediği suç türleri
CATEGORIES = [
    "Hırsızlık",
    "Saldırı",
    "Dolandırıcılık",
    "Vandalizm",
    "Uyuşturucu",
    "Araç Hırsızlığı",
    "Gasp",
    "Cinayet",
]

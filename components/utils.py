# components/utils.py
from datetime import datetime

# İsterseniz burayı dinamik hale getirip bir metadata JSON'dan da çekebilirsiniz.
MODEL_VERSION = "v1.0.0"
MODEL_LAST_TRAIN = datetime(2025, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M UTC")

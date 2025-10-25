# services/cache.py

import time
from typing import Any, Dict, Tuple

class SimpleTTLCache:
def __init__(self):
self._store: Dict[str, Tuple[float, Any]] = {}

def get(self, key: str):
rec = self._store.get(key)
if not rec:
return None
exp, val = rec
if time.time() > exp:
self._store.pop(key, None)
return None
return val

def set(self, key: str, val: Any, ttl: int):
self._store[key] = (time.time() + ttl, val)

CACHE = SimpleTTLCache()

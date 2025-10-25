# ui/state.py
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TimeMode(str, Enum):
NOW = "now"
HOURLY = "hourly"
DAILY = "daily"


@dataclass
class TimeQuery:
mode: TimeMode = TimeMode.NOW
offset: int = 0 # HOURLY için saat, DAILY için gün
tz: str = "America/Los_Angeles"


def clamp(self):
if self.mode == TimeMode.NOW:
self.offset = 0
elif self.mode == TimeMode.HOURLY:
self.offset = max(1, min(24, int(self.offset)))
elif self.mode == TimeMode.DAILY:
self.offset = max(1, min(3, int(self.offset)))
return self

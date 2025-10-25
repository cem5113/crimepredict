# services/time_utils.py
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from ui.state import TimeMode, TimeQuery

def compute_time_window(now_utc: datetime, t: TimeQuery):
"""Seçilen moda göre [start,end) zaman penceresi üretir."""
tz = ZoneInfo(t.tz)
now = now_utc.astimezone(tz)
if t.mode == TimeMode.NOW:
start = now.replace(minute=0, second=0, microsecond=0)
end = start + timedelta(hours=1)
elif t.mode == TimeMode.HOURLY:
start = (now + timedelta(hours=t.offset)).replace(minute=0, second=0, microsecond=0)
end = start + timedelta(hours=1)
else: # DAILY
day = (now.date() + timedelta(days=t.offset))
start = datetime.combine(day, time(0, 0), tzinfo=tz)
end = start + timedelta(days=1)
return start, end

def label_time_window(start, end):
"""UI etiket metni (mutlak zaman) döner."""
fmt = "%Y-%m-%d %H:%M %Z"
return f"{start.strftime(fmt)} → {end.strftime(fmt)}"

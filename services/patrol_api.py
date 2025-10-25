# services/patrol_api.py
from fastapi import APIRouter, Query
from datetime import datetime, timezone
from ui.state import TimeQuery, TimeMode
from patrol.engine import generate_recommendations
from config import settings


router = APIRouter(prefix="/patrol", tags=["patrol"])


@router.get("/predict")
def predict(
mode: TimeMode = Query(TimeMode.NOW),
offset: int = Query(0, ge=0, le=24),
tz: str = Query(settings.DEFAULT_TZ),
):
# Validasyon
tq = TimeQuery(mode=mode, offset=offset, tz=tz).clamp()
now_utc = datetime.now(timezone.utc)
out = generate_recommendations(now_utc, tq)
return out

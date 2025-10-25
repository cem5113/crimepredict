# ui/tab_planning.py
import streamlit as st
import requests
from datetime import datetime
from zoneinfo import ZoneInfo


from ui.state import TimeMode
from config import settings


API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")


st.header("🚓 Devriye Planlama")


# --- Zaman Modu Seçici ---
mode = st.segmented_control("Zaman modu", [TimeMode.NOW.value, TimeMode.HOURLY.value, TimeMode.DAILY.value],
default=TimeMode.NOW.value, key="time_mode")


if mode == TimeMode.NOW.value:
offset = 0
elif mode == TimeMode.HOURLY.value:
offset = st.slider("Saatlik ufuk (saat)", 1, 24, 3, 1)
else:
offset = st.slider("Günlük ufuk (gün)", 1, 3, 1, 1)


# Opsiyonel risk multiplier kontrolü
if st.toggle("Risk çarpanı aktif", value=settings.RISK_MULTIPLIER_ENABLED, key="rm_en"):
rm = st.slider("Risk çarpanı", 0.5, 2.0, float(settings.RISK_MULTIPLIER), 0.05)
else:
rm = settings.RISK_MULTIPLIER


# Parametre hash (basit):
param_key = f"{mode}:{offset}:{rm}"


# --- Öneri Al ---
if st.button("Devriye Öner") or st.session_state.get("last_params") != param_key:
st.session_state["last_params"] = param_key
with st.spinner("Öneriler oluşturuluyor..."):
resp = requests.get(f"{API_BASE}/patrol/predict", params={
"mode": mode,
"offset": offset,
"tz": settings.DEFAULT_TZ,
})
if resp.ok:
data = resp.json()
st.session_state["patrol_resp"] = data
else:
st.error(f"API hata: {resp.status_code}")


# --- Gösterim ---
resp = st.session_state.get("patrol_resp")
if resp:
tw = resp.get("time_window", {})
st.caption(f"Zaman penceresi: {tw.get('start','?')} → {tw.get('end','?')}")


# Skor paneli
scores = resp.get("scores", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kapsama", f"{scores.get('coverage',0):.2f}")
c2.metric("Çeşitlilik", f"{scores.get('diversity',0):.2f}")
c3.metric("Örtüşme", f"{scores.get('overlap',0):.2f}")
c4.metric("Toplam Skor", f"{scores.get('combined',0):.2f}")


# Karusel/kart arayüzü
recs = resp.get("recommendations", [])
if recs:
idx = st.slider("Öneri", 1, len(recs), 1)
rec = recs[idx-1]
st.subheader(f"Alternatif #{idx}")
st.write(f"Mesafe ~ {rec['distance_km']:.2f} km | Süre ~ {rec['duration_min']:.0f} dk")
with st.expander("Rota noktaları"):
st.json(rec["route"]) # Projede burada gerçek harita çizimi kullanın
st.button("Amir Onayı İste", key=f"approve_{rec['id']}")

# 🧭 Crime Forecast – Üç Motorlu Zaman–Mekân Suç Tahmin Pipeline

Bu proje, **makine öğrenmesi tabanlı üç kademeli (short / mid / long horizon)** bir **suç tahmin sistemi (forecast pipeline)** sunar.  
Sistem, geçmiş verilerden (suç, 911/311 çağrıları, hava durumu, POI, nüfus, toplu taşıma, polis binaları vb.) **zaman–mekân–suç türü odaklı tahmin** üretir.

Modelleme stratejisi, **stacking tabanlı ensemble öğrenme** yapısı ve **horizon’a göre özelleşmiş üç motor** üzerine kuruludur:

| Motor | Zaman Ufku | Yaklaşım | Amaç |
|--------|-------------|-----------|------|
| **SHORT** | 0–72 saat | XGBoost + LightGBM + RF + meta LR | Günlük/haftalık tahmin (nowcasting) |
| **MID** | 3–30 gün (72–720 saat) | Orta vadeli stacking | Kısa trend + hafta örüntüleri |
| **LONG** | 1–6 ay (960–2160 saat) | Mevsimsel baseline + kalibrasyon | Uzun vadeli planlama |

---

## 📁 Proje Yapısı


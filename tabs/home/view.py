# tabs/home/view.py
import streamlit as st
from core import data as core_data
from core import mapkit

def _kpi(k, v, help_txt: str = ""):
    st.metric(label=k, value=v, help=help_txt)

def _goto(tab_key: str):
    st.session_state["__active_tab__"] = tab_key
    st.experimental_rerun()

def render(state=None, services=None):
    st.title("Suç Tahmini — Ana Sayfa")
    st.caption("Basit ama etkileyici: Güncel veriden risk haritası, planlama ve raporlar.")
    st.divider()

    c_reload_l, c_reload_r = st.columns([1, 0.18])
    with c_reload_r:
        if st.button("↻ Yeniden yükle", use_container_width=True,
                     help="Cache temizle & sayfayı tazele"):
            try:
                # Streamlit cache'i temizle (varsa)
                st.cache_data.clear()
            except Exception:
                pass
            # Yeniden çalıştır
            try:
                st.rerun()  # yeni sürümler
            except AttributeError:
                st.experimental_rerun() 

    from components.last_update import show_last_update_badge
    from utils.constants import MODEL_VERSION, MODEL_LAST_TRAIN

    kpis = core_data.get_latest_kpis()  # kpis zaten aşağıda da kullanılıyor
    show_last_update_badge(
        app_name="SUTAM",
        data_upto=kpis.get("last_update") or "-",
        model_version=MODEL_VERSION,
        last_train=MODEL_LAST_TRAIN,
        daily_update_hour_sf=19,
        show_times=True,
        tz_label="SF",
        show_actions=True,       # sadece “↻ Yeniden yükle” butonu için
        on_pipeline_click=None   # diğer butonları boş bırak
    )

    # ── KPI'lar
    col1, col2, col3, col4 = st.columns(4)
    kpis = core_data.get_latest_kpis()
    with col1: _kpi("Kayıt", f"{kpis['rows']:,}")
    with col2: _kpi("Son Güncelleme", kpis["last_update"])
    with col3: _kpi("Ortalama Risk", kpis["avg_risk"])
    with col4: _kpi("Yüksek Risk Oranı", kpis["high_rate"])

    st.write(f"**Veri Kaynağı:** `{kpis['member']}` (artefact zip içi)")

    # ── Veri Tanılama
    with st.expander("🔎 Veri Tanılama", expanded=False):
        members = core_data.list_members()
        st.write("Bulunan Parquet dosyaları:", members[:20])
        if members:
            sel = st.selectbox(
                "Bir dosya seç ve şemasını göster",
                members,
                index=members.index(kpis["member"]) if kpis["member"] in members else 0,
            )
            try:
                cols = core_data._read_schema(sel)  # sadece debug
                st.write(f"**{sel}** şema:", cols)
            except Exception as e:
                st.warning(f"Şema okunamadı: {e}")

    # ── Kısayollar
    st.markdown("### Hızlı Erişim")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🔮 Suç Tahmini", use_container_width=True):
            _goto("forecast")
    with c2:
        if st.button("🚓 Devriye Planlama", use_container_width=True):
            _goto("planning")
    with c3:
        if st.button("📊 İstatistikler", use_container_width=True):
            _goto("stats")
    with c4:
        if st.button("🧾 Raporlar", use_container_width=True):
            _goto("reports")

    # ── Harita
    st.markdown("### Şehir Anlık Görünüm")
    st.caption(
        "Bu ekran, **renkli risk katmanı + ısı katmanı** ile mekânsal yoğunluğu gösterir. "
        "Detaylı katmanlar ve analiz için **Suç Tahmini** sekmesine geçebilirsiniz."
    )

    df_map = core_data.sample_for_map(limit=50_000)
    if df_map.empty:
        st.info("Harita için veri bulunamadı. Veri bağlantısı kurulunca katmanlar otomatik görünecek.")
    else:
        # core/mapkit.home_deck artık utils.deck.build_map_fast_deck ile çalışıyor
        deck = mapkit.home_deck(df_map)
        st.pydeck_chart(deck, use_container_width=True)

    st.caption(
        "İpucu: Hover/tıklamada tooltip’te **GEOID**, mahalle adı, **E[olay]** (beklenen olay) ve risk seviyesi görünür."
    )

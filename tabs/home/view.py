# tabs/home/view.py
import streamlit as st

# --- Güvenli import: hem paketli (crimepredict.*) hem yerel (core.*) destekler ---
try:
    from crimepredict.core import mapkit
    from crimepredict.core import data as core_data
except ImportError:
    from core import mapkit
    from core import data as core_data


def _kpi(k, v, help_txt: str = ""):
    """KPI göstergeleri için kısa yardımcı"""
    st.metric(label=k, value=v, help=help_txt)


def _goto(tab_key: str):
    """Sekmeler arasında geçiş (state tabanlı)"""
    st.session_state["__active_tab__"] = tab_key
    st.experimental_rerun()


def render(state=None, services=None):
    st.title("Suç Tahmini — Ana Sayfa")
    st.caption("Basit ama etkileyici: Güncel veriden risk haritası, planlama ve raporlar.")
    st.divider()

    # --- ↻ Yeniden yükle butonu ---
    c_reload_l, c_reload_r = st.columns([1, 0.18])
    with c_reload_r:
        if st.button("↻ Yeniden yükle", use_container_width=True, help="Cache temizle ve sayfayı yenile"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

    # --- KPI'lar ---
    col1, col2, col3, col4 = st.columns(4)
    kpis = core_data.get_latest_kpis()
    with col1:
        _kpi("Kayıt", f"{kpis['rows']:,}")
    with col2:
        _kpi("Son Güncelleme", kpis["last_update"])
    with col3:
        _kpi("Ortalama Risk", kpis["avg_risk"])
    with col4:
        _kpi("Yüksek Risk Oranı", kpis["high_rate"])

    st.write(f"**Veri Kaynağı:** `{kpis['member']}` (artefact zip içi)")

    # --- Veri Tanılama ---
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
                cols = core_data._read_schema(sel)
                st.write(f"**{sel}** şema:", cols)
            except Exception as e:
                st.warning(f"Şema okunamadı: {e}")

    # --- Hızlı Erişim Kısayolları ---
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

    # --- Şehir Haritası ---
    st.markdown("### Şehir Anlık Görünüm")
    st.caption(
        "Bu ekran, **renkli risk katmanı + ısı katmanı** ile mekânsal yoğunluğu gösterir. "
        "Detaylı katmanlar ve analiz için **Suç Tahmini** sekmesine geçebilirsiniz."
    )

    df_map = core_data.sample_for_map(limit=50_000)
    if df_map.empty:
        st.info("Harita için veri bulunamadı. Veri bağlantısı kurulunca katmanlar otomatik görünecek.")
    else:
        deck = mapkit.home_deck(df_map)
        st.pydeck_chart(deck, use_container_width=True)

    st.caption(
        "İpucu: Hover/tıklamada tooltip’te **GEOID**, mahalle adı, **E[olay]** (beklenen olay) ve risk seviyesi görünür."
    )

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# --- 1. Veri yükleme ---
# Kullanılacak veri: cem5113/crime_prediction_data/artifact/fr-crime-pipeline-output.zip içindeki fr_crime_09.csv
# Stacking metrikleri: cem5113/crime_prediction_data/artifact/sf-crime-pipeline-output.zip içindeki metrics_stacking_ohe.csv

@st.cache_data
def load_data():
    # Ana suç verisi (fr_crime_09.csv)
    df = pd.read_csv('https://github.com/cem5113/crime_prediction_data/raw/main/artifact/fr-crime-pipeline-output/fr_crime_09.csv')

    # Stacking metrikleri (metrics_stacking_ohe.csv)
    metrics = pd.read_csv('https://github.com/cem5113/crime_prediction_data/raw/main/artifact/sf-crime-pipeline-output/metrics_stacking_ohe.csv')
    
    return df, metrics

df, metrics = load_data()

# --- 2. Arayüz Başlığı ---
st.title('🔎 Suç Tahmin Modülü (Yalnız Kolluğa Yararlı)')
st.markdown('Zaman, mekân ve kategori bazlı risk tahminleri — yalnız kolluk için anlamlı sonuçlar gösterilir.')

# --- 3. Filtreler ---
col1, col2, col3 = st.columns(3)
with col1:
    date_selected = st.date_input('Tarih seçin')
with col2:
    hour_selected = st.slider('Saat aralığı seçin', 0, 23, (18, 23))
with col3:
    category_selected = st.selectbox('Suç kategorisi', sorted(df['Category'].dropna().unique()))

show_only_relevant = st.toggle('🔒 Yalnız kolluğa yararlı sonuçları göster', value=True)

# --- 4. Risk Filtreleme ---
if 'risk_score' in df.columns:
    q75 = df['risk_score'].quantile(0.75)
else:
    st.warning('⚠️ Veri kümesinde risk_score sütunu bulunamadı. risk olasılığı hesaplanmalı.')
    q75 = 0.5

filtered = df[df['risk_score'] >= q75] if show_only_relevant else df

# Saat ve kategori filtreleme
if 'hour' in df.columns:
    filtered = filtered[(filtered['hour'] >= hour_selected[0]) & (filtered['hour'] <= hour_selected[1])]
if category_selected:
    filtered = filtered[filtered['Category'] == category_selected]

# --- 5. Harita Oluşturma ---
center = [37.77, -122.42]
m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')

for _, row in filtered.iterrows():
    if 'latitude' in row and 'longitude' in row:
        popup_text = f"GEOID: {row['GEOID']}<br>Risk: {row['risk_score']:.2f}<br>Saat: {row['hour']}<br>Kategori: {row['Category']}"
        color = 'red' if row['risk_score'] >= q75 else 'orange'
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

st_folium(m, width=750, height=550)

# --- 6. Özet Tablo ---
st.subheader('📊 Yüksek Riskli Noktalar')
st.dataframe(filtered[['GEOID', 'hour', 'Category', 'risk_score']].sort_values(by='risk_score', ascending=False).head(20))

# --- 7. Stacking Performans Özeti ---
st.subheader('📈 Model Performans Özeti (Stacking)')
st.dataframe(metrics)

# --- 8. Veri Dışa Aktarım ---
st.download_button('⬇️ Hotspot verisini indir (CSV)', filtered.to_csv(index=False), 'high_risk_hotspots.csv', 'text/csv')

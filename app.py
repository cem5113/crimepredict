# SUTAM - Suç Tahmin Modeli


path = st.session_state.get('geojson_path_ui') or st.secrets.get("geojson_path", "sf_cells.geojson")
url_override = st.secrets.get("geojson_url", "")


# 1) Artifact içinden dene
try:
zip_bytes = fetch_latest_artifact_zip(OWNER, REPO, ARTIFACT_NAME)
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
memlist = zf.namelist()
st.session_state['zip_manifest'] = memlist # teşhis için
# Klasör içindeyse basit eşleştirme yapalım
if path in memlist:
with zf.open(path) as f:
return json.load(io.TextIOWrapper(f, encoding="utf-8"))
# Aynı isim farklı klasörde olabilir → suffix araması
cand = [n for n in memlist if n.endswith('/'+path) or n.endswith(path)]
if cand:
with zf.open(cand[0]) as f:
return json.load(io.TextIOWrapper(f, encoding="utf-8"))
except Exception as e:
st.session_state['zip_error'] = str(e)


headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}


# 2) Repo contents API
try:
api = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}"
r = requests.get(api, headers=headers, timeout=30)
st.session_state['repo_contents_status'] = r.status_code
if r.status_code == 200:
b64 = r.json().get('content', '')
if b64:
import base64
data = base64.b64decode(b64)
return json.loads(data.decode('utf-8'))
except Exception as e:
st.session_state['repo_error'] = str(e)


# 3) Direkt URL override
if url_override:
r = requests.get(url_override, headers=headers if GITHUB_TOKEN else None, timeout=30)
st.session_state['geojson_url_status'] = r.status_code
if r.status_code == 200:
return r.json()


return {}


# GeoJSON seçenekleri
st.sidebar.header("Harita Sınırları (GeoJSON)")
st.sidebar.caption("Varsayılan: sf_cells.geojson (artifact veya repo içinde)")
st.session_state['geojson_path_ui'] = st.sidebar.text_input("GeoJSON dosya adı / yolu", value=st.secrets.get("geojson_path", "sf_cells.geojson"))
geojson_file = st.sidebar.file_uploader("(Opsiyonel) Yerel GeoJSON yükle", type=["json","geojson"])


geojson = load_geojson(geojson_file) if geojson_file else fetch_geojson_auto()


st.subheader(f"Harita — {sel_date}")
if not geojson:
st.warning("GeoJSON otomatik bulunamadı. Aşağıdaki teşhis bilgilerini kontrol edin veya dosya adını doğru girin.")
else:
enriched = inject_properties(geojson, one_day)
make_map(enriched)


with st.expander("Teşhis (GeoJSON)"):
st.write({
'geojson_path_attempted': st.session_state.get('geojson_path_ui') or st.secrets.get('geojson_path','sf_cells.geojson'),
'zip_error': st.session_state.get('zip_error'),
'zip_manifest_head': (st.session_state.get('zip_manifest') or [])[:20],
'repo_status': st.session_state.get('repo_contents_status'),
'repo_error': st.session_state.get('repo_error'),
'geojson_url_status': st.session_state.get('geojson_url_status'),
})


# Tablo + indirme
st.subheader("Seçilen Gün Tablosu")
st.dataframe(one_day.sort_values('risk_score_daily', ascending=False), use_container_width=True)


csv = one_day.drop(columns=['q25','q50','q75'], errors='ignore').to_csv(index=False).encode('utf-8')
st.download_button(
label="Günlük tabloyu CSV indir",
data=csv,
file_name=f"risk_daily_{sel_date}.csv",
mime="text/csv",
)


with st.expander("Nasıl çalışır?"):
st.markdown(
"""
- **Veri Kaynağı:** GitHub Actions artifact → **`sf-crime-parquet.zip`** (repo: `cem5113/crime_prediction_data`).
- **Kimlik Doğrulama:** Streamlit secrets içine `github_token` ekleyin (Actions → Artifacts okuma yetkisiyle).
- **Günlük Ortalama:** `hour_range` göz ardı edilerek aynı gün içindeki kayıtların ortalaması alınır.
- **4 Seviye:** Günlük dağılıma göre çeyrekler: `low (≤Q25)`, `medium (Q25–Q50)`, `high (Q50–Q75)`, `critical (>Q75)`.
- **Harita:** GeoJSON'daki **GEOID** eşleşir; renkler: low=yeşil, medium=sarı, high=turuncu, critical=kırmızı.
- **Yenile:** Sol menüden artifact'i yeniden çekebilirsiniz.
"""
)

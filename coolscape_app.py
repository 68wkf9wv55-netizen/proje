"""
CoolScape AI – v6 prototip (Siemens Türkiye Climathon)
Optimizasyon tabanlı karar destek: kısıtlı soğutma suyunu saat saat ısı stresine göre dağıtır.

Ne gerçek, ne vekil, ne temsili:
- Hava GERÇEK: Open-Meteo. Kayıtlı günler için geçmiş arşiv (ERA5 tabanlı), "Yarın" için tahmin API'si. Tek nokta;
  kıyı ile iç kesim arasındaki nem farkı modelde yok.
- Isı yüzeyi GERÇEK: Landsat 9 C2 L2 yüzey sıcaklığı (25.07.2025), H3 altıgenler (isi_yuzeyi_h3.csv).
- NDVI GERÇEK ama KISMİ: Sentinel-2 (24.07.2025); mevcut dosyada yalnızca kuzeydoğu köşedeki altıgenlerde var.
  NDVI olmayan yerde ağaç gölgesi 0 kabul edilir (ndvi_hazirla.py ile tüm alan için yeniden üretilebilir).
- Hassas tesisler GERÇEK: OpenStreetMap (tesisler_osm.json; yoksa Overpass API'den canlı).
- Parklar: 5 pilot park (koordinat gerçek, işletme değerleri TEMSİLİ) + OpenStreetMap park poligonları (parklar_osm.json;
  yoksa Overpass'tan canlı). OSM park alanı GERÇEK; sulanan pay, hat kapasitesi, pompa katsayısı TEMSİLİ.
- Nüfus GERÇEK: WorldPop 2025 toplam nüfus (nufus_h3.csv). 65+ sayısı VEKİL: toplam nüfus × ilçe 65+ oranı.
- Kırılganlık göstergesi VEKİL: 400 m içindeki OSM okul/kreş, sağlık, huzurevi sayısı.
- TEMSİLİ: hat kapasitesi, sulanan alan/pay, pompa katsayısı, kişi/saat erişim katsayıları, gölge etkinliği,
  saatlik yüzey sıcaklığı ölçeğinin sınırları, sisleme istasyonu alanı ve yayılım ölçekleri.
- Ekonomi: ASAT 2026 tarifesi VEKİL (parklara uygulandığı doğrulanmadı); arıtılmış su maliyeti ve yatırım kullanıcı girdisi.
- Soğutma eğrisi LİTERATÜR ÖLÇEKLİ (Hendel vd. 2016; Parison vd. 2020; Traill vd. 2024; Broadbent vd. 2018;
  Vanos vd. 2022). Antalya'da yerinde doğrulanmadı; doğrulama 2027 saha pilotunun işi.

Çalıştırma:  pip install -r requirements.txt  &&  python -m streamlit run coolscape_app.py
"""

import base64
import io
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import h3
import matplotlib
import numpy as np
import pandas as pd
import pydeck as pdk
import pulp
import requests
import streamlit as st
from pythermalcomfort.models import utci

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import folium  # noqa: E402
import branca.colormap as bcm  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

SURUM = "v6"

# ---------------------------------------------------------------------------
# SABİTLER VE VARSAYIMLAR
# ---------------------------------------------------------------------------
ANTALYA_LAT, ANTALYA_LON = 36.89, 30.71
SAATLER = list(range(8, 21))  # 08:00–20:00 karar ufku
GUNLER = {
    "25 Temmuz 2025 · kayıtlı sıcak hava dalgası (resmî ölçüm 46,1 °C)": ("arsiv", "2025-07-25"),
    "10 Ağustos 2025 · kayıtlı sıcak gün (MGM: Kepez 45,2 °C)": ("arsiv", "2025-08-10"),
    "Yarın · Open-Meteo tahmini": ("tahmin", None),
}
ISI_YUZEYI_CSV = "isi_yuzeyi_h3.csv"
NUFUS_CSV = "nufus_h3.csv"
PARK_JSON = "parklar_osm.json"   # overpass-turbo "ham veri" dışa aktarımı (leisure=park, out geom)
SU_ESIGI_C = 36.0   # bu yüzey sıcaklığının altındaki altıgenler deniz/su (Temmuz LST iki tepeli)
UTCI_ESIK = 32.0    # "güçlü ısı stresi" alt sınırı
OYNATMA_ADIM_S = 1.0
BBOX = (36.85, 30.62, 36.97, 30.80)  # güney, batı, kuzey, doğu

UTCI_SINIFLARI = [
    (-99, 26, "Isı stresi yok", (120, 144, 156)),
    (26, 32, "Orta", (254, 204, 92)),
    (32, 38, "Güçlü", (253, 141, 60)),
    (38, 46, "Çok güçlü", (240, 59, 32)),
    (46, 99, "Aşırı", (189, 0, 38)),
]

# Soğutma eğrisi (model sabiti; spec bölüm 5)
E_REF = 1.0
E_ARALIK = (0.5, 2.0)
VPD_DOYMA = 2.5
X0_MM = 0.5
X0_ARALIK = (0.5, 2.0)
KIRILIM = [0, 2, 4, 6, 8, 12, 16, 24, 32]
MIN_DOZ = 2.0
CEVRE_YARICAP_M = 500
YURUME_5DK_M = 400
YURUME_10DK_M = 800

# MRT bileşenleri
ZEMIN_GORUS = 0.5
SOLARCAL = {"sharp": 90, "f_svv": 0.5, "f_bes": 1.0, "asw": 0.7, "posture": "standing"}
# v6 – B1: ağaç gölgesi. Örtü = (NDVI − 0,2) / (0,6 − 0,2), 0–1 (TEMSİLİ doğrusal eşleme).
# Güneşin MRT katkısı (1 − GOLGE_ETKINLIK × örtü) ile çarpılır (TEMSİLİ: örtülü alanda doğrudan güneşin kesilen payı).
GOLGE_NDVI = (0.2, 0.6)
GOLGE_ETKINLIK = 0.8
# v6 – B3: yüzey sıcaklığı farkının saatlik ölçeği g = K↓(saat) / K↓(Landsat geçiş saati), [G_MIN, G_MAX] aralığında.
# Landsat ~11:20'de geçer; Open-Meteo saatlik ışınımı önceki saatin ortalaması → 12:00 değeri (11–12 ort.) referans.
LST_REF_SAAT = 12
LST_OLCEK_SINIR = (0.3, 1.3)   # TEMSİLİ: akşam depolanan ısı alt sınırı, öğle üst sınırı

# ---------------------------------------------------------------------------
# YÖNTEMLER
# ---------------------------------------------------------------------------
SISLEME_E = 4.4               # °C UTCI, Vanos vd. 2022 (sislenen alan ortalaması)
SISLEME_BIRIM_M2 = 100        # Vanos ölçümüne karşılık gelen birim alan (~10 × 10 m)
SISLEME_BIRIM_SU = 0.12       # m³/saat birim başına (VARSAYIM: ~20 nozul × 0,1 L/dk)
SISLEME_ALAN_M2 = 300         # v6 – B4: istasyon = 3 birim, ~300 m² meydan/durak (TEMSİLİ)
SISLEME_SU_M3_SAAT = SISLEME_BIRIM_SU * SISLEME_ALAN_M2 / SISLEME_BIRIM_M2
ISLATMA_ALAN_M2 = 3000        # VARSAYIM: ~300 m cadde × 10 m
ISLATMA_KAPASITE = 5.0        # m³/saat, bir arazöz (VARSAYIM)
# Yayılım çekirdeği: alan içi (yarıçap R = √(alan/π)) tam etki, dışında e^(−(d−R)/ölçek), R+kesmeden sonra 0.
# v6 – B4: noktasal değil, alan tabanlı. Ölçekler TEMSİLİ.
YONTEMLER = {
    # tür: (etiket, renk, ölçek m, kesme m)
    "park": ("Park sulama", (30, 136, 229), 150, 500),
    "sisleme": ("Sisleme istasyonu", (0, 188, 212), 10, 40),
    "islatma": ("Yol ıslatma", (149, 117, 205), 60, 200),
}
ADAY_SISLEME, ADAY_ISLATMA = 4, 3
ADAY_ARALIK_M = 1000
ADAY_PARK_UZAKLIK_M = (600, 3000)
ADAY_DOGU_SINIRI = 30.785

# Erişim (kişi/saat) – TEMSİLİ. Sisleme değeri 100 m² başına; istasyon alanıyla ölçeklenir.
ERISIM_VARSAYILAN = {"park_carpan": 50, "sisleme": 40, "islatma": 150}
SEHIR_KIRILGAN_YUZDE = 15

ON_AYARLAR = {
    "Dengeli": {"kirilgan_pay": 40, "w_k": 0.5},
    "Kırılgan nüfusa öncelik": {"kirilgan_pay": 60, "w_k": 1.5},
    "En çok toplam serinlik": {"kirilgan_pay": 0, "w_k": 0.0},
}

# ---------------------------------------------------------------------------
# PARKLAR
# ---------------------------------------------------------------------------
PARKLAR = pd.DataFrame({
    "park": ["Karaalioğlu Parkı", "Atatürk Parkı", "Düden Park", "Dokuma Park", "Tek Sarnıç Kent Parkı"],
    "ilce": ["Muratpaşa", "Muratpaşa", "Muratpaşa", "Kepez", "Kepez"],
    "lat": [36.87964, 36.88385, 36.85378, 36.91285, 36.90318],
    "lon": [30.70488, 30.68648, 30.78207, 30.67538, 30.63917],
    "yuzey_farki_c": [0.7, -0.3, -2.3, 2.7, 1.5],
    "yesil_skoru": [7, 6, 8, 4, 6],
    "nufus_skoru": [8, 7, 9, 5, 7],
    "hassas_yuzde": [20, 15, 25, 10, 18],
    "pompa_katsayisi": [1.2, 1.1, 1.5, 1.8, 1.3],
    "hat_kapasitesi_m3_saat": [15, 12, 20, 10, 15],
    "sulanan_alan_ha": [1.5, 1.2, 2.0, 1.0, 1.5],
    "aritilmis_su_erisimi": [False, False, True, True, False],
})
PARKLAR["tur"] = "park"
PARKLAR["pilot"] = True
PARKLAR["alan_ha"] = np.nan
PARKLAR["yaricap_m"] = 0.0
PARKLAR["park_kaynak"] = "pilot (koordinat Google Haritalar; işletme değerleri temsili)"

PARK_MIN_HA = 0.5          # bu alanın altındaki OSM parkları plana alınmaz
SULANAN_PAY = 0.6          # TEMSİLİ: park alanının sulanan payı
HAT_HA_BASINA = 10.0       # TEMSİLİ: m³/saat hat kapasitesi / sulanan ha
HAT_UST = 20.0             # TEMSİLİ: tek park hat kapasitesi üst sınırı
PILOT_YAKIN_M = 150        # pilot parka bu kadar yakın OSM poligonu aynı park sayılır
OSM_PARK_SAYISI = 10       # MILP'e alınacak OSM park sayısı (varsayılan)

# ---------------------------------------------------------------------------
# EKONOMİ
# ---------------------------------------------------------------------------
ASAT_TARIFE_TL_M3 = 33.31      # ASAT Haziran 2026, Resmi Daireler ve Belediyeler, kademesiz, KDV hariç (VEKİL)
ASAT_TARIFE_NOT = ("ASAT 2026 Haziran tarifesi, cetvel 3 \"Resmi Daireler ve Belediyeler\", kademesiz, KDV hariç; "
                   "her ay (ÜFE+TÜFE)/2 oranında artar. Parkların bu cetvelle faturalandığı doğrulanmadı (vekil).")
SULAMA_MM_GUN = 5.0            # TEMSİLİ: mevsim ortalaması günlük sulama (mm = L/m²)
SULAMA_GUN = 180               # TEMSİLİ: yıllık sulama günü


# ---------------------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------------------
HAVA_DEGISKENLERI = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,direct_normal_irradiance"


def _hava_tablosu(h):
    df = pd.DataFrame({
        "zaman": pd.to_datetime(h["time"]),
        "sicaklik_c": h["temperature_2m"], "bagil_nem": h["relative_humidity_2m"],
        "ruzgar_ms": h["wind_speed_10m"], "gunes_wm2": h["shortwave_radiation"],
    })
    if "direct_normal_irradiance" in h:
        df["dni_wm2"] = h["direct_normal_irradiance"]
    df["saat"] = df["zaman"].dt.hour
    return df


def _hava_istek(url, tarih):
    params = {"latitude": ANTALYA_LAT, "longitude": ANTALYA_LON, "start_date": tarih, "end_date": tarih,
              "hourly": HAVA_DEGISKENLERI, "wind_speed_unit": "ms", "timezone": "Europe/Istanbul"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return _hava_tablosu(r.json()["hourly"])


@st.cache_data(show_spinner=False)
def hava_verisi_cek(tarih: str) -> pd.DataFrame:
    return _hava_istek("https://archive-api.open-meteo.com/v1/archive", tarih)


@st.cache_data(show_spinner=False, ttl=3600)
def hava_tahmini_cek(tarih: str) -> pd.DataFrame:
    return _hava_istek("https://api.open-meteo.com/v1/forecast", tarih)


def csv_hava_verisi(dosya) -> pd.DataFrame:
    df = pd.read_csv(dosya)
    eksik = {"saat", "sicaklik_c", "bagil_nem", "ruzgar_ms", "gunes_wm2"} - set(df.columns)
    if eksik:
        raise ValueError(f"CSV'de eksik sütunlar: {', '.join(sorted(eksik))}")
    return df


def kiyi_siniflari(df):
    """Su ve kıyı karışık piksel maskesi: ≥2 su komşusu olan ya da 1 su komşusu olup iç kesim kara altıgenlerinin
    5. yüzdeliğinden serin olan altıgenler karışık sayılır ve analiz dışı bırakılır."""
    su = set(df.loc[df["lst_c"] < SU_ESIGI_C, "hex"])
    kara = df[df["lst_c"] >= SU_ESIGI_C]
    komsu = np.array([sum(n in su for n in h3.grid_ring(h, 1)) for h in kara["hex"]])
    ic = kara["lst_c"].values[komsu == 0]
    p5 = float(np.percentile(ic, 5)) if len(ic) else -np.inf
    karisik = (komsu >= 2) | ((komsu == 1) & (kara["lst_c"].values < p5))
    return su, set(kara["hex"].values[karisik])


def ndvi_ortu(ndvi):
    """NDVI → ağaç/bitki örtüsü payı 0–1 (TEMSİLİ doğrusal eşleme). NDVI yoksa 0."""
    a, b = GOLGE_NDVI
    return np.clip((np.nan_to_num(np.asarray(ndvi, dtype=float), nan=a) - a) / (b - a), 0, 1)


@st.cache_data(show_spinner=False)
def isi_yuzeyi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    df = pd.read_csv(yol)
    if not {"hex", "lst_c", "lat", "lon"} <= set(df.columns):
        return None
    df = df.dropna(subset=["lst_c"])
    su, karisik = kiyi_siniflari(df)
    df = df[~df["hex"].isin(su | karisik)].reset_index(drop=True)
    df["lst_fark_c"] = df["lst_c"] - df["lst_c"].median()
    if "ndvi" not in df:
        df["ndvi"] = np.nan
    df["ndvi_var"] = df["ndvi"].notna()
    df["golge_ortu"] = ndvi_ortu(df["ndvi"])
    df.attrs["su_altigen"] = len(su)
    df.attrs["kiyi_altigen"] = len(karisik)
    df.attrs["ndvi_pay"] = float(df["ndvi_var"].mean())
    return df


OSM_KATEGORILER = {
    "Okul / kreş": (r'["amenity"~"^(school|kindergarten)$"]', (245, 245, 245)),
    "Sağlık": (r'["amenity"~"^(hospital|clinic)$"]', (240, 98, 146)),
    "Huzurevi": (r'["social_facility"="nursing_home"]', (174, 213, 129)),
    "İçme suyu": (r'["amenity"="drinking_water"]', (178, 235, 242)),
}
OVERPASS_ADRESLERI = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
                      "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                      "https://overpass.kumi.systems/api/interpreter"]
TESIS_CSV = "tesisler_osm.csv"
TESIS_JSON = "tesisler_osm.json"


def _tesis_kategori(t):
    if t.get("amenity") in ("school", "kindergarten"):
        return "Okul / kreş"
    if t.get("amenity") in ("hospital", "clinic"):
        return "Sağlık"
    if t.get("social_facility") == "nursing_home" or t.get("amenity") == "nursing_home":
        return "Huzurevi"
    if t.get("amenity") == "drinking_water":
        return "İçme suyu"
    return ""


def _tesis_satirlari(elements):
    satirlar = []
    for e in elements:
        t = e.get("tags", {})
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        kat = _tesis_kategori(t)
        if lat is None or lon is None or not kat:
            continue
        satirlar.append({"kategori": kat, "ad": t.get("name", kat), "lat": lat, "lon": lon})
    return pd.DataFrame(satirlar, columns=["kategori", "ad", "lat", "lon"])


@st.cache_data(show_spinner=False)
def tesis_dosyasi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    if yol.lower().endswith(".json"):
        with open(yol, encoding="utf-8") as f:
            df = _tesis_satirlari(json.load(f).get("elements", []))
    else:
        t = pd.read_csv(yol, sep=None, engine="python")
        if {"@lat", "@lon"} <= set(t.columns):
            t = t.rename(columns={"@lat": "lat", "@lon": "lon"})
            for c in ("amenity", "social_facility", "name"):
                if c not in t:
                    t[c] = None
            t["kategori"] = [_tesis_kategori({"amenity": a, "social_facility": s})
                             for a, s in zip(t["amenity"], t["social_facility"])]
            t["ad"] = t["name"].fillna(t["kategori"])
        df = t
    if df is None or not {"kategori", "lat", "lon"} <= set(df.columns):
        return None
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["kategori"].isin(list(OSM_KATEGORILER))].copy()
    if "ad" not in df:
        df["ad"] = df["kategori"]
    return df[["kategori", "ad", "lat", "lon"]].reset_index(drop=True) if len(df) else None


def _overpass(sorgu):
    son_hata = None
    for adres in OVERPASS_ADRESLERI:
        try:
            r = requests.post(adres, data={"data": sorgu}, timeout=60,
                              headers={"User-Agent": f"CoolScape-AI-prototip/{SURUM} (Climathon)",
                                       "Accept": "application/json"})
            r.raise_for_status()
            return r.json().get("elements", []), None
        except Exception as hata:
            son_hata = f"{adres.split('/')[2]}: {type(hata).__name__}"
    return None, str(son_hata)


@st.cache_data(show_spinner=False, ttl=86400)
def osm_tesisleri_cek():
    """OpenStreetMap'ten pilot bölgedeki hassas tesisler. Erişilemezse (None, hata); uydurma veri yok."""
    g, b, k, d = BBOX
    parcalar = [f"nwr{etiket}({g},{b},{k},{d});" for etiket, _ in OSM_KATEGORILER.values()]
    parcalar.append(f'nwr["amenity"="nursing_home"]({g},{b},{k},{d});')
    elemanlar, hata = _overpass(f"[out:json][timeout:25];({''.join(parcalar)});out center tags;")
    if elemanlar is None:
        return None, hata
    return _tesis_satirlari(elemanlar).drop_duplicates(), None


# ---------------------------------------------------------------------------
# OSM PARKLARI (B2): poligon alanı ve ağırlık merkezi
# ---------------------------------------------------------------------------
def poligon_alan_merkez(halka):
    """halka: [(lat, lon), ...]. Yerel düzlem yaklaşımıyla alan (m²) ve ağırlık merkezi (lat, lon)."""
    p = np.asarray(halka, dtype=float)
    if len(p) < 3:
        return 0.0, None
    lat0, lon0 = p[:, 0].mean(), p[:, 1].mean()
    kx, ky = 111_320 * np.cos(np.radians(lat0)), 110_540
    x, y = (p[:, 1] - lon0) * kx, (p[:, 0] - lat0) * ky
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    capraz = x * y2 - x2 * y
    a = 0.5 * capraz.sum()
    if abs(a) < 1e-9:
        return 0.0, None
    cx = ((x + x2) * capraz).sum() / (6 * a)
    cy = ((y + y2) * capraz).sum() / (6 * a)
    return float(abs(a)), (float(lat0 + cy / ky), float(lon0 + cx / kx))


def _park_satiri(e):
    t = e.get("tags", {})
    halkalar = []
    if e.get("type") == "way" and e.get("geometry"):
        halkalar.append([(g["lat"], g["lon"]) for g in e["geometry"]])
    elif e.get("type") == "relation":
        for m in e.get("members", []):
            if m.get("role") in ("outer", "") and m.get("geometry"):
                halkalar.append([(g["lat"], g["lon"]) for g in m["geometry"]])
    toplam, mx, my = 0.0, 0.0, 0.0
    for h in halkalar:
        a, c = poligon_alan_merkez(h)
        if c is None:
            continue
        toplam += a
        mx += a * c[0]
        my += a * c[1]
    if toplam <= 0:
        return None
    return {"osm_id": f"{e.get('type', '?')[0]}{e.get('id')}", "ad": t.get("name"),
            "lat": mx / toplam, "lon": my / toplam, "alan_ha": toplam / 10_000}


def osm_park_tablosu(elemanlar):
    satirlar = [s for s in (_park_satiri(e) for e in elemanlar
                            if e.get("tags", {}).get("leisure") == "park") if s]
    df = pd.DataFrame(satirlar, columns=["osm_id", "ad", "lat", "lon", "alan_ha"])
    g, b, k, d = BBOX
    return df[(df["lat"].between(g, k)) & (df["lon"].between(b, d))].drop_duplicates("osm_id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def park_dosyasi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    with open(yol, encoding="utf-8") as f:
        df = osm_park_tablosu(json.load(f).get("elements", []))
    return df if len(df) else None


@st.cache_data(show_spinner=False, ttl=86400)
def osm_parklari_cek():
    g, b, k, d = BBOX
    sorgu = (f'[out:json][timeout:60];(way["leisure"="park"]({g},{b},{k},{d});'
             f'relation["leisure"="park"]({g},{b},{k},{d}););out geom tags;')
    elemanlar, hata = _overpass(sorgu)
    if elemanlar is None:
        return None, hata
    df = osm_park_tablosu(elemanlar)
    return (df if len(df) else None), None


def osm_parklari_hazirla(osm, pilot):
    """OSM poligonlarını model satırlarına çevirir. Pilot parka 150 m'den yakın poligon pilot parkın kendisi sayılır:
    pilot satırına gerçek alanı yazılır, ayrıca eklenmez."""
    pilot = pilot.copy()
    if osm is None or not len(osm):
        return pilot, pilot.iloc[0:0].copy()
    osm = osm.copy()
    for i, r in pilot.iterrows():
        d = mesafe_m(r["lat"], r["lon"], osm["lat"].values, osm["lon"].values)
        if d.min() <= PILOT_YAKIN_M:
            j = int(np.argmin(d))
            pilot.at[i, "alan_ha"] = round(float(osm.iloc[j]["alan_ha"]), 2)
            osm = osm.drop(osm.index[j])
    osm = osm[osm["alan_ha"] >= PARK_MIN_HA].reset_index(drop=True)
    sulanan = osm["alan_ha"] * SULANAN_PAY
    yeni = pd.DataFrame({
        "park": [a if isinstance(a, str) and a.strip() else f"Park {i}" for a, i in zip(osm["ad"], osm["osm_id"])],
        "ilce": "–", "lat": osm["lat"], "lon": osm["lon"], "yuzey_farki_c": 0.0, "yesil_skoru": 5.0,
        "nufus_skoru": 5.0, "hassas_yuzde": float(SEHIR_KIRILGAN_YUZDE), "pompa_katsayisi": 1.0,
        "hat_kapasitesi_m3_saat": np.minimum(HAT_HA_BASINA * sulanan, HAT_UST).round(1),
        "sulanan_alan_ha": sulanan.round(2), "aritilmis_su_erisimi": False, "tur": "park", "pilot": False,
        "alan_ha": osm["alan_ha"].round(2), "yaricap_m": np.sqrt(osm["alan_ha"] * 10_000 / np.pi).round(0),
        "park_kaynak": "OpenStreetMap poligonu (alan gerçek; sulanan pay, hat, pompa temsili)"})
    ad_say = yeni.groupby("park").cumcount()
    yeni["park"] = [a if n == 0 else f"{a} ({n + 1})" for a, n in zip(yeni["park"], ad_say)]
    yeni = yeni[~yeni["park"].isin(set(pilot["park"]))]
    return pilot, yeni.reset_index(drop=True)


def parklari_guncelle(parklar, yuzey):
    """Isı yüzeyi varsa: yüzey sıcaklığı farkı parkın ~500 m çevresinden; yeşil skoru ve gölge örtüsü parkın
    kendisinden (en yakın altıgenler, yalnızca NDVI olanlar)."""
    p = parklar.copy()
    p[["yuzey_farki_c", "yesil_skoru"]] = p[["yuzey_farki_c", "yesil_skoru"]].astype(float)
    p["yuzey_kaynak"] = "temsili"
    p["yesil_kaynak"] = "temsili"
    p["golge_ortu"] = 0.0
    p["golge_kaynak"] = "NDVI yok: gölge 0"
    if yuzey is None or not len(p):
        return p
    cozunurluk = h3.get_resolution(yuzey["hex"].iloc[0])
    aralik_m = h3.average_hexagon_edge_length(cozunurluk, unit="m") * 3 ** 0.5
    k_cevre = max(1, int(round(CEVRE_YARICAP_M / aralik_m)))
    idx = yuzey.set_index("hex")
    for i, r in p.iterrows():
        merkez = h3.latlng_to_cell(r["lat"], r["lon"], cozunurluk)
        cevre = [c for c in h3.grid_disk(merkez, k_cevre) if c in idx.index]
        k_ic = max(1, int(round(float(r.get("yaricap_m", 0) or 0) / aralik_m)))
        yakin = [c for c in h3.grid_disk(merkez, k_ic) if c in idx.index]
        if cevre:
            p.at[i, "yuzey_farki_c"] = round(float(idx.loc[cevre, "lst_fark_c"].mean()), 2)
            p.at[i, "yuzey_kaynak"] = f"Landsat (~{CEVRE_YARICAP_M} m çevre)"
        if yakin and idx.loc[yakin, "ndvi"].notna().any():
            ndvi = float(idx.loc[yakin, "ndvi"].mean())
            p.at[i, "yesil_skoru"] = round(float(np.clip(1 + ndvi / 0.8 * 9, 1, 10)), 1)
            p.at[i, "yesil_kaynak"] = "Sentinel-2 NDVI"
            p.at[i, "golge_ortu"] = round(float(ndvi_ortu([ndvi])[0]), 3)
            p.at[i, "golge_kaynak"] = "Sentinel-2 NDVI (örtü eşlemesi temsili)"
    return p


# ---------------------------------------------------------------------------
# MEKÂNSAL ÖNCELİK
# ---------------------------------------------------------------------------
ONCELIK_AGIRLIK = {"c_isi": 0.40, "c_yasli": 0.25, "c_park": 0.20, "c_tesis": 0.15}
ONCELIK_OLCUT_AD = {"c_isi": "yüzey ısısı", "c_yasli": "65+ sayısı (ilçe oranıyla)", "c_park": "parka uzaklık",
                    "c_tesis": "okul ve sağlık tesisi"}
NUFUS_YERLESIM_ESIK = 50
PARK_NUFUS_YARICAP_M = 800
SINIF_ADLARI = ["Çok düşük", "Düşük", "Orta", "Yüksek", "Çok yüksek"]
SINIF_RENK = {0: (205, 205, 205), 1: (255, 247, 188), 2: (254, 227, 145), 3: (254, 196, 79),
              4: (236, 112, 20), 5: (153, 0, 13)}
YERLESIM_YARICAP_M = 1000


@st.cache_data(show_spinner=False)
def nufus_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    df = pd.read_csv(yol)
    if not {"hex", "nufus", "yasli_65"} <= set(df.columns):
        return None
    return df[["hex", "nufus", "yasli_65"]]


def _yakin_say(lat, lon, t_lat, t_lon, yaricap):
    kx = 111_320 * np.cos(np.radians(36.89))
    dx = (lon[:, None] - t_lon[None, :]) * kx
    dy = (lat[:, None] - t_lat[None, :]) * 110_540
    return (np.hypot(dx, dy) <= yaricap).sum(axis=1)


def park_kenar_uzakligi(parklar, lat, lon):
    """Her altıgenden en yakın park kenarına uzaklık: merkez uzaklığı − eşdeğer yarıçap (≥0)."""
    if not len(parklar):
        return np.full(len(lat), np.inf)
    return np.min([np.maximum(mesafe_m(r["lat"], r["lon"], lat, lon) - float(r.get("yaricap_m", 0) or 0), 0)
                   for _, r in parklar.iterrows()], axis=0)


@st.cache_data(show_spinner=False)
def oncelik_hesapla(yuzey, parklar, tesisler, nufus=None):
    """Her kara altıgeni için soğutma önceliği (ağırlıklı çakıştırma). Parka uzaklık tüm parklardan (OSM varsa),
    pilot park erişimi ayrıca raporlanır."""
    y = yuzey.copy()
    oznitelik = dict(yuzey.attrs)
    if nufus is not None:
        y = y.drop(columns=[c for c in ("nufus", "yasli_65") if c in y]).merge(nufus, on="hex", how="left")
        y[["nufus", "yasli_65"]] = y[["nufus", "yasli_65"]].fillna(0.0)
    y.attrs.update(oznitelik)
    lat, lon = y["lat"].values, y["lon"].values
    y["d_park_m"] = park_kenar_uzakligi(parklar, lat, lon)
    pilot = parklar[parklar["pilot"]] if "pilot" in parklar else parklar
    y["d_pilot_m"] = park_kenar_uzakligi(pilot, lat, lon)
    p5, p95 = np.nanpercentile(y["lst_fark_c"], [5, 95])
    y["c_isi"] = np.clip((y["lst_fark_c"] - p5) / max(p95 - p5, 1e-6), 0, 1)
    y["c_park"] = np.clip((y["d_park_m"] - 400) / 1100, 0, 1)
    agirlik = dict(ONCELIK_AGIRLIK)
    hassas = None if tesisler is None else tesisler[tesisler["kategori"].isin(["Okul / kreş", "Sağlık", "Huzurevi"])]
    if hassas is not None and len(hassas):
        y["n_tesis_400"] = _yakin_say(lat, lon, hassas["lat"].values, hassas["lon"].values, 400)
        y["n_tesis_1km"] = _yakin_say(lat, lon, hassas["lat"].values, hassas["lon"].values, YERLESIM_YARICAP_M)
        y["c_tesis"] = np.clip(y["n_tesis_400"] / max(np.percentile(y["n_tesis_400"], 95), 1), 0, 1)
        y["yerlesim"] = y["n_tesis_1km"] >= 1
        kural = f"{YERLESIM_YARICAP_M} m içinde en az bir okul veya sağlık tesisi (nüfus verisi yerine vekil)"
    else:
        y["n_tesis_400"], y["n_tesis_1km"], y["c_tesis"] = 0, 0, 0.0
        agirlik.pop("c_tesis")
        y["yerlesim"] = True
        kural = "tesis verisi yok; tüm kara alanı"
    if "nufus" in y:
        y["c_yasli"] = np.clip(y["yasli_65"] / max(np.percentile(y["yasli_65"], 95), 1e-6), 0, 1)
        y["yerlesim"] = y["nufus"] >= NUFUS_YERLESIM_ESIK
        kural = f"WorldPop 2025: altıgende en az {NUFUS_YERLESIM_ESIK} kişi"
    else:
        y["c_yasli"] = 0.0
        agirlik.pop("c_yasli")
    y["oncelik"] = sum(y[k] * w for k, w in agirlik.items()) / sum(agirlik.values())
    y["sinif_no"] = 0
    yer = y["yerlesim"]
    if yer.sum() >= 5:
        y.loc[yer, "sinif_no"] = pd.qcut(y.loc[yer, "oncelik"].rank(method="first"), 5,
                                         labels=[1, 2, 3, 4, 5]).astype(int)
    y["sinif"] = y["sinif_no"].map(lambda i: SINIF_ADLARI[i - 1] if i > 0 else "Seyrek yerleşim")
    alan = float(np.mean([h3.cell_area(h, unit="km^2") for h in y["hex"].iloc[:50]]))
    yuksek = y["sinif_no"] >= 4

    def erisim(sutun):
        return {
            "park400_pct": float(100 * (y.loc[yer, sutun] <= 400).mean()) if yer.any() else 0.0,
            "yuksek_park_disi_pct": float(100 * (y.loc[yuksek, sutun] > 400).mean()) if yuksek.any() else 0.0,
            "park400_nufus_pct": (float(100 * y.loc[yer & (y[sutun] <= 400), "nufus"].sum()
                                        / max(y.loc[yer, "nufus"].sum(), 1)) if "nufus" in y else None)}
    bilgi = {
        "agirlik": " · ".join(f"{ONCELIK_OLCUT_AD[k]} {v:.2f}".replace(".", ",") for k, v in agirlik.items()),
        "kural": kural, "yerlesim_km2": float(yer.sum() * alan), "yuksek_km2": float(yuksek.sum() * alan),
        "nufus_var": "nufus" in y,
        "yerlesim_nufus": float(y.loc[yer, "nufus"].sum()) if "nufus" in y else None,
        "pilot": erisim("d_pilot_m"), "tum": erisim("d_park_m"),
        "park_sayisi": int(len(parklar)), "pilot_sayisi": int(len(pilot)),
    }
    return y, bilgi


def park_nufus_guncelle(parklar, yuzey):
    """Parkın 800 m içindeki nüfus (WorldPop) → nüfus skoru 1–10 (parklar arasında göreli). Yoksa OSM vekili."""
    p = parklar.copy()
    p["nufus_kaynak"] = "temsili"
    p[["nufus_skoru", "hassas_yuzde"]] = p[["nufus_skoru", "hassas_yuzde"]].astype(float)
    p["nufus_800"] = np.nan
    if yuzey is None or not len(p):
        return p
    lat, lon = yuzey["lat"].values, yuzey["lon"].values
    olcu, kaynak = [], None
    for _, r in p.iterrows():
        icinde = mesafe_m(r["lat"], r["lon"], lat, lon) <= PARK_NUFUS_YARICAP_M
        if "nufus" in yuzey:
            olcu.append(float(yuzey.loc[icinde, "nufus"].sum()))
            kaynak = f"WorldPop 2025 ({PARK_NUFUS_YARICAP_M} m içi nüfus)"
        elif "n_tesis_1km" in yuzey and yuzey["n_tesis_1km"].sum() > 0:
            olcu.append(float(yuzey.loc[icinde, "n_tesis_400"].sum()))
            kaynak = f"OSM okul/sağlık yoğunluğu ({PARK_NUFUS_YARICAP_M} m, vekil)"
    if kaynak and olcu and max(olcu) > 0:
        p["nufus_skoru"] = [round(1 + 9 * o / max(olcu), 1) for o in olcu]
        p["nufus_800"] = olcu
        p["nufus_kaynak"] = kaynak
    return p


def park_sec(parklar, yuzey, n_osm):
    """MILP'e alınacak parklar: pilot parkların tümü + 800 m içindeki (nüfus × öncelik) toplamı en yüksek n OSM parkı."""
    pilot = parklar[parklar["pilot"]]
    osm = parklar[~parklar["pilot"]].copy()
    if not len(osm) or n_osm <= 0 or yuzey is None or "oncelik" not in yuzey:
        return pilot.reset_index(drop=True)
    agirlik = yuzey["oncelik"].values * (yuzey["nufus"].values if "nufus" in yuzey else 1.0)
    osm["secim_skoru"] = [float(agirlik[mesafe_m(r["lat"], r["lon"], yuzey["lat"].values, yuzey["lon"].values)
                                        <= PARK_NUFUS_YARICAP_M].sum()) for _, r in osm.iterrows()]
    secilen = osm.sort_values("secim_skoru", ascending=False).head(n_osm).drop(columns="secim_skoru")
    return pd.concat([pilot, secilen], ignore_index=True)


def kirilganlik_ata(noktalar, tesisler):
    """Tüm müdahale noktalarına aynı kuralla kırılganlık göstergesi (0–1): 400 m içindeki okul/kreş, sağlık,
    huzurevi sayısı / noktalar arasındaki en yüksek değer (vekil). Tesis yoksa temsili kırılgan % / 30."""
    p = noktalar.copy()
    hassas = None if tesisler is None else tesisler[tesisler["kategori"].isin(["Okul / kreş", "Sağlık", "Huzurevi"])]
    if hassas is not None and len(hassas):
        n = np.array([int((mesafe_m(r["lat"], r["lon"], hassas["lat"].values, hassas["lon"].values) <= 400).sum())
                      for _, r in p.iterrows()])
        p["hassas_tesis_400"] = n
        p["kirilgan_endeks"] = n / max(int(n.max()), 1)
        kaynak = "OSM: 400 m içinde okul/kreş, sağlık, huzurevi sayısı (vekil)"
    else:
        p["kirilgan_endeks"] = np.clip(p["hassas_yuzde"].astype(float) / 30.0, 0, 1)
        kaynak = "temsili kırılgan %"
    p["kirilgan_endeks"] = p["kirilgan_endeks"].round(3)
    p["kirilgan_kaynak"] = kaynak
    esik = float(p["kirilgan_endeks"].median())
    kirilgan = tuple(sorted(p.loc[(p["kirilgan_endeks"] >= esik) & (p["kirilgan_endeks"] > 0), "park"]))
    return p, kirilgan, kaynak


def aday_noktalar(yuzey, parklar):
    """Sisleme ve ıslatma ADAY konumları: en öncelikli yerleşim altıgenlerinden, birbirinden ve parklardan ≥1 km."""
    if yuzey is None:
        return parklar.iloc[0:0].copy()
    y = yuzey[yuzey["lon"] <= ADAY_DOGU_SINIRI].copy()
    if "oncelik" in y.columns:
        y = y[y["yerlesim"]]
    y["d_park"] = park_kenar_uzakligi(parklar, y["lat"].values, y["lon"].values)
    y = y[(y["d_park"] >= ADAY_PARK_UZAKLIK_M[0]) & (y["d_park"] <= ADAY_PARK_UZAKLIK_M[1])]
    y = y.sort_values("oncelik" if "oncelik" in y.columns else "lst_c", ascending=False)
    secilen = []
    for _, r in y.iterrows():
        if all(mesafe_m(r["lat"], r["lon"], a["lat"], a["lon"]) >= ADAY_ARALIK_M for a in secilen):
            secilen.append(r)
        if len(secilen) >= ADAY_SISLEME + ADAY_ISLATMA:
            break
    satirlar = []
    for i, r in enumerate(secilen):
        tur = "sisleme" if i < ADAY_SISLEME else "islatma"
        no = i + 1 if tur == "sisleme" else i - ADAY_SISLEME + 1
        alan = SISLEME_ALAN_M2 if tur == "sisleme" else ISLATMA_ALAN_M2
        satirlar.append({
            "park": f"{'Sisleme' if tur == 'sisleme' else 'Islatma'} {no} (aday)", "ilce": "–",
            "lat": r["lat"], "lon": r["lon"], "yuzey_farki_c": round(float(r["lst_fark_c"]), 2),
            "yesil_skoru": 0.0, "nufus_skoru": 0.0, "hassas_yuzde": float(SEHIR_KIRILGAN_YUZDE), "pompa_katsayisi": 1.0,
            "hat_kapasitesi_m3_saat": SISLEME_SU_M3_SAAT if tur == "sisleme" else ISLATMA_KAPASITE,
            "sulanan_alan_ha": alan / 10_000, "aritilmis_su_erisimi": tur == "islatma", "tur": tur, "pilot": False,
            "alan_ha": alan / 10_000, "yaricap_m": round(float(np.sqrt(alan / np.pi)), 1),
            "park_kaynak": "aday (sıcak nokta analizi)", "yuzey_kaynak": "Landsat (altıgenin kendisi)",
            "yesil_kaynak": "–", "golge_ortu": round(float(r.get("golge_ortu", 0.0)), 3),
            "golge_kaynak": "Sentinel-2 NDVI" if bool(r.get("ndvi_var", False)) else "NDVI yok: gölge 0"})
    return pd.DataFrame(satirlar)


# ---------------------------------------------------------------------------
# FİZİK
# ---------------------------------------------------------------------------
def vpd_kpa(sicaklik_c, bagil_nem):
    t = np.asarray(sicaklik_c, dtype=float)
    rh = np.asarray(bagil_nem, dtype=float)
    try:
        from metpy.calc import saturation_vapor_pressure
        from metpy.units import units
        es = saturation_vapor_pressure(t * units.degC).to("kPa").magnitude
    except Exception:
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh / 100.0)


def gunes_yuksekligi(zaman, lat=ANTALYA_LAT, lon=ANTALYA_LON):
    """Güneş yüksekliği (°), NOAA yaklaşımı. Saatlik ışınım önceki saatin ortalaması olduğundan saat ortası alınır."""
    t = pd.to_datetime(pd.Series(zaman)) - pd.Timedelta(hours=3, minutes=30)
    doy = t.dt.dayofyear.values
    saat = t.dt.hour.values + t.dt.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (saat - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    dek = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((saat * 60 + eqt + 4 * lon) / 4 - 180)
    f = np.radians(lat)
    return np.degrees(np.arcsin(np.clip(np.sin(f) * np.sin(dek) + np.cos(f) * np.cos(dek) * np.cos(ha), -1, 1)))


def gunes_mrt_ekle(hava):
    """Her saat için güneşin MRT katkısı (dmrt_gunes, °C) ve yüzey sıcaklığı farkının saatlik ölçeği (lst_olcek)."""
    h = hava.copy()
    if "dni_wm2" in h and h["dni_wm2"].notna().all() and "zaman" in h:
        from pythermalcomfort.models import solar_gain
        alt_ = gunes_yuksekligi(h["zaman"])
        d = []
        for a, dni in zip(alt_, h["dni_wm2"]):
            if a <= 0 or dni <= 0:
                d.append(0.0)
                continue
            r = solar_gain(sol_altitude=float(min(a, 90)), sharp=SOLARCAL["sharp"], sol_radiation_dir=float(dni),
                           sol_transmittance=1.0, f_svv=SOLARCAL["f_svv"], f_bes=SOLARCAL["f_bes"],
                           asw=SOLARCAL["asw"], posture=SOLARCAL["posture"])
            d.append(float(r["delta_mrt"] if isinstance(r, dict) else r.delta_mrt))
        h["gunes_yuksekligi"] = np.round(alt_, 1)
        h["dmrt_gunes"] = d
        yontem = "SolarCal (ASHRAE 55)"
    else:
        h["dmrt_gunes"] = 0.02 * h["gunes_wm2"]
        yontem = "kaba yaklaşım (0,02 × küresel ışınım)"
    ref = h.loc[h["saat"] == LST_REF_SAAT, "gunes_wm2"]
    ref = float(ref.iloc[0]) if len(ref) and float(ref.iloc[0]) > 0 else float(h["gunes_wm2"].max() or 1.0)
    h["lst_olcek"] = np.clip(h["gunes_wm2"].astype(float) / max(ref, 1.0), *LST_OLCEK_SINIR)
    h.attrs["mrt_yontem"] = yontem
    return h


def mrt_hesapla(t, dmrt_gunes, golge_ortu, lst_fark, lst_olcek):
    """v6 MRT = T + (1 − GOLGE_ETKINLIK × örtü) × ΔMRT_güneş + 0,5 × LST farkı × g(saat)."""
    return t + (1 - GOLGE_ETKINLIK * np.asarray(golge_ortu, dtype=float)) * dmrt_gunes \
        + ZEMIN_GORUS * np.asarray(lst_fark, dtype=float) * lst_olcek


def utci_hesapla(tdb, mrt, ruzgar, nem):
    tdb = np.asarray(tdb, dtype=float)
    mrt = np.clip(np.asarray(mrt, dtype=float), tdb - 30, tdb + 70)
    v = np.clip(np.asarray(ruzgar, dtype=float), 0.5, 17.0)
    sonuc = utci(tdb=tdb, tr=mrt, v=v, rh=np.asarray(nem, dtype=float))
    return np.asarray(getattr(sonuc, "utci", sonuc), dtype=float)


def park_saat_tablosu(hava, parklar):
    h = hava[hava["saat"].isin(SAATLER)]
    parcalar = []
    for _, p in parklar.iterrows():
        d = h.copy()
        d["park"] = p["park"]
        d["mrt_c"] = mrt_hesapla(d["sicaklik_c"], d["dmrt_gunes"], p.get("golge_ortu", 0.0), p["yuzey_farki_c"],
                                 d["lst_olcek"])
        parcalar.append(d)
    df = pd.concat(parcalar, ignore_index=True)
    df["utci_c"] = utci_hesapla(df["sicaklik_c"], df["mrt_c"], df["ruzgar_ms"], df["bagil_nem"])
    df["vpd_kpa"] = vpd_kpa(df["sicaklik_c"], df["bagil_nem"])
    df["tehlike"] = (df["utci_c"] - UTCI_ESIK).clip(lower=0).fillna(0)
    return df


def yuzey_utci(yuzey, hava_satiri):
    t = hava_satiri["sicaklik_c"]
    n = len(yuzey)
    mrt = mrt_hesapla(t, hava_satiri["dmrt_gunes"], yuzey["golge_ortu"].values, yuzey["lst_fark_c"].values,
                      hava_satiri["lst_olcek"])
    return utci_hesapla(np.full(n, t), mrt, np.full(n, hava_satiri["ruzgar_ms"]), np.full(n, hava_satiri["bagil_nem"]))


def kent_utci(satir):
    """Kent geneli referans: açık güneşte, ortanca yüzey (gölge ve yüzey farkı yok)."""
    return float(utci_hesapla([satir["sicaklik_c"]], [satir["sicaklik_c"] + satir["dmrt_gunes"]],
                              [satir["ruzgar_ms"]], [satir["bagil_nem"]])[0])


def utci_sinif(deger):
    for alt_s, ust_s, ad, renk in UTCI_SINIFLARI:
        if alt_s <= deger < ust_s:
            return ad, renk
    return UTCI_SINIFLARI[-1][2], UTCI_SINIFLARI[-1][3]


# ---------------------------------------------------------------------------
# OPTİMİZASYON (MILP)
# ---------------------------------------------------------------------------
def sogutma_egrisi(x_m3, emax, x0):
    return emax * (1 - np.exp(-np.asarray(x_m3, dtype=float) / np.asarray(x0, dtype=float)))


def girdileri_hazirla(tablo, noktalar, w_kirilganlik, erisim):
    p = noktalar.set_index("park")
    t = tablo.copy()
    t["tur"] = t["park"].map(p["tur"])
    sisleme_kisi = erisim["sisleme"] * SISLEME_ALAN_M2 / SISLEME_BIRIM_M2
    t["kisi"] = np.select([t["tur"] == "park", t["tur"] == "sisleme"],
                          [t["park"].map(p["nufus_skoru"]) * erisim["park_carpan"], sisleme_kisi],
                          default=erisim["islatma"]).astype(float)
    endeks = p["kirilgan_endeks"] if "kirilgan_endeks" in p else p["hassas_yuzde"] / 30.0
    t["oncelik"] = t["tehlike"] * t["kisi"] / 100.0 * (1 + w_kirilganlik * t["park"].map(endeks))
    vpd_f = np.minimum(t["vpd_kpa"] / VPD_DOYMA, 1.0)
    t["emax"] = np.where(t["tur"] == "sisleme", SISLEME_E, E_REF) * vpd_f
    t["alan_m2"] = t["park"].map(p["sulanan_alan_ha"]) * 10_000
    t["x0"] = np.where(t["tur"] == "sisleme", SISLEME_SU_M3_SAAT / 10.0, X0_MM * t["alan_m2"] / 1000.0)
    t["kapasite"] = t["park"].map(p["hat_kapasitesi_m3_saat"]).astype(float)
    t["pompa"] = t["park"].map(p["pompa_katsayisi"]).astype(float)
    return t


def degisken(model, ad, alt_s=None, ust_s=None, tur="Continuous"):
    if hasattr(model, "add_variable"):
        return model.add_variable(ad, lowBound=alt_s, upBound=ust_s, cat=tur)
    return pulp.LpVariable(ad, lowBound=alt_s, upBound=ust_s, cat=tur)


def cozucu_sec():
    for ad in ("PULP_CBC_CMD", "HiGHS", "HiGHS_CMD", "COIN_CMD"):
        sinif = getattr(pulp, ad, None)
        if sinif is None:
            continue
        try:
            c = sinif(msg=False)
            if c.available():
                return c
        except Exception:
            continue
    raise RuntimeError("MILP çözücüsü bulunamadı (pulp==2.9.0 kurulu olmalı).")


TUM_TURLER = ("park", "sisleme", "islatma")


def optimize_et(t, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0_carpan=1.0,
                turler=TUM_TURLER, park_kesme=18, park_gunluk_mm=6.0):
    """Öncelik ağırlıklı UTCI düşüşünü en çoklayan MILP. Park/ıslatma parçalı doğrusal azalan getiri; sisleme aç/kapa."""
    if float(t["oncelik"].sum()) <= 0:
        sonuc = t.copy()
        sonuc["su_m3"] = 0.0
        return sonuc, "Optimal"
    model = pulp.LpProblem("CoolScape", pulp.LpMaximize)
    su, amac, toplam, kirilgan, park_gunluk = {}, [], [], [], {}
    for idx, r in t.iterrows():
        y = degisken(model, f"y_{idx}", tur="Binary")
        kapali = (r["tur"] not in turler or r["vpd_kpa"] < vpd_min
                  or (r["tur"] == "park" and r["saat"] >= park_kesme))
        if kapali:
            model += y == 0
        if r["tur"] == "sisleme":
            x = SISLEME_SU_M3_SAAT * y
            amac.append(r["oncelik"] * float(sogutma_egrisi(SISLEME_SU_M3_SAAT, r["emax"], r["x0"])) * y)
            su[idx] = ("sabit", y)
        else:
            segs = []
            for k in range(len(KIRILIM) - 1):
                a, b = KIRILIM[k], min(KIRILIM[k + 1], r["kapasite"])
                if b <= a:
                    break
                v = degisken(model, f"s_{idx}_{k}", alt_s=0, ust_s=float(b - a))
                x0 = r["x0"] * x0_carpan
                egim = (sogutma_egrisi(b, r["emax"], x0) - sogutma_egrisi(a, r["emax"], x0)) / (b - a)
                amac.append(r["oncelik"] * egim * v)
                if enerji_opt:
                    amac.append(-0.01 * r["pompa"] * v)
                segs.append(v)
            x = pulp.lpSum(segs)
            model += x <= r["kapasite"] * y
            model += x >= min(MIN_DOZ, r["kapasite"]) * y
            su[idx] = ("parca", segs)
            if r["tur"] == "park":
                park_gunluk.setdefault(r["park"], ([], r["alan_m2"]))[0].append(x)
        toplam.append(x)
        if r["park"] in kirilgan_parklar and r["tur"] in turler:
            kirilgan.append(x)
    for _, (xs, alan) in park_gunluk.items():
        model += pulp.lpSum(xs) <= park_gunluk_mm * alan / 1000.0
    model += pulp.lpSum(amac)
    model += pulp.lpSum(toplam) <= butce
    if kirilgan and kirilgan_pay > 0:
        model += pulp.lpSum(kirilgan) >= kirilgan_pay * pulp.lpSum(toplam)
    model.solve(cozucu_sec())
    durum = pulp.LpStatus[model.status] if hasattr(pulp, "LpStatus") else str(model.status)
    sonuc = t.copy()
    degerler = []
    for i in t.index:
        tip, v = su[i]
        degerler.append(round(SISLEME_SU_M3_SAAT * (v.value() or 0), 3) if tip == "sabit"
                        else round(sum((s_.value() or 0) for s_ in v), 2))
    sonuc["su_m3"] = degerler
    return sonuc, durum


def esit_dagitim(t, butce, park_kesme=18, park_gunluk_mm=6.0):
    """Karşılaştırma: bütçe yalnızca parklara ve izin verilen gündüz saatlerine eşit bölünür; aynı kapasite ve çim
    güvenliği kurallarına uyar."""
    sonuc = t.copy()
    sonuc["su_m3"] = 0.0
    izin = (sonuc["tur"] == "park") & (sonuc["saat"] < park_kesme)
    if izin.sum() == 0:
        return sonuc
    pay_ = np.minimum(butce / int(izin.sum()), sonuc.loc[izin, "kapasite"])
    tavan = (park_gunluk_mm * sonuc.loc[izin, "alan_m2"] / 1000.0) / sonuc.loc[izin].groupby("park")["saat"].transform("count")
    sonuc.loc[izin, "su_m3"] = np.minimum(pay_, tavan)
    return sonuc


def etki(sonuc, x0_carpan=1.0, emax_carpan=1.0):
    s = sonuc.copy()
    x0 = np.where(s["tur"] == "sisleme", s["x0"], s["x0"] * x0_carpan)
    s["utci_dusus_c"] = sogutma_egrisi(s["su_m3"], s["emax"] * emax_carpan, x0)
    s["su_mm"] = s["su_m3"] / (s["alan_m2"] / 1000.0)
    s["agirlikli_fayda"] = s["oncelik"] * s["utci_dusus_c"]
    s["kisi_derece_saat"] = s["kisi"] * s["utci_dusus_c"] * (s["tehlike"] > 0)
    s["utci_sonra_c"] = s["utci_c"] - s["utci_dusus_c"]
    return s


@st.cache_data(show_spinner="Su planı hesaplanıyor...")
def plan_hesapla(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, turler, park_kesme, park_mm):
    kirilgan_parklar = set(kirilgan_parklar)
    ayar = dict(turler=turler, park_kesme=park_kesme, park_gunluk_mm=park_mm)
    plan, durum = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, **ayar)
    if durum != "Optimal":
        return None, None, None, durum
    esit = esit_dagitim(girdi, butce, park_kesme, park_mm)
    plan_e, esit_e = etki(plan), etki(esit)
    fayda_esit = esit_e["agirlikli_fayda"].sum()
    artislar = []
    for c in X0_ARALIK:
        p2, d2 = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0_carpan=c, **ayar)
        fz = etki(esit, c)["agirlikli_fayda"].sum()
        if d2 == "Optimal" and fz > 0:
            artislar.append(etki(p2, c)["agirlikli_fayda"].sum() / fz - 1)
    yalniz_park = None
    if set(turler) != {"park"}:
        pp, dp = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt,
                             turler=("park",), park_kesme=park_kesme, park_gunluk_mm=park_mm)
        if dp == "Optimal" and fayda_esit > 1e-9:
            yalniz_park = etki(pp)["agirlikli_fayda"].sum() / fayda_esit - 1
    if artislar and fayda_esit > 1e-9:
        artislar.append(plan_e["agirlikli_fayda"].sum() / fayda_esit - 1)
    sulanan = plan_e[plan_e["su_m3"] > 0.001]
    ozet = {
        "isi_stresi_var": fayda_esit > 1e-9,
        "artis": plan_e["agirlikli_fayda"].sum() / fayda_esit - 1 if fayda_esit > 1e-9 else 0.0,
        "artis_aralik": (min(artislar), max(artislar)) if artislar else None,
        "yalniz_park": yalniz_park,
        "kds": float(plan_e["kisi_derece_saat"].sum()), "kds_esit": float(esit_e["kisi_derece_saat"].sum()),
        "dusus_aralik": (float(sulanan["utci_dusus_c"].min()), float(sulanan["utci_dusus_c"].max()))
        if len(sulanan) else None,
        "su_tur": plan_e.groupby("tur")["su_m3"].sum().to_dict(),
    }
    return plan_e, esit_e, ozet, durum


def operator_komutlari(sonuc, parklar, aritilmis_senaryo):
    p = parklar.set_index("park")
    satirlar = []
    for park, g in sonuc[sonuc["su_m3"] > 0.001].sort_values("saat").groupby("park"):
        miktar = g.set_index("saat")["su_m3"]
        saatler = list(miktar.index)
        tur = p.loc[park, "tur"]
        if tur == "sisleme":
            kaynak = "İçme suyu (ihtiyat: aerosol, lejyonella)"
        elif aritilmis_senaryo and p.loc[park, "aritilmis_su_erisimi"]:
            kaynak = "Arıtılmış su (senaryo; Ek-7 ve Bakanlık uygun görüşü gerekir)" + (", arazöz" if tur == "islatma" else "")
        else:
            kaynak = "Arazöz, mevcut kaynak" if tur == "islatma" else "Mevcut kaynak"
        bas = onceki = saatler[0]
        for sa in saatler[1:] + [None]:
            if sa is None or sa != onceki + 1:
                satirlar.append({"Nokta": park, "Yöntem": YONTEMLER[tur][0], "Başlangıç": f"{bas:02d}:00",
                                 "Bitiş": f"{onceki + 1:02d}:00",
                                 "Su (m³)": round(float(miktar.loc[bas:onceki].sum()), 2), "Kaynak": kaynak})
                bas = sa
            onceki = sa if sa is not None else onceki
    return pd.DataFrame(satirlar).sort_values(["Başlangıç", "Nokta"]) if satirlar else pd.DataFrame()


# ---------------------------------------------------------------------------
# EKONOMİ (A4)
# ---------------------------------------------------------------------------
def ekonomi_hesapla(sulanan_ha, mm_gun, gun, birim_mevcut, birim_aritilmis, aritilmis_pay, yatirim, isletme_tl_yil,
                    gun_plan_m3):
    """Yıllık sulama suyu ve arıtılmış su senaryosu. None girdiler 'girilmedi' döner; uydurma değer yok."""
    yillik_m3 = sulanan_ha * 10.0 * mm_gun * gun
    ikame_m3 = yillik_m3 * aritilmis_pay
    s = {"yillik_m3": yillik_m3, "ikame_m3": ikame_m3,
         "mevcut_tl": yillik_m3 * birim_mevcut if birim_mevcut is not None else None,
         "gun_plan_tl": gun_plan_m3 * birim_mevcut if birim_mevcut is not None else None,
         "tasarruf_tl": None, "geri_donus_yil": None}
    if birim_mevcut is not None and birim_aritilmis is not None:
        s["tasarruf_tl"] = ikame_m3 * (birim_mevcut - birim_aritilmis) - (isletme_tl_yil or 0.0)
        if yatirim is not None and s["tasarruf_tl"] > 0:
            s["geri_donus_yil"] = yatirim / s["tasarruf_tl"]
    return s


# ---------------------------------------------------------------------------
# HARİTA ORTAK
# ---------------------------------------------------------------------------
SICAK_DURAKLAR = [(0.00, (255, 255, 178, 0)), (0.15, (254, 204, 92, 120)), (0.40, (253, 141, 60, 170)),
                  (0.70, (240, 59, 32, 205)), (1.00, (189, 0, 38, 230))]
SOGUK_DURAKLAR = [(0.00, (198, 219, 239, 0)), (0.08, (158, 202, 225, 150)), (0.30, (107, 174, 214, 190)),
                  (0.60, (49, 130, 189, 215)), (1.00, (8, 81, 156, 235))]
SOGUMA_UST_C = E_REF
YUKSEKLIK_M = 900


def renk_tablosu(t, duraklar, alfa_carpan=1.0, taban_alfa=0):
    konum = np.array([d[0] for d in duraklar])
    renk = np.array([d[1] for d in duraklar], dtype=float)
    rgba = np.stack([np.interp(t, konum, renk[:, i]) for i in range(4)], axis=-1)
    rgba[:, 3] = np.where(t < 0.02, taban_alfa, rgba[:, 3] * alfa_carpan)
    return rgba.astype(int).tolist()


def mesafe_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    f1, f2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((f2 - f1) / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def yayilim_cekirdegi(d, yaricap, olcek, kesme):
    """Alan içi tam etki; dışında e^(−(d−R)/ölçek); R+kesmeden sonra 0."""
    d = np.asarray(d, dtype=float)
    disari = np.maximum(d - yaricap, 0.0)
    return np.where(disari <= kesme, np.exp(-disari / olcek), 0.0)


def mahalle_sogumasi(yuzey, parklar, plan, saat):
    """Her konum için o saatteki tahmini UTCI düşüşü: noktadaki düşüş × yayılım çekirdeği (yönteme göre)."""
    sog = np.zeros(len(yuzey))
    if plan is None:
        return sog
    h = plan[plan["saat"] == saat].set_index("park")
    for _, p in parklar.iterrows():
        dusus = float(h["utci_dusus_c"].get(p["park"], 0.0))
        if dusus <= 0:
            continue
        _, _, olcek, kesme = YONTEMLER[p["tur"]]
        yaricap = float(p.get("yaricap_m", 0) or 0)
        if p["tur"] == "park" and yaricap == 0:
            yaricap = float(np.sqrt(p["sulanan_alan_ha"] * 10_000 / np.pi))
        d = mesafe_m(p["lat"], p["lon"], yuzey["lat"].values, yuzey["lon"].values)
        sog += dusus * yayilim_cekirdegi(d, yaricap, olcek, kesme)
    return sog


def harita_katmanlari(saat, hava, yuzey, parklar, plan, sicak_pay, mod, secenekler, tesisler, boyut3=True):
    katmanlar, ist = [], None
    satir = hava[hava["saat"] == saat].iloc[0]
    if yuzey is not None:
        utci_once = yuzey_utci(yuzey, satir)
        sog = mahalle_sogumasi(yuzey, parklar, plan, saat)
        utci_sonra = utci_once - sog
        esik = float(np.nanquantile(utci_once, 1 - sicak_pay))
        ust = float(max(np.nanquantile(utci_once, 0.99), esik + 0.5))
        alan = float(np.mean([h3.cell_area(h, unit="km^2") for h in yuzey["hex"].iloc[:50]]))
        ist = {"once": int(np.sum(utci_once > esik)), "sonra": int(np.sum(utci_sonra > esik)),
               "alan_km2": float(np.sum(sog >= 0.1) * alan), "azami": float(sog.max()) if len(sog) else 0.0,
               "esik": esik, "ust": ust}
        hx = pd.DataFrame({"hex": yuzey["hex"].values})
        sinif = yuzey["sinif"].values if "sinif" in yuzey else np.array(["–"] * len(yuzey))
        if mod == "Öncelik" and "sinif_no" in yuzey:
            no = yuzey["sinif_no"].values
            hx["renk"] = [list(SINIF_RENK[int(i)]) + ([215] if i > 0 else [70]) for i in no]
            hx["yukseklik"] = np.where(no > 0, yuzey["oncelik"].values, 0.0) * YUKSEKLIK_M
            hx["ad"] = [f"{c} öncelik" if c != "Seyrek yerleşim" else c for c in sinif]
        elif mod == "Soğuma":
            t = np.clip(np.nan_to_num(sog / SOGUMA_UST_C), 0, 1)
            hx["renk"] = renk_tablosu(t, SOGUK_DURAKLAR, 1.0, taban_alfa=0)
            hx["yukseklik"] = t * YUKSEKLIK_M
            hx["ad"] = [f"Soğuma {v:.2f} °C" for v in sog]
        else:
            deger = utci_sonra if mod == "Sonra" else utci_once
            t = np.nan_to_num(np.clip((deger - esik) / (ust - esik), 0, 1) * (deger > esik))
            alfa = float(np.clip((kent_utci(satir) - 26) / 20, 0.35, 1.0))
            hx["renk"] = renk_tablosu(t, SICAK_DURAKLAR, alfa, taban_alfa=28)
            hx["yukseklik"] = t * YUKSEKLIK_M
            hx["ad"] = [f"UTCI {v:.1f} °C" for v in deger]
        d_park = yuzey["d_park_m"].values if "d_park_m" in yuzey else np.full(len(yuzey), np.nan)
        n_t = yuzey["n_tesis_400"].values if "n_tesis_400" in yuzey else np.zeros(len(yuzey))
        hx["bilgi"] = [f"UTCI önce {a:.1f} → sonra {b:.1f} °C · yüzey farkı {f:+.1f} °C\n"
                       f"Öncelik: {c} · en yakın park {d:,.0f} m · 400 m'de {int(n)} okul/sağlık"
                       for a, b, f, c, d, n in zip(utci_once, utci_sonra, yuzey["lst_fark_c"].values, sinif, d_park, n_t)]
        if mod == "Soğuma":
            hx = hx[np.array([r[3] for r in hx["renk"]]) > 0]
        if len(hx):
            katmanlar.append(pdk.Layer(
                "H3HexagonLayer", hx, id="altigen", get_hexagon="hex", get_fill_color="renk",
                get_elevation="yukseklik", elevation_scale=1, extruded=boyut3, coverage=0.92, stroked=False,
                filled=True, pickable=True, auto_highlight=True, highlight_color=[255, 255, 255, 120]))
    p = parklar[["park", "lat", "lon", "tur"]].copy()
    if plan is not None:
        h = plan[plan["saat"] == saat].set_index("park")
        p["su"] = p["park"].map(h["su_m3"]).fillna(0)
        p["once"] = p["park"].map(h["utci_c"])
        p["sonra"] = p["park"].map(h["utci_sonra_c"])
        p["yaricap"] = np.where(p["tur"] == "park", np.sqrt(p["su"]) * 20 + 45, 45)
        p["renk"] = [list(YONTEMLER[t][1]) + [235] if s_ > 0 else [140, 140, 140, 200]
                     for t, s_ in zip(p["tur"], p["su"])]
        p["bilgi"] = [f"{YONTEMLER[t][0]} · {s_:.2f} m³ · UTCI {a:.1f} → {b:.1f} °C"
                      for t, s_, a, b in zip(p["tur"], p["su"], p["once"], p["sonra"])]
    else:
        p["yaricap"], p["renk"], p["bilgi"] = 90, [[140, 140, 140, 200]] * len(p), ""
    p["sekil"] = [[255, 255, 255] if t == "park" else list(YONTEMLER[t][1]) for t in p["tur"]]
    p["ad"] = p["park"]
    p["kisa"] = [kisa_ad(a, t) for a, t in zip(p["park"], p["tur"])]
    if "Yürüme mesafesi (5 ve 10 dk)" in secenekler:
        pk = p[p["tur"] == "park"]
        halka = pd.concat([pk.assign(r=YURUME_5DK_M), pk.assign(r=YURUME_10DK_M)])
        katmanlar.append(pdk.Layer("ScatterplotLayer", halka, id="yurume", get_position="[lon, lat]",
                                   get_radius="r", filled=False, stroked=True, get_line_color=[150, 150, 150, 200],
                                   line_width_min_pixels=1.5))
    if "Hassas tesisler (OSM)" in secenekler and tesisler is not None and not tesisler.empty:
        t_df = tesisler.copy()
        t_df["renk"] = t_df["kategori"].map(lambda k: list(OSM_KATEGORILER[k][1]) + [230])
        t_df["bilgi"] = t_df["kategori"]
        katmanlar.append(pdk.Layer("ScatterplotLayer", t_df, id="tesis", get_position="[lon, lat]",
                                   get_radius=30, radius_min_pixels=3, get_fill_color="renk", pickable=True))
    if boyut3:
        p["yukseklik_nokta"] = YUKSEKLIK_M * 1.2
        katmanlar.append(pdk.Layer("ColumnLayer", p, id="parklar", get_position="[lon, lat]",
                                   get_elevation="yukseklik_nokta", radius=70, extruded=True, disk_resolution=16,
                                   get_fill_color="renk", pickable=True))
    else:
        katmanlar.append(pdk.Layer("ScatterplotLayer", p, id="parklar", get_position="[lon, lat]",
                                   get_radius="yaricap", get_fill_color="renk", pickable=True, radius_min_pixels=7,
                                   stroked=True, get_line_color="sekil", line_width_min_pixels=2))
    if "Nokta adları" in secenekler:
        p["z"] = YUKSEKLIK_M * 1.3 if boyut3 else 0
        katmanlar.append(pdk.Layer("TextLayer", p, id="etiket", get_position="[lon, lat, z]", get_text="kisa",
                                   get_size=13, get_color=[255, 255, 255], get_pixel_offset=[0, -14],
                                   character_set=pdk.types.String("auto"), font_weight=600, background=True,
                                   font_family=pdk.types.String("Helvetica, Arial, sans-serif"),
                                   get_background_color=[20, 20, 20, 190], background_padding=[4, 2]))
    merkez = yuzey if yuzey is not None else parklar
    gorunum = pdk.ViewState(latitude=float(merkez["lat"].mean()) - (0.035 if boyut3 else 0),
                            longitude=float(merkez["lon"].mean()), zoom=11.25 if boyut3 else 11.6,
                            pitch=50 if boyut3 else 0, bearing=-12 if boyut3 else 0)
    return pdk.Deck(layers=katmanlar, initial_view_state=gorunum, tooltip={"text": "{ad}\n{bilgi}"}), ist


def kisa_ad(ad, tur):
    if tur == "park":
        return ad
    return ("S" if tur == "sisleme" else "I") + ad.split()[1]


def gradyan(duraklar, genislik=150):
    renkler = ", ".join(f"rgba({r},{g},{b},{max(a, 70) / 255:.2f}) {k * 100:.0f}%" for k, (r, g, b, a) in duraklar)
    return (f"<span style='display:inline-block;width:{genislik}px;height:10px;vertical-align:middle;"
            f"border-radius:2px;background:linear-gradient(to right, {renkler})'></span>")


def lejant_html(mod, ist, sicak_pay, tesis_var, boyut3=False):
    parca = []
    if ist is not None and mod == "Öncelik":
        kutu = "".join(f"<span style='display:inline-block;width:14px;height:10px;margin:0 2px 0 6px;"
                       f"background:rgb{SINIF_RENK[i]}'></span>{SINIF_ADLARI[i - 1].lower()}" for i in range(1, 6))
        parca.append(f"<b>Soğutma önceliği</b>{kutu}<span style='display:inline-block;width:14px;height:10px;"
                     f"margin:0 2px 0 6px;background:rgb{SINIF_RENK[0]}'></span>seyrek yerleşim")
    elif ist is not None and mod == "Soğuma":
        parca.append(f"<b>Soğuma etkisi</b> (tahmini UTCI düşüşü) &nbsp;0 °C {gradyan(SOGUK_DURAKLAR)} "
                     f"{SOGUMA_UST_C:.1f} °C")
    elif ist is not None:
        baslik = "Plan uygulanmış" if mod == "Sonra" else "Mevcut durum"
        parca.append(f"<b>{baslik}: sıcak noktalar</b> (kentin en sıcak %{sicak_pay * 100:.0f}'lik alanı) &nbsp;"
                     f"{ist['esik']:.1f} °C {gradyan(SICAK_DURAKLAR)} {ist['ust']:.1f} °C UTCI")
    else:
        parca.append("<b>Isı yüzeyi dosyası yok:</b> yalnızca parklar gösteriliyor")
    parca.append(" &nbsp;".join(f"<span style='color:rgb{YONTEMLER[k][1]}'>●</span> {YONTEMLER[k][0].lower()}"
                                for k in YONTEMLER) + " &nbsp;<span style='color:#8c8c8c'>●</span> bu saatte kapalı")
    if tesis_var:
        parca.append(" &nbsp;".join(f"<span style='color:rgb{OSM_KATEGORILER[k][1]}'>●</span> {k.lower()}"
                                    for k in OSM_KATEGORILER))
    if boyut3 and ist is not None:
        parca.append("<span style='color:gray'>sütun yüksekliği renkteki değerle orantılı · sağ tıkla sürükleyerek döndürün</span>")
    return "<div style='font-size:0.82rem;line-height:1.7'>" + "<br>".join(parca) + "</div>"


def isi_seridi_html(satir):
    u = kent_utci(satir)
    ad, (r, g, b) = utci_sinif(u)
    yazi = "#000" if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else "#fff"
    return (f"<div style='background:rgb({r},{g},{b});color:{yazi};padding:6px 10px;border-radius:4px;"
            f"font-weight:600'>Kent geneli UTCI {u:.0f} °C (açık güneş) · {ad.lower()} ısı stresi</div>")


def saat_karti(saat, hava, plan, vpd_min):
    satir = hava[hava["saat"] == saat].iloc[0]
    st.markdown(f"#### {saat:02d}:00")
    st.markdown(isi_seridi_html(satir), unsafe_allow_html=True)
    st.caption(f"Hava {satir['sicaklik_c']:.1f} °C · nem %{satir['bagil_nem']:.0f} · "
               f"rüzgâr {satir['ruzgar_ms']:.1f} m/s · güneş {satir['gunes_wm2']:.0f} W/m² · "
               f"güneşte MRT +{satir['dmrt_gunes']:.0f} °C · yüzey farkı ölçeği {satir['lst_olcek']:.2f}".replace(".", ","))
    if plan is None:
        return
    h = plan[plan["saat"] == saat].sort_values("su_m3", ascending=False)
    sulanan = h[h["su_m3"] > 0.001]
    if sulanan.empty:
        if h["tehlike"].sum() == 0:
            st.write("Bu saatte güçlü ısı stresi yok; sulama gerekmiyor.")
        elif (h["vpd_kpa"] < vpd_min).all():
            st.write("Bu saatte sulama yok: hava nemli, su az serinletir (nem kapısı).")
        else:
            st.write("Bu saatte sulama yok: su daha verimli olacağı saatlere ayrıldı.")
    sulanan = sulanan.sort_values("agirlikli_fayda", ascending=False)

    def sayi(x, f):
        return format(x, f).replace(".", ",")
    for _, r in sulanan.head(6).iterrows():
        st.markdown(f"**{r['park']}** · {YONTEMLER[r['tur']][0].lower()} · {sayi(r['su_m3'], '.2g')} m³  \n"
                    f"UTCI {sayi(r['utci_c'], '.1f')} → {sayi(r['utci_sonra_c'], '.1f')} °C · "
                    f"<span style='color:gray'>{utci_sinif(r['utci_c'])[0].lower()} stres, "
                    f"VPD {sayi(r['vpd_kpa'], '.1f')} kPa</span>", unsafe_allow_html=True)
    if len(sulanan) > 6:
        st.caption(f"ve {len(sulanan) - 6} nokta daha (tam liste aşağıdaki onay listesinde)")


def plan_matrisi(plan):
    d = plan[["park", "saat", "su_m3", "utci_dusus_c"]].copy()
    d["saat_et"] = d["saat"].map(lambda s: f"{s:02d}:00")
    d["etiket"] = d["su_m3"].map(lambda v: f"{v:.0f}" if v >= 0.5 else ("•" if v > 0.001 else ""))
    taban = alt.Chart(d).encode(x=alt.X("saat_et:O", title="Saat"), y=alt.Y("park:N", title=None))
    hucre = taban.mark_rect().encode(
        color=alt.Color("su_m3:Q", title="Su (m³)", scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("park:N", title="Nokta"), alt.Tooltip("saat_et:O", title="Saat"),
                 alt.Tooltip("su_m3:Q", title="Su (m³)", format=".2f"),
                 alt.Tooltip("utci_dusus_c:Q", title="UTCI düşüşü (°C)", format=".2f")])
    yazi = taban.mark_text(fontSize=11).encode(
        text="etiket:N", color=alt.condition("datum.su_m3 > 8", alt.value("white"), alt.value("black")))
    return (hucre + yazi).properties(height=28 * d["park"].nunique() + 30)


def park_ozet_tablosu(parklar, plan, tesisler):
    satirlar = []
    for _, p in parklar.iterrows():
        g = plan[plan["park"] == p["park"]]
        saatler = g.loc[g["su_m3"] > 0.001, "saat"]
        r = {"Nokta": p["park"], "Yöntem": YONTEMLER[p["tur"]][0],
             "Alan (ha)": float(p.get("alan_ha", np.nan)),
             "Günlük su (m³)": round(float(g["su_m3"].sum()), 2),
             "Çalışma saatleri": f"{saatler.min():02d}:00–{saatler.max() + 1:02d}:00" if len(saatler) else "–",
             "En yüksek UTCI (°C)": round(float(g["utci_c"].max()), 1),
             "Gölge örtüsü": round(float(p.get("golge_ortu", 0.0)), 2),
             "Erişim (kişi/saat, temsili)": int(g["kisi"].iloc[0]),
             "Kişi·°C·saat": round(float(g["kisi_derece_saat"].sum()), 0),
             "Kaynak": p.get("park_kaynak", "")}
        if tesisler is not None:
            d = mesafe_m(p["lat"], p["lon"], tesisler["lat"].values, tesisler["lon"].values) <= YURUME_5DK_M
            for k in ("Okul / kreş", "Sağlık", "Huzurevi"):
                r[f"{k} (400 m)"] = int(((tesisler["kategori"] == k).values & d).sum())
        satirlar.append(r)
    return pd.DataFrame(satirlar).sort_values("Günlük su (m³)", ascending=False)


# ---------------------------------------------------------------------------
# CANLI HARİTA (Folium)
# ---------------------------------------------------------------------------
IZGARA_DERECE = 0.0006
YUMUSATMA_PIKSEL = 3.0
CANLI_SAYDAMLIK = 0.58
CANLI_BUYUTME = 3
UTCI_ACIKLIK_C = 4.0
UTCI_CIZGI_MIN_C = 0.5
ALTLIKLAR = {
    "Uydu (Esri World Imagery)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"
                                  "{z}/{y}/{x}",
                                  "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, "
                                  "Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"),
    "Sokak (OpenStreetMap)": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                              "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> katkıcıları"),
}
UYDU_ETIKETLER = [
    ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
     "Yollar"),
    ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/"
     "{z}/{y}/{x}", "Yer adları"),
]
NOKTA_RENK = {"park": "#2e7d32", "sisleme": "#00838f", "islatma": "#6a1b9a"}


@st.cache_data(show_spinner=False)
def su_hexleri_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return (), ()
    su, karisik = kiyi_siniflari(pd.read_csv(yol, usecols=["hex", "lst_c"]).dropna())
    return tuple(sorted(su)), tuple(sorted(karisik))


@st.cache_data(show_spinner=False)
def izgara_hazirla(kara_hex: tuple, su_hex: tuple, kiyi_hex: tuple = ()):
    res = h3.get_resolution(kara_hex[0])
    merkez = np.array([h3.cell_to_latlng(h) for h in kara_hex + su_hex + kiyi_hex])
    pay = 0.006
    lat = np.arange(merkez[:, 0].min() - pay, merkez[:, 0].max() + pay, IZGARA_DERECE)
    lon = np.arange(merkez[:, 1].min() - pay, merkez[:, 1].max() + pay, IZGARA_DERECE)
    LON, LAT = np.meshgrid(lon, lat)
    hucre = [h3.latlng_to_cell(a, o, res) for a, o in zip(LAT.ravel(), LON.ravel())]
    konum = {h: i for i, h in enumerate(kara_hex)}
    su = set(su_hex)
    kiyi = set(kiyi_hex)
    idx = np.array([konum.get(c, -2 if c in kiyi else -1) for c in hucre]).reshape(LAT.shape)
    deniz = np.array([c in su for c in hucre], dtype=float).reshape(LAT.shape)
    return lon, lat, idx, deniz


def _bulanik(a, s=YUMUSATMA_PIKSEL):
    r = int(3 * s)
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * s * s))
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def _yumusak(degerler, idx, gecerli=None):
    degerler = np.asarray(degerler, dtype=float)
    var = idx >= 0
    v = np.where(var, degerler[np.clip(idx, 0, None)], 0.0)
    if gecerli is not None:
        var = var & np.asarray(gecerli)[np.clip(idx, 0, None)]
    v = np.where(var, np.nan_to_num(v), 0.0)
    w = _bulanik(var.astype(float))
    return np.where(w > 1e-3, _bulanik(v) / np.maximum(w, 1e-3), np.nan)


def _guzel_adim(aralik, hedef=8):
    for a in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0):
        if aralik / a <= hedef:
            return a
    return 10.0


def canli_alan(mod, saat, hava, yuzey, parklar, plan):
    su_hex, kiyi_hex = su_hexleri_oku(ISI_YUZEYI_CSV, os.path.getmtime(ISI_YUZEYI_CSV)
                                      if os.path.exists(ISI_YUZEYI_CSV) else 0)
    lon, lat, idx, _ = izgara_hazirla(tuple(yuzey["hex"]), su_hex, kiyi_hex)
    LON, LAT = np.meshgrid(lon, lat)
    kara = _bulanik((idx != -1).astype(float)) > 0.5
    satir = hava[hava["saat"] == saat].iloc[0]
    if mod == "LST":
        hexd = yuzey["lst_c"].values
        z = _yumusak(hexd, idx)
        bilgi = {"baslik": "Yüzey sıcaklığı (°C)", "cmap": "RdYlBu_r", "birim": "°C",
                 "not": f"Landsat 9 ölçümü, {yuzey['landsat_tarih'].iloc[0] if 'landsat_tarih' in yuzey else '?'}, "
                        "yerel saatle ~11:20 geçiş; o anın yüzey sıcaklığı (saat seçiminden bağımsız)"}
    elif mod == "Öncelik" and "oncelik" in yuzey:
        yer = yuzey["yerlesim"].values.astype(bool)
        hexd = np.where(yer, yuzey["oncelik"].values, np.nan)
        z = _yumusak(yuzey["oncelik"].values, idx, gecerli=yer)
        z = np.where(_yumusak(yer.astype(float), idx) >= 0.5, z, np.nan)
        bilgi = {"baslik": "Soğutma önceliği (0–1)", "cmap": "YlOrRd", "birim": "",
                 "not": "Ağırlıklı çakıştırma; seyrek yerleşim boş bırakıldı"}
    else:
        u = yuzey_utci(yuzey, satir)
        sog = np.zeros(LAT.shape)
        if plan is not None and mod in ("Sonra", "Soğuma"):
            sog[kara] = mahalle_sogumasi(pd.DataFrame({"lat": LAT[kara], "lon": LON[kara]}), parklar, plan, saat)
        if mod == "Soğuma":
            hexd = mahalle_sogumasi(yuzey, parklar, plan, saat)
            z = _bulanik(sog, 1.0)
            z = np.where(z >= 0.05, z, np.nan)
            bilgi = {"baslik": f"Soğuma etkisi, {saat:02d}:00 (UTCI düşüşü, °C)", "cmap": "Blues", "birim": "°C",
                     "cmin": 0.35,
                     "not": "Planın yarattığı tahmini UTCI düşüşü; 0,05 °C altı gösterilmez. Etki müdahale alanının içinde "
                            "tam, dışında azalır (park 150 m, ıslatma 60 m, sisleme 10 m ölçekle; varsayım)"}
        else:
            hexd = u - (mahalle_sogumasi(yuzey, parklar, plan, saat) if mod == "Sonra" else 0)
            z = _yumusak(u, idx) - (sog if mod == "Sonra" else 0)
            bilgi = {"baslik": f"UTCI, {saat:02d}:00 (°C){' · plan uygulanmış' if mod == 'Sonra' else ''}",
                     "cmap": "YlOrRd", "birim": "°C",
                     "not": f"Kent geneli {kent_utci(satir):.1f} °C ({utci_sinif(kent_utci(satir))[0].lower()} ısı stresi, "
                            "açık güneş); mekânsal fark yüzey sıcaklığının saatlik ölçekli MRT katkısından ve (NDVI olan "
                            "yerde) ağaç gölgesinden gelir"}
    z = np.where(kara, z, np.nan)
    gecerli = hexd[np.isfinite(hexd)]
    if mod == "Soğuma":
        vmin, vmax = 0.0, max(float(np.nanmax(z)) if np.isfinite(z).any() else 0.1, 0.1)
    else:
        vmin, vmax = (float(v) for v in np.nanpercentile(gecerli, [2, 98]))
        if mod in ("Önce", "Sonra"):
            orta = (vmin + vmax) / 2
            yari = max((vmax - vmin) / 2, UTCI_ACIKLIK_C / 2)
            vmin, vmax = orta - yari, orta + yari
        if vmax - vmin < 1e-6:
            vmax = vmin + 0.1
    adim = max(_guzel_adim(vmax - vmin), UTCI_CIZGI_MIN_C if mod in ("Önce", "Sonra") else 0)
    vmin, vmax = np.floor(vmin / adim + 1e-9) * adim, np.ceil(vmax / adim - 1e-9) * adim
    bilgi.update(vmin=float(vmin), vmax=float(vmax), adim=float(adim))
    return lon, lat, z, hexd, bilgi


def _buyut(a, k):
    def eksen(x, ax):
        n = x.shape[ax]
        eski, yeni = np.arange(n), np.linspace(0, n - 1, (n - 1) * k + 1)
        return np.apply_along_axis(lambda v: np.interp(yeni, eski, v), ax, x)
    return eksen(eksen(a, 0), 1)


def _renkli_png(z, bilgi):
    cmap = plt.get_cmap(bilgi["cmap"])
    t = np.clip((z - bilgi["vmin"]) / (bilgi["vmax"] - bilgi["vmin"]), 0, 1)
    t = bilgi.get("cmin", 0.0) + (1 - bilgi.get("cmin", 0.0)) * t
    rgba = cmap(np.nan_to_num(t))
    rgba[..., 3] = np.where(np.isfinite(z), CANLI_SAYDAMLIK, 0.0)
    rgba[..., :3] *= rgba[..., 3:4]
    rgba = _buyut(rgba, CANLI_BUYUTME)
    a = rgba[..., 3:4]
    rgba[..., :3] = np.where(a > 1e-6, rgba[..., :3] / np.maximum(a, 1e-6), 0)
    rgba[..., 3] = np.where(a[..., 0] >= CANLI_SAYDAMLIK * 0.5, CANLI_SAYDAMLIK, 0.0)
    rgba = np.clip(rgba, 0, 1)
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _es_deger_cizgileri(lon, lat, z, bilgi):
    if not np.isfinite(z).any():
        return [], None
    adim = bilgi["adim"]
    seviyeler = np.arange(np.ceil(bilgi["vmin"] / adim) * adim, bilgi["vmax"] + 1e-9, adim)
    if bilgi["birim"] == "" or len(seviyeler) == 0:
        return [], adim
    fig, ax = plt.subplots()
    cs = ax.contour(*np.meshgrid(lon, lat), np.ma.masked_invalid(z), levels=seviyeler)
    cizgiler = []
    for sev, parcalar in zip(cs.levels, cs.allsegs):
        for p in parcalar:
            if len(p) >= 10:
                cizgiler.append((float(sev), [(float(y), float(x)) for x, y in p]))
    plt.close(fig)
    return cizgiler, adim


def canli_harita(mod, saat, hava, yuzey, parklar, plan, secenekler, tesisler, altlik):
    lon, lat, z, hexd, bilgi = canli_alan(mod, saat, hava, yuzey, parklar, plan)
    d = IZGARA_DERECE / (2 * CANLI_BUYUTME)
    sinir = [[float(lat.min() - d), float(lon.min() - d)], [float(lat.max() + d), float(lon.max() + d)]]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=12, tiles=None, control_scale=True,
                   prefer_canvas=True)
    for ad, (url, atif) in ALTLIKLAR.items():
        folium.TileLayer(url, attr=atif, name=ad, max_zoom=19, show=(ad == altlik)).add_to(m)
    folium.raster_layers.ImageOverlay(_renkli_png(z, bilgi), bounds=sinir, name="Isı katmanı", opacity=1.0,
                                      interactive=False, zindex=2).add_to(m)
    folium.map.CustomPane("etiketler", z_index=650, pointer_events=False).add_to(m)
    for url, ad in UYDU_ETIKETLER:
        folium.TileLayer(url, attr="Esri", name=ad, overlay=True, control=True, show=True, max_zoom=19,
                         pane="etiketler").add_to(m)
    cizgiler, adim = _es_deger_cizgileri(lon, lat, z, bilgi)
    if cizgiler:
        fmt = (lambda v: f"{v:.1f}".replace(".", ",")) if adim < 1 else (lambda v: f"{v:.0f}")
        grup = folium.FeatureGroup(name="Eş değer çizgileri", show=False)
        etiket_say = 0
        for sev, nokta in sorted(cizgiler, key=lambda c: -len(c[1])):
            folium.PolyLine(nokta, color="#1a1a1a", weight=1.1, opacity=0.75,
                            tooltip=f"{fmt(sev)} {bilgi['birim']}").add_to(grup)
            if len(nokta) >= 40 and etiket_say < 30:
                etiket_say += 1
                orta = nokta[len(nokta) // 2]
                folium.Marker(orta, icon=folium.DivIcon(
                    icon_size=(40, 14), icon_anchor=(20, 7),
                    html=f"<div style='font:600 11px Arial;color:#111;text-align:center;"
                         f"text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff'>{fmt(sev)}</div>")).add_to(grup)
        grup.add_to(m)
    oz = []
    for h, v, f in zip(yuzey["hex"].values, hexd, yuzey["lst_fark_c"].values):
        if not np.isfinite(v):
            continue
        kenar = [[round(b, 6), round(a, 6)] for a, b in h3.cell_to_boundary(h)]
        deger = (f"{v:.2f}" if bilgi["birim"] == "" or mod == "Soğuma" else f"{v:.1f}").replace(".", ",")
        oz.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [kenar + [kenar[0]]]},
                   "properties": {"b": f"{bilgi['baslik']}: {deger} · yüzey farkı {f:+.1f} °C".replace(".", ",")}})
    folium.GeoJson({"type": "FeatureCollection", "features": oz}, name="Altıgen değerleri", control=False,
                   style_function=lambda _: {"fillOpacity": 0, "weight": 0, "color": "#000"},
                   highlight_function=lambda _: {"weight": 1, "color": "#000", "fillOpacity": 0.05},
                   tooltip=folium.GeoJsonTooltip(fields=["b"], labels=False)).add_to(m)
    if "Yürüme mesafesi (5 ve 10 dk)" in secenekler:
        g = folium.FeatureGroup(name="Yürüme mesafesi (400 / 800 m)")
        for _, p in parklar[parklar["tur"] == "park"].iterrows():
            for r_, cizgi in ((YURUME_5DK_M, "6 4"), (YURUME_10DK_M, "2 4")):
                folium.Circle([p["lat"], p["lon"]], radius=r_ + float(p.get("yaricap_m", 0) or 0), color="#1b5e20",
                              weight=1.5, fill=False, dash_array=cizgi).add_to(g)
        g.add_to(m)
    if "Hassas tesisler (OSM)" in secenekler and tesisler is not None and len(tesisler):
        g = folium.FeatureGroup(name="Tesisler (OpenStreetMap)")
        renk = {"Okul / kreş": "#111111", "Sağlık": "#d62728", "Huzurevi": "#2e7d32", "İçme suyu": "#1565c0"}
        for _, t in tesisler.iterrows():
            folium.CircleMarker([t["lat"], t["lon"]], radius=3, color="#ffffff", weight=0.8, fill=True,
                                fill_color=renk.get(t["kategori"], "#555"), fill_opacity=1,
                                tooltip=f"{t['kategori']}: {t['ad']}").add_to(g)
        g.add_to(m)
    su = plan[plan["saat"] == saat].set_index("park") if plan is not None else None
    g = folium.FeatureGroup(name="Müdahale noktaları")
    for _, p in parklar.iterrows():
        acik = su is not None and float(su["su_m3"].get(p["park"], 0)) > 0.001
        ipucu = f"<b>{p['park']}</b><br>{YONTEMLER[p['tur']][0]}"
        if su is not None and p["park"] in su.index:
            r = su.loc[p["park"]]
            ipucu += (f"<br>{saat:02d}:00 · {r['su_m3']:.2f} m³ · UTCI {r['utci_c']:.1f} → {r['utci_sonra_c']:.1f} °C"
                      .replace(".", ","))
        folium.CircleMarker([p["lat"], p["lon"]], radius=8 if p["tur"] == "park" else 6, color="#000", weight=1.5,
                            fill=True, fill_color=NOKTA_RENK[p["tur"]] if acik or su is None else "#9e9e9e",
                            fill_opacity=1, tooltip=ipucu).add_to(g)
        if "Nokta adları" in secenekler:
            folium.Marker([p["lat"], p["lon"]], icon=folium.DivIcon(
                icon_size=(160, 16), icon_anchor=(80, -10) if p["tur"] == "park" else (80, 22),
                html=f"<div style='font:700 11px Arial;color:#0d2b12;white-space:nowrap;text-align:center;"
                     f"text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff'>{kisa_ad(p['park'], p['tur'])}</div>")).add_to(g)
    g.add_to(m)
    renkler = [matplotlib.colors.to_hex(plt.get_cmap(bilgi["cmap"])(x))
               for x in np.linspace(bilgi.get("cmin", 0.0), 1, 9)]
    adim_l = bilgi["adim"]
    while (bilgi["vmax"] - bilgi["vmin"]) / adim_l > 8:
        adim_l *= 2
    isaret = [round(float(v), 3) for v in np.arange(bilgi["vmin"], bilgi["vmax"] + 1e-9, adim_l)]
    lejant = bcm.LinearColormap(renkler, vmin=round(bilgi["vmin"], 3), vmax=round(bilgi["vmax"], 3),
                                tick_labels=isaret if len(isaret) >= 2 else None)
    lejant.caption = bilgi["baslik"]
    lejant.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds(sinir)
    return m, bilgi


def canli_harita_goster(alan, m, yukseklik=640):
    html = m.get_root().render()
    with alan:
        if hasattr(st, "iframe"):
            st.iframe(html, height=yukseklik)
        else:
            components.html(html, height=yukseklik)


# ---------------------------------------------------------------------------
# ARAYÜZ
# ---------------------------------------------------------------------------
def tl(x, ondalik=0):
    if x is None:
        return "girilmedi"
    return f"{x:,.{ondalik}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ana():
    st.set_page_config(page_title="CoolScape AI", layout="wide", initial_sidebar_state="expanded")
    for anahtar, deger in {"saat": 14, "oynuyor": False}.items():
        st.session_state.setdefault(anahtar, deger)

    st.title("CoolScape AI")
    st.markdown(f"Kentsel serinlik ve su tahsis motoru · Antalya Muratpaşa–Kepez pilotu · öğrenci karar destek "
                f"prototipi ({SURUM})")

    st.sidebar.header("Günün planı")
    gun = st.sidebar.selectbox("Gün", list(GUNLER))
    butce = st.sidebar.number_input("Günlük su bütçesi (m³)", value=300, step=25, min_value=25,
                                    help="08:00–20:00 arasında parklara, sisleme istasyonlarına ve yol ıslatmaya "
                                         "dağıtılacak toplam su.")
    on_ayar = st.sidebar.radio("Öncelik", list(ON_AYARLAR), help="Hazır ayar. Ayrıntılar aşağıda.")
    v = ON_AYARLAR[on_ayar]
    yontem_etiket = {v_[0]: k for k, v_ in YONTEMLER.items()}
    secili_yontem = st.sidebar.multiselect("Yöntemler", list(yontem_etiket), default=list(yontem_etiket),
                                           help="Sisleme ve yol ıslatma noktaları sıcak nokta haritasından otomatik "
                                                "önerilen adaylardır.")
    turler = tuple(sorted({yontem_etiket[x] for x in secili_yontem} | {"park"}))
    with st.sidebar.expander("Gelişmiş ayarlar"):
        kirilgan_pay = st.slider("Kırılgan bölgelere asgari su payı (%)", 0, 80, v["kirilgan_pay"],
                                 key=f"pay_{on_ayar}") / 100
        w_kirilganlik = st.slider("Ağırlık: kırılganlık göstergesi", 0.0, 2.0, v["w_k"], 0.1, key=f"wk_{on_ayar}")
        vpd_min = st.slider("Nem kapısı: en düşük VPD (kPa)", 0.0, 2.0, 0.8, 0.1,
                            help="Buhar basıncı açığı bunun altındaysa su az serinletir; o saatte sulama yapılmaz.")
        enerji_opt = st.checkbox("Pompalama enerjisini gözet", value=True)
        n_osm = st.slider("Plana alınacak OSM parkı (pilotlara ek)", 0, 30, OSM_PARK_SAYISI,
                          help="OpenStreetMap parkları arasından 800 m içindeki nüfus × öncelik toplamı en yüksek "
                               "olanlar. Park verisi yoksa yalnızca 5 pilot park kullanılır.")
        st.markdown("**Çim güvenliği (park sulama)**")
        park_kesme = st.slider("Akşam kesme saati", 14, 20, 18, format="%d:00",
                               help="Bu saatten sonra park sulanmaz (gece ıslak yaprak, mantar riski).")
        park_mm = st.slider("Park başına günlük üst sınır (mm = L/m²)", 2.0, 12.0, 6.0, 0.5)
        st.markdown("**Erişim varsayımları (kişi/saat, temsili)**")
        erisim = {"park_carpan": st.slider("Park: nüfus skoru başına kişi", 10, 100,
                                           ERISIM_VARSAYILAN["park_carpan"], 5),
                  "sisleme": st.slider("Sisleme: 100 m² başına kişi", 5, 200, ERISIM_VARSAYILAN["sisleme"], 5),
                  "islatma": st.slider("Islatılan cadde", 20, 500, ERISIM_VARSAYILAN["islatma"], 10)}
        aritilmis_senaryo = st.checkbox("Arıtılmış su senaryosu", value=False,
                                        help="Altyapı kurulursa hangi noktaların arıtılmış suyla beslenebileceğini "
                                             "gösterir. Mevcut bağlantı: BİLGİ YETERSİZ.")
        sicak_pay = st.slider("Haritada gösterilecek en sıcak alan (%)", 5, 50, 20, 5) / 100

    # --- Veri ---
    tur, tarih = GUNLER[gun]
    if tur == "tahmin":
        tarih = (datetime.now(ZoneInfo("Europe/Istanbul")).date() + timedelta(days=1)).isoformat()
    try:
        hava = hava_tahmini_cek(tarih) if tur == "tahmin" else hava_verisi_cek(tarih)
        hava_kaynak = f"Open-Meteo tahmini ({tarih})" if tur == "tahmin" else "Open-Meteo arşivi (ERA5 tabanlı)"
    except Exception as hata:
        st.error(f"Open-Meteo'ya erişilemedi ({hata}). Uydurma veriyle devam edilmez. Saatlik veriyi CSV olarak "
                 "yükleyebilirsiniz (sütunlar: saat, sicaklik_c, bagil_nem, ruzgar_ms, gunes_wm2).")
        yuklenen = st.file_uploader("Saatlik hava verisi (CSV)", type="csv")
        if yuklenen is None:
            st.stop()
        hava = csv_hava_verisi(yuklenen)
        hava_kaynak = "yüklenen CSV"
    if not set(SAATLER) <= set(hava["saat"]):
        st.error("Hava verisinde 08:00–20:00 saatlerinin bir kısmı eksik.")
        st.stop()
    hava = gunes_mrt_ekle(hava)
    mrt_yontem = hava.attrs.get("mrt_yontem", "")

    def mtime(y):
        return os.path.getmtime(y) if os.path.exists(y) else 0

    yuzey = isi_yuzeyi_oku(ISI_YUZEYI_CSV, mtime(ISI_YUZEYI_CSV))
    tesis_yolu = TESIS_CSV if os.path.exists(TESIS_CSV) else TESIS_JSON
    tesisler = tesis_dosyasi_oku(tesis_yolu, mtime(tesis_yolu))
    if tesisler is not None:
        osm_hata, osm_kaynak = None, f"OpenStreetMap ({tesis_yolu}), {len(tesisler)} tesis"
    else:
        tesisler, osm_hata = osm_tesisleri_cek()
        osm_kaynak = (f"OpenStreetMap (canlı), {len(tesisler)} tesis" if tesisler is not None
                      else f"OpenStreetMap'e erişilemedi; {TESIS_JSON} dosyasını repoya ekleyin")
    osm_parklar = park_dosyasi_oku(PARK_JSON, mtime(PARK_JSON))
    park_hata = None
    if osm_parklar is not None:
        park_veri_kaynak = f"OpenStreetMap ({PARK_JSON})"
    else:
        osm_parklar, park_hata = osm_parklari_cek()
        park_veri_kaynak = "OpenStreetMap (canlı)" if osm_parklar is not None else "yalnızca 5 pilot park"
    pilot, osm_satir = osm_parklari_hazirla(osm_parklar, PARKLAR)
    parklar_tum = parklari_guncelle(pd.concat([pilot, osm_satir], ignore_index=True), yuzey)
    nufus = nufus_oku(NUFUS_CSV, mtime(NUFUS_CSV))
    obilgi = None
    if yuzey is not None:
        yuzey, obilgi = oncelik_hesapla(yuzey, parklar_tum, tesisler, nufus)
    parklar_tum = park_nufus_guncelle(parklar_tum, yuzey)
    parklar_g = park_sec(parklar_tum, yuzey, n_osm)
    adaylar = aday_noktalar(yuzey, parklar_tum)
    parklar = pd.concat([parklar_g, adaylar], ignore_index=True)
    parklar, kirilgan_parklar, kirilgan_kaynak = kirilganlik_ata(parklar, tesisler)
    tablo = park_saat_tablosu(hava, parklar)
    girdi = girdileri_hazirla(tablo, parklar, w_kirilganlik, erisim)

    if yuzey is not None:
        tarih_bilgi = yuzey["landsat_tarih"].iloc[0] if "landsat_tarih" in yuzey else "?"
        yuzey_kaynak = (f"Landsat 9 ({tarih_bilgi}, sabit taban harita, saatlik ölçekli), {len(yuzey)} kara altıgeni; "
                        f"{yuzey.attrs.get('su_altigen', 0)} deniz/su ve {yuzey.attrs.get('kiyi_altigen', 0)} kıyı karışık "
                        f"piksel altıgeni analiz dışı; NDVI kara altıgenlerinin %{100 * yuzey.attrs.get('ndvi_pay', 0):.0f}'"
                        "sinde var (olmayanda gölge 0)")
    else:
        yuzey_kaynak = f"yok ({ISI_YUZEYI_CSV} bulunamadı)"
    st.caption(f"Hava: {hava_kaynak} · Güneşin MRT katkısı: {mrt_yontem} · Isı yüzeyi: {yuzey_kaynak} · "
               f"Tesisler: {osm_kaynak} · Parklar: {park_veri_kaynak}"
               f"{'' if park_hata is None else ' (' + park_hata + ')'}; plana {len(parklar_g)} park · "
               f"Nüfus: {'WorldPop 2025 toplam nüfus; 65+ yalnızca ilçe oranıyla' if nufus is not None else 'yok'} · "
               "Temsili: hat kapasitesi, sulanan alan/pay, pompa, kişi/saat erişim, gölge etkinliği, saatlik yüzey "
               "ölçeği sınırları, sisleme istasyon alanı · Etki eğrileri Paris, Melbourne, Adelaide ve Arizona saha "
               "ölçümlerine göre ölçeklendi; Antalya saha doğrulaması 2027 pilotunda.")

    plan, esit, ozet, durum = plan_hesapla(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt,
                                           turler, park_kesme, park_mm)
    if plan is None:
        st.error(f"Plan bulunamadı (çözücü durumu: {durum}). Gelişmiş ayarlarda kırılgan payı veya nem kapısını gevşetin.")
        st.stop()

    if not ozet["isi_stresi_var"]:
        st.info("Bu gün hiçbir saatte güçlü ısı stresi (UTCI ≥ 32 °C) beklenmiyor; soğutma sulaması önerilmiyor.")
    else:
        def pay(d):
            return d.loc[d["park"].isin(kirilgan_parklar), "su_m3"].sum() / max(d["su_m3"].sum(), 1e-9) * 100
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Serinlik kazancı", f"{ozet['artis'] * 100:+.0f}%", "gündüz eşit park sulamasına göre",
                  delta_color="off", help="Öncelik ağırlıklı UTCI düşüşü, aynı su miktarıyla. Model simülasyonudur.")
        alt_satir = []
        if ozet["artis_aralik"]:
            a, b = ozet["artis_aralik"]
            alt_satir.append(f"Belirsizlik aralığı {a * 100:+.0f}% … {b * 100:+.0f}%")
        if ozet["yalniz_park"] is not None:
            alt_satir.append(f"Yeni yöntem eklemeden, yalnız park sulamasının saat ve yer seçimiyle "
                             f"{ozet['yalniz_park'] * 100:+.0f}%")
        if alt_satir:
            m1.caption(" · ".join(alt_satir))
        m2.metric("Serinlik (kişi·°C·saat)", f"{ozet['kds']:,.0f}".replace(",", "."),
                  f"eşit dağıtım {ozet['kds_esit']:,.0f}".replace(",", "."), delta_color="off",
                  help="Her saat, ulaşılan kişi × UTCI düşüşü toplamı (güçlü ısı stresi saatleri). Erişim temsilidir. "
                       "Bugünkü gece sulamasının gündüz katkısı 0 kabul edildiği için bu değer gece tabanına göre "
                       "kazancın kendisidir.")
        m3.metric("Kullanılan su", f"{plan['su_m3'].sum():.0f} m³", f"bütçe {butce} m³", delta_color="off")
        m3.caption(" · ".join(f"{YONTEMLER[k][0]} {v_:.1f} m³" for k, v_ in ozet["su_tur"].items() if v_ > 0.001))
        m4.metric("Kırılgan noktalara giden su", f"%{pay(plan):.0f}", f"eşit dağıtımda %{pay(esit):.0f}",
                  delta_color="off", help=f"Gösterge kaynağı: {kirilgan_kaynak}.")
        if ozet["dusus_aralik"]:
            a, b = ozet["dusus_aralik"]
            m4.caption(f"Noktada UTCI düşüşü {a:.1f}–{b:.1f} °C")
        kullanilan = float(plan["su_m3"].sum())
        if butce - kullanilan > max(5.0, 0.05 * butce):
            neden = ("kırılgan bölgelere asgari pay kuralı ve çim güvenliği sınırları"
                     if kirilgan_pay > 0 else "çim güvenliği sınırları ve hat kapasiteleri")
            st.info(f"Bütçenin {butce - kullanilan:.0f} m³'ü kullanılmadı: {neden} nedeniyle daha fazla su faydalı "
                    "biçimde dağıtılamıyor.")
        st.caption("Bugünkü yaygın uygulama gece sulamasıdır; gündüz ısı stresine doğrudan katkısı yok kabul edilir. "
                   "Karşılaştırma, daha zor bir referans olan gündüz eşit park sulamasıyla yapılır.")

    # --- Harita ---
    st.subheader("Isı haritası ve saatlik plan")
    gorunumler = {"Yüzey sıcaklığı": "LST", "UTCI: mevcut": "Önce", "UTCI: planla": "Sonra", "Soğuma etkisi": "Soğuma",
                  "Soğutma önceliği": "Öncelik"}
    if obilgi is not None:
        def yuzde(x):
            return str(round(x, 1)).replace(".", ",")
        pb, tb = obilgi["pilot"], obilgi["tum"]
        with st.container(border=True):
            metin = (f"**Park erişim boşluğu.** {obilgi['pilot_sayisi']} pilot parkın 5 dakikalık (400 m) yürüme alanı, "
                     f"yerleşim alanının (~{obilgi['yerlesim_km2']:.0f} km²) yalnızca **%{yuzde(pb['park400_pct'])}**'ini "
                     f"kapsıyor. Yüksek ve çok yüksek öncelikli {obilgi['yuksek_km2']:.0f} km²'nin "
                     f"**%{pb['yuksek_park_disi_pct']:.0f}**'i pilot parklara 400 m'den uzak.")
            if obilgi.get("nufus_var"):
                metin += (f" Yerleşimdeki ~{obilgi['yerlesim_nufus']:,.0f}".replace(",", ".")
                          + f" kişinin **%{yuzde(pb['park400_nufus_pct'])}**'i bir pilot parka 5 dk yürüme mesafesinde "
                          "(WorldPop 2025).")
            if obilgi["park_sayisi"] > obilgi["pilot_sayisi"]:
                metin += (f" OpenStreetMap parklarıyla birlikte {obilgi['park_sayisi']} park (park kenarına uzaklık) hesaba "
                          f"katılınca: yerleşim "
                          f"alanının %{yuzde(tb['park400_pct'])}'i"
                          + (f", nüfusun %{yuzde(tb['park400_nufus_pct'])}'i" if obilgi.get("nufus_var") else "")
                          + " bir parka 400 m içinde.")
            st.markdown(metin)
            st.caption(f"Öncelik: {obilgi['agirlik']} (ağırlıklı çakıştırma; parka uzaklık ölçütü tüm parklardan). "
                       f"Yerleşim alanı: {obilgi['kural']}.")
    katman_secenekleri = ["Nokta adları", "Yürüme mesafesi (5 ve 10 dk)", "Hassas tesisler (OSM)"]

    def oynat_degistir():
        if st.session_state.oynuyor:
            st.session_state.oynuyor = False
        else:
            st.session_state.oynuyor = True
            st.session_state.oyun_yeni = True
            st.session_state.saat = SAATLER[0]

    cbs_klasor = "haritalar"
    cbs_haritalar = {"Yüzey sıcaklığı (Landsat)": "harita_1", "Soğutma önceliği": "harita_2",
                     "Erişim boşluğu ve müdahale planı": "harita_3"}
    cbs_var = all(os.path.exists(os.path.join(cbs_klasor, f + ".png")) for f in cbs_haritalar.values())
    canli = "Uydu haritası (etkileşimli)"
    harita_turleri = [canli] + (["CBS haritası (QGIS)"] if cbs_var else []) + ["Etkileşimli 3B"]
    c_btn, c_gor, c_tur, c_kat = st.columns([0.8, 2.6, 1.6, 2.2], vertical_alignment="bottom")
    c_btn.button("Durdur" if st.session_state.oynuyor else "Günü oynat", on_click=oynat_degistir,
                 type="primary", key="oynat_dugmesi")
    cbs_secim = None
    if yuzey is not None:
        harita_turu = c_tur.selectbox("Harita", harita_turleri, index=0)
        if harita_turu == "CBS haritası (QGIS)":
            cbs_secim = c_gor.radio("CBS haritası", list(cbs_haritalar), horizontal=True)
            mod = "Öncelik" if cbs_secim == "Soğutma önceliği" else "Önce"
        else:
            mod = gorunumler[c_gor.radio("Görünüm", list(gorunumler), index=0, horizontal=True)]
    else:
        mod, harita_turu = "Önce", "Etkileşimli 2B"
    boyut3 = harita_turu == "Etkileşimli 3B"
    secenekler = c_kat.multiselect("Ek katmanlar", katman_secenekleri, default=katman_secenekleri[:2])

    @st.fragment(run_every=OYNATMA_ADIM_S if st.session_state.oynuyor else None)
    def zaman_paneli():
        if st.session_state.oynuyor:
            if not st.session_state.pop("oyun_yeni", False):
                st.session_state.saat = min(st.session_state.saat + 1, SAATLER[-1])
            if st.session_state.saat >= SAATLER[-1]:
                st.session_state.oynuyor = False
                st.rerun()
        st.select_slider("Saat", options=SAATLER, key="saat", format_func=lambda s: f"{s:02d}:00",
                         label_visibility="collapsed")
        saat = st.session_state.saat
        c_harita, c_kart = st.columns([3.3, 1])
        tesis_k = tesisler if "Hassas tesisler (OSM)" in secenekler else None
        if cbs_secim is not None:
            dosya = os.path.join(cbs_klasor, cbs_haritalar[cbs_secim])
            c_harita.image(dosya + ".png", width="stretch")
            c_harita.caption("QGIS 3.34 ile üretilmiş sabit harita (v5 modeliyle). Saat saat ısı stresi için "
                             "\"Uydu haritası\"nı seçin.")
            if os.path.exists(dosya + ".pdf") and not st.session_state.oynuyor:
                with open(dosya + ".pdf", "rb") as f:
                    c_harita.download_button("Haritayı indir (PDF)", f.read(),
                                             file_name=f"coolscape_{cbs_haritalar[cbs_secim]}.pdf",
                                             mime="application/pdf", key="cbs_indir")
        elif harita_turu == canli and yuzey is not None:
            m, bilgi = canli_harita(mod, saat, hava, yuzey, parklar, plan, secenekler, tesis_k,
                                    "Uydu (Esri World Imagery)")
            canli_harita_goster(c_harita, m)
            c_harita.caption(bilgi["not"] + ". Renk geçişi altıgen değerlerinin ~180 m yumuşatılmış hâlidir; üzerine "
                             "gelince altıgenin kendi değeri görünür.")
        else:
            mod_3b = "Önce" if mod == "LST" else mod
            deck, ist = harita_katmanlari(saat, hava, yuzey, parklar, plan, sicak_pay, mod_3b, secenekler, tesis_k,
                                          boyut3)
            c_harita.markdown(lejant_html(mod_3b, ist, sicak_pay, tesis_k is not None, boyut3), unsafe_allow_html=True)
            c_harita.pydeck_chart(deck, height=600, key="harita")
            if ist is not None:
                c_harita.caption(f"{saat:02d}:00 · Planın soğuttuğu alan (≥0,1 °C): ~{ist['alan_km2']:.1f} km², en çok "
                                 f"{ist['azami']:.1f} °C · sıcak nokta alanı {ist['once']} → {ist['sonra']} altıgen.")
        with c_kart.container(border=True):
            saat_karti(saat, hava, plan, vpd_min)

    zaman_paneli()
    if "Hassas tesisler (OSM)" in secenekler and tesisler is None:
        st.caption(f"OpenStreetMap tesisleri yüklenemedi ({osm_hata}); katman gösterilmiyor.")

    st.subheader("Müdahale noktaları")
    st.dataframe(park_ozet_tablosu(parklar, plan, tesisler), hide_index=True)
    st.caption("Sisleme ve ıslatma noktaları soğutma önceliği en yüksek yerleşim altıgenlerinden seçilen, birbirinden ve "
               "park kenarlarından en az 600 m uzak ADAY konumlardır. Kesin yer sahada belediyeyle seçilir. "
               f"Sisleme istasyonu ~{SISLEME_ALAN_M2} m² (temsili), saatte {SISLEME_SU_M3_SAAT:.2f} m³ içme suyu.")

    st.subheader("Günün su planı (nokta × saat, m³; • = sisleme açık)")
    st.altair_chart(plan_matrisi(plan))

    st.subheader("Operatör onay listesi")
    komut = operator_komutlari(plan, parklar, aritilmis_senaryo)
    if komut.empty:
        st.write("Bu koşullarda sulama önerilmiyor.")
    else:
        komut.insert(0, "Onay", False)
        st.data_editor(komut, hide_index=True, disabled=["Nokta", "Yöntem", "Başlangıç", "Bitiş", "Su (m³)", "Kaynak"])
        st.download_button("Planı CSV olarak indir", komut.drop(columns="Onay").to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"coolscape_plan_{tarih}.csv", mime="text/csv")

    ekonomi_bolumu(plan, parklar_g)

    with st.expander("Nem kapısı kontrolü (kıyı)"):
        nem_kapisi_bolumu(hava, girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, turler,
                          park_kesme, park_mm)

    with st.expander("Bütçe–serinlik eğrisi: ek su ne kazandırır?"):
        if st.button("Eğriyi hesapla"):
            satirlar = []
            for b in sorted({int(butce * k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)}):
                p_b, d_b = optimize_et(girdi, b, set(kirilgan_parklar), kirilgan_pay, vpd_min, enerji_opt,
                                       turler=turler, park_kesme=park_kesme, park_gunluk_mm=park_mm)
                if d_b == "Optimal":
                    satirlar.append({"Bütçe (m³)": b, "Senaryo": "CoolScape", "Fayda": etki(p_b)["agirlikli_fayda"].sum()})
                satirlar.append({"Bütçe (m³)": b, "Senaryo": "Gündüz eşit park sulaması",
                                 "Fayda": etki(esit_dagitim(girdi, b, park_kesme, park_mm))["agirlikli_fayda"].sum()})
            egri = pd.DataFrame(satirlar)
            st.altair_chart(alt.Chart(egri).mark_line(point=True).encode(
                x="Bütçe (m³):Q", y=alt.Y("Fayda:Q", title="Öncelik ağırlıklı serinlik"),
                color=alt.Color("Senaryo:N", scale=alt.Scale(range=["#1e88e5", "#999999"]))))

    with st.expander("Neden? (teknik ayrıntı ve varsayımlar)"):
        s = st.session_state.saat
        d = plan[plan["saat"] == s][["park", "tur", "mrt_c", "utci_c", "vpd_kpa", "kisi", "oncelik", "emax", "su_m3",
                                     "su_mm", "utci_dusus_c"]].round(2)
        d["tur"] = d["tur"].map(lambda t: YONTEMLER[t][0])
        d.columns = ["Nokta", "Yöntem", "MRT (°C)", "UTCI (°C)", "VPD (kPa)", "Erişim (kişi)", "Öncelik",
                     "Azami etki (°C)", "Su (m³)", "Su (mm = L/m²)", "UTCI düşüşü (°C)"]
        st.markdown(f"**{s:02d}:00 saatinin hesap tablosu**")
        st.dataframe(d, hide_index=True)
        st.dataframe(parklar_g[["park", "alan_ha", "park_kaynak", "yuzey_farki_c", "yuzey_kaynak", "golge_ortu",
                                "golge_kaynak", "nufus_skoru", "nufus_kaynak"]]
                     .rename(columns={"park": "Park", "alan_ha": "OSM alanı (ha)", "park_kaynak": "Park kaynağı",
                                      "yuzey_farki_c": "Çevre yüzey farkı (°C)", "yuzey_kaynak": "Kaynak",
                                      "golge_ortu": "Gölge örtüsü (0–1)", "golge_kaynak": "Kaynak ",
                                      "nufus_skoru": "Nüfus skoru (1–10)", "nufus_kaynak": "Kaynak  "}),
                     hide_index=True)
        kd = parklar[["park", "kirilgan_endeks"]].copy()
        kd["Kırılgan sayılır"] = kd["park"].isin(kirilgan_parklar).map({True: "evet", False: "hayır"})
        st.markdown(f"**Kırılganlık göstergesi** (kaynak: {kirilgan_kaynak}). Kırılgan nokta: göstergesi tüm noktaların "
                    "medyanında ya da üstünde.")
        st.dataframe(kd.rename(columns={"park": "Nokta", "kirilgan_endeks": "Gösterge (0–1)"}), hide_index=True)
        st.markdown(
            "- **Öncelik** = max(UTCI − 32, 0) × erişim (kişi) / 100 × (1 + kırılganlık ağırlığı × gösterge).\n"
            f"- **MRT ({SURUM})** = T + (1 − {GOLGE_ETKINLIK:.1f} × gölge örtüsü) × güneş katkısı + 0,5 × yüzey sıcaklığı "
            f"farkı × g(saat). Gölge örtüsü = (NDVI − {GOLGE_NDVI[0]}) / ({GOLGE_NDVI[1]} − {GOLGE_NDVI[0]}), 0–1 "
            "(temsili eşleme; NDVI yoksa 0). g(saat) = o saatin küresel ışınımı / 11–12 arası ışınım (Landsat geçişi), "
            f"{LST_OLCEK_SINIR[0]}–{LST_OLCEK_SINIR[1]} aralığında (sınırlar temsili). Güneş katkısı: {mrt_yontem}; "
            "SolarCal varsayımları: ayakta kişi, gökyüzü görüş oranı 0,5, güneş yandan, giysi yutuculuğu 0,7. "
            "Bina gölgesi hesaba katılmaz.\n"
            f"- **Sisleme ({SURUM}):** istasyon ~{SISLEME_ALAN_M2} m² ({SISLEME_ALAN_M2 // SISLEME_BIRIM_M2} × "
            f"{SISLEME_BIRIM_M2} m² birim), saatte {SISLEME_SU_M3_SAAT:.2f} m³; alan içinde {SISLEME_E} °C × "
            "min(VPD / 2,5 kPa, 1) UTCI düşüşü, dışında 10 m ölçekle azalır. Yalnızca içme suyuyla çalışır.\n"
            "- **Park sulama ve yol ıslatma:** UTCI düşüşü = 1,0 °C × min(VPD / 2,5, 1) × (1 − e^(−doz/x₀)), "
            "x₀ = 0,5 mm/saat (duyarlılık 0,25–1,0 mm). Belirsizlik bandı 0,5–2,0 °C.\n"
            f"- **Çim güvenliği:** park sulaması {park_kesme}:00'dan sonra yok; park başına günlük en çok {park_mm} mm.\n"
            "- **Parklar:** 5 pilot park + OpenStreetMap parkları (≥0,5 ha). OSM park alanı gerçek; sulanan pay "
            f"%{SULANAN_PAY * 100:.0f}, hat kapasitesi {HAT_HA_BASINA:.0f} m³/saat/ha (en çok {HAT_UST:.0f}) temsili. "
            "Parka uzaklık park kenarına (merkez uzaklığı − eşdeğer yarıçap) göredir.\n"
            "- **Soğutma önceliği:** yüzey ısısı 0,40, 65+ sayısı (ilçe oranıyla) 0,25, parka uzaklık 0,20, 400 m "
            "içindeki okul/sağlık tesisi 0,15. Ağırlıklar uzman varsayımıdır.\n"
            "- **Landsat** yerel sabah geçer; taban harita yüzeyin göreli ısınma desenini gösterir, saatlik ölçek "
            "bu deseni gün içinde büyütür ya da küçültür.\n"
            "- **UTCI sınıf eşikleri** resmîdir (26/32/38/46 °C); renkler bu uygulamanın seçimidir.")
        st.markdown("**Etki eğrilerinin dayandığı saha ölçümleri**")
        st.dataframe(pd.DataFrame([
            ["Hendel vd. 2016, Urban Climate", "Paris, Louvre", "Kaldırım ıslatma", "−0,79 °C", "−1,03 °C", "Merkezi değer"],
            ["Hendel vd. 2016, Urban Climate", "Paris, Belleville", "Kaldırım ıslatma", "−0,60 °C", "−0,93 °C", "Merkezi değer"],
            ["Parison vd. 2020", "Paris (2013–2015)", "Kaldırım ıslatma", "−1,02 °C", "−1,93 °C", "Üst band"],
            ["Traill vd. 2024, Urban Climate", "Melbourne", "Kaldırım ıslatma (~1 mm)", "en çok −0,6 °C", "−0,2 … −2,0 °C", "Band"],
            ["Broadbent vd. 2018, Urban Climate", "Adelaide (model)", "Zemin/çim sulama", "günlük ort. en çok −2,3 °C",
             "–", "Doyma: ~20 L/m²/gün üzeri ek fayda yok"],
            ["Vanos vd. 2022, Int. J. Biometeorology", "Tempe, Arizona", "Sisleme (restoran terasları)", "–",
             "ort. −4,4 °C (gölgeyle −9,7 °C)", "Sisleme etkisi"],
        ], columns=["Çalışma", "Yer", "Uygulama", "Hava sıcaklığı", "UTCI", "Modelde kullanımı"]), hide_index=True)
        st.caption("Sınırlar: Paris ve Melbourne ölçümleri kaldırım ıslatmadır; park çim sulamasında etki farklı olabilir. "
                   "Sisleme etkisi Arizona'nın kuru koşulunda ölçülmüştür. Hava tek noktadandır; kıyı-iç nem farkı "
                   "modelde yok. Eğri yerel ölçümle doğrulanmalıdır (2027 saha pilotu).")


def ekonomi_bolumu(plan, parklar_g):
    st.subheader("Su kaynağı ve ekonomi (senaryo)")
    st.caption("Danışman geri bildirimi: parkların bugün hangi suyla sulandığı, maliyeti ve arıtılmış suya geçişte "
               "tasarruf. Pilot parklarla arıtma tesisleri arasında sulama hattı bağlantısı: BİLGİ YETERSİZ. Arıtılmış "
               "su kullanımı için Teknik Usuller Tebliği Ek-7 kriterleri ve Bakanlık uygun görüşü gerekir "
               "(RG 25.10.2022, 31994, md.18).")
    park_ha = float(parklar_g["sulanan_alan_ha"].sum())
    c1, c2, c3 = st.columns(3)
    kaynak = c1.selectbox("Bugünkü sulama kaynağı", ["Bilinmiyor (BİLGİ YETERSİZ)", "ASAT şebekesi", "Kuyu", "Dere/kanal"])
    birim_mevcut = c1.number_input("Mevcut su birim maliyeti (TL/m³)", value=ASAT_TARIFE_TL_M3, min_value=0.0,
                                   step=0.5, help=ASAT_TARIFE_NOT)
    senaryo = c1.radio("Senaryo", ["S1: arıtma hattı bağlantısı var", "S2: bağlantı yok, yatırım gerekir"])
    sulanan_ha = c2.number_input("Sulanan alan (ha)", value=round(park_ha, 2), min_value=0.0, step=0.5,
                                 help="Varsayılan: plandaki parkların sulanan alanı (temsili).")
    mm_gun = c2.number_input("Mevsim ortalaması günlük sulama (mm)", value=SULAMA_MM_GUN, min_value=0.0, step=0.5)
    gun = c2.number_input("Yıllık sulama günü", value=SULAMA_GUN, min_value=0, step=10)
    birim_aritilmis = c3.number_input("Arıtılmış su birim maliyeti (TL/m³)", value=None, min_value=0.0, step=0.5,
                                      placeholder="BİLGİ YETERSİZ: girin")
    aritilmis_pay = c3.slider("Arıtılmış suyla karşılanan pay (%)", 0, 100, 100 if senaryo.startswith("S1") else 80) / 100
    yatirim = c3.number_input("Yatırım (TL)", value=None, min_value=0.0, step=1_000_000.0,
                              placeholder="S2: hat + ileri arıtma + depo; girin")
    isletme = c3.number_input("Ek yıllık işletme (TL/yıl)", value=0.0, min_value=0.0, step=100_000.0)
    gun_plan_m3 = float(plan.loc[plan["tur"].isin(["park", "islatma"]), "su_m3"].sum())
    e = ekonomi_hesapla(sulanan_ha, mm_gun, gun, birim_mevcut, birim_aritilmis, aritilmis_pay, yatirim, isletme,
                        gun_plan_m3)
    birim_etiket = ("vekil (ASAT tarifesi; parklara uygulandığı doğrulanmadı)"
                    if abs(birim_mevcut - ASAT_TARIFE_TL_M3) < 1e-9 else "kullanıcı girdisi")
    tablo = pd.DataFrame([
        ["Bugünkü kaynak", kaynak, "kullanıcı girdisi" if not kaynak.startswith("Bilinmiyor") else "BİLGİ YETERSİZ"],
        ["Mevcut birim maliyet", f"{tl(birim_mevcut, 2)} TL/m³", birim_etiket],
        ["Yıllık sulama suyu", f"{tl(e['yillik_m3'])} m³", "temsili (alan × mm × gün)"],
        ["Yıllık mevcut maliyet", f"{tl(e['mevcut_tl'])} TL" if e["mevcut_tl"] is not None else "girilmedi",
         "türetilmiş"],
        ["Sıcak gün planı (park + ıslatma)", f"{tl(gun_plan_m3, 1)} m³ · {tl(e['gun_plan_tl'])} TL",
         "model çıktısı × birim maliyet"],
        ["Arıtılmış suyla ikame edilen", f"{tl(e['ikame_m3'])} m³/yıl", "temsili (pay)"],
        ["Arıtılmış su birim maliyeti",
         f"{tl(birim_aritilmis, 2)} TL/m³" if birim_aritilmis is not None else "girilmedi",
         "kullanıcı girdisi" if birim_aritilmis is not None else "BİLGİ YETERSİZ"],
        ["Yıllık net tasarruf", f"{tl(e['tasarruf_tl'])} TL" if e["tasarruf_tl"] is not None else "girilmedi", "türetilmiş" if e["tasarruf_tl"] is not None else "hesaplanamadı"],
        ["Yatırım", f"{tl(yatirim)} TL" if yatirim is not None else "girilmedi",
         "kullanıcı girdisi" if yatirim is not None else "BİLGİ YETERSİZ"],
        ["Basit geri dönüş",
         f"{tl(e['geri_donus_yil'], 1)} yıl" if e["geri_donus_yil"] is not None else "hesaplanamadı",
         "türetilmiş"],
    ], columns=["Kalem", "Değer", "Etiket"])
    st.dataframe(tablo, hide_index=True)
    st.caption("Sisleme her senaryoda içme suyuyla çalışır ve tasarrufa dahil edilmez. Finansman kanalları (Dünya Bankası "
               "GFC/İLBANK, AFD) ve kaynaklar: CoolScape_MEVZUAT_EKONOMI.md.")


def nem_kapisi_bolumu(hava, girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, turler, park_kesme,
                      park_mm):
    h = hava[hava["saat"].isin(SAATLER)].copy()
    h["vpd_kpa"] = vpd_kpa(h["sicaklik_c"], h["bagil_nem"])
    h["kapali"] = h["vpd_kpa"] < vpd_min
    h["etki_carpani"] = np.minimum(h["vpd_kpa"] / VPD_DOYMA, 1.0)
    st.markdown(f"Kapı {vpd_min:.1f} kPa: {int(h['kapali'].sum())} / {len(h)} saat kapalı. Hava tek noktadandır "
                f"({ANTALYA_LAT}, {ANTALYA_LON}); kıyı şeridinin daha nemli olup olmadığı bu veriyle ayırt edilemez "
                "(BİLGİ YETERSİZ; MGM kıyı istasyonu gerekir).")
    st.dataframe(h[["saat", "sicaklik_c", "bagil_nem", "vpd_kpa", "etki_carpani", "kapali"]].round(2).rename(
        columns={"saat": "Saat", "sicaklik_c": "Sıcaklık (°C)", "bagil_nem": "Nem (%)", "vpd_kpa": "VPD (kPa)",
                 "etki_carpani": "Etki çarpanı", "kapali": "Kapı kapalı"}), hide_index=True)
    if st.button("Eşik duyarlılığını hesapla"):
        satirlar = []
        for esik in (0.0, 0.4, 0.8, 1.2, 1.6):
            p, d = optimize_et(girdi, butce, set(kirilgan_parklar), kirilgan_pay, esik, enerji_opt, turler=turler,
                               park_kesme=park_kesme, park_gunluk_mm=park_mm)
            if d == "Optimal":
                e = etki(p)
                satirlar.append({"Eşik (kPa)": esik, "Kullanılan su (m³)": round(float(e["su_m3"].sum()), 1),
                                 "Öncelik ağırlıklı serinlik": round(float(e["agirlikli_fayda"].sum()), 2),
                                 "Kişi·°C·saat": round(float(e["kisi_derece_saat"].sum()), 0)})
        st.dataframe(pd.DataFrame(satirlar), hide_index=True)
        st.caption("Etki zaten VPD ile orantılı küçüldüğü için kapı yalnızca suyun düşük getirili saatlere gitmesini "
                   "engeller. Serinlik eşikle birlikte belirgin düşüyorsa kapı fazla sıkıdır.")


if __name__ == "__main__":
    ana()

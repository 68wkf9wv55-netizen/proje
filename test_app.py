"""CoolScape AI testleri. Sentetik hava YALNIZCA test içindir; çıktı yüzdeleri raporlanmaz.
Sandbox Open-Meteo, Overpass, harita karoları ve Streamlit sitesine erişemez: ağ çağrıları taklit edilir."""
import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
import coolscape_app as ca  # noqa: E402


def sentetik_saatlik(tarih="2025-07-25"):
    """TEST VERİSİ: tipik sıcak gün şekli; gerçek gözlem değildir."""
    saat = np.arange(24)
    t = 30 + 12 * np.clip(np.sin((saat - 6) / 14 * np.pi), 0, None)
    rh = 70 - 45 * np.clip(np.sin((saat - 6) / 14 * np.pi), 0, None)
    sw = 950 * np.clip(np.sin((saat - 6) / 13 * np.pi), 0, None)
    return {"time": [f"{tarih}T{h:02d}:00" for h in saat], "temperature_2m": t.round(1).tolist(),
            "relative_humidity_2m": rh.round(0).tolist(), "wind_speed_10m": [2.5] * 24,
            "shortwave_radiation": sw.round(0).tolist(), "direct_normal_irradiance": (sw * 0.85).round(0).tolist()}


class _Yanit:
    def __init__(self, veri):
        self._v = veri

    def raise_for_status(self):
        pass

    def json(self):
        return self._v


def sahte_get(url, params=None, timeout=None, **_):
    if "open-meteo" in url:
        return _Yanit({"hourly": sentetik_saatlik(params["start_date"])})
    raise ConnectionError("sandbox: ağ yok")


def sahte_post(*_, **__):
    raise ConnectionError("sandbox: Overpass yok")


@pytest.fixture
def calisma(tmp_path, monkeypatch):
    for f in ("coolscape_app.py", "isi_yuzeyi_h3.csv", "nufus_h3.csv", "tesisler_osm.json"):
        shutil.copy(os.path.join(KOK, f), tmp_path / f)
    monkeypatch.chdir(tmp_path)
    import requests
    monkeypatch.setattr(requests, "get", sahte_get)
    monkeypatch.setattr(requests, "post", sahte_post)
    return tmp_path


def test_poligon_alani():
    # 100 m × 100 m kare ≈ 1 ha
    dlat, dlon = 100 / 110_540, 100 / (111_320 * math.cos(math.radians(36.9)))
    kare = [(36.9, 30.7), (36.9, 30.7 + dlon), (36.9 + dlat, 30.7 + dlon), (36.9 + dlat, 30.7)]
    alan, merkez = ca.poligon_alan_merkez(kare)
    assert abs(alan - 10_000) < 50
    assert abs(merkez[0] - (36.9 + dlat / 2)) < 1e-6


def test_golge_ve_saatlik_olcek():
    m0 = ca.mrt_hesapla(40.0, 30.0, 0.0, 2.0, 1.0)
    m1 = ca.mrt_hesapla(40.0, 30.0, 1.0, 2.0, 1.0)
    assert m0 == pytest.approx(40 + 30 + 1.0)
    assert m1 == pytest.approx(40 + 0.2 * 30 + 1.0)
    assert ca.ndvi_ortu([np.nan, 0.1, 0.4, 0.9]).tolist() == pytest.approx([0, 0, 0.5, 1])
    h = pd.DataFrame({"zaman": pd.to_datetime(sentetik_saatlik()["time"])})
    v = sentetik_saatlik()
    h = ca._hava_tablosu(v)
    h = ca.gunes_mrt_ekle(h)
    assert h["lst_olcek"].between(*ca.LST_OLCEK_SINIR).all()
    assert float(h.loc[h["saat"] == ca.LST_REF_SAAT, "lst_olcek"].iloc[0]) == pytest.approx(1.0)


def test_yayilim_cekirdegi():
    k = ca.yayilim_cekirdegi([0, 9, 19, 200], 9.8, 10, 40)
    assert k[0] == 1 and k[1] == 1 and 0 < k[2] < 1 and k[3] == 0


def test_ekonomi():
    e = ca.ekonomi_hesapla(10, 5, 180, 33.31, None, 1.0, None, 0, 100)
    assert e["yillik_m3"] == pytest.approx(10 * 10 * 5 * 180)
    assert e["tasarruf_tl"] is None and e["geri_donus_yil"] is None
    e = ca.ekonomi_hesapla(10, 5, 180, 33.31, 10.0, 0.5, 1_000_000, 0, 100)
    assert e["tasarruf_tl"] == pytest.approx(90_000 * 0.5 * 23.31)
    assert e["geri_donus_yil"] == pytest.approx(1_000_000 / e["tasarruf_tl"])


def _girdi():
    h = ca.gunes_mrt_ekle(ca._hava_tablosu(sentetik_saatlik()))
    p = ca.PARKLAR.copy()
    p["golge_ortu"] = 0.0
    p, kir, _ = ca.kirilganlik_ata(p, None)
    return ca.girdileri_hazirla(ca.park_saat_tablosu(h, p), p, 0.5, ca.ERISIM_VARSAYILAN), kir


def test_milp_kisitlari():
    g, kir = _girdi()
    plan, durum = ca.optimize_et(g, 200, set(kir), 0.4, 0.8, True, park_kesme=18, park_gunluk_mm=6.0)
    assert durum == "Optimal"
    assert plan["su_m3"].sum() <= 200 + 1e-6
    assert plan.loc[plan["saat"] >= 18, "su_m3"].sum() == 0
    for park, gr in plan.groupby("park"):
        assert gr["su_m3"].sum() <= 6.0 * gr["alan_m2"].iloc[0] / 1000 + 1e-6
    assert (plan.loc[plan["vpd_kpa"] < 0.8, "su_m3"] == 0).all()
    assert (plan["su_m3"] <= plan["kapasite"] + 1e-6).all()


def test_osm_park_ayrisma():
    veri = json.load(open(os.path.join(KOK, "tests", "parklar_test.json"), encoding="utf-8"))
    df = ca.osm_park_tablosu(veri["elements"])
    assert len(df) == 3
    pilot, yeni = ca.osm_parklari_hazirla(df, ca.PARKLAR)
    assert pilot["alan_ha"].notna().sum() == 1      # Karaalioğlu'na 150 m içindeki poligon pilot sayılır
    assert len(yeni) == 1                           # 0,5 ha altı elenir


def _uygulama():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.abspath("coolscape_app.py"), default_timeout=600)
    at.run()
    return at


def test_uygulama_pilot(calisma):
    at = _uygulama()
    assert not at.exception, at.exception
    assert any("Su kaynağı ve ekonomi" in s.value for s in at.subheader)
    assert any("yalnızca 5 pilot park" in c.value for c in at.caption)
    metrikler = {m.label for m in at.metric}
    assert "Kullanılan su" in metrikler


def test_uygulama_osm_parklari(calisma):
    shutil.copy(os.path.join(KOK, "tests", "parklar_test.json"), calisma / "parklar_osm.json")
    at = _uygulama()
    assert not at.exception, at.exception
    assert any("parklar_osm.json" in c.value for c in at.caption)


def test_uygulama_etkilesim(calisma):
    at = _uygulama()
    at.sidebar.radio[0].set_value("Kırılgan nüfusa öncelik").run()
    assert not at.exception, at.exception
    for b in at.button:
        if b.label == "Eşik duyarlılığını hesapla":
            b.click().run()
            break
    assert not at.exception, at.exception

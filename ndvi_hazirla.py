"""
CoolScape AI – NDVI'yi tüm H3 altıgenleri için yeniden üretir (Sentinel-2 L2A, 10 m).

Neden: mevcut isi_yuzeyi_h3.csv içinde NDVI yalnızca kuzeydoğu köşedeki 165 altıgende var (sahne pilot alanın
tamamını kapsamıyordu). Gölge/ağaç örtüsü (MRT) ve park yeşil skoru NDVI'ye bağlı.

Girdi: aynı güne ait Sentinel-2 L2A karoları. Her karo için B04 (kırmızı) ve B08 (yakın kızılötesi) 10 m bandı,
isteğe bağlı SCL (sahne sınıflandırması, 20 m) bulut/gölge/su maskesi için. Pilot alan birden çok karoya düşebilir;
tüm karoları verin, pikseller altıgen bazında birleştirilir.
İndirme (elle): Copernicus Browser (https://browser.dataspace.copernicus.eu) → Sentinel-2 L2A → tarih 24.07.2025
(Landsat 25.07.2025'e en yakın açık gün) → pilot alanı kapsayan karolar → B04, B08, SCL (.jp2).

Çıktı: isi_yuzeyi_h3.csv'nin ndvi ve s2_tarih sütunları güncellenir (diğer sütunlar değişmez). Yedek: *.bak.

Kurulum: pip install rasterio pyproj h3 pandas numpy
Çalıştırma:
  python ndvi_hazirla.py --tarih 2025-07-24 \
      --karo T36SUF_B04_10m.jp2 T36SUF_B08_10m.jp2 T36SUF_SCL_20m.jp2 \
      --karo T36SVF_B04_10m.jp2 T36SVF_B08_10m.jp2 T36SVF_SCL_20m.jp2
  (SCL yoksa yalnızca B04 ve B08 verin.)

Kaynak: Copernicus Sentinel-2 verisi (ESA), Copernicus lisansı. Atıf korunur.
"""

import argparse
import os
import shutil

import h3
import numpy as np
import pandas as pd

BBOX = (30.62, 36.85, 30.80, 36.97)          # batı, güney, doğu, kuzey (uygulamadakiyle aynı)
SCL_GECERSIZ = {0, 1, 3, 6, 8, 9, 10, 11}     # veri yok, doymuş, bulut gölgesi, su, bulut (orta/yüksek), sirrus, kar
L2A_OFSET = 1000                              # PB ≥ 04.00 (2022 sonrası) L2A yansıtma ofseti: (DN − 1000) / 10000


def karo_oku(b04, b08, scl=None, ofset=L2A_OFSET):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform, transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(b04) as kr, rasterio.open(b08) as nir:
        sinir = transform_bounds("EPSG:4326", kr.crs, *BBOX)
        w = from_bounds(*sinir, transform=kr.transform).round_offsets().round_lengths()
        w = w.intersection(rasterio.windows.Window(0, 0, kr.width, kr.height))
        r = kr.read(1, window=w).astype("float64")
        n = nir.read(1, window=w).astype("float64")
        t = kr.window_transform(w)
        crs = kr.crs
        gecerli = (r > 0) & (n > 0)
        if scl:
            with rasterio.open(scl) as s:
                sw = from_bounds(*rasterio.windows.bounds(w, kr.transform), transform=s.transform)
                sc = s.read(1, window=sw, out_shape=r.shape, resampling=Resampling.nearest)
            gecerli &= ~np.isin(sc, list(SCL_GECERSIZ))
    r = (r - ofset) / 10000.0
    n = (n - ofset) / 10000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (n - r) / (n + r)
    gecerli &= np.isfinite(ndvi) & (np.abs(ndvi) <= 1)
    satir, sutun = np.nonzero(gecerli)
    x = t.c + (sutun + 0.5) * t.a
    y = t.f + (satir + 0.5) * t.e
    lon, lat = transform(crs, "EPSG:4326", x.tolist(), y.tolist())
    return pd.DataFrame({"lat": lat, "lon": lon, "ndvi": ndvi[satir, sutun]})


def altigene_topla(pikseller, cozunurluk):
    pikseller = pikseller.copy()
    pikseller["hex"] = [h3.latlng_to_cell(a, b, cozunurluk) for a, b in zip(pikseller["lat"], pikseller["lon"])]
    g = pikseller.groupby("hex")["ndvi"]
    return pd.DataFrame({"ndvi": g.mean(), "ndvi_piksel": g.size()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--karo", nargs="+", action="append", required=True,
                    help="B04 B08 [SCL] dosya yolları; her karo için ayrı --karo")
    ap.add_argument("--tarih", required=True, help="Sentinel-2 alım tarihi, YYYY-AA-GG")
    ap.add_argument("--yuzey", default="isi_yuzeyi_h3.csv")
    ap.add_argument("--min-piksel", type=int, default=50, help="altıgende en az geçerli 10 m piksel (res 9 ≈ 1.050)")
    ap.add_argument("--ofset", type=int, default=L2A_OFSET, help="L2A yansıtma ofseti (eski PB için 0)")
    a = ap.parse_args()
    parcalar = []
    for k in a.karo:
        if len(k) not in (2, 3):
            raise SystemExit("--karo: B04 B08 [SCL] verin")
        print("okunuyor:", ", ".join(os.path.basename(f) for f in k))
        parcalar.append(karo_oku(k[0], k[1], k[2] if len(k) == 3 else None, a.ofset))
    pikseller = pd.concat(parcalar, ignore_index=True)
    y = pd.read_csv(a.yuzey)
    cozunurluk = h3.get_resolution(y["hex"].iloc[0])
    ag = altigene_topla(pikseller, cozunurluk)
    ag = ag[ag["ndvi_piksel"] >= a.min_piksel]
    shutil.copy(a.yuzey, a.yuzey + ".bak")
    y = y.drop(columns=[c for c in ("ndvi", "ndvi_piksel") if c in y]).merge(ag, left_on="hex", right_index=True,
                                                                             how="left")
    y["s2_tarih"] = a.tarih
    y["ndvi"] = y["ndvi"].round(4)
    y.to_csv(a.yuzey, index=False)
    kara = y["lst_c"] >= 36 if "lst_c" in y else np.ones(len(y), bool)
    print(f"Yazıldı: {a.yuzey} · NDVI dolu kara altıgeni %{100 * y.loc[kara, 'ndvi'].notna().mean():.1f} "
          f"({int(y['ndvi'].notna().sum())}/{len(y)}) · yedek {a.yuzey}.bak")


if __name__ == "__main__":
    main()

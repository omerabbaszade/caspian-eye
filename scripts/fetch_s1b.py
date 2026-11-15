# Sentinel-1 RTC (orbit 108, descending): reads the Kura Delta window onto a 20 m UTM 39N grid and writes dB*100 int16 GeoTIFFs
import sys, os, json, time, datetime as dt
import numpy as np, rasterio, pystac_client, planetary_computer
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
OUT = sys.argv[1]
CRS = "EPSG:32639"; RES = 20
l, b, r, t = transform_bounds("EPSG:4326", CRS, 48.85, 38.85, 49.6, 39.6)
l, b = np.floor(l/RES)*RES, np.floor(b/RES)*RES; r, t = np.ceil(r/RES)*RES, np.ceil(t/RES)*RES
W, H = int((r-l)/RES), int((t-b)/RES); T = from_origin(l, t, RES, RES)
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
def scenes(dt_range):
    its = list(cat.search(collections=["sentinel-1-rtc"], bbox=[48.85,38.85,49.6,39.6], datetime=dt_range,
                          query={"sat:relative_orbit": {"eq": 108}}).items())
    by = {}
    for it in its: by.setdefault(str(it.datetime.date()), []).append(it)
    return by
targets = []
for y in range(2017, 2027):  # closest to 15 October each year
    by = scenes(f"{y}-09-20/{y}-11-10")
    if by:
        d = min(by, key=lambda s: abs((dt.date.fromisoformat(s) - dt.date(y, 10, 15)).days)); targets.append((d, by[d]))
by26 = scenes("2026-06-01/2026-11-14")
for m in [6, 7, 8, 9, 11]:  # 2026 seasonal series (October is already above)
    d = min(by26, key=lambda s: abs((dt.date.fromisoformat(s) - dt.date(2026, m, 20 if m == 6 else 15)).days)); targets.append((d, by26[d]))
from concurrent.futures import ThreadPoolExecutor
uniq = {}
for d, its in targets: uniq[d] = its
def job(arg):
    d, items = arg
    fn = os.path.join(OUT, f"s1_{d}.tif")
    if os.path.exists(fn): return
    t0 = time.time(); out = np.zeros((2, H, W), np.int16); out[:] = -32768
    for bi, band in enumerate(["vv", "vh"]):
        acc = np.full((H, W), np.nan, np.float32)
        for it in items:
            # OVERVIEW_LEVEL=0 -> 20 m overview level (4x less data)
            with rasterio.open(it.assets[band].href, OVERVIEW_LEVEL=0) as src, WarpedVRT(src, crs=CRS, transform=T, width=W, height=H,
                    resampling=Resampling.average, src_nodata=-32768, nodata=np.nan, dtype="float32") as v:
                a = v.read(1)
            m = np.isnan(acc) & np.isfinite(a) & (a > 0)
            acc[m] = a[m]
        ok = np.isfinite(acc)
        out[bi][ok] = np.clip(np.round(10*np.log10(np.maximum(acc[ok], 1e-5))*100), -6000, 3000).astype(np.int16)
    prof = dict(driver="GTiff", width=W, height=H, count=2, dtype="int16", crs=CRS, transform=T, nodata=-32768, compress="deflate", tiled=True)
    with rasterio.open(fn + ".part", "w", **prof) as dst:
        dst.write(out); dst.set_band_description(1, "VV_dBx100"); dst.set_band_description(2, "VH_dBx100")
    os.replace(fn + ".part", fn)
    print(d, [i.id[:3] for i in items], f"{time.time()-t0:.0f}s", flush=True)
json.dump([dict(date=d, items=[i.id for i in its], platform=its[0].properties.get("platform")) for d, its in sorted(uniq.items())],
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("targets", sorted(uniq), "grid", W, H, flush=True)
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE"):
    with ThreadPoolExecutor(3) as ex:
        for _ in ex.map(job, sorted(uniq.items())): pass
print("done", flush=True)

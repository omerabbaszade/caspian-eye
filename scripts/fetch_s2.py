# Sentinel-2 L2A: picks cloud-free days and reads RGB (TCI), B03/B08 for NDWI and SCL onto the same 20 m grid as S1
import sys, os, json, time
import numpy as np, rasterio, pystac_client, planetary_computer
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from shapely.geometry import shape, box
from shapely.ops import unary_union
OUT = sys.argv[1]
CRS = "EPSG:32639"; RES = 20
l, b, r, t = transform_bounds("EPSG:4326", CRS, 48.85, 38.85, 49.6, 39.6)
l, b = np.floor(l/RES)*RES, np.floor(b/RES)*RES; r, t = np.ceil(r/RES)*RES, np.ceil(t/RES)*RES
W, H = int((r-l)/RES), int((t-b)/RES); T = from_origin(l, t, RES, RES)
aoi = box(48.85, 38.85, 49.6, 39.6)
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
meta = []
for label, rng in [("2026", "2026-09-15/2026-11-12"), ("2017", "2017-09-15/2017-11-15"), ("2026jun", "2026-06-05/2026-07-10")]:
    its = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(aoi.bounds), datetime=rng,
                          query={"eo:cloud_cover": {"lt": 20}}).items())
    by = {}
    for it in its: by.setdefault(str(it.datetime.date()), []).append(it)
    cands = []
    for d, g in by.items():
        cov = unary_union([shape(i.geometry) for i in g]).intersection(aoi).area / aoi.area
        cc = max(i.properties["eo:cloud_cover"] for i in g)
        cands.append((cov >= 0.97, -cc, cov, d))
    cands.sort(reverse=True)
    print(label, [(c[3], round(c[2], 2), -c[1]) for c in cands[:6]], flush=True)
    d = cands[0][3]; items = by[d]
    t0 = time.time()
    layers = {}
    for band, rs in [("B04", Resampling.average), ("B03", Resampling.average), ("B02", Resampling.average), ("B08", Resampling.average), ("SCL", Resampling.nearest)]:
        acc = np.zeros((H, W), np.float32); have = np.zeros((H, W), bool)
        for it in items:
            # read the 20 m overview level of the 10 m bands (4x less data)
            with rasterio.open(it.assets[band].href, **({"OVERVIEW_LEVEL": 0} if band != "SCL" else {})) as src, WarpedVRT(src, crs=CRS, transform=T, width=W, height=H, resampling=rs, src_nodata=0, nodata=0) as v:
                a = v.read(1).astype(np.float32)
            m = (~have) & (a > 0); acc[m] = a[m]; have |= m
        layers[band] = acc
    # L2A offset: baseline 04.00+ (from 2022-01-25) DN-1000
    off = 1000.0 if d >= "2022-01-25" else 0.0
    refl = {k: np.clip((layers[k] - off) / 10000.0, 0, 1) for k in ["B04", "B03", "B02", "B08"]}
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (refl["B03"] - refl["B08"]) / (refl["B03"] + refl["B08"])
    prof = dict(driver="GTiff", width=W, height=H, dtype="int16", crs=CRS, transform=T, nodata=-32768, compress="deflate", tiled=True)
    out = np.stack([np.round(refl[k]*10000).astype(np.int16) for k in ["B04", "B03", "B02", "B08"]] +
                   [np.where(np.isfinite(ndwi), np.round(ndwi*1000), -32768).astype(np.int16), layers["SCL"].astype(np.int16)])
    with rasterio.open(os.path.join(OUT, f"s2_{d}.tif"), "w", count=out.shape[0], **prof) as dst:
        dst.write(out)
    meta.append(dict(label=label, date=d, items=[i.id for i in items], cloud=[i.properties["eo:cloud_cover"] for i in items]))
    print(label, d, f"{time.time()-t0:.0f}s", flush=True)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

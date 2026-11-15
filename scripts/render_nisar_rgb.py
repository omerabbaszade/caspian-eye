# NISAR GCOV RGB display images: R=HH (−22…−4 dB), G=HV (−26…−12 dB), B=HH−HV (0…12 dB)
import sys, os, glob, math, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nisar_classifier import smooth_db
D, WEB = sys.argv[1], sys.argv[2]
AOI = (48.85, 38.85, 49.6, 39.6); DW = 1000; DLON = (AOI[2] - AOI[0]) / DW
DLAT = DLON * math.cos(math.radians((AOI[1] + AOI[3]) / 2)); DH = int(round((AOI[3] - AOI[1]) / DLAT)); DT = from_origin(AOI[0], AOI[3], DLON, DLAT)
for fn in sorted(glob.glob(os.path.join(D, "nisar_gcov", "gcov_*.tif"))):
    d8 = os.path.basename(fn)[-12:-4]; d = f"{d8[:4]}-{d8[4:6]}-{d8[6:]}"
    with rasterio.open(fn) as src:
        a = src.read().astype(np.float32); t, crs = src.transform, src.crs
    v = (a[0] > -32768) & (a[1] > -32768)
    hh = smooth_db(a[0] / 100, v, 3); hv = smooth_db(a[1] / 100, v, 3); del a
    chans = [np.clip((hh + 22) / 18, 0, 1), np.clip((hv + 26) / 14, 0, 1), np.clip((hh - hv) / 12, 0, 1)]
    out = []
    for c in chans:
        o = np.full((DH, DW), np.nan, np.float32)
        reproject(np.where(v, c, np.nan).astype(np.float32), o, src_transform=t, src_crs=crs, dst_transform=DT, dst_crs="EPSG:4326", resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        out.append(o)
    img = np.where(np.isfinite(out), np.clip(np.nan_to_num(np.stack(out)) * 255, 0, 255), 0).astype(np.uint8)
    Image.fromarray(np.moveaxis(img, 0, -1), "RGB").save(os.path.join(WEB, "img", f"ngc_{d}.jpg"), quality=84)
    print(d, flush=True)

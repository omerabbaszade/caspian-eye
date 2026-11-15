# Stable water / stable land mask over all 2026 S1 dates (reference areas for the NISAR classifier)
import sys, os, json, glob, numpy as np, rasterio
from scipy import ndimage
D = sys.argv[1]
thr = json.load(open(os.path.join(D, "web", "data.json")))["thresholds"]
fs = sorted(glob.glob(os.path.join(D, "s1", "s1_2026-*.tif")))
allw = allland = None
for f in fs:
    d = os.path.basename(f)[3:13]
    with rasterio.open(f) as src:
        vh = src.read(2).astype(np.float32); prof = src.profile
    v = vh > -32768
    lin = np.where(v, 10 ** (vh / 1000), 0); del vh
    s = 10 * np.log10(np.maximum(ndimage.uniform_filter(lin, 3), 1e-7)); del lin
    w = ndimage.median_filter((v & (s < thr[d]["vh"])).astype(np.uint8), 3).astype(bool) & v; del s
    allw = w if allw is None else (allw & w)
    allland = (v & ~w) if allland is None else (allland & v & ~w)
    print(d, flush=True)
# stay 2 pixels away from edges
sw = ndimage.binary_erosion(allw, iterations=2); sl = ndimage.binary_erosion(allland, iterations=2)
out = np.zeros(sw.shape, np.uint8); out[sl] = 1; out[sw] = 2
prof.update(count=1, dtype="uint8", nodata=0)
with rasterio.open(os.path.join(D, "s1", "stable2026.tif"), "w", **prof) as dst: dst.write(out, 1)
print("stable water km2", round(sw.sum() * 0.0004, 1), "stable land km2", round(sl.sum() * 0.0004, 1))

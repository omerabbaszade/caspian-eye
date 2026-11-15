# NISAR L2 GCOV: reads the Kura Delta window (HH, HV γ0) remotely and writes it on the same 20 m grid as S1
# Usage: python fetch_nisar_gcov.py <caspian-eye folder> <stack, e.g. 165_D_069_2005>
import sys, os, json, glob, time
import numpy as np, rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, transform_bounds, Resampling
from affine import Affine
from concurrent.futures import ThreadPoolExecutor
D, STACK = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(D, "scripts"))
from h5open import open_h5
OUT = os.path.join(D, "nisar_gcov"); os.makedirs(OUT, exist_ok=True)
with rasterio.open(sorted(glob.glob(os.path.join(D, "s1", "s1_*.tif")))[0]) as ref:
    GT, GCRS, H, W = ref.transform, ref.crs, ref.height, ref.width
gx0, gy0 = GT.c, GT.f  # top-left corner of the grid
rank = {"V1": 2, "PROVISIONAL_V1": 1, "BETA_V1": 0}
best = {}
for f in glob.glob(os.path.join(D, "data", "cmr_*.json")):
    coll = os.path.basename(f)[len("cmr_NISAR_L2_GCOV_"):-5]
    for it in json.load(open(f))["items"]:
        g = it["umm"]; p = g["GranuleUR"].split("_")
        if "_".join(p[5:9]) != STACK: continue
        d = p[11][:8]
        url = [r["URL"] for r in g["RelatedUrls"] if r.get("Type") == "GET DATA" and r["URL"].startswith("https") and r["URL"].endswith(".h5")][0]
        cand = (rank.get(coll, 0), g["GranuleUR"], url, coll)
        if d not in best or cand > best[d]: best[d] = cand
print(STACK, len(best), "dates:", sorted(best), flush=True)
meta_path = os.path.join(OUT, "meta.json")
def job(item):
    d, (_, ur, url, coll) = item
    fn = os.path.join(OUT, f"gcov_{STACK}_{d}.tif")
    if os.path.exists(fn):
        # already downloaded: return metadata only
        with rasterio.open(fn) as src: valid = float((src.read(1) > -32768).mean())
        return dict(date=f"{d[:4]}-{d[4:6]}-{d[6:]}", granule=ur, collection=coll, stack=STACK, valid=round(valid, 3), dx=20.0)
    t0 = time.time()
    for attempt in range(3):
        try:
            h = open_h5(url)
            g = h["science/LSAR/GCOV/grids/frequencyA"]
            x = g["xCoordinates"][:]; y = g["yCoordinates"][:]
            dx = float(g["xCoordinateSpacing"][()]); dy = float(g["yCoordinateSpacing"][()])
            # The file's own projection (UTM 38N or 39N depending on the frame)
            src_crs = CRS.from_epsg(int(g["projection"][()]))
            # NISAR coordinates are pixel centres
            ex0, ey0 = x[0] - dx / 2, y[0] - dy / 2
            # Our grid bounds in the file's CRS (+1 km margin)
            l, b, r, t = transform_bounds(GCRS, src_crs, gx0, gy0 - H * 20, gx0 + W * 20, gy0)
            sc0 = max(int((l - 1000 - ex0) / dx), 0); sc1 = min(int((r + 1000 - ex0) / dx) + 1, len(x))
            sr0 = max(int((t + 1000 - ey0) / dy), 0); sr1 = min(int((b - 1000 - ey0) / dy) + 1, len(y))
            out = np.full((2, H, W), -32768, np.int16)
            if sc1 > sc0 and sr1 > sr0:
                st = Affine(dx, 0, ex0 + sc0 * dx, 0, dy, ey0 + sr0 * dy)
                for bi, term in enumerate(["HHHH", "HVHV"]):
                    a = g[term][sr0:sr1, sc0:sc1].astype(np.float32)
                    a[~(np.isfinite(a) & (a > 0))] = np.nan
                    dst = np.full((H, W), np.nan, np.float32)
                    if src_crs == GCRS and abs(dx) == 20:
                        # same grid: copy directly
                        reproject(a, dst, src_transform=st, src_crs=src_crs, dst_transform=GT, dst_crs=GCRS, resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
                    else:
                        reproject(a, dst, src_transform=st, src_crs=src_crs, dst_transform=GT, dst_crs=GCRS, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
                    ok = np.isfinite(dst) & (dst > 0)
                    out[bi][ok] = np.clip(np.round(10 * np.log10(dst[ok]) * 100), -6000, 3000).astype(np.int16)
            h.close()
            prof = dict(driver="GTiff", width=W, height=H, count=2, dtype="int16", crs=GCRS, transform=GT, nodata=-32768, compress="deflate", tiled=True)
            with rasterio.open(fn + ".part", "w", **prof) as dst:
                dst.write(out); dst.set_band_description(1, "HH_gamma0_dBx100"); dst.set_band_description(2, "HV_gamma0_dBx100")
            os.replace(fn + ".part", fn)
            valid = float((out[0] > -32768).mean())
            print(d, coll, f"valid={valid:.2f}", f"{time.time() - t0:.0f}s", flush=True)
            return dict(date=f"{d[:4]}-{d[4:6]}-{d[6:]}", granule=ur, collection=coll, stack=STACK, valid=round(valid, 3), dx=dx)
        except Exception as e:
            print(d, "error", type(e).__name__, str(e)[:200], "attempt", attempt + 1, flush=True); time.sleep(5)
    return None
with ThreadPoolExecutor(int(os.environ.get('NISAR_THREADS', '1'))) as ex:
    res = [r for r in ex.map(job, sorted(best.items())) if r]
old = json.load(open(meta_path)) if os.path.exists(meta_path) else []
keep = {(m["stack"], m["date"]): m for m in old}
for r in res: keep[(r["stack"], r["date"])] = r
json.dump(sorted(keep.values(), key=lambda m: (m["stack"], m["date"])), open(meta_path, "w"), indent=1)
print("done", flush=True)

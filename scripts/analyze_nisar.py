# NISAR L2 GCOV (L-band, HH+HV γ0): water and flooded-vegetation classification for the Kura Delta
# Input: nisar_gcov/gcov_*.tif (HH, HV dB*100, 20 m UTM39N), s1/*.tif, zones.json, web/data.json (S1 thresholds)
# Output: web/img/ngc_*.jpg (RGB), web/img/ncls_*.png (classes), web/nisar_gcov.json
import sys, os, json, glob, math
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import rasterize
from scipy import ndimage
from PIL import Image
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as stransform
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nisar_classifier import smooth_db, fit_predict

D = sys.argv[1]
WEB = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "web")  # folder of the published page
AOI = (48.85, 38.85, 49.6, 39.6)
PX_KM2 = 0.0004
DW = 1000; DLON = (AOI[2] - AOI[0]) / DW; DLAT = DLON * math.cos(math.radians((AOI[1] + AOI[3]) / 2))
DH = int(round((AOI[3] - AOI[1]) / DLAT)); DT = from_origin(AOI[0], AOI[3], DLON, DLAT)
S1D = json.load(open(os.path.join(WEB, "data.json")))
ZONES = json.load(open(os.path.join(D, "zones.json")))


def to_display(arr, t, crs, rs, dtype=np.float32, nodata=np.nan):
    out = np.full((DH, DW), nodata, dtype)
    reproject(arr.astype(dtype), out, src_transform=t, src_crs=crs, dst_transform=DT, dst_crs="EPSG:4326",
              resampling=rs, src_nodata=nodata, dst_nodata=nodata)
    return out


def otsu(x, lo, hi, bins=256):
    h, e = np.histogram(x, bins=bins, range=(lo, hi)); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def smooth(db, valid, k=3):
    lin = np.where(valid, 10 ** (db / 10), 0)
    num = ndimage.uniform_filter(lin, k); den = ndimage.uniform_filter(valid.astype(np.float32), k)
    return np.where(valid, 10 * np.log10(np.maximum(num / np.maximum(den, 1e-6), 1e-7)), np.nan)


def read_db(fn):
    with rasterio.open(fn) as src:
        a = src.read().astype(np.float32); t, crs = src.transform, src.crs
    valid = (a[0] > -32768) & (a[1] > -32768)
    return np.where(valid, a[0] / 100, np.nan), np.where(valid, a[1] / 100, np.nan), valid, t, crs


files = sorted(glob.glob(os.path.join(D, "nisar_gcov", "gcov_*.tif")))
meta = {(m["stack"], m["date"]): m for m in json.load(open(os.path.join(D, "nisar_gcov", "meta.json")))}
_, _, _, GT, GCRS = read_db(files[0])
SHAPE = rasterio.open(files[0]).shape
tr = Transformer.from_crs("EPSG:4326", GCRS.to_string(), always_xy=True)
ZM = {z["id"]: rasterize([(mapping(stransform(lambda x, y: tr.transform(x, y), Polygon(z["coords"]))), 1)],
                         out_shape=SHAPE, transform=GT, fill=0, dtype="uint8").astype(bool) for z in ZONES}
ZM["AOI"] = np.ones(SHAPE, bool)
with rasterio.open(os.path.join(D, "s1", "stable2026.tif")) as src:
    STABLE = src.read(1)  # 1 = land on every 2026 S1 date, 2 = water on every date (training areas)

# --- S1 water maps (same rules, 20 m) for comparison
s1_cache = {}


def s1_water(d):
    if d in s1_cache: return s1_cache[d]
    _, vh, v, _, _ = read_db(os.path.join(D, "s1", f"s1_{d}.tif"))
    w = v & (smooth(vh, v) < S1D["thresholds"][d]["vh"])
    w = ndimage.median_filter(w.astype(np.uint8), 3).astype(bool) & v
    s1_cache.clear(); s1_cache[d] = (w, v)
    return w, v


# Common footprint: pixels imaged on every NISAR date (so coverage gaps do not distort km²)
COMMON = None
for fn in files:
    with rasterio.open(fn) as src:
        vv_ = src.read(1) > -32768
    COMMON = vv_ if COMMON is None else (COMMON & vv_)
common_area = {z: round(float((COMMON & m).sum() * PX_KM2), 1) for z, m in ZM.items()}
print("common footprint km2", common_area, flush=True)
rows, cls_list, conf_list, dates, hist = [], [], [], [], {}
for fn in files:
    stack = "_".join(os.path.basename(fn)[5:-13].split("_"))
    d8 = os.path.basename(fn)[-12:-4]; d = f"{d8[:4]}-{d8[4:6]}-{d8[6:]}"
    hh, hv, v, _, _ = read_db(fn)
    hh_s, hv_s = smooth_db(hh, v), smooth_db(hv, v); del hh, hv
    # Histogram (diagnostics)
    hist[d] = dict(hh=np.percentile(hh_s[v], [5, 25, 50, 75, 95]).round(1).tolist(), hv=np.percentile(hv_s[v], [5, 25, 50, 75, 95]).round(1).tolist())
    # L-band: per-scene quadratic discriminant (HH, HV) trained on stable reference areas
    prob, cls_params = fit_predict(hh_s, hv_s, v, STABLE)
    water = ndimage.median_filter((v & (prob > 0.5)).astype(np.uint8), 3).astype(bool) & v
    # Flooded vegetation (double bounce): very strong HH and a large HH−HV difference, on non-water pixels
    fveg = v & ~water & (hh_s > -3.0) & ((hh_s - hv_s) > 10.0)
    cls = np.zeros(SHAPE, np.uint8); cls[v] = 1; cls[water] = 2; cls[fveg] = 3
    conf = np.where(v, np.abs(np.nan_to_num(prob) - 0.5) * 2 * 255, 0).astype(np.uint8); del prob
    # Comparison with S1 (nearest date)
    near = min(S1D["dates"], key=lambda x: abs((np.datetime64(x) - np.datetime64(d)).astype(int)))
    w1, v1 = s1_water(near)
    agree = {}
    for z, m in ZM.items():
        mm = m & v & v1
        n = max(mm.sum(), 1)
        agree[z] = dict(agreement=round(float(((water == w1) & mm).sum() / n), 4),
                        iou=round(float((water & w1 & mm).sum() / max(((water | w1) & mm).sum(), 1)), 4))
    per = {z: dict(water=round(float((water & m).sum() * PX_KM2), 2), fveg=round(float((fveg & m).sum() * PX_KM2), 2),
                   valid=round(float((v & m).sum() / max(m.sum(), 1)), 3),
                   hi_conf=round(float(((conf > 168) & m & v).sum() / max((m & v).sum(), 1)), 3)) for z, m in ZM.items()}
    for z, m in ZM.items():
        per[z]["water_common"] = round(float((water & COMMON & m).sum() * PX_KM2), 2)
    mt = meta.get((stack, d), {})
    # Reliability check: agreement also outside the S1 reference areas
    nonref = v & v1 & (STABLE == 0)
    agree_nonref = round(float(((water == w1) & nonref).sum() / max(nonref.sum(), 1)), 4)
    rows.append(dict(date=d, stack=stack, granule=mt.get("granule"), collection=mt.get("collection"), classes=cls_params, agree_nonref=agree_nonref,
                     s1_date=near, gap_days=int(abs((np.datetime64(near) - np.datetime64(d)).astype(int))), per_zone=per, vs_s1=agree))
    dates.append(d); cls_list.append(cls); conf_list.append(conf)
    # Display: RGB = HH, HV, HH−HV
    # Display stretch as in render_nisar_rgb.py: HH −22…−4, HV −26…−12, HH−HV 0…12 dB
    r = np.clip((hh_s + 22) / 18, 0, 1); g = np.clip((hv_s + 26) / 14, 0, 1); b = np.clip((hh_s - hv_s) / 12, 0, 1)
    rgb = np.stack([to_display(np.where(v, c, np.nan), GT, GCRS, Resampling.average) for c in (r, g, b)])
    img = np.where(np.isfinite(rgb), np.clip(rgb * 255, 0, 255), 0).astype(np.uint8)
    Image.fromarray(np.moveaxis(img, 0, -1), "RGB").save(os.path.join(WEB, "img", f"ngc_{d}.jpg"), quality=84)
    dcls = to_display(cls, GT, GCRS, Resampling.mode, np.uint8, 0)
    dconf = to_display(conf, GT, GCRS, Resampling.average, np.uint8, 0)
    dconf = (np.round(dconf / 85.0) * 85).astype(np.uint8)
    Image.fromarray(np.dstack([dcls, dconf, np.zeros_like(dcls)]), "RGB").save(os.path.join(WEB, "img", f"ncls_{d}.png"), optimize=True)
    print(d, cls_params, "water AOI", per["AOI"]["water"], "fveg B", per["B"]["fveg"], "agree S1", agree["AOI"]["agreement"], agree["B"]["agreement"], hist[d], flush=True)

# Pairwise change (only pixels valid on both dates)
pairs = {}
for i in range(len(dates)):
    for j in range(i + 1, len(dates)):
        a, b = cls_list[i], cls_list[j]
        both = (a > 0) & (b > 0)
        wa, wb = (a == 2) & both, (b == 2) & both
        fa, fb = (a == 3) & both, (b == 3) & both
        hc = (conf_list[i] > 168) & (conf_list[j] > 168)
        rec = {}
        for z, m in ZM.items():
            g_, l_ = ~wa & wb & m, wa & ~wb & m
            n = g_.sum() + l_.sum()
            rec[z] = dict(gain=round(float(g_.sum() * PX_KM2), 2), loss=round(float(l_.sum() * PX_KM2), 2),
                          stable=round(float((wa & wb & m).sum() * PX_KM2), 2),
                          wa=round(float((wa & m).sum() * PX_KM2), 2), wb=round(float((wb & m).sum() * PX_KM2), 2),
                          fa=round(float((fa & m).sum() * PX_KM2), 2), fb=round(float((fb & m).sum() * PX_KM2), 2),
                          area=round(float((both & m).sum() * PX_KM2), 1),
                          change_hi_conf=round(float(((g_ | l_) & hc).sum() / max(n, 1)), 3))
        pairs[f"{dates[i]}|{dates[j]}"] = rec

# 2026 L-band frequency: on how many dates flooded vegetation appears (Pilot B)
FV = np.stack([c == 3 for c in cls_list]).astype(np.float32).mean(0)
fv_any = {z: round(float(((FV > 0) & m).sum() * PX_KM2), 2) for z, m in ZM.items()}
fv_half = {z: round(float(((FV >= 0.5) & m).sum() * PX_KM2), 2) for z, m in ZM.items()}
# S1 (2026) water area on the same common footprint, to compare the two sensors
s1_common = {}
for d in [x for x in S1D["dates"] if x.startswith("2026")]:
    w1, v1 = s1_water(d)
    s1_common[d] = {z: round(float((w1 & COMMON & m).sum() * PX_KM2), 2) for z, m in ZM.items()}
json.dump(dict(dates=dates, rows=rows, pairs=pairs, fveg_any=fv_any, fveg_half=fv_half, hist=hist,
               common=dict(area=common_area, s1=s1_common)),
          open(os.path.join(WEB, "nisar_gcov.json"), "w"))
print("fveg any", fv_any, "fveg >=50%", fv_half)
print("done", len(dates))

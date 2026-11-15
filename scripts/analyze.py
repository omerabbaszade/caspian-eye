# CASPIAN EYE: Kura Delta pilot analysis
# Input: s1/*.tif (VV, VH dB*100, 20 m UTM39N), s2/*.tif, nisar_browse/raw + meta
# Output: PNG layers and data.json (statistics) in web/
import sys, os, json, glob, math
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin, Affine
from scipy import ndimage
from PIL import Image

S = sys.argv[1]
WEB = os.path.join(S, "web"); os.makedirs(os.path.join(WEB, "img"), exist_ok=True)
AOI = (48.85, 38.85, 49.6, 39.6)
PX_KM2 = 0.02 * 0.02  # 20 m pixel = 0.0004 km²

# Pilot zones (lon, lat), refined after visual checks
ZONES = json.load(open(os.path.join(S, "zones.json")))

# --- Display grid: EPSG:4326 with roughly square pixels in metres
DW = 1000
DLON = (AOI[2] - AOI[0]) / DW
DLAT = DLON * math.cos(math.radians((AOI[1] + AOI[3]) / 2))
DH = int(round((AOI[3] - AOI[1]) / DLAT))
DT = from_origin(AOI[0], AOI[3], DLON, DLAT)


def to_display(arr, src_t, src_crs, resampling, dtype=np.float32, nodata=np.nan):
    out = np.full((DH, DW), nodata, dtype)
    reproject(arr.astype(dtype), out, src_transform=src_t, src_crs=src_crs, dst_transform=DT,
              dst_crs="EPSG:4326", resampling=resampling, src_nodata=nodata, dst_nodata=nodata)
    return out


def otsu(x, lo, hi, bins=256):
    h, e = np.histogram(x, bins=bins, range=(lo, hi))
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(c[np.argmax(var)])


def zone_masks(transform, shape, crs):
    """Rasterize the zone polygons on the given grid."""
    from rasterio.features import rasterize
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as stransform
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    out = {}
    for z in ZONES:
        poly = stransform(lambda x, y: tr.transform(x, y), Polygon(z["coords"]))
        out[z["id"]] = rasterize([(mapping(poly), 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    out["AOI"] = np.ones(shape, bool)
    return out


# ================= Sentinel-1 =================
s1_files = sorted(glob.glob(os.path.join(S, "s1", "s1_*.tif")))
s1_meta = {m["date"]: m for m in json.load(open(os.path.join(S, "s1", "meta.json")))}
with rasterio.open(s1_files[0]) as src:
    GT, GCRS, GSHAPE = src.transform, src.crs, (src.height, src.width)
ZM = zone_masks(GT, GSHAPE, GCRS)

dates, cls_stack, conf_stack, thr = [], [], [], {}
for f in s1_files:
    d = os.path.basename(f)[3:13]
    with rasterio.open(f) as src:
        a = src.read().astype(np.float32)
    valid = (a[0] > -32768) & (a[1] > -32768)
    vv = np.where(valid, a[0] / 100, np.nan); vh = np.where(valid, a[1] / 100, np.nan)
    # Speckle: 3x3 mean on linear power
    def smooth(db):
        lin = np.where(valid, 10 ** (db / 10), 0)
        num = ndimage.uniform_filter(lin, 3); den = ndimage.uniform_filter(valid.astype(np.float32), 3)
        return np.where(valid, 10 * np.log10(np.maximum(num / np.maximum(den, 1e-6), 1e-6)), np.nan)
    vv_s, vh_s = smooth(vv), smooth(vh)
    t_vh = min(max(otsu(vh_s[valid], -35, -5), -26.0), -18.0)
    t_vv = min(max(otsu(vv_s[valid], -30, 5), -22.0), -12.0)
    water = valid & (vh_s < t_vh)
    # Wind-roughened sea can be bright in VV while VH stays low, so VH is the main criterion.
    # Flooded-vegetation candidate (double bounce): very strong VV and a large VV-VH difference
    fveg = valid & ~water & (vv_s > -3.0) & ((vv_s - vh_s) > 9.0)
    # Remove small specks (3x3 median)
    water = ndimage.median_filter(water.astype(np.uint8), 3).astype(bool) & valid
    cls = np.zeros(GSHAPE, np.uint8)
    cls[valid] = 1; cls[water] = 2; cls[fveg & ~water] = 3
    # Confidence: distance from the threshold (dB), 3 dB = full confidence
    conf = np.where(valid, np.clip(np.abs(vh_s - t_vh) / 3.0, 0, 1) * 255, 0).astype(np.uint8)  # 0..255
    dates.append(d); cls_stack.append(cls); conf_stack.append(conf)
    thr[d] = dict(vh=round(t_vh, 2), vv=round(t_vv, 2))
    # Display layers: grey radar image (VV) and class PNG (R=class, G=confidence)
    disp_vv = to_display(vv_s, GT, GCRS, Resampling.average)
    g = np.clip((disp_vv + 25) / 25 * 255, 0, 255)
    g = np.where(np.isfinite(disp_vv), g, 0).astype(np.uint8)
    Image.fromarray(g, "L").save(os.path.join(WEB, "img", f"s1vv_{d}.jpg"), quality=82)
    dcls = to_display(cls, GT, GCRS, Resampling.mode, np.uint8, 0)
    dconf = to_display(conf, GT, GCRS, Resampling.average, np.uint8, 0)
    dconf = (np.round(dconf / 85.0) * 85).astype(np.uint8)  # 0, 85, 170, 255
    Image.fromarray(np.dstack([dcls, dconf, np.zeros_like(dcls)]), "RGB").save(os.path.join(WEB, "img", f"cls_{d}.png"), optimize=True)
    print(d, thr[d], "water km2", round(water.sum() * PX_KM2, 1), flush=True)

cls_stack = np.stack(cls_stack); conf_stack = np.stack(conf_stack)
valid_all = (cls_stack > 0).all(0)
for k in ZM: ZM[k] = ZM[k] & valid_all  # compare only pixels that are valid on every date

# Zone areas per date
per_date = {}
for i, d in enumerate(dates):
    w = cls_stack[i] == 2; fv = cls_stack[i] == 3
    per_date[d] = {z: dict(water=round(float((w & m).sum() * PX_KM2), 2), fveg=round(float((fv & m).sum() * PX_KM2), 2),
                           area=round(float(m.sum() * PX_KM2), 1),
                           hi_conf=round(float(((conf_stack[i] > 168) & m).sum() / max(m.sum(), 1)), 3))
                   for z, m in ZM.items()}

# Pairwise change
from pyproj import Transformer
_tr = Transformer.from_crs(GCRS.to_string(), "EPSG:4326", always_xy=True)


def centroid(mask):
    """Centroid (lon, lat) of the changed pixels."""
    if mask.sum() < 25: return None
    r, c = np.nonzero(mask[::4, ::4]); r = r * 4; c = c * 4
    if len(r) == 0: return None
    x, y = GT * (c.mean() + 0.5, r.mean() + 0.5)
    lo, la = _tr.transform(x, y)
    return [round(lo, 4), round(la, 4)]


STRUCT8 = np.ones((3, 3), bool)
ZM2 = {z: m[::2, ::2] for z, m in ZM.items()}


def patches(mask):
    """Label connected areas in the mask decimated to 40 m."""
    lab, n = ndimage.label(mask[::2, ::2], structure=STRUCT8)
    return lab, n


def largest_in_zone(lab, n, zm2):
    if n == 0: return None
    cnt = np.bincount(lab[zm2], minlength=n + 1); cnt[0] = 0
    k = int(cnt.argmax())
    if cnt[k] < 10: return None
    r, c = np.nonzero((lab == k) & zm2)
    x, y = GT * (2 * c.mean() + 1, 2 * r.mean() + 1)
    lo, la = _tr.transform(x, y)
    return dict(km2=round(float(cnt[k] * 4 * PX_KM2), 1), lon=round(lo, 3), lat=round(la, 3))


pairs = {}
for i in range(len(dates)):
    for j in range(i + 1, len(dates)):
        wa, wb = cls_stack[i] == 2, cls_stack[j] == 2
        gain, loss = ~wa & wb, wa & ~wb
        hc = (conf_stack[i] > 168) & (conf_stack[j] > 168)
        lg, ng_ = patches(gain); ll, nl = patches(loss)
        rec = {}
        for z, m in ZM.items():
            g_, l_ = gain & m, loss & m
            n = g_.sum() + l_.sum()
            rec[z] = dict(gain=round(float(g_.sum() * PX_KM2), 2), loss=round(float(l_.sum() * PX_KM2), 2),
                          stable=round(float((wa & wb & m).sum() * PX_KM2), 2),
                          change_hi_conf=round(float(((g_ | l_) & hc).sum() / max(n, 1)), 3),
                          gain_patch=largest_in_zone(lg, ng_, ZM2[z]), loss_patch=largest_in_zone(ll, nl, ZM2[z]))
        pairs[f"{dates[i]}|{dates[j]}"] = rec

# Long term: October series each year -> "year the sea floor emerged"
yearly = [d for d in dates if d[5:7] in ("09", "10", "11") and (d[:4] != "2026" or d[5:7] == "10")]
yearly = sorted({d[:4]: d for d in yearly}.values())
Y = np.stack([cls_stack[dates.index(d)] == 2 for d in yearly])
years = [int(d[:4]) for d in yearly]
emerg = np.zeros(GSHAPE, np.uint16)
# Pixel was water in the first year and stayed dry in EVERY October from a given year on
for k in range(1, len(years)):
    cond = Y[0] & Y[k - 1] & (~Y[k:]).all(0) & valid_all
    emerg[cond & (emerg == 0)] = years[k]
emerg_km2 = {y: round(float(((emerg == y) & ZM["AOI"]).sum() * PX_KM2), 2) for y in years[1:]}
emerg_zone = {z: {y: round(float(((emerg == y) & m).sum() * PX_KM2), 2) for y in years[1:]} for z, m in ZM.items()}
# Opposite direction: lasting inundation (dry in the first year, water ever after)
flooded = np.zeros(GSHAPE, np.uint16)
for k in range(1, len(years)):
    cond = (~Y[0]) & (~Y[k - 1]) & Y[k:].all(0) & valid_all
    flooded[cond & (flooded == 0)] = years[k]

# 2026 seasonal water frequency
s26 = [d for d in dates if d.startswith("2026")]
F = np.stack([cls_stack[dates.index(d)] == 2 for d in s26]).mean(0)
freq_cls = np.zeros(GSHAPE, np.uint8)
freq_cls[valid_all & (F == 0)] = 1
freq_cls[valid_all & (F > 0) & (F < 0.99)] = 2  # seasonal / variable water
freq_cls[valid_all & (F >= 0.99)] = 3  # permanent water
freq_km2 = {z: dict(seasonal=round(float(((freq_cls == 2) & m).sum() * PX_KM2), 2),
                    permanent=round(float(((freq_cls == 3) & m).sum() * PX_KM2), 2)) for z, m in ZM.items()}

# Display PNGs: emergence year (R=year-2000), frequency class
de = to_display(emerg, GT, GCRS, Resampling.mode, np.uint16, 0)
df = to_display(flooded, GT, GCRS, Resampling.mode, np.uint16, 0)
dq = to_display(freq_cls, GT, GCRS, Resampling.mode, np.uint8, 0)
dfreq = to_display((F * 255).astype(np.uint8), GT, GCRS, Resampling.average, np.uint8, 0)
Image.fromarray(np.dstack([np.where(de > 0, de - 2000, 0).astype(np.uint8), np.where(df > 0, df - 2000, 0).astype(np.uint8), dq]), "RGB").save(
    os.path.join(WEB, "img", "trend.png"), optimize=True)
Image.fromarray(np.dstack([dfreq, dq, np.zeros_like(dq)]), "RGB").save(os.path.join(WEB, "img", "freq2026.png"), optimize=True)

# ================= Sentinel-2 (optical check) =================
s2 = []
s2_meta = {m["date"]: m for m in json.load(open(os.path.join(S, "s2", "meta.json")))} if os.path.exists(os.path.join(S, "s2", "meta.json")) else {}
for f in sorted(glob.glob(os.path.join(S, "s2", "s2_*.tif"))):
    d = os.path.basename(f)[3:13]
    with rasterio.open(f) as src:
        a = src.read().astype(np.float32)
    rgb = a[:3] / 10000.0; ndwi = a[4] / 1000.0; scl = a[5]
    ok = (a[0] > 0) & ~np.isin(scl, [0, 1, 3, 8, 9, 10])  # clouds, shadows and no-data removed
    # true-colour display image
    disp = np.stack([to_display(rgb[k], GT, GCRS, Resampling.average) for k in range(3)])
    img = np.clip((np.nan_to_num(disp) / 0.25) ** (1 / 1.6) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(np.moveaxis(img, 0, -1), "RGB").save(os.path.join(WEB, "img", f"s2_{d}.jpg"), quality=84)
    # nearest S1 date
    near = min(dates, key=lambda x: abs((np.datetime64(x) - np.datetime64(d)).astype(int)))
    i = dates.index(near)
    w1 = cls_stack[i] == 2; v1 = cls_stack[i] > 0
    w2 = ndwi > 0.0
    m = ok & v1
    tp = (w1 & w2 & m).sum(); fp = (w1 & ~w2 & m).sum(); fn = (~w1 & w2 & m).sum(); tn = (~w1 & ~w2 & m).sum()
    rec = dict(date=d, s1_date=near, gap_days=int(abs((np.datetime64(near) - np.datetime64(d)).astype(int))),
               overall_agreement=round(float((tp + tn) / max(m.sum(), 1)), 4),
               water_iou=round(float(tp / max(tp + fp + fn, 1)), 4),
               water_f1=round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4),
               s1_water_km2=round(float((w1 & m).sum() * PX_KM2), 1), s2_water_km2=round(float((w2 & m).sum() * PX_KM2), 1),
               compared_km2=round(float(m.sum() * PX_KM2), 1), clouds=s2_meta.get(d, {}).get("cloud"))
    s2.append(rec); print("S2 val", rec, flush=True)

# ================= NISAR browse (indicative) =================
nmeta = json.load(open(os.path.join(S, "nisar_browse", "nisar_browse_meta.json")))
nisar = []
for m in sorted(nmeta, key=lambda x: (x["stack"], x["date"])):
    a = np.asarray(Image.open(os.path.join(S, "nisar_browse", "raw", m["granule"] + ".png")).convert("RGB")).astype(np.float32)
    st = Affine(m["dx"], 0, m["ox"], 0, -m["dy"], m["oy"])
    valid = a.sum(2) > 0
    # R channel ~ HH (browse composite); 8-bit visualization, not calibrated
    hh = np.where(valid, a[..., 0], np.nan)
    disp = np.stack([to_display(np.where(valid, a[..., k], np.nan), st, "EPSG:4326", Resampling.average) for k in range(3)])
    dvalid = np.isfinite(disp[0])
    img = np.where(dvalid, np.nan_to_num(disp), 0).astype(np.uint8)
    tag = m["stack"].replace("_", "")[:7] + "_" + m["date"]
    Image.fromarray(np.moveaxis(img, 0, -1), "RGB").save(os.path.join(WEB, "img", f"nisar_{tag}.jpg"), quality=84)
    # Dark (water) class inside the AOI: Otsu, self-normalizing within the image
    lum = disp.mean(0)
    sel = dvalid
    t = otsu(lum[sel], 0, 255)
    wat = sel & (lum < t)
    px_km2 = (DLON * 111.32 * math.cos(math.radians(39.225))) * (DLAT * 110.95)
    # Agreement with 20 m S1 (nearest date, on the display grid)
    d_iso = f"{m['date'][:4]}-{m['date'][4:6]}-{m['date'][6:]}"
    near = min(dates, key=lambda x: abs((np.datetime64(x) - np.datetime64(d_iso)).astype(int)))
    c1 = np.asarray(Image.open(os.path.join(WEB, "img", f"cls_{near}.png")))[..., 0]
    mm = sel & (c1 > 0)
    w1 = c1 == 2
    agree = float(((w1 == wat) & mm).sum() / max(mm.sum(), 1))
    iou = float((w1 & wat & mm).sum() / max(((w1 | wat) & mm).sum(), 1))
    nisar.append(dict(stack=m["stack"], date=d_iso, granule=m["granule"], tag=tag, aoi_valid=m["aoi_valid"],
                      dark_km2=round(float(wat.sum() * px_km2), 1), valid_km2=round(float(sel.sum() * px_km2), 1),
                      s1_date=near, agreement_vs_s1=round(agree, 4), water_iou_vs_s1=round(iou, 4)))
    print("NISAR", tag, nisar[-1]["dark_km2"], round(agree, 3), round(iou, 3), flush=True)

out = dict(aoi=AOI, display=dict(w=DW, h=DH, dlon=DLON, dlat=DLAT), grid=dict(res_m=20, crs="EPSG:32639", shape=GSHAPE),
           zones=ZONES, dates=dates, s1_meta=s1_meta, thresholds=thr, per_date=per_date, pairs=pairs,
           yearly=yearly, emergence_km2=emerg_km2, emergence_zone=emerg_zone, freq2026=freq_km2, s26=s26,
           s2_validation=s2, nisar=nisar)
json.dump(out, open(os.path.join(WEB, "data.json"), "w"))
print("done", len(dates), "dates")

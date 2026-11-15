# Downloads NISAR GCOV LATLON browse images, georeferences them and crops them to the AOI
import json, sys, glob, os, urllib.request
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
data_dir, out = sys.argv[1], sys.argv[2]
AOI = (48.85, 38.85, 49.6, 39.6)  # lon_min, lat_min, lon_max, lat_max
WANT = {('165','D','069','2005'), ('056','A','021','2005')}
meta = []
seen = set()
for f in sorted(glob.glob(os.path.join(data_dir, 'cmr_*.json'))):
    for it in json.load(open(f))['items']:
        g = it['umm']; ur = g['GranuleUR']; p = ur.split('_')
        key = (p[5], p[6], p[7], p[8])
        if key not in WANT: continue
        date = p[11][:8]
        if (key, date) in seen: continue  # duplicate versions
        seen.add((key, date))
        url = [r['URL'] for r in g['RelatedUrls'] if r['URL'].startswith('https') and r['URL'].endswith('_LATLON.png')][0]
        pts = g['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
        raw = os.path.join(out, 'raw', ur + '.png')
        if not os.path.exists(raw):
            urllib.request.urlretrieve(url, raw)
        a = np.asarray(Image.open(raw).convert('RGB'))
        m = a.sum(2) > 0
        ys, xs = np.where(m)
        # corner pixels <-> extreme points of the footprint polygon
        lons = [q['Longitude'] for q in pts]; lats = [q['Latitude'] for q in pts]
        x0p, x1p = xs.min(), xs.max(); y0p, y1p = ys.min(), ys.max()
        lon0, lon1 = min(lons), max(lons); lat0, lat1 = min(lats), max(lats)
        dx = (lon1 - lon0) / (x1p - x0p); dy = (lat1 - lat0) / (y1p - y0p)
        ox = lon0 - x0p * dx; oy = lat1 + y0p * dy
        # check: each polygon vertex should fall on the edge of the valid area
        resid = []
        for lo, la in zip(lons[:-1], lats[:-1]):
            px = int(round((lo - ox) / dx)); py = int(round((oy - la) / dy))
            px = min(max(px, 0), a.shape[1]-1); py = min(max(py, 0), a.shape[0]-1)
            win = m[max(py-4,0):py+5, max(px-4,0):px+5]
            resid.append(bool(win.any()))
        c0 = int((AOI[0]-ox)/dx); c1 = int((AOI[2]-ox)/dx)
        r0 = int((oy-AOI[3])/dy); r1 = int((oy-AOI[1])/dy)
        crop = a[r0:r1, c0:c1]
        valid = (crop.sum(2) > 0).mean()
        name = f"{p[5]}{p[6]}{p[7]}_{date}"
        Image.fromarray(crop).save(os.path.join(out, 'crop', name + '.png'))
        meta.append(dict(stack=f"{p[5]}_{p[6]}_{p[7]}_{p[8]}", date=date, granule=ur, url=url,
                         dx=dx, dy=dy, ox=ox, oy=oy, corners_ok=all(resid), aoi_valid=round(float(valid),3),
                         crop_shape=list(crop.shape), version=os.path.basename(f)[4:-5]))
        print(name, crop.shape, 'valid', round(float(valid),2), 'corners_ok', all(resid), round(dx*1e4,3), round(dy*1e4,3))
json.dump(meta, open(os.path.join(out, 'nisar_browse_meta.json'), 'w'), indent=1)

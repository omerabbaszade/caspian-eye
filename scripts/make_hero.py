# Builds the website hero map (img/hero.jpg) and NISAR thumbnails (img/thumb/)
# Usage: python make_hero.py <web folder>
import sys, os, json
import numpy as np
from PIL import Image
WEB = sys.argv[1]; IMG = os.path.join(WEB, "img")
D = json.load(open(os.path.join(WEB, "data.json")))
NG = json.load(open(os.path.join(WEB, "nisar_gcov.json")))
last = D["yearly"][-1]
RAMP = ["#6e2a0c", "#8a3510", "#a84215", "#c5501c", "#dd6128", "#ec7a45", "#f19566", "#f5b08b", "#f9cbb2"]
hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
vv = np.asarray(Image.open(f"{IMG}/s1vv_{last}.jpg").convert("L"), np.float32) / 255
H, W = vv.shape
cls = np.asarray(Image.open(f"{IMG}/cls_{last}.png").convert("RGB").resize((W, H), Image.NEAREST))[..., 0]
tr = np.asarray(Image.open(f"{IMG}/trend.png").convert("RGB").resize((W, H), Image.NEAREST))
ey, fy = tr[..., 0].astype(int), tr[..., 1]
# Land: radar texture with a cool tint; water: dark sea colour with faint texture
land = np.stack([vv * 0.80 + 0.03, vv * 0.88 + 0.05, vv * 0.88 + 0.06], -1)
sea = np.stack([0.035 + vv * 0.10, 0.12 + vv * 0.14, 0.20 + vv * 0.16], -1)
out = np.where((cls == 2)[..., None], sea, land) * 255
years = [int(d[:4]) for d in D["yearly"]][1:]
for k, y in enumerate(years):
    m = ey == (y - 2000)
    out[m] = out[m] * 0.08 + hexrgb(RAMP[min(k, len(RAMP) - 1)]) * 0.92
m = (ey == 0) & (fy > 0)
out[m] = out[m] * 0.55 + hexrgb("#3987e5") * 0.45  # land→water (mostly farmland noise): shown faintly
nodata = vv <= 0.002
out[nodata] = hexrgb("#061a1f")
im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
im.resize((800, round(800 * H / W)), Image.LANCZOS).save(f"{IMG}/hero.jpg", quality=86, optimize=True, progressive=True)
os.makedirs(f"{IMG}/thumb", exist_ok=True)
for d in NG["dates"]:
    t = Image.open(f"{IMG}/ngc_{d}.jpg").convert("RGB")
    t.resize((240, round(240 * t.height / t.width)), Image.LANCZOS).save(f"{IMG}/thumb/ngc_{d}.jpg", quality=82, optimize=True)
print("hero", im.size, os.path.getsize(f"{IMG}/hero.jpg") // 1024, "KB; thumbs", len(NG["dates"]))

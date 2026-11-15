# Map composites for the video in the website colours: water maps, change map, NISAR water map
import sys, json, os
import numpy as np
from PIL import Image
WEB, OUT = sys.argv[1], sys.argv[2]; IMG = f"{WEB}/img"
D = json.load(open(f"{WEB}/data.json")); NG = json.load(open(f"{WEB}/nisar_gcov.json"))
WATER, SOFT, LOSS, VEG = [np.array(c, np.float32) for c in ([42, 120, 214], [158, 197, 244], [235, 104, 52], [27, 175, 122])]
def base_gray(path):
    g = np.asarray(Image.open(path).convert("L"), np.float32) * 0.62
    return np.stack([g, g, g], -1)
def base_rgb(path):  # brightness .62, saturate .6
    a = np.asarray(Image.open(path).convert("RGB"), np.float32); l = a.mean(-1, keepdims=True)
    return (l + (a - l) * 0.6) * 0.62
def cls(path, shape):
    return np.asarray(Image.open(path).convert("RGB").resize((shape[1], shape[0]), Image.NEAREST))[..., 0]
def over(img, mask, col, a):
    img[mask] = img[mask] * (1 - a) + col * a
def save(a, name):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(f"{OUT}/{name}", quality=90)
for d in D["yearly"]:
    b = base_gray(f"{IMG}/s1vv_{d}.jpg"); c = cls(f"{IMG}/cls_{d}.png", b.shape)
    over(b, c == 2, WATER, 165 / 255); over(b, c == 3, VEG, 230 / 255); save(b, f"s1w_{d}.jpg")
a, z = D["yearly"][0], D["yearly"][-1]
b = base_gray(f"{IMG}/s1vv_{z}.jpg"); ca = cls(f"{IMG}/cls_{a}.png", b.shape); cb = cls(f"{IMG}/cls_{z}.png", b.shape)
ok = (ca > 0) & (cb > 0); wa, wb = ca == 2, cb == 2
over(b, ok & wa & wb, SOFT, 120 / 255); over(b, ok & ~wa & wb, WATER, 235 / 255); over(b, ok & wa & ~wb, LOSS, 235 / 255); save(b, "change.jpg")
for d in NG["dates"]:
    b = base_rgb(f"{IMG}/ngc_{d}.jpg"); Image.open(f"{IMG}/ngc_{d}.jpg").convert("RGB").save(f"{OUT}/ngc_{d}.jpg", quality=90)
last = NG["dates"][-1]
b = base_rgb(f"{IMG}/ngc_{last}.jpg"); c = cls(f"{IMG}/ncls_{last}.png", b.shape)
over(b, c == 2, WATER, 165 / 255); over(b, c == 3, VEG, 230 / 255); save(b, "nwater.jpg")
print("ok", len(os.listdir(OUT)))

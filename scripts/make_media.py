# Time-lapses for video: shoreline 2017→2026 (S1) and NISAR L-band 2026, as GIF and MP4
import sys, os, json, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
D = sys.argv[1]; WEB = os.path.join(D, "web"); OUT = os.path.join(D, "media"); os.makedirs(OUT, exist_ok=True)
data = json.load(open(os.path.join(WEB, "data.json"))); ng = json.load(open(os.path.join(WEB, "nisar_gcov.json")))
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
W, H = 1000, 1285


def label(img, title, sub, note):
    im = img.convert("RGB"); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 118], fill=(8, 16, 17))
    d.text((28, 18), title, font=ImageFont.truetype(FB, 44), fill=(235, 242, 240))
    d.text((28, 72), sub, font=ImageFont.truetype(FR, 26), fill=(170, 190, 188))
    d.rectangle([0, H - 54, W, H], fill=(8, 16, 17))
    d.text((28, H - 42), note, font=ImageFont.truetype(FR, 20), fill=(150, 170, 168))
    return im


def save_anim(frames, name, ms):
    # GIF (compact) and MP4 (for video editors)
    small = [f.resize((600, int(600 * H / W)), Image.LANCZOS) for f in frames]
    small[0].save(os.path.join(OUT, name + ".gif"), save_all=True, append_images=small[1:], duration=ms, loop=0, optimize=True)
    tmp = os.path.join(OUT, "_frames"); os.makedirs(tmp, exist_ok=True)
    for k, f in enumerate(frames): f.save(os.path.join(tmp, f"{k:03d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(1000 / ms), "-i", os.path.join(tmp, "%03d.png"),
                    "-vf", "scale=1000:1286:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-r", "30", os.path.join(OUT, name + ".mp4")], check=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)


# 1) Shoreline: October 2017 compared with each later October
y0 = data["yearly"][0]; a = np.asarray(Image.open(os.path.join(WEB, "img", f"cls_{y0}.png")))[..., 0]
frames = []
for d in data["yearly"]:
    b = np.asarray(Image.open(os.path.join(WEB, "img", f"cls_{d}.png")))[..., 0]
    base = np.asarray(Image.open(os.path.join(WEB, "img", f"s1vv_{d}.jpg")).convert("RGB")).astype(np.float32) * 0.62
    v = (a > 0) & (b > 0); wa, wb = a == 2, b == 2
    o = base.copy()
    o[v & wa & wb] = o[v & wa & wb] * 0.45 + np.array([158, 197, 244]) * 0.55
    o[v & ~wa & wb] = [42, 120, 214]; o[v & wa & ~wb] = [235, 104, 52]
    loss = data["pairs"].get(f"{y0}|{d}", {}).get("AOI", {}).get("loss", 0.0)
    sub = f"October {d[:4]} vs October {y0[:4]}  ·  water → land: {loss:.0f} km²" if d != y0 else f"October {y0[:4]}  ·  baseline"
    frames.append(label(Image.fromarray(o.astype(np.uint8)), "Kura Delta coastline", sub, "Orange: water → land   Blue: land → water   Sentinel-1 radar, Caspian Eye"))
frames += [frames[-1]] * 2  # hold the last frame
save_anim(frames, "coastline_2017_2026", 1100)

# 2) NISAR L-band 2026
frames = []
for r in ng["rows"]:
    img = Image.open(os.path.join(WEB, "img", f"ngc_{r['date']}.jpg"))
    frames.append(label(img, "NISAR L-band, Kura Delta", f"{r['date']}  ·  open water {r['per_zone']['AOI']['water_common']:.0f} km² (area imaged on all dates)",
                        "NASA–ISRO NISAR L2 GCOV, track 165 · HH, HV, HH−HV · Caspian Eye"))
frames += [frames[-1]] * 2
save_anim(frames, "nisar_2026", 1000)
for f in sorted(os.listdir(OUT)): print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")

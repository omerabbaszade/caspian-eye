# Renders film.html frame by frame: python render_film.py <video folder> <web folder>
import asyncio, json, sys, os
from playwright.async_api import async_playwright
V, WEB = sys.argv[1], sys.argv[2]
D = json.load(open(f"{WEB}/data.json")); NG = json.load(open(f"{WEB}/nisar_gcov.json"))
src = open(f"{V}/film.html").read().replace("__YEARS__", json.dumps(D["yearly"])).replace("__NDATES__", json.dumps(NG["dates"]))
open(f"{V}/film_r.html", "w").write(src)
FR = f"{V}/frames"; os.makedirs(FR, exist_ok=True)
ONLY = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(f"file://{V}/film_r.html"); await pg.evaluate("window.ready")
        await pg.wait_for_timeout(500)
        ts = ONLY if ONLY else [f / 30 for f in range(900)]
        for k, t in enumerate(ts):
            await pg.evaluate(f"setT({t})")
            name = f"{FR}/t{t:05.2f}.jpg" if ONLY else f"{FR}/f{k:04d}.jpg"
            await pg.screenshot(path=name, type="jpeg", quality=93)
        print("frames", len(ts), "errors", errs or "yox")
        await b.close()
asyncio.run(main())

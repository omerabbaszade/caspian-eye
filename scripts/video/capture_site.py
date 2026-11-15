# Captures swipe frames of the Water map layer on ceye.online
import asyncio, sys
from playwright.async_api import async_playwright
OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1200}, color_scheme="dark")
        await pg.goto("https://ceye.online/", wait_until="networkidle")
        await pg.click("#modes button:has-text('Water map')")
        await pg.select_option("#selA", "2017-10-21"); await pg.select_option("#selB", "2026-10-17")
        await pg.wait_for_timeout(1500)
        await pg.evaluate("""() => { const f = document.querySelector('#modes'); window.scrollTo(0, f.getBoundingClientRect().top + scrollY - 96); }""")
        await pg.wait_for_timeout(800)
        for i in range(N):
            v = 92 - 84 * (i / max(N - 1, 1))  # right to left: 2017 → 2026
            await pg.evaluate(f"""() => {{ const s = document.querySelector('#swipe'); s.value = {v:.2f}; s.dispatchEvent(new Event('input', {{bubbles: true}})); }}""")
            await pg.wait_for_timeout(30)
            await pg.screenshot(path=f"{OUT}/s{i:03d}.jpg", type="jpeg", quality=90)
        await b.close()
asyncio.run(main())

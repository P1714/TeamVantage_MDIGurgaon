"""Capture console screenshots for docs/img (README, deck)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/img"
with sync_playwright() as pw:
    b = pw.chromium.launch()
    p = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.25)
    p.goto((ROOT / "dashboard/madhutwin_console.html").as_uri())
    p.wait_for_timeout(1500)
    p.screenshot(path=str(IMG / "console_now.png"))
    p.screenshot(path=str(IMG / "console_hero.png"), clip={"x": 290, "y": 0, "width": 1310, "height": 520})
    p.click("#heroWhatif") if p.locator("#heroWhatif").count() else p.click("[data-tab=whatif]")
    p.wait_for_timeout(800)
    p.screenshot(path=str(IMG / "console_whatif.png"))
    p.click("[data-tab=model]"); p.wait_for_timeout(800)
    p.locator("#twinvalue").locator("xpath=..").screenshot(path=str(IMG / "console_twinvalue.png"))
    p.click("[data-tab=app]"); p.wait_for_timeout(800)
    p.screenshot(path=str(IMG / "console_app.png"))
    for t, f in (("week", "console_agp.png"), ("record", "console_record.png")):
        p.click(f"[data-tab={t}]"); p.wait_for_timeout(800); p.screenshot(path=str(IMG / f))
    b.close()
print("screenshots written")

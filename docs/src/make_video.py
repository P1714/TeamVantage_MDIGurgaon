"""Produce the narrated demo video (docs/MadhuTwin_Demo.mp4).

Requirements (optional, only for re-making the video):
    pip install kokoro-onnx soundfile playwright
    kokoro-v1.0.int8.onnx + voices-v1.0.bin from github.com/thewh1teagle/kokoro-onnx/releases
    ffmpeg, pdftoppm; slides rendered to <build>/slides/s-NN.png (1920x1080)

Usage: python docs/src/make_video.py <build_dir> <kokoro_dir> [script.json] [output.mp4]

The long version (docs/src/video_script_long.json) also needs the API running for its last demo step:
    uvicorn madhutwin.api:app --port 8000
"""
import json
import subprocess
import time
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
B = Path(sys.argv[1])
KD = Path(sys.argv[2])
SCRIPT = json.loads(Path(sys.argv[3] if len(sys.argv) > 3 else ROOT / "docs/src/video_script.json").read_text())
FINAL = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "docs" / "MadhuTwin_Demo.mp4"
(B / "audio").mkdir(parents=True, exist_ok=True)
(B / "seg").mkdir(exist_ok=True)


def sh(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


# 1. narration -----------------------------------------------------------------
from kokoro_onnx import Kokoro  # noqa: E402

tts = Kokoro(str(KD / "kokoro-v1.0.int8.onnx"), str(KD / "voices-v1.0.bin"))
durs = []
for i, seg in enumerate(SCRIPT["segments"]):
    wav = B / "audio" / f"{i:02d}.wav"
    if not wav.exists():
        samples, sr = tts.create(seg["text"], voice=SCRIPT["voice"], speed=SCRIPT["speed"], lang="en-us")
        sf.write(wav, samples, sr)
    durs.append(sf.info(wav).duration)
    print(f"audio {i:02d} {durs[-1]:.1f}s", flush=True)

# 2. console screen recording, paced by the narration -----------------------------
from playwright.sync_api import sync_playwright  # noqa: E402

demo = [(i, s) for i, s in enumerate(SCRIPT["segments"]) if "demo" in s]
CURSOR = """
const c=document.createElement('div');c.id='fakecursor';
c.style.cssText='position:fixed;z-index:99999;width:18px;height:18px;border-radius:50%;background:rgba(201,130,10,.35);border:2px solid #c9820a;pointer-events:none;transform:translate(-50%,-50%);left:-50px;top:-50px;transition:left .35s ease,top .35s ease';
document.body.appendChild(c);addEventListener('mousemove',e=>{c.style.left=e.clientX+'px';c.style.top=e.clientY+'px'});
"""
PRETTY = """
const t=document.body.innerText;let j;try{j=JSON.stringify(JSON.parse(t),null,2)}catch(e){j=t}
document.body.innerHTML='';const h=document.createElement('div');h.textContent='MadhuTwin REST API  ·  '+location.pathname;
h.style.cssText='font:600 22px DejaVu Sans,sans-serif;padding:24px 40px;background:#15222a;color:#fff';
const pre=document.createElement('pre');pre.textContent=j;pre.style.cssText='font:17px/1.45 DejaVu Sans Mono,monospace;padding:20px 40px;margin:0;white-space:pre-wrap';
document.body.style.cssText='margin:0;background:#fbfbf9;color:#15222a';document.body.append(h,pre);
"""
vid_dir = B / "rec"
with sync_playwright() as pw:
    br = pw.chromium.launch()
    ctx = br.new_context(viewport={"width": 1600, "height": 900}, record_video_dir=str(vid_dir),
                         record_video_size={"width": 1600, "height": 900})
    p = ctx.new_page()
    p.goto((ROOT / "dashboard/madhutwin_console.html").as_uri())
    p.wait_for_timeout(1500)
    p.evaluate(CURSOR)
    t_start = (time.monotonic() * 1000)
    marks = []

    def glide(sel, dx=0.5, dy=0.5):
        box = p.locator(sel).first.bounding_box()
        p.mouse.move(box["x"] + box["width"] * dx, box["y"] + box["height"] * dy, steps=12)

    for i, seg in demo:
        marks.append((time.monotonic() * 1000) - t_start)
        d = durs[i] * 1000 + 500
        t0 = (time.monotonic() * 1000)
        name = seg["demo"]
        if name == "triage":
            glide("#trisum")
            p.wait_for_timeout(2500)
            glide(".pitem[aria-current=true]")
        elif name == "hero":
            glide("#hero .h-head")
            p.wait_for_timeout(3500)
            glide("#hero .reason")
            p.wait_for_timeout(3500)
            glide("#hero .act-big")
        elif name == "whatif":
            glide("#heroWhatif")
            p.click("#heroWhatif")
            p.wait_for_timeout(2000)
            glide("#wichart", 0.4, 0.3)
            p.wait_for_timeout(2500)
            glide("#widelta", 0.5, 0.5)
        elif name == "wimore":
            glide("[data-preset=reset]")
            p.click("[data-preset=reset]")
            p.wait_for_timeout(1200)
            glide("[data-preset=millet]")
            p.click("[data-preset=millet]")
            p.wait_for_timeout(4000)
            glide("#widelta", 0.85, 0.5)
            p.wait_for_timeout(2500)
            glide("[data-preset=portion]")
            p.click("[data-preset=portion]")
            p.wait_for_timeout(4000)
            p.locator("#wivalid").scroll_into_view_if_needed()
            glide("#wivalid", 0.5, 0.4)
        elif name == "state":
            p.click("[data-tab=now]")
            p.wait_for_timeout(300)
            p.locator("#twinstate").scroll_into_view_if_needed()
            glide("#twinstate", 0.5, 0.3)
            p.wait_for_timeout(2500)
            glide("#todaybase", 0.5, 0.5)
        elif name == "alerts":
            p.locator("#alerts").scroll_into_view_if_needed()
            glide("#alerts", 0.3, 0.3)
            p.wait_for_timeout(4500)
            btn = p.locator("#alerts button").first
            if btn.count():
                glide("#alerts button")
                btn.click()
            p.wait_for_timeout(3500)
            p.evaluate("window.scrollTo(0,0)")
            glide("#jump")
        elif name == "reveal":
            p.evaluate("window.scrollTo(0,0)")
            glide("#mainchart", 0.5, 0.3)
            p.wait_for_timeout(5000)
            glide("#mainchart", 0.5, 0.75)
            p.wait_for_timeout(2500)
            glide("#showfuture")
            p.click("#showfuture")
            p.wait_for_timeout(600)
            glide("#mainchart", 0.95, 0.2)
        elif name == "play":
            p.evaluate("window.scrollTo(0,0)")
            if p.locator("#showfuture").is_checked():
                p.click("#showfuture")
            glide("#play")
            p.click("#play")
            remaining = d - ((time.monotonic() * 1000) - t0) - 400
            p.wait_for_timeout(max(0, remaining))
            p.click("#play")
        elif name == "evidence":
            glide("[data-tab=model]")
            p.click("[data-tab=model]")
            p.wait_for_timeout(1500)
            glide("#twinvalue", 0.5, 0.4)
            p.wait_for_timeout(int(d * 0.4))
            p.locator("#oppoint").scroll_into_view_if_needed()
            glide("#oppoint", 0.5, 0.4)
            p.wait_for_timeout(int(d * 0.2))
            p.locator("#calib").scroll_into_view_if_needed()
            glide("#calib", 0.5, 0.5)
        elif name == "sick":
            p.click("[data-tab=now]")
            p.wait_for_timeout(400)
            p.locator("#sickcard").scroll_into_view_if_needed()
            glide("#sickcard", 0.5, 0.3)
            p.wait_for_timeout(3000)
            glide("#sickcard", 0.5, 0.7)
        elif name == "evidence2":
            for sel in ("#ops", "#bridge", "#realworld", "#prov", "#leak", "#fair", "#latency", "#limits"):
                if p.locator(sel).count():
                    p.locator(sel).scroll_into_view_if_needed()
                    glide(sel, 0.5, 0.3)
                    p.wait_for_timeout(max(900, (d - 2000) / 8 - 600))
        elif name == "api":
            p.goto("http://127.0.0.1:8000/v1/patients/P0163/risk")
            p.evaluate(PRETTY)
            p.wait_for_timeout(max(2000, d * 0.55))
            p.goto("http://127.0.0.1:8000/v1/patients/P0163/fhir")
            p.evaluate(PRETTY)
        elif name == "app":
            glide("[data-tab=app]")
            p.click("[data-tab=app]")
            p.wait_for_timeout(2500)
            glide("#phone", 0.5, 0.45)
            p.wait_for_timeout(2000)
            glide("[data-lang=hi]")
            p.click("[data-lang=hi]")
            p.wait_for_timeout(1500)
            glide("#phone", 0.5, 0.55)
        elif name == "model":
            glide("[data-tab=model]")
            p.click("[data-tab=model]")
            p.wait_for_timeout(1200)
            p.mouse.wheel(0, 700)
            p.wait_for_timeout(1500)
            glide("#realworld", 0.5, 0.3)
        elif name == "agp":
            glide("[data-tab=week]")
            p.click("[data-tab=week]")
            p.wait_for_timeout(2500)
            glide("#agp", 0.5, 0.5)
            p.wait_for_timeout(4000)
            glide("#tirbar", 0.5, 0.5)
            p.wait_for_timeout(3000)
            glide("#wkstats", 0.5, 0.5)
        elif name == "record":
            glide("[data-tab=record]")
            p.click("[data-tab=record]")
            p.wait_for_timeout(2500)
            glide("#labs", 0.5, 0.4)
            p.wait_for_timeout(3500)
            glide("#twinphys", 0.3, 0.5)
            p.wait_for_timeout(3500)
            p.locator("#habits").scroll_into_view_if_needed()
            glide("#habits", 0.5, 0.4)
        left = d - ((time.monotonic() * 1000) - t0)
        if left > 0:
            p.wait_for_timeout(left)
    total_demo = (time.monotonic() * 1000) - t_start
    lead_in = t_start  # page time before narration starts (trimmed below)
    video_path = p.video.path()
    ctx.close()
    br.close()

# The recording starts at context creation; approximate the offset of t_start.
webm = Path(video_path)
probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(webm)],
                       capture_output=True, text=True).stdout.strip()
vid_len = float(probe)
offset = max(0.0, vid_len - total_demo / 1000)

# demo narration track: each clip placed at its mark
demo_audio = B / "demo_audio.wav"
inputs, filters = [], []
for k, ((i, _), m) in enumerate(zip(demo, marks)):
    inputs += ["-i", str(B / "audio" / f"{i:02d}.wav")]
    filters.append(f"[{k}:a]adelay={int(m)}|{int(m)},aresample=48000[a{k}]")
mix = "".join(f"[a{k}]" for k in range(len(demo))) + f"amix=inputs={len(demo)}:normalize=0[out]"
sh(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters) + ";" + mix, "-map", "[out]", "-t",
    f"{total_demo / 1000:.2f}", str(demo_audio)])
sh(["ffmpeg", "-y", "-ss", f"{offset:.2f}", "-i", str(webm), "-i", str(demo_audio), "-map", "0:v", "-map", "1:a",
    "-vf", "scale=1920:1080:flags=lanczos,fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-shortest", str(B / "seg" / "demo.mp4")])

# 3. slide segments ------------------------------------------------------------------
parts = []
demo_done = False
for i, seg in enumerate(SCRIPT["segments"]):
    if "demo" in seg:
        if not demo_done:
            parts.append(B / "seg" / "demo.mp4")
            demo_done = True
        continue
    img = B / "slides" / f"s-{seg['slide']:02d}.png"
    out = B / "seg" / f"slide_{i:02d}.mp4"
    dur = durs[i] + 0.9
    sh(["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-i", str(img), "-i", str(B / "audio" / f"{i:02d}.wav"),
        "-af", "adelay=350|350,apad", "-t", f"{dur:.2f}",
        "-vf", "scale=1920:1080,format=yuv420p,fade=t=in:st=0:d=0.3", "-c:v", "libx264", "-preset", "medium",
        "-tune", "stillimage", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", str(out)])
    parts.append(out)

lst = B / "concat.txt"
lst.write_text("".join(f"file '{x}'\n" for x in parts))
final = FINAL
sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-preset", "medium", "-crf", "21",
    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(final)])
print("wrote", final)

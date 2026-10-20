"""Fill in the public links once the repo is on GitHub and the video is on YouTube.

    python scripts/set_links.py --video https://youtu.be/XXXX --repo https://github.com/<user>/TeamVantage_MDIGurgaon

Updates team.json and README.md (no other tools needed).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="unlisted YouTube link")
ap.add_argument("--repo", required=True, help="public GitHub repository link")
a = ap.parse_args()
tj = ROOT / "team.json"
t = json.loads(tj.read_text())
old_v, old_r = t["demo_video_url"], t["repo_url"]
t["demo_video_url"], t["repo_url"] = a.video, a.repo
tj.write_text(json.dumps(t, indent=2, ensure_ascii=False))
import subprocess  # noqa: E402
# regenerate the README so the checklist marks switch from pending to done once real links exist
subprocess.run([sys.executable, str(ROOT / "scripts" / "build_readme.py")], check=True)
rd = ROOT / "README.md"
assert "<unlisted YouTube link>" not in rd.read_text() and "<your-username>" not in rd.read_text()
print("team.json and README.md updated:", a.video, a.repo)
print("Now open both links in a signed-out (incognito) window before submitting.")

"""One-command reproduction of MadhuTwin.

    python run.py            # quick: shipped synthetic cohort -> twin -> models -> evaluation -> stress test -> console
                             #        (reuses the saved open-cohort results; see README "Clean-clone verification" for timings)
    python run.py --full     # also re-simulates the cohort, recomputes the open-cohort experiment and rebuilds docs
    python run.py --serve    # after a build: open the REST API + console on http://127.0.0.1:8000/console

Everything is seeded (config.SEED = 2026) so numbers reproduce exactly on the same library versions
(see requirements-lock.txt).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sh(*args):
    print("\n$", " ".join(args), flush=True)
    t = time.time()
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"  done in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="regenerate the synthetic cohort and run real-world validation")
    ap.add_argument("--serve", action="store_true", help="start the API + console server")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()

    if a.serve:
        subprocess.run([sys.executable, "-m", "uvicorn", "madhutwin.api:app", "--port", "8000"], cwd=ROOT)
        return
    if not a.no_tests:
        sh("-m", "pytest", "-q")
    pipe = ["-m", "madhutwin.pipeline"]
    if a.full:
        pipe.append("--regen")
    else:
        pipe.append("--skip-real")
    sh(*pipe)
    sh("scripts/stress_test.py")
    sh("scripts/benchmark.py")
    sh("scripts/build_dashboard.py")
    if a.full:
        sh("scripts/render_docs.py")
        sh("scripts/build_model_card.py")
        sh("scripts/build_readme.py")
    print("\nOpen dashboard/madhutwin_console.html in a browser, or run:  python run.py --serve")
    print("Key numbers: reports/metrics.json   Model card: docs/model_card.md")


if __name__ == "__main__":
    main()

"""The browser what-if simulator must match the Python twin exactly."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from madhutwin import physiology as PH
from madhutwin import whatif as W

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_twin_matches_python():
    src = (ROOT / "dashboard" / "console_fragment.html").read_text()
    fn = re.search(r"function simulate\(.*?\n}\n", src, re.S).group(0)
    physio = {"P2": PH.P2, "N_INS": PH.N_INS, "ALPHA": PH.ALPHA, "K_FAST": PH.K_FAST, "K_SLOW": PH.K_SLOW,
              "F_BIO": PH.F_BIO, "K_EX": PH.K_EX, "RENAL_THR": PH.RENAL_THR, "RENAL_K": PH.RENAL_K,
              "SGLT2_THR": PH.SGLT2_THR, "SGLT2_K": PH.SGLT2_K, "TAU_ISF": PH.TAU_ISF, "SG": 0.016}
    tw = {"SI": 2.2e-4, "sigma": 0.035, "Gb": 138.0, "dawn": 12.0, "V": 118.0}
    js = f"const PH={json.dumps(physio)};\n{fn}\nconsole.log(JSON.stringify(simulate({json.dumps(tw)},85,72,20,15,false,750)));"
    out = np.array(json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout))
    prm = {"SG": np.array([0.016]), "SI": np.array([tw["SI"]]), "sigma": np.array([tw["sigma"]]),
           "Gb": np.array([tw["Gb"]]), "h": np.array([tw["Gb"]]), "dawn": np.array([tw["dawn"]]),
           "V": np.array([tw["V"]]), "sglt2": np.array([0.0])}
    py = W.run(prm, 85, 72, walk=(15, 20, 105), start_tod=750)[0]
    np.testing.assert_allclose(out, py, atol=1e-6)

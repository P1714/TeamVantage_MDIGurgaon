"""Build the doctor console.

dashboard/console_fragment.html  (source: markup + CSS + JS, no data)
        -> dashboard/index.html              (loads dashboard/data.js; serve the folder or open directly)
        -> dashboard/madhutwin_console.html  (single self-contained file with the data inlined)
"""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "dashboard"
frag = (D / "console_fragment.html").read_text()
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
title_end = frag.index("</style>") + len("</style>")
page_head, body = frag[:title_end], frag[title_end:]
full = head + page_head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
(D / "index.html").write_text(full)
data = (D / "data.js").read_text()
# merge the latest latency benchmark (scripts/benchmark.py) into the console data
import json as _json
_extra = {"latency": D.parent / "reports" / "latency.json", "stress_test": D.parent / "reports" / "stress_test.json"}
if any(f.exists() for f in _extra.values()):
    _obj = _json.loads(data[data.index("=") + 1:].rstrip().rstrip(";"))
    for _k, _f in _extra.items():
        if _f.exists():
            _obj["metrics"][_k] = _json.loads(_f.read_text())
    data = "window.MADHU_DATA = " + _json.dumps(_obj, separators=(",", ":")) + ";\n"
    (D / "data.js").write_text(data)
standalone = full.replace('<script src="data.js"></script>', "<script>\n" + data + "\n</script>")
(D / "madhutwin_console.html").write_text(standalone)
# fragment form (for hosts that add their own <html>/<head> skeleton)
(D.parent / "dist").mkdir(exist_ok=True)
(D.parent / "dist" / "madhutwin_console_fragment.html").write_text(
    frag.replace('<script src="data.js"></script>', "<script>\n" + data + "\n</script>"))
print("built", D / "index.html", "and", D / "madhutwin_console.html")

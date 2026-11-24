"""Build report.pdf (repo root) from report/report_draft.md.

    python scripts/build_report.py

Needs pandoc and Google Chrome (or Chromium) on the PATH. The draft-status box and the
HTML comments of the Markdown file are removed, the figures are embedded, and Chrome
prints the page to PDF (A4, Times 10.5 pt).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = """<style>
@page{size:A4;margin:22mm 20mm}
body{font-family:'Times New Roman',Times,serif;font-size:10.5pt;line-height:1.35;max-width:none;margin:0;padding:0;color:#111}
header#title-block-header h1{font-size:16pt;text-align:center;margin:0} .author{text-align:center;margin:4px 0 14px}
h2{font-size:12.5pt;margin:14px 0 4px} h3{font-size:11pt;margin:10px 0 3px}
p{margin:0 0 6px;text-align:justify}
figure{margin:6px 0 10px;text-align:center;page-break-inside:avoid} figcaption{font-size:9pt;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:8.5pt;margin:4px 0 8px} tr{page-break-inside:avoid}
th,td{border-top:0.5px solid #999;border-bottom:0.5px solid #999;padding:2px 4px;text-align:left}
code{font-size:9pt} h2#appendix{page-break-before:always}
</style>"""


def find_chrome() -> str:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("Chrome or Chromium not found on the PATH")


def main() -> None:
    text = (ROOT / "report" / "report_draft.md").read_text(encoding="utf-8")
    text = re.sub(r"> \*\*Draft status\.\*\*.*?PDF\.\n", "", text, flags=re.S)  # note to myself
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    with tempfile.TemporaryDirectory() as tmp:
        md, page = Path(tmp) / "report.md", Path(tmp) / "report.html"
        md.write_text(text, encoding="utf-8")
        subprocess.run(
            ["pandoc", str(md), "-o", str(page), "--standalone", "--embed-resources",
             f"--resource-path={ROOT / 'report'}"],
            check=True,
        )
        page.write_text(page.read_text(encoding="utf-8").replace("</head>", CSS + "</head>"), encoding="utf-8")
        subprocess.run(
            [find_chrome(), "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={ROOT / 'report.pdf'}", page.as_uri()],
            check=True, capture_output=True,
        )
    print("written", ROOT / "report.pdf")


if __name__ == "__main__":
    main()

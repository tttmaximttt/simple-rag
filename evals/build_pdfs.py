"""Rebuild the evaluation PDFs in evals/pdfs/ from their published sources, pinned to fixed commits.

The PDFs are committed, so this only needs to run to check or refresh them. It downloads the source
files from raw.githubusercontent.com, turns Markdown, reStructuredText and AsciiDoc into plain text,
and lays it out as simple text PDFs with a running header and footer (which the app's cleaner should
remove). Page numbers in dataset.jsonl refer to these PDFs, so a rebuild that changes the layout or
the pinned commits needs the labels checked again: tests/test_evals.py does that.

Run: python -m evals.build_pdfs [--cache DIR]
"""
import argparse
import html
import re
import textwrap
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PDF_DIR = HERE / "pdfs"

LINES_PER_PAGE = 56
WRAP = 96

# name: (title, repo, commit, [paths]). Sources and licenses are listed in evals/README.md.
SOURCES = {
    "nist-sp800-63b.pdf": (
        "NIST SP 800-63B Digital Identity Guidelines: Authentication and Lifecycle Management",
        "usnistgov/800-63-3", "7dee3cad9d429dcefc27ba1976a55704bff0800a",
        ["sp800-63b/sec4_aal.md", "sp800-63b/sec5_authenticators.md", "sp800-63b/sec7_session.md",
         "sp800-63b/appA_memorized.md"]),
    "pep8-style-guide.pdf": (
        "PEP 8 Style Guide for Python Code",
        "python/peps", "28581cca58bda3ad476e8a210be6df4733e88c97", ["peps/pep-0008.rst"]),
    "raspberry-pi-hardware.pdf": (
        "Raspberry Pi hardware documentation (selected chapters)",
        "raspberrypi/documentation", "53074241f8ee8119973c645f73a8703d722e8e94",
        [f"documentation/asciidoc/computers/raspberry-pi/{n}.adoc" for n in (
            "power-supplies", "revision-codes", "otp-bits", "frequency-management", "rtc", "power-button")]),
    "mdn-http-es.pdf": (
        "Referencia HTTP de MDN: códigos de estado y encabezados (selección)",
        "mdn/translated-content", "0862a7d392286f5e62ef09caa14049a93fe94e21",
        [f"files/es/web/http/reference/status/{c}/index.md" for c in (
            "100", "101", "200", "201", "202", "203", "206", "302", "304", "400", "401", "403", "404", "405",
            "408", "413", "418", "500", "502", "503", "504", "505")]
        + [f"files/es/web/http/reference/headers/{h}/index.md" for h in (
            "cache-control", "strict-transport-security", "content-security-policy", "set-cookie",
            "x-frame-options", "etag", "content-disposition", "x-content-type-options", "referrer-policy",
            "access-control-allow-origin")]),
}


def fetch(repo, commit, path, cache: Path) -> str:
    local = cache / repo / commit / path
    if not local.exists():
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    return local.read_text(encoding="utf-8")


# --- source formats to plain text -------------------------------------------------------------------------

def from_markdown(src: str) -> str:
    title = None
    if src.startswith("---\n"):  # MDN front matter: keep the title as a heading
        front, _, src = src[4:].partition("\n---\n")
        m = re.search(r"^title:\s*(.+)$", front, re.M)
        title = m and m[1].strip().strip("'\"")
    src = re.sub(r"</t[dh]>", " | ", src)
    src = re.sub(r"</tr>|<br\s*/?>|</p>|</li>", "\n", src)
    # Only real HTML tags: MDN writes placeholders like <cookie-name> that are part of the text.
    src = re.sub(r"</?(?:a|b|i|em|strong|code|div|span|table|tr|td|th|img|p|br|ul|ol|li|sup|sub)\b[^>]*>", "", src)
    src = html.unescape(src)
    src = re.sub(r"\{\{\s*(?:Glossary|HTTPHeader|HTTPStatus|HTTPMethod|domxref|jsxref|CSP)\(\s*\"([^\"]+)\"[^}]*\}\}",
                 r"\1", src)
    src = re.sub(r"\{\{[^}]*\}\}", "", src)  # other MDN macros (compat tables, sidebars)
    src = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", src)  # images
    src = re.sub(r"\[\[([^\]]+)\]\]\([^)]*\)", r"[\1]", src)  # NIST reference links: [[RFC 20]](#RFC20)
    src = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", src)
    src = re.sub(r"^> \[!(NOTE|WARNING|CALLOUT)\]\s*$", r"\1:", src, flags=re.M)
    src = re.sub(r"^>\s?", "", src, flags=re.M)
    src = re.sub(r"^\s*\|?\s*-{3,}[-|: ]*$", "", src, flags=re.M)  # table rules
    src = re.sub(r"^```\w*\s*$", "", src, flags=re.M)
    src = re.sub(r"^#{1,6}\s*", "\n", src, flags=re.M)
    src = re.sub(r"^\s*-\s+:\s*", "  ", src, flags=re.M)  # MDN definition lists
    src = src.replace("**", "").replace("`", "")
    src = re.sub(r"(?<![\w*])_([^_\n]+)_(?![\w*])", r"\1", src)
    return (f"{title}\n\n" if title else "") + src


def from_rst(src: str) -> str:
    src = re.sub(r":\w+:`([^`<]+?)\s*<[^>]+>`", r"\1", src)  # :pep:`text <8>`
    src = re.sub(r":(\w+):`([^`]+)`", lambda m: f"PEP {m[2]}" if m[1] == "pep" else m[2], src)
    src = re.sub(r"`([^`<]+?)\s*<[^>]+>`_+", r"\1", src)
    src = src.replace("``", "")
    src = re.sub(r"^\.\. (?:code-block|code|rubric|note|warning)::\s*\w*\s*$", "", src, flags=re.M)
    src = re.sub(r"^\s*:\w+:.*$", "", src, flags=re.M)  # directive options
    src = re.sub(r"^\.\. _[^:]+:.*$", "", src, flags=re.M)
    src = re.sub(r"^[=\-~^\"']{3,}\s*$", "", src, flags=re.M)  # heading underlines
    src = re.sub(r"\s*::$", ":", src, flags=re.M)
    src = re.sub(r"(\[\d+\])_", r"\1", src)
    return src


def from_asciidoc(src: str) -> str:
    out, row, in_table = [], [], False
    for line in src.splitlines():
        s = line.strip()
        if s == "|===":
            if row:
                out.append(" | ".join(row))
            row, in_table = [], not in_table
            continue
        if in_table:
            if s.startswith("|"):
                cells = [c.strip() for c in s[1:].split("|")]
                if len(cells) > 1 and not row:  # a header or one-line row
                    out.append(" | ".join(cells))
                else:
                    row += cells
            elif not s:
                if row:
                    out.append(" | ".join(row))
                row = []
            else:
                row.append(s)
            continue
        if re.match(r"^(\[\[.*\]\]|\[[a-z]+[^\]]*\]|image::.*|ifdef::.*|endif::.*|include::.*|:\w.*:.*|----|\.\.\.\.|\*\*\*\*)$", s):
            continue
        if s.startswith(".") and len(s) > 1 and s[1] != ".":
            s = s[1:]  # block titles
        out.append(s)
    text = "\n".join(out)
    text = re.sub(r"^=+\s*", "\n", text, flags=re.M)
    text = re.sub(r"(?:https?://|xref:)\S+?\[([^\]]*)\]", r"\1", text)
    text = re.sub(r"<<[^,>]+,([^>]+)>>", r"\1", text)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = text.replace("`", "").replace("*", "")
    return text


CONVERT = {".md": from_markdown, ".rst": from_rst, ".adoc": from_asciidoc}


# --- layout -------------------------------------------------------------------------------------------------

def paragraphs(text: str) -> list[str]:
    """Wrapped lines; prose paragraphs are re-flowed, short and table-like lines are kept as they are."""
    lines = []
    for block in re.split(r"\n\s*\n", text):
        block_lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if not block_lines:
            continue
        keep = any("|" in l or l.startswith("    ") or l.lstrip().startswith(("*", "-", "+ ")) for l in block_lines)
        if keep:
            for l in block_lines:
                lines += textwrap.wrap(l.strip(), WRAP, subsequent_indent="  ") or [""]
        else:
            lines += textwrap.wrap(" ".join(l.strip() for l in block_lines), WRAP)
        lines.append("")
    return lines


def pages_of(title: str, lines: list[str]) -> list[list[str]]:
    body = LINES_PER_PAGE - 4
    chunks = [lines[i:i + body] for i in range(0, len(lines), body)]
    return [[title, ""] + c + ["", f"Page {n} of {len(chunks)}"] for n, c in enumerate(chunks, start=1)]


_ASCII = {"→": "->", "←": "<-", "≤": "<=", "≥": ">=", "≠": "!=", "×": "x",
          "−": "-", "‑": "-", " ": " ", " ": " ", "​": "", "✓": "yes",
          "✗": "no", " ": " "}


def _pdf_string(s: str) -> bytes:
    s = "".join(_ASCII.get(ch, ch) for ch in s)
    b = s.encode("cp1252", errors="replace")
    return b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """Text-only PDF, Helvetica in WinAnsiEncoding (so Spanish accents survive), one list of lines per page."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    kids = []
    for lines in pages:
        body = b"BT /F1 10 Tf 13 TL 50 800 Td " + b" ".join(b"(" + _pdf_string(l) + b") Tj T*" for l in lines) + b" ET"
        objs.append(b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream")
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                    b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % len(objs))
        kids.append(b"%d 0 R" % len(objs))
    objs[1] = b"<< /Type /Pages /Kids [" + b" ".join(kids) + b"] /Count %d >>" % len(kids)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def build(cache: Path):
    PDF_DIR.mkdir(exist_ok=True)
    for name, (title, repo, commit, paths) in SOURCES.items():
        text = "\n\n".join(CONVERT[Path(p).suffix](fetch(repo, commit, p, cache)) for p in paths)
        pages = pages_of(title, paragraphs(text))
        (PDF_DIR / name).write_bytes(make_pdf(pages))
        print(f"{name}: {len(pages)} pages")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=HERE / ".cache", help="where downloaded sources are kept")
    build(ap.parse_args().cache)

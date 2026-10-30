"""PDF text extraction, kept free of app imports so extraction worker processes start quickly."""
import re
import unicodedata
from collections import Counter

from pypdf import PdfReader


def page_count(path: str) -> int:
    return len(PdfReader(path).pages)


def extract(path: str, start: int, stop: int) -> list[str]:
    """Text of pages [start, stop), zero-based."""
    reader = PdfReader(path)
    return [_page_text(reader, i) for i in range(start, stop)]


def _page_text(reader, i: int) -> str:
    """One malformed page shouldn't fail the whole document; it just contributes no text."""
    try:
        return reader.pages[i].extract_text() or ""
    except Exception:  # noqa: BLE001
        return ""


_PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
# Some PDFs come out letter by letter, with words two spaces apart: "C h e c k  t h e  p e d a l".
_LETTER_RUN = re.compile(r"(?<!\S)(?:[^\W\d_] ){2,}[^\W\d_](?!\S)")
_SPACED_RUN = re.compile(r"(?<!\S)(?:\S ){2,}\S(?!\S)")
_ONE_WORD = re.compile(r"(?:\S )+\S")
_DIGITS = re.compile(r"\d+")


def clean_pages(pages: list[str]) -> list[str]:
    """Remove what repeats on many pages (running headers, footers, printer's marks), then clean each page."""
    if len(pages) >= 5:
        seen = Counter(key for page in pages for key in {_line_key(l) for l in page.splitlines()})
        boilerplate = {k for k, n in seen.items() if len(k) >= 12 and n >= max(3, 0.3 * len(pages))}
    else:
        boilerplate = set()
    return [clean_text("\n".join(l for l in page.splitlines() if _line_key(l) not in boilerplate)) for page in pages]


def clean_text(text: str) -> str:
    """Drop glyphs from private-use fonts and rejoin letter-spaced lines, which both dilute the embeddings."""
    return _PRIVATE_USE.sub(" ", "\n".join(_unspace(l) for l in text.splitlines()))


def _unspace(line: str) -> str:
    """Rejoin a letter-spaced line. Only lines that are mostly runs of single letters with two-space word gaps
    count, so table rows like "1 2 3", "Y N N Y" or "x = a + b" are left alone."""
    letters = sum(len(m.group(0)) for m in _LETTER_RUN.finditer(line))
    if "  " not in line.strip() or letters < max(11, 0.5 * len(line.strip())):
        return line
    words = re.split(r" {2,}", line)
    # Inside a spaced line, every two-space-separated part is one word, even a short one like "i s".
    return " ".join(w.replace(" ", "") if _ONE_WORD.fullmatch(w) else
                    _SPACED_RUN.sub(lambda m: m.group(0).replace(" ", ""), w) for w in words)


def _line_key(line: str) -> str:
    # Page numbers and dates differ from page to page in otherwise identical lines.
    return _DIGITS.sub("#", unicodedata.normalize("NFKC", line).strip())


# Words common in one language and rare in the others; enough to tell which language a document is in.
_STOPWORDS = {
    "English": "the and of to is with that this you your are for be from",
    "Spanish": "el los las del por con para una está más también como son",
    "French": "le les des et est une pour dans du avec sur sont pas vous",
    "German": "der die das und ist mit den nicht ein eine für auf sie wird",
    "Italian": "il gli che di per con una sono non della delle nel anche",
    "Portuguese": "os do da em não uma são com para também pelo pela mais",
    "Dutch": "het een en van is dat met voor op niet zijn worden deze",
}
_STOPWORD_SETS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}


def detect_language(text: str) -> str | None:
    """The document's language among a few common ones, by counting their frequent words; None if unclear."""
    words = re.findall(r"[^\W\d_]+", text[:50000].lower())
    if len(words) < 30:
        return None
    hits = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORD_SETS.items()}
    lang, n = max(hits.items(), key=lambda x: x[1])
    return lang if n >= 0.03 * len(words) else None

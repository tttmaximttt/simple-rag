"""The labeled questions in dataset.jsonl, and the page texts of the PDFs they refer to.

Each line is one question:
  id         stable name, e.g. "rpi-01"
  question   what a user would ask
  doc        file name in evals/pdfs/
  pages      1-based pages of doc that answer it (any one of them counts as a hit)
  evidence   verbatim snippets from those pages; `pages` is exactly the set of pages that contain one,
             which tests/test_evals.py checks against the PDFs
  answer     optional reference answer, for reading the reports
  tags       "exact" (codes, numbers, names), "cross-language" (English question, Spanish document)
"""
import json
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent
DATASET = HERE / "dataset.jsonl"
PDF_DIR = HERE / "pdfs"


def load(path: Path = DATASET) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["id"] for q in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate question ids")
    return items


def normalize(text: str) -> str:
    """Case and spacing don't matter when looking for a snippet on a page. A line broken after a hyphen
    ("self-" / "asserted") is joined back without the space; the hyphen is part of the word."""
    text = re.sub(r"-\s+", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@lru_cache(maxsize=None)
def page_texts(doc: str) -> tuple[str, ...]:
    """The pages of an eval PDF as the app sees them: extracted and cleaned by app.pdftext."""
    from app import pdftext
    path = str(PDF_DIR / doc)
    return tuple(pdftext.clean_pages(pdftext.extract(path, 0, pdftext.page_count(path))))


@lru_cache(maxsize=None)
def _normalized_pages(doc: str) -> tuple[str, ...]:
    return tuple(normalize(t) for t in page_texts(doc))


def pages_with(doc: str, snippets) -> list[int]:
    """The 1-based pages of doc that contain any of the snippets."""
    wanted = [normalize(s) for s in snippets]
    return [n for n, text in enumerate(_normalized_pages(doc), start=1) if any(w in text for w in wanted)]

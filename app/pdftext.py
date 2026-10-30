"""PDF text extraction, kept free of app imports so extraction worker processes start quickly."""
from pypdf import PdfReader


def page_count(path: str) -> int:
    return len(PdfReader(path).pages)


def extract(path: str, start: int, stop: int) -> list[str]:
    """Text of pages [start, stop), zero-based."""
    reader = PdfReader(path)
    return [reader.pages[i].extract_text() or "" for i in range(start, stop)]

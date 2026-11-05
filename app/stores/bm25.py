"""A BM25 keyword index kept in memory beside a vector store that has no keyword search of its own (Chroma, and
Qdrant here). It is rebuilt from the chunks stored in the collection, so it needs no extra data on disk and no
re-index: the first keyword search after a start or a change loads every chunk's text and indexes it.

The index for a collection is reused while its signature (the chunk count and a version bumped by this
process's own upserts and deletes) stays the same. A change made by another process that keeps the count the
same goes unnoticed until the next restart or local change; the app has one ingest worker, so that's rare."""
import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict

K1, B = 1.5, 0.75

# Common words in the languages pdftext.detect_language knows, plus question words. They'd otherwise get a high
# weight from being rare in documents ("what", "how") and match noise.
_STOP = set("""
a an and are as at be been but by can do does for from has have how i if in into is it its may must not of on or
should so than that the their them then there these they this to was we what when where which who why will with
would you your about any all also more most other some such only own same very just over both each few
el la los las un una unos unas y o de del al en por con para que como es son está están se su sus lo le les
qué cuál cuáles cómo cuándo dónde quién por qué no sí más también este esta estos estas ese esa eso
""".split())
_WORD = re.compile(r"[^\W_]+(?:[.\-/][^\W_]+)*")
_PARTS = re.compile(r"[.\-/]")


def _fold(word: str) -> str:
    """Lowercase without accents, so "código" matches "codigo"."""
    return "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))


def terms(text: str) -> list[str]:
    """Words, with codes such as "802.3at-2009" or "dvfs=2" kept whole and also split into their parts."""
    out = []
    for w in _WORD.findall(text):
        w = _fold(w)
        out.append(w)
        if _PARTS.search(w):
            out += [p for p in _PARTS.split(w) if p]
    return [t for t in out if t not in _STOP]


class Index:
    """BM25 (Okapi) over a list of chunk payloads, each with a "text"."""

    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, p in enumerate(payloads):
            tf = Counter(terms(p["text"]))
            self.lengths.append(sum(tf.values()))
            for t, n in tf.items():
                self.postings[t].append((i, n))
        self.avg = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def search(self, query: str, k: int) -> list[dict]:
        n_docs, scores = len(self.payloads), defaultdict(float)
        for t in dict.fromkeys(terms(query)):
            posting = self.postings.get(t)
            if not posting:
                continue
            idf = math.log((n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1)
            for i, tf in posting:
                norm = tf + K1 * (1 - B + B * self.lengths[i] / (self.avg or 1))
                scores[i] += idf * tf * (K1 + 1) / norm
        best = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [{**self.payloads[i], "score": s} for i, s in best]


_lock = threading.Lock()  # guards the dicts below; never held while loading or building
_indexes: dict[tuple, tuple[tuple, Index]] = {}
_versions: Counter = Counter()
_building: dict[tuple, threading.Lock] = {}


def changed(key: tuple) -> None:
    """Call after an upsert or delete in the collection, so the next search rebuilds the index."""
    with _lock:
        _versions[key] += 1


def search(key: tuple, count: int, load, query: str, k: int) -> list[dict]:
    """Keyword search in the collection `key`, (re)building its index with load() -> payloads when needed.

    One rebuild per collection runs at a time, outside the shared lock. While it runs, other searches use the
    previous index (a few chunks behind during an ingest) instead of queueing or rebuilding again; only a
    search with no index at all waits for it."""
    if not count:
        return []
    with _lock:
        signature = (count, _versions[key])
        cached = _indexes.get(key)
        building = _building.setdefault(key, threading.Lock())
    if cached is None or cached[0] != signature:
        if building.acquire(blocking=cached is None):
            try:
                with _lock:  # another search may have rebuilt it while this one waited
                    signature, cached = (count, _versions[key]), _indexes.get(key)
                if cached is None or cached[0] != signature:
                    cached = (signature, Index(load()))
                    with _lock:
                        _indexes[key] = cached
            finally:
                building.release()
    return cached[1].search(query, k)


def clear() -> None:
    """Forget every index (tests, or a restart)."""
    with _lock:
        _indexes.clear()
        _versions.clear()
        _building.clear()

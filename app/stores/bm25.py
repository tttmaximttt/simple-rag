"""A BM25 keyword index kept in memory beside a vector store that has no keyword search of its own (Chroma, and
Qdrant here). It is built from the chunks stored in the collection, so it needs no extra data on disk and no
re-index: the first keyword search after a start loads every chunk's text and indexes it.

After that the store keeps it current: upsert() and delete_doc() call added() and removed(), which update the
index in place, so an ingest doesn't rebuild it once per batch. A search whose index no longer has the store's
chunk count (another process wrote to the collection, or an update was lost) rebuilds it. A change by another
process that keeps the count the same goes unnoticed until the next restart; the app has one ingest worker.

At most MAX_INDEXES collections are kept, least recently used dropped first, so switching collections doesn't
keep an old one's text in memory."""
import math
import re
import threading
import unicodedata
from collections import Counter, OrderedDict, defaultdict

K1, B = 1.5, 0.75
MAX_INDEXES = 2


def _fold(word: str) -> str:
    """Lowercase without accents, so "código" matches "codigo"."""
    return "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))


# Common words in the languages pdftext.detect_language knows, plus question words. They'd otherwise get a high
# weight from being rare in documents ("what", "how") and match noise. Folded like the words they're compared to.
_STOP = {_fold(w) for w in """
a an and are as at be been but by can do does for from has have how i if in into is it its may must not of on or
should so than that the their them then there these they this to was we what when where which who why will with
would you your about any all also more most other some such only own same very just over both each few
el la los las un una unos unas y o de del al en por con para que como es son está están se su sus lo le les
qué cuál cuáles cómo cuándo dónde quién por qué no sí más también este esta estos estas ese esa eso
""".split()}
_WORD = re.compile(r"[^\W_]+(?:[.\-/=][^\W_]+)*")
_PARTS = re.compile(r"[.\-/=]")


def terms(text: str) -> list[str]:
    """Words, with codes such as "802.3at-2009" or "dvfs=2" kept whole and also split into their parts."""
    out = []
    for w in _WORD.findall(text):
        w = _fold(w)
        out.append(w)
        if _PARTS.search(w):
            out += [p for p in _PARTS.split(w) if p]
    return [t for t in out if t not in _STOP]


def _chunk_id(payload: dict) -> tuple:
    return payload["doc_id"], payload["chunk"]


class Index:
    """BM25 (Okapi) over chunk payloads, each with doc_id, chunk and text. Chunks can be added and removed;
    removed ones leave a tombstone in the postings that searches skip."""

    def __init__(self, payloads=()):
        self.payloads, self.alive, self.lengths = [], [], []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.df: Counter = Counter()
        self.where: dict[tuple, int] = {}  # chunk id -> position
        self.n = self.total = 0
        self.add(payloads)

    def add(self, payloads) -> None:
        for p in payloads:
            self._remove_at(self.where.get(_chunk_id(p)))  # an upsert replaces the chunk
            tf = Counter(terms(p["text"]))
            i = len(self.payloads)
            self.payloads.append(p)
            self.alive.append(True)
            self.lengths.append(sum(tf.values()))
            self.where[_chunk_id(p)] = i
            for t, n in tf.items():
                self.postings[t].append((i, n))
                self.df[t] += 1
            self.n += 1
            self.total += self.lengths[i]

    def remove_doc(self, doc_id) -> None:
        for cid in [c for c in self.where if c[0] == doc_id]:
            self._remove_at(self.where[cid])

    def _remove_at(self, i) -> None:
        if i is None or not self.alive[i]:
            return
        self.alive[i] = False
        del self.where[_chunk_id(self.payloads[i])]
        for t in set(terms(self.payloads[i]["text"])):
            self.df[t] -= 1
        self.n -= 1
        self.total -= self.lengths[i]

    def search(self, query: str, k: int) -> list[dict]:
        avg, scores = (self.total / self.n if self.n else 1.0), defaultdict(float)
        for t in dict.fromkeys(terms(query)):
            df = self.df.get(t, 0)
            if df <= 0:
                continue
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for i, tf in self.postings[t]:
                if self.alive[i]:
                    scores[i] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.lengths[i] / (avg or 1)))
        best = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [{**self.payloads[i], "score": s} for i, s in best]


class _Entry:
    def __init__(self):
        self.lock = threading.Lock()  # held while building, updating or searching this collection's index
        self.index: Index | None = None


_lock = threading.Lock()  # guards _entries
_entries: "OrderedDict[tuple, _Entry]" = OrderedDict()


def _entry(key: tuple, create: bool) -> _Entry | None:
    with _lock:
        entry = _entries.get(key)
        if entry is None and create:
            entry = _entries[key] = _Entry()
            while len(_entries) > MAX_INDEXES:
                _entries.popitem(last=False)
        if entry is not None:
            _entries.move_to_end(key)
        return entry


def added(key: tuple, payloads: list[dict]) -> None:
    """Call after the store upserted these chunks. Nothing to do if no search has built the index yet."""
    if entry := _entry(key, create=False):
        with entry.lock:
            if entry.index is not None:
                entry.index.add(payloads)


def removed(key: tuple, doc_id) -> None:
    """Call after the store deleted a document's chunks."""
    if entry := _entry(key, create=False):
        with entry.lock:
            if entry.index is not None:
                entry.index.remove_doc(doc_id)


def search(key: tuple, count: int, load, query: str, k: int) -> list[dict]:
    """Keyword search in the collection `key`, which holds `count` chunks; the index is (re)built with load() ->
    payloads when it has none or a different number of chunks."""
    if not count:
        return []
    entry = _entry(key, create=True)
    with entry.lock:
        if entry.index is None or entry.index.n != count:
            entry.index = Index(load())
        return entry.index.search(query, k)


def clear() -> None:
    """Forget every index (tests, or a restart)."""
    with _lock:
        _entries.clear()

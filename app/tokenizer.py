"""Token counting and token-based chunking.

Uses tiktoken's cl100k_base when it is available (exact for OpenAI embedding models,
a close approximation for others). If the encoding file can't be loaded (offline
host), falls back to a regex approximation of roughly one token per word piece.
"""
import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

# Max input tokens of common embedding models. Inputs above the limit are silently
# truncated by most providers, which loses the end of the chunk.
MODEL_LIMITS = {
    "BAAI/bge-small-en-v1.5": 512,
    "BAAI/bge-base-en-v1.5": 512,
    "BAAI/bge-large-en-v1.5": 512,
    "sentence-transformers/all-MiniLM-L6-v2": 256,
    "nomic-ai/nomic-embed-text-v1.5": 8192,
    "nomic-embed-text": 2048,  # Ollama's default context, though the model supports 8192
    "mxbai-embed-large": 512,
    "all-minilm": 256,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}
DEFAULT_LIMIT = 512
# cl100k counts fewer tokens than the WordPiece/SentencePiece tokenizers most open
# models use, so leave headroom for anything that isn't an OpenAI model.
NON_OPENAI_SAFETY = 0.75

_WORDISH = re.compile(r"\s*(?:\w+|[^\w\s])", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class _ApproxEncoding:
    name = "approximate (regex)"

    def encode(self, text):
        # Long words become several tokens, as BPE would split them.
        out = []
        for m in _WORDISH.finditer(text):
            w = m.group()
            out += [w[i:i + 6] for i in range(0, len(w), 6)]
        return out

    def decode(self, tokens):
        return "".join(tokens).strip()


@lru_cache(maxsize=1)
def encoding():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # noqa: BLE001
        log.warning("tiktoken unavailable (%s); using approximate token counts", e)
        return _ApproxEncoding()


def tokenizer_name():
    enc = encoding()
    return enc.name if isinstance(enc, _ApproxEncoding) else f"tiktoken {enc.name}"


def count(text):
    return len(encoding().encode(text))


def model_limit(cfg):
    e = cfg["embeddings"]
    model = e[e["provider"]]["model"]
    limit = MODEL_LIMITS.get(model, DEFAULT_LIMIT)
    return limit if e["provider"] == "openai" else int(limit * NON_OPENAI_SAFETY)


def effective_chunk_tokens(cfg):
    """Chunk size from settings, capped so chunks fit the embedding model."""
    return max(32, min(int(cfg["rag"]["chunk_tokens"]), model_limit(cfg)))


def _split_long(sentence, size):
    enc = encoding()
    toks = enc.encode(sentence)
    return [enc.decode(toks[i:i + size]) for i in range(0, len(toks), size)]


def chunk_text(text, size, overlap):
    """Pack whole sentences into chunks of at most `size` tokens, carrying the last
    ~`overlap` tokens of sentences into the next chunk."""
    text = " ".join(text.split())
    if not text:
        return []
    overlap = min(overlap, size // 2)
    sentences = []
    for s in _SENTENCE_END.split(text):
        n = count(s)
        if n > size:
            sentences += [(p, count(p)) for p in _split_long(s, size)]
        elif s:
            sentences.append((s, n))

    chunks, cur, cur_n = [], [], 0
    for s, n in sentences:
        if cur and cur_n + n > size:
            chunks.append(" ".join(x for x, _ in cur))
            # Keep trailing sentences as overlap, as long as they fit.
            keep, keep_n = [], 0
            for x, xn in reversed(cur):
                if keep_n + xn > overlap or keep_n + xn + n > size:
                    break
                keep.insert(0, (x, xn))
                keep_n += xn
            cur, cur_n = keep, keep_n
        cur.append((s, n))
        cur_n += n
    if cur:
        chunks.append(" ".join(x for x, _ in cur))
    return chunks

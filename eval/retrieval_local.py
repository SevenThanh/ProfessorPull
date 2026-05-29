import numpy as np
from rag.ingest import chunk
from rag.embeddings import embed

_cache = {}


def _emb(old_file):
    h = hash(old_file)
    if h not in _cache:
        parts = chunk(old_file)
        if parts:
            ce = np.array([embed(c, "document") for c in parts])
            ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-12)
        else:
            ce = np.zeros((0, 768))
        _cache[h] = (parts, ce)
    return _cache[h]


def retrieve(old_file, query, k=5):
    parts, ce = _emb(old_file or "")
    if not parts:
        return []
    qe = np.array(embed(query, "query"))
    qe = qe / (np.linalg.norm(qe) + 1e-12)
    scores = ce @ qe
    idx = np.argsort(scores)[-k:][::-1]
    return [parts[i] for i in idx]


if __name__ == "__main__":
    from eval.dataset import load_examples
    ex = load_examples()[0]
    res = retrieve(ex["old_file"], ex["diff_hunk"], k=3)
    print(f"got {len(res)} chunks")
    for i, r in enumerate(res):
        print(f"--- chunk {i} ---")
        print(r[:200])

import numpy as np
from rag.embeddings import embed


def chunk(text, size=400):
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def retrieve(old_file, query, k=3):
    chunks = chunk(old_file)
    if not chunks:
        return []
    ce = np.array([embed(c, "document") for c in chunks])
    qe = np.array(embed(query, "query"))
    eps = 1e-12
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + eps)
    qe = qe / (np.linalg.norm(qe) + eps)
    scores = ce @ qe
    idx = np.argsort(scores)[-k:][::-1]
    return [chunks[i] for i in idx]


if __name__ == "__main__":
    from eval.dataset import load_examples
    ex = load_examples()[0]
    res = retrieve(ex["old_file"], ex["diff_hunk"], k=3)
    print(f"got {len(res)} chunks")
    for i, r in enumerate(res):
        print(f"--- chunk {i} ---")
        print(r[:200])

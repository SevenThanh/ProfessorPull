from rag.embeddings import embed
from rag.pinecone_db import query


def get_context(files, repo_name):
    hits = {}
    for f in files:
        patch = f.get("patch")
        if not patch:
            continue
        vec = embed(patch, task="query")
        for m in query(vec, repo_name, k=5):
            mid = m["id"]
            if mid not in hits or m["score"] > hits[mid]["score"]:
                hits[mid] = m

    ranked = sorted(hits.values(), key=lambda x: x["score"], reverse=True)[:10]
    print(f"Retrieved {len(ranked)} context chunks for {len(files)} files")
    return [{"path": r["filename"], "content": r["text"]} for r in ranked]

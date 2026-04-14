import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
IDX = "professor-pull"

if IDX not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=IDX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

idx = pc.Index(IDX)

def upsert(chunks, repo):
    vecs = [(c["id"], c["vec"], {"text": c["text"], "filename": c["filename"], "repo": repo}) for c in chunks]
    idx.upsert(vectors=vecs, namespace=repo)

def query(vec, repo, k=5):
    res = idx.query(vector=vec, top_k=k, namespace=repo, include_metadata=True)
    return [{"id": m.id, "score": m.score, **m.metadata} for m in res.matches]
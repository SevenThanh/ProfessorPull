import json, argparse, os, re
from rag.embeddings import embed
from rag.pinecone_db import upsert

SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".wasm", ".pyc"}
SKIP_NAME = {"package-lock.json", "yarn.lock", "poetry.lock", ".DS_Store"}
SKIP_DIR = {"node_modules/", "__pycache__/", ".git/"}


def load(path):
    raw = open(path).read()
    raw = raw.split("========")[0].strip()
    return json.loads(raw)["repo_context"]["files"]

def skip(path):
    if os.path.basename(path) in SKIP_NAME:
        return True
    if any(d in path for d in SKIP_DIR):
        return True
    if os.path.splitext(path)[1].lower() in SKIP_EXT:
        return True
    return False

def chunk(text, max_tok=400):
    max_chars = max_tok * 4
    sections = re.split(r"\n\n+", text)
    chunks = []
    for sec in sections:
        if len(sec) <= max_chars:
            if sec.strip():
                chunks.append(sec)
        else:
            lines = sec.split("\n")
            buf = ""
            for ln in lines:
                if len(buf) + len(ln) + 1 > max_chars:
                    if buf.strip():
                        chunks.append(buf)
                    buf = ln
                else:
                    buf = buf + "\n" + ln if buf else ln
            if buf.strip():
                chunks.append(buf)
    return chunks

def run(path, repo):
    files = load(path)
    print(f"Loaded {len(files)} files from {path}")
    kept = [f for f in files if not skip(f["path"])]
    print(f"Kept {len(kept)} files after filtering")
    batch = []
    for f in kept:
        fname = f["path"]
        parts = chunk(f["content"])
        safe = fname.replace("/", "_")
        print(f"  {fname} -> {len(parts)} chunks")
        for i, txt in enumerate(parts):
            vec = embed(txt, task="document")
            batch.append({"id": f"{repo}__{safe}__{i}", "vec": vec, "text": txt, "filename": fname})
    print(f"Upserting {len(batch)} vectors to namespace '{repo}'")
    upsert(batch, repo)
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    run(args.path, args.repo)

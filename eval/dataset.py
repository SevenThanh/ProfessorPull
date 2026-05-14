import os
import json
import random
import zipfile
import urllib.request

URL = "https://zenodo.org/records/6900648/files/Comment_Generation.zip"
RAW = "eval/results/comment_gen.zip"
def _cache_path(n):
    return f"eval/results/examples_{n}.json"


def load_examples(n=50, seed=42):
    cache = _cache_path(n)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    os.makedirs("eval/results", exist_ok=True)
    if not os.path.exists(RAW):
        urllib.request.urlretrieve(URL, RAW)

    rows = []
    with zipfile.ZipFile(RAW) as z:
        name = next(n for n in z.namelist() if n.endswith("msg-test.jsonl"))
        with z.open(name) as f:
            for line in f:
                r = json.loads(line)
                msg = r.get("msg", "")
                if isinstance(msg, str) and 0 < len(msg) < 500:
                    rows.append(r)

    random.seed(seed)
    picked = random.sample(rows, n)

    res = [{"old_file": r["oldf"], "diff_hunk": r["patch"], "comment": r["msg"]} for r in picked]
    with open(cache, "w") as f:
        json.dump(res, f)
    return res


if __name__ == "__main__":
    exs = load_examples()
    print(len(exs))
    ex = exs[0]
    for k in ("old_file", "diff_hunk", "comment"):
        print(f"{k}: {str(ex[k])[:100]}")

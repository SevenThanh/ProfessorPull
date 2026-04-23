import os
import csv
import asyncio
import argparse
from datetime import datetime
from tqdm import tqdm
from eval.dataset import load_examples
from eval import adapters

CSV = "eval/results/scores.csv"
CFGS = {
    "full_pp": adapters.full_pp,
    "gemma": adapters.gemma,
    "no_multiagent": adapters.no_multiagent,
    "baseline": adapters.baseline,
}
FIELDS = ["ex_id", "config", "gen", "ref"]


def done():
    if not os.path.exists(CSV):
        return set()
    with open(CSV) as f:
        return {(int(r["ex_id"]), r["config"]) for r in csv.DictReader(f)}


def append(row):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def _work(name, i, ex):
    fn = CFGS[name]
    try:
        gen = fn(ex)
        ref = ex["comment"]
        return gen, ref, None
    except Exception as e:
        return "", ex["comment"], e


async def run_one(name, i, ex, sem, lock, bar):
    async with sem:
        gen, ref, err = await asyncio.to_thread(_work, name, i, ex)
    if err is not None:
        print(f"[{name} ex{i}] ERROR: {err}")
        with open("eval/results/errors.log", "a") as ef:
            ef.write(f"{datetime.now().isoformat()}\t{name}\tex{i}\t{type(err).__name__}: {err}\n")
    async with lock:
        append({"ex_id": i, "config": name, "gen": gen, "ref": ref})
        bar.update(1)


async def _main_async(names, exs, skip):
    sem = asyncio.Semaphore(5)
    lock = asyncio.Lock()
    total = len(names) * len(exs)
    with tqdm(total=total) as bar:
        tasks = []
        for name in names:
            for i, ex in enumerate(exs):
                if (i, name) in skip:
                    bar.update(1)
                    continue
                tasks.append(run_one(name, i, ex, sem, lock, bar))
        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="full_pp,no_rag,no_multiagent,baseline")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    names = args.configs.split(",")
    exs = load_examples(n=args.n)
    skip = done()

    asyncio.run(_main_async(names, exs, skip))

    counts = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            counts[r["config"]] = counts.get(r["config"], 0) + 1

    print(f"\n{'config':<16} {'n':>6}")
    for c, n in counts.items():
        print(f"{c:<16} {n:>6}")


if __name__ == "__main__":
    main()

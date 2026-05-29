import os
import csv
import json
import time
import asyncio
import anthropic
from datetime import datetime
from tqdm import tqdm
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("Missing env var ANTHROPIC_API_KEY")

SCORES = "eval/results/scores.csv"
OUT = "eval/results/judge.csv"
EXS = "eval/results/examples_200.json"
ERRLOG = "eval/results/judge_errors.log"
FIELDS = ["ex_id", "config", "correctness", "actionability", "depth"]
MODEL = "claude-haiku-4-5-20251001"

SYSTEM = """You are evaluating a code review comment. You will see the code diff and the review. Score the review on three dimensions from 1 to 5.

Correctness: Does the review identify real issues actually present in the diff, without inventing problems? 1 = wrong or hallucinated. 3 = partially correct. 5 = accurate and grounded in the diff.

Actionability: Can a developer act on this review without asking follow-up questions? 1 = vague. 3 = somewhat specific. 5 = concrete, located, fixable.

Depth: Does the review go beyond style nitpicks to logic, design, or architectural concerns? 1 = surface only. 3 = mixed. 5 = substantive reasoning about code quality.

Call the score tool with your three integer ratings."""

TOOL = {
    "name": "score",
    "description": "Submit integer scores for the three review dimensions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
            "actionability": {"type": "integer", "minimum": 1, "maximum": 5},
            "depth": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["correctness", "actionability", "depth"],
    },
}

_cli = Anthropic()


def done():
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {(int(r["ex_id"]), r["config"]) for r in csv.DictReader(f)}


def append(row):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def _load_rows():
    with open(SCORES) as f:
        return list(csv.DictReader(f))


def _load_exs():
    with open(EXS) as f:
        return json.load(f)


def _call(diff, gen):
    delays = [10, 20, 40, 80, 160]
    for attempt in range(5):
        try:
            r = _cli.messages.create(
                model=MODEL,
                max_tokens=300,
                temperature=0,
                system=SYSTEM,
                tools=[TOOL],
                tool_choice={"type": "tool", "name": "score"},
                messages=[{"role": "user", "content": f"DIFF:\n{diff}\n\nREVIEW:\n{gen}"}],
            )
            for b in r.content:
                if b.type == "tool_use" and b.name == "score":
                    return b.input
            raise RuntimeError("no tool_use in response")
        except anthropic.RateLimitError:
            if attempt == 4:
                raise
            time.sleep(delays[attempt])


async def judge_one(row, exs, sem, lock, bar):
    ex_id = int(row["ex_id"])
    name = row["config"]
    gen = row.get("gen", "") or ""
    if not gen.strip():
        async with lock:
            bar.update(1)
        return
    diff = exs[ex_id]["diff_hunk"]
    async with sem:
        try:
            s = await asyncio.to_thread(_call, diff, gen)
        except Exception as e:
            with open(ERRLOG, "a") as ef:
                ef.write(f"{datetime.now().isoformat()}\t{name}\tex{ex_id}\t{type(e).__name__}: {e}\n")
            async with lock:
                bar.update(1)
            return
    async with lock:
        append({
            "ex_id": ex_id,
            "config": name,
            "correctness": int(s["correctness"]),
            "actionability": int(s["actionability"]),
            "depth": int(s["depth"]),
        })
        bar.update(1)


async def _main_async():
    rows = _load_rows()
    exs = _load_exs()
    skip = done()
    todo = [r for r in rows if (int(r["ex_id"]), r["config"]) not in skip]
    sem = asyncio.Semaphore(3)
    lock = asyncio.Lock()
    with tqdm(total=len(todo)) as bar:
        tasks = [judge_one(r, exs, sem, lock, bar) for r in todo]
        await asyncio.gather(*tasks, return_exceptions=True)


def _summary():
    sums = {}
    with open(OUT) as f:
        for r in csv.DictReader(f):
            c = r["config"]
            s = sums.setdefault(c, {"c": 0.0, "a": 0.0, "d": 0.0, "n": 0})
            s["c"] += float(r["correctness"])
            s["a"] += float(r["actionability"])
            s["d"] += float(r["depth"])
            s["n"] += 1
    print(f"\n{'config':<16} {'correct':>8} {'action':>8} {'depth':>8}  n")
    for c, s in sums.items():
        n = s["n"]
        print(f"{c:<16} {s['c']/n:>8.3f} {s['a']/n:>8.3f} {s['d']/n:>8.3f}  {n}")


def main():
    asyncio.run(_main_async())
    _summary()


if __name__ == "__main__":
    main()

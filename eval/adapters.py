import os
import time
import requests
from openai import OpenAI
from agents.agent1_summarizer import run_agent1
from agents.agent2_reasoner import run_agent2, verify_findings
from agents.agent3_reviewer import run_agent3, run_agent3_comment
from eval.retrieval_local import retrieve

OR_KEY = os.environ["OPENROUTER_API_KEY"]
_oai = OpenAI()

PROMPT = """You are a code reviewer. Given the file and diff below, write a single concise review comment identifying any bug, issue, or improvement. Write only the comment, nothing else.

FILE:
{file}

DIFF:
{diff}"""


def to_files(ex):
    patch = ex["diff_hunk"]
    adds = sum(1 for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in patch.splitlines() if l.startswith("-") and not l.startswith("---"))
    return [{
        "filename": "source",
        "status": "modified",
        "patch": patch,
        "additions": adds,
        "deletions": dels,
        "content": ex["old_file"],
    }]


def full_pp(ex):
    files = to_files(ex)
    chunks = retrieve(ex["old_file"] or "", ex["diff_hunk"], k=5)
    ctx = [{"path": "ctx", "content": c} for c in chunks]
    s = run_agent1(files)
    f = run_agent2(s, ctx, files)
    kept = verify_findings(f.get("findings", []), files)
    kept = [x for x in kept if x.get("confidence", "high") != "low"]
    f["findings"] = kept
    return run_agent3_comment(s, f, ex["diff_hunk"])


def no_rag(ex):
    files = to_files(ex)
    s = run_agent1(files)
    f = run_agent2(s, [], files)
    kept = verify_findings(f.get("findings", []), files)
    kept = [x for x in kept if x.get("confidence", "high") != "low"]
    f["findings"] = kept
    return run_agent3_comment(s, f, ex["diff_hunk"])


def gemma(ex):
    msg = PROMPT.format(file=ex["old_file"], diff=ex["diff_hunk"])
    delay = 4
    for attempt in range(6):
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OR_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/SevenThanh/ProfessorPull",
                "X-Title": "ProfessorPull",
            },
            json={
                "model": "google/gemma-4-26b-a4b-it",
                "messages": [{"role": "user", "content": msg}],
                "max_tokens": 256,
                "temperature": 0.2,
            },
            timeout=120,
        )
        if r.status_code == 200:
            d = r.json()
            if "choices" in d:
                return d["choices"][0]["message"]["content"]
            print(f"gemma missing choices, body: {d}")
        else:
            print(f"gemma HTTP {r.status_code}: {r.text[:300]}")
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError("gemma failed after 6 retries")


def no_multiagent(ex):
    msg = PROMPT.format(file=ex["old_file"], diff=ex["diff_hunk"])
    for attempt in range(3):
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OR_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/SevenThanh/ProfessorPull",
                "X-Title": "ProfessorPull",
            },
            json={
                "model": "meta-llama/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": msg}],
                "max_tokens": 256,
                "temperature": 0.2,
            },
            timeout=120,
        )
        if r.status_code == 200:
            d = r.json()
            if "choices" in d:
                return d["choices"][0]["message"]["content"]
            print(f"no_multiagent missing choices, body: {d}")
        else:
            print(f"no_multiagent HTTP {r.status_code}: {r.text[:300]}")
        time.sleep(2)
    raise RuntimeError("no_multiagent failed after 3 retries")


def baseline(ex):
    from openai import RateLimitError, APIError
    msg = PROMPT.format(file=ex["old_file"], diff=ex["diff_hunk"])
    delay = 4
    for attempt in range(6):
        try:
            r = _oai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": msg}],
                max_completion_tokens=512,
            )
            return r.choices[0].message.content
        except (RateLimitError, APIError) as e:
            if attempt == 5:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)


if __name__ == "__main__":
    from eval.dataset import load_examples
    ex = load_examples()[0]
    print("=== ref ===")
    print(ex["comment"])
    for name, fn in [("full_pp", full_pp), ("no_rag", no_rag), ("gemma", gemma), ("no_multiagent", no_multiagent), ("baseline", baseline)]:
        print(f"\n=== {name} ===")
        try:
            print(fn(ex))
        except Exception as e:
            print(f"ERROR: {e}")


import json
import os
import csv
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DIR = os.path.join(BASE_DIR, "scrapedFiles")
INPUT_FILE  = os.path.join(SCRAPED_DIR, "bug_fix_prs.json")
OUTPUT_JSON = os.path.join(SCRAPED_DIR, "copilot_results.json")
OUTPUT_CSV  = os.path.join(SCRAPED_DIR, "copilot_results.csv")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

MAX_PRS = None  


def call_copilot(pr: dict) -> str:
    prompt = f"""You are a senior software engineer doing a code review.
Review the following pull request and identify any bugs, logic errors, or issues with the implementation.
Be specific about what file and line the problem is on if possible.

PR Title: {pr['title']}

PR Description:
{(pr.get('body') or '')[:1000]}

Diff:
{pr['diff'][:6000]}

Give a concise technical review. Focus on real bugs and logic issues."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"



def judge(tool_output: str, human_comments: list[dict]) -> tuple[str, str]:
    if tool_output.startswith("ERROR"):
        return "N", f"Tool failed: {tool_output}"

    ground_truth = "\n".join(
        f"- {c['body']}" for c in human_comments if c.get("body")
    )

    judge_prompt = f"""You are evaluating an AI code review tool.

A human reviewer left this feedback on a pull request (this is the ground truth):
--- HUMAN REVIEW ---
{ground_truth}
---

The AI tool produced this review:
--- AI OUTPUT ---
{tool_output[:2000]}
---

Did the AI tool catch the SAME issue the human flagged?
It does NOT need to match word-for-word — it just needs to identify the same underlying problem.

Reply with exactly this format:
VERDICT: Y
REASON: <one sentence>

or

VERDICT: N
REASON: <one sentence>"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=120,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        verdict = "Y" if "VERDICT: Y" in text else "N"
        reason  = text.split("REASON:")[-1].strip() if "REASON:" in text else text
        return verdict, reason
    except Exception as e:
        return "?", f"Judge error: {e}"


#Main

def run():
    with open(INPUT_FILE) as f:
        prs = json.load(f)

    if MAX_PRS:
        prs = prs[:MAX_PRS]

    print(f"\nTesting Copilot on {len(prs)} PRs...\n")

    results = []

    for i, pr in enumerate(prs, 1):
        num   = pr["pr_number"]
        repo  = pr["repo"]
        title = pr["title"][:65]
        print(f"[{i}/{len(prs)}] {repo} PR #{num} — {title}")

        # Review
        print("  → Copilot reviewing...")
        output = call_copilot(pr)
        time.sleep(1)

        # Judge
        print("  → Judging...")
        caught, reason = judge(output, pr["human_comments"])
        time.sleep(0.5)

        print(f"  ✓ Caught: {caught} — {reason[:80]}\n")

        results.append({
            "pr_number":      num,
            "repo":           repo,
            "title":          pr["title"],
            "bug_confirmed":  pr.get("bug_confirmed", False),
            "human_comments": [c["body"] for c in pr["human_comments"]],
            "copilot_caught": caught,
            "reason":         reason,
            "copilot_output": output,
        })

    
    os.makedirs(SCRAPED_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results → {OUTPUT_JSON}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pr_number", "repo", "title", "bug_confirmed",
            "copilot_caught", "reason"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f" CSV → {OUTPUT_CSV}")

    total   = len(results)
    caught  = sum(1 for r in results if r["copilot_caught"] == "Y")
    missed  = total - caught

    print(f"""
Total PRs tested:  {total:<18}║
Bugs caught:       {caught:<18}║
Bugs missed:       {missed:<18}║
Detection rate:    {f"{caught/total*100:.1f}%":<18}
""")

    confirmed_prs  = [r for r in results if r["bug_confirmed"]]
    if confirmed_prs:
        conf_caught = sum(1 for r in confirmed_prs if r["copilot_caught"] == "Y")
        print(f"  On confirmed bugs only ({len(confirmed_prs)} PRs): "
              f"{conf_caught}/{len(confirmed_prs)} caught "
              f"({conf_caught/len(confirmed_prs)*100:.1f}%)")


if __name__ == "__main__":
    run()

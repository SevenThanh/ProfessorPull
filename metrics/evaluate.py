import json
import os
import sys
import time
from rouge_score import rouge_scorer
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DIR = os.path.join(BASE_DIR, "scrapedFiles")

sys.path.insert(0, os.path.join(BASE_DIR, ".."))

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

from rag.pipeline import run_review

def run_professor_pull(diff, repo_name):
    files = [{
        "filename": "diff",
        "status": "modified",
        "additions": diff.count("\n+"),
        "deletions": diff.count("\n-"),
        "patch": diff,
    }]
    return run_review(files, repo_name)

def compute_rouge(prediction, references):
    if not references:
        return 0.0
    scores = [
        scorer.score(ref, prediction)["rougeL"].fmeasure
        for ref in references
    ]
    return round(max(scores), 4)

def evaluate_on_dataset(prs, system_name, agent_fn):
    results = []

    for i, pr in enumerate(prs):
        print(f"  [{i+1}/{len(prs)}] PR #{pr['pr_number']} — {pr['title'][:50]}...")

        diff = pr["diff"]
        human_comments = [c["body"] for c in pr["human_comments"]]

        try:
            output = agent_fn(diff, pr.get("repo", ""))
        except Exception as e:
            print(f"    ⚠️  Agent error: {e}")
            output = ""

        rouge = compute_rouge(output, human_comments)

        results.append({
            "pr_number": pr["pr_number"],
            "repo": pr["repo"],
            "title": pr["title"],
            "system": system_name,
            "output": output,
            "rouge_l": rouge,
        })

        time.sleep(1)

    return results

def print_summary(all_results):
    from collections import defaultdict

    by_system = defaultdict(list)
    for r in all_results:
        by_system[r["system"]].append(r)

    print("\n" + "="*40)
    print(f"{'System':<25} {'ROUGE-L':>8}")
    print("="*40)

    for system, results in by_system.items():
        avg_rouge = round(sum(r["rouge_l"] for r in results) / len(results), 3)
        print(f"{system:<25} {avg_rouge:>8}")

    print("="*40)

if __name__ == "__main__":
    with open(os.path.join(SCRAPED_DIR, "test_set.json"), "r") as f:
        all_prs = json.load(f)

    print(f"Loaded {len(all_prs)} PRs\n")

    print("Running Professor Pull...")
    results = evaluate_on_dataset(
        all_prs,
        system_name="Professor Pull",
        agent_fn=run_professor_pull
    )

    with open(os.path.join(SCRAPED_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print("\nFull results saved to eval_results.json")
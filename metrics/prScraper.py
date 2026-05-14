import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DIR = os.path.join(BASE_DIR, "scrapedFiles")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_prs_with_reviews(owner, repo, max_prs=10):
    print(f"\nScraping {owner}/{repo}...")
    results = []
    page = 1

    while len(results) < max_prs:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {"state": "closed", "per_page": 30, "page": page}
        prs = requests.get(url, headers=HEADERS, params=params).json()

        if not prs:
            break

        for pr in prs:
            if len(results) >= max_prs:
                break

            pr_number = pr["number"]

            # Only keep PRs that have real review comments
            comments_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
            comments = requests.get(comments_url, headers=HEADERS).json()

            if len(comments) < 2:
                continue

            # Fetch the diff
            diff_headers = {**HEADERS, "Accept": "application/vnd.github.v3.diff"}
            diff_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
            diff = requests.get(diff_url, headers=diff_headers).text

            # Skip massive diffs
            if len(diff) > 20000:
                continue

            results.append({
                "repo": f"{owner}/{repo}",
                "pr_number": pr_number,
                "title": pr["title"],
                "body": pr.get("body", ""),
                "merged": pr["merged_at"] is not None,
                "diff": diff,
                "human_comments": [
                    {
                        "body": c["body"],
                        "path": c.get("path", ""),
                        "line": c.get("line")
                    }
                    for c in comments
                ]
            })

            print(f"  ✓ PR #{pr_number} — {pr['title'][:60]}")
            time.sleep(0.5) 

        page += 1

    return results


# Scrape both repos
dataset = []
dataset += get_prs_with_reviews("tiangolo", "fastapi", max_prs=50)
dataset += get_prs_with_reviews("pallets", "flask", max_prs=50)

os.makedirs(SCRAPED_DIR, exist_ok=True)
with open(os.path.join(SCRAPED_DIR, "test_set.json"), "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\nDone. {len(dataset)} PRs saved to scrapedFiles/test_set.json")
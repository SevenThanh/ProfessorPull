import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DIR = os.path.join(BASE_DIR, "scrapedFiles")


BUG_KEYWORDS = ["fix", "bug", "error", "issue", "crash", "broken", "patch", "regression", "incorrect", "wrong", "fail"]

def is_bug_fix(pr):
    title = pr.get("title", "").lower()
    body = (pr.get("body") or "").lower()
    return any(kw in title or kw in body for kw in BUG_KEYWORDS)

def get_full_pr_details(repo, pr_number):
    """Fetch the full PR body/title in case the scraper didn't save it"""
    owner, name = repo.split("/")
    url = f"https://api.github.com/repos/{owner}/{name}/pulls/{pr_number}"
    return requests.get(url, headers=HEADERS).json()

# Load your existing test set
with open(os.path.join(SCRAPED_DIR, "test_set.json"), "r") as f:
    test_set = json.load(f)


print(f"Loaded {len(test_set)} PRs from test_set.json")

# Filter for bug-fix PRs
bug_prs = []
non_bug_prs = []

for pr in test_set:
    # Fetch full details if body is missing (scraper may not have saved it)
    if not pr.get("body"):
        details = get_full_pr_details(pr["repo"], pr["pr_number"])
        pr["body"] = details.get("body", "")
        pr["title"] = details.get("title", pr.get("title", ""))

    if is_bug_fix(pr):
        pr["pr_type"] = "bug_fix"
        pr["bug_confirmed"] = False   
        pr["bug_description"] = ""   
        bug_prs.append(pr)
        print(f" BUG  | PR #{pr['pr_number']} [{pr['repo']}] — {pr['title'][:70]}")
    else:
        pr["pr_type"] = "feature_or_refactor"
        non_bug_prs.append(pr)
        print(f" OTHER | PR #{pr['pr_number']} [{pr['repo']}] — {pr['title'][:70]}")

# Save split datasets
with open(os.path.join(SCRAPED_DIR, "bug_fix_prs.json"), "w") as f:
    json.dump(bug_prs, f, indent=2)

with open(os.path.join(SCRAPED_DIR, "non_bug_prs.json"), "w") as f:
    json.dump(non_bug_prs, f, indent=2)

with open(os.path.join(SCRAPED_DIR, "test_set.json"), "w") as f:
    json.dump(test_set, f, indent=2)

print(f"\n--- Summary ---")
print(f"Bug-fix PRs:  {len(bug_prs)}  → saved to bug_fix_prs.json")
print(f"Other PRs:    {len(non_bug_prs)} → saved to non_bug_prs.json")
print(f"test_set.json updated with pr_type labels")
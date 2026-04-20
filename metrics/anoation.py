import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAPED_DIR = os.path.join(BASE_DIR, "scrapedFiles")

with open(os.path.join(SCRAPED_DIR, "bug_fix_prs.json"), "r") as f:
    bug_prs = json.load(f)

# Classify each PR by whether the diff touches real code files
def has_code_changes(diff):
    code_extensions = [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"]
    return any(ext in diff for ext in code_extensions)

for pr in bug_prs:
    pr["has_code_diff"] = has_code_changes(pr["diff"])
    diff_lines = [l for l in pr["diff"].split("\n") if l.startswith("+") and not l.startswith("+++")]
    pr["lines_added"] = len(diff_lines)

# Split into code-bug PRs (precision/recall) vs docs-bug PRs (ROUGE only)
code_bug_prs = [pr for pr in bug_prs if pr["has_code_diff"]]
docs_bug_prs  = [pr for pr in bug_prs if not pr["has_code_diff"]]

print(f"Real code bugs (precision/recall set): {len(code_bug_prs)}")
for pr in code_bug_prs:
    print(f"  #{pr['pr_number']} +{pr['lines_added']} lines — {pr['title'][:70]}")

print(f"\nDocs-only bugs (ROUGE set only): {len(docs_bug_prs)}")
for pr in docs_bug_prs:
    print(f"  #{pr['pr_number']} — {pr['title'][:70]}")

with open(os.path.join(SCRAPED_DIR, "code_bug_prs.json"), "w") as f:
    json.dump(code_bug_prs, f, indent=2)

print("\nSaved code_bug_prs.json")
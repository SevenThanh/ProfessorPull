import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

OR_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OR_KEY:
    raise RuntimeError("Missing env var OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"

HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/SevenThanh/ProfessorPull",
    "X-Title": "ProfessorPull",
}

SYSTEM_PROMPT = """You are a senior software engineer performing a thorough code review.
You will be given:
1. The raw diffs from the pull request — the actual lines added and removed
2. A summary of what changed (from a previous analysis step)
3. Relevant files from the codebase for context

Analyze the raw diffs directly — do not rely solely on the summary.

Surface concerns across these dimensions:
- Correctness: logic errors, off-by-one, null/undefined handling, type mismatches, broken imports, syntax breaks
- Performance: expensive work in loops, blocking calls, memory leaks, redundant computation, re-render triggers
- Design: unclear invariants, hidden coupling, leaking abstractions, API-surface regressions, inconsistent error handling
- Maintainability: duplicated logic, violations of existing patterns in the same file, confusing control flow, dead code
- Safety: input validation gaps, unchecked external calls, race conditions, resource leaks

For every finding, explain *why* it matters — the specific scenario where it breaks, the assumption it violates, or the downstream consequence. Root-cause reasoning, not surface observation. Reference exact function names, variable names, or line regions.

Grounding is mandatory. Every finding must include an "evidence" field containing the exact added or removed line from the diff (prefixed with + or -) that proves the finding. If you cannot quote a specific line, do not include the finding. Assign "confidence" of high, medium, or low: high means the evidence line unambiguously demonstrates the issue; low means the issue depends on unseen code or framework behavior you are unsure about.

IMPORTANT:
- Respond ONLY with valid JSON. No explanation, no markdown, no code fences.
- If you find no issues, return an empty findings array — do not invent problems.
- A syntax error that breaks the build is always high severity.
- Prefer one sharp finding with deep reasoning over three shallow ones.
"""


def _prompt(agent1_summary, context_files, pr_files):
    diff_sections = []
    for f in pr_files:
        filename = f.get("filename", "unknown")
        status = f.get("status", "modified")
        additions = f.get("additions", 0)
        deletions = f.get("deletions", 0)
        patch = f.get("patch", "")
        if not patch:
            continue
        diff_sections.append(
            f"FILE: {filename}\n"
            f"STATUS: {status} (+{additions} / -{deletions})\n"
            f"DIFF:\n{patch}"
        )
    diffs_block = "\n\n---\n\n".join(diff_sections) if diff_sections else "No diff content available."

    summary_block = json.dumps(agent1_summary, indent=2)

    file_blocks = []
    for f in context_files:
        path = f.get("path", "unknown")
        content = f.get("content", "")
        if content:
            file_blocks.append(f"FILE: {path}\n```\n{content}\n```")
    files_block = "\n\n".join(file_blocks) if file_blocks else "No additional context files available."

    return f"""Here are the raw diffs from the pull request. Analyze these directly for any issues:

{diffs_block}

Here is the summary from the previous analysis step (use for context, but do not rely on it exclusively):

{summary_block}

Here are relevant files from the codebase for additional context:

{files_block}

Based on the raw diffs and codebase context, identify any bugs, syntax errors, or performance issues.

Return this exact JSON structure:
{{
  "findings": [
    {{
      "filename": "path/to/file.ext",
      "type": "bug | performance | design | maintainability | safety",
      "severity": "low | medium | high",
      "confidence": "high | medium | low",
      "evidence": "the exact +/- line from the diff that proves this finding",
      "title": "Short title for this finding.",
      "description": "Explain the root cause and the concrete scenario where this breaks or degrades the code.",
      "suggestion": "Concrete, specific fix — reference function/variable names or exact changes to make."
    }}
  ],
  "overall_risk": "low | medium | high",
  "risk_reasoning": "One paragraph explaining the overall risk assessment.",
  "approved": true or false
}}

Set approved to true only if there are no high severity findings.
"""


def _call(user_prompt, retries=2):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=180)
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"Agent 2: request timed out, retrying (attempt {attempt + 1}/{retries})...")
                continue
            raise
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {r.status_code}: {r.text}")
        return r.json()["choices"][0]["message"]["content"]


def _parse(raw):
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in model output:\n{raw[:500]}")
    return json.loads(m.group(), strict=False)


def run_agent2(agent1_summary, context_files, pr_files):
    if not agent1_summary.get("changed_files"):
        return {
            "findings": [],
            "overall_risk": "low",
            "risk_reasoning": "No files were changed.",
            "approved": True,
        }
    user_prompt = _prompt(agent1_summary, context_files, pr_files)
    raw = _call(user_prompt)
    return _parse(raw)


VERIFY_SYSTEM = """You verify code review findings against a raw diff. For each finding you receive, decide whether the quoted evidence actually appears in the diff and whether the finding's claim is directly supported by visible diff content.

Keep only findings where both are true:
1. The evidence line (verbatim or near-verbatim) appears in the diff.
2. The claim in description follows from code visible in the diff — no reliance on unseen context, framework assumptions, or guesses.

Drop everything else. Do not rewrite findings — only keep or drop.

Respond ONLY with valid JSON. No markdown, no fences.
"""


def _verify_prompt(findings, pr_files):
    diff_sections = []
    for f in pr_files:
        filename = f.get("filename", "unknown")
        patch = f.get("patch", "")
        if not patch:
            continue
        diff_sections.append(f"FILE: {filename}\nDIFF:\n{patch}")
    diffs_block = "\n\n---\n\n".join(diff_sections) if diff_sections else "No diff content available."

    return f"""Raw diff:
{diffs_block}

Candidate findings:
{json.dumps(findings, indent=2)}

Return only the findings that pass verification, preserving their fields exactly:
{{
  "findings": [ ... ]
}}
"""


def verify_findings(findings, pr_files):
    if not findings:
        return []
    user_prompt = _verify_prompt(findings, pr_files)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=180)
        except requests.exceptions.Timeout:
            if attempt < 2:
                continue
            raise
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {r.status_code}: {r.text}")
        raw = r.json()["choices"][0]["message"]["content"]
        try:
            return _parse(raw).get("findings", [])
        except Exception:
            return findings


if __name__ == "__main__":
    import json

    sample_agent1_summary = {
        "changed_files": [
            {
                "filename": "src/components/sections/About.jsx",
                "status": "modified",
                "change_type": "cosmetic",
                "summary": "Updated the placeholder text in the About section.",
                "risk": "low",
                "risk_reason": "Only a text string was changed, no logic affected."
            }
        ],
        "overall_summary": "This PR updates placeholder text in the About component.",
        "overall_change_type": "cosmetic",
        "overall_risk": "low",
        "flags": []
    }

    sample_context_files = [
        {
            "path": "src/components/sections/About.jsx",
            "content": "export const About = () => {\n  return (\n    <div>\n      <h3>Here you can put what you want to say about yourself. new changes for a practice commit</h3>\n    </div>\n  );\n};"
        },
        {
            "path": "src/App.jsx",
            "content": "import { About } from './components/sections/About';\n\nfunction App() {\n  return <About />;\n}\n\nexport default App;"
        }
    ]

    sample_pr_files = [
        {
            "filename": "src/components/sections/About.jsx",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "patch": (
                "@@ -7,7 +7,7 @@ export const About = () => {\n"
                "                 </h2>\n"
                "                 <div className=\"glass rounded-xl\">\n"
                "                     <h3 className=\"text-white text-lg\">\n"
                "-                        Here you can put what you want to say about yourself. Some differece for testing \n"
                "+                        Here you can put what you want to say about yourself. new changes for a practice commit\n"
                "                     </h3>\n"
                "                 </div>\n"
                "             </div>"
            ),
        }
    ]

    result = run_agent2(sample_agent1_summary, sample_context_files, sample_pr_files)
    print(json.dumps(result, indent=2))

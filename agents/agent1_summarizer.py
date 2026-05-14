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
MODEL = "qwen/qwen3-coder"

HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/SevenThanh/ProfessorPull",
    "X-Title": "ProfessorPull",
}

SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
You will be given a list of file diffs from a GitHub pull request.
Your job is to analyze what changed and produce a structured JSON summary.

IMPORTANT:
- Respond ONLY with valid JSON. No explanation, no markdown, no code fences.
- Be specific about what logic changed, not just what lines changed.
- If a change is purely cosmetic (whitespace, comments, text copy), mark it as such.
"""


def _prompt(pr_files):
    sections = []
    for f in pr_files:
        filename = f.get("filename", "unknown")
        status = f.get("status", "modified")
        additions = f.get("additions", 0)
        deletions = f.get("deletions", 0)
        patch = f.get("patch", "")
        if not patch:
            continue
        sections.append(
            f"FILE: {filename}\n"
            f"STATUS: {status} (+{additions} / -{deletions})\n"
            f"DIFF:\n{patch}"
        )
    if not sections:
        return "No diff content available."
    joined = "\n\n---\n\n".join(sections)
    return f"""Analyze the following pull request diffs and return a JSON summary.

{joined}

Return this exact JSON structure:
{{
  "changed_files": [
    {{
      "filename": "...",
      "status": "added | modified | removed | renamed",
      "change_type": "logic | cosmetic | config | dependency | test | refactor",
      "summary": "One sentence describing what changed in this file.",
      "risk": "low | medium | high",
      "risk_reason": "Why this risk level was assigned."
    }}
  ],
  "overall_summary": "One paragraph describing the PR as a whole.",
  "overall_change_type": "feature | bugfix | refactor | config | cosmetic | test | mixed",
  "overall_risk": "low | medium | high",
  "flags": ["List any concerns worth flagging, or empty array if none."]
}}"""


def _call(user_prompt, retries=2):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"Agent 1: request timed out, retrying (attempt {attempt + 1}/{retries})...")
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


def run_agent1(pr_files):
    if not pr_files:
        return {
            "changed_files": [],
            "overall_summary": "No files changed.",
            "overall_change_type": "cosmetic",
            "overall_risk": "low",
            "flags": [],
        }
    user_prompt = _prompt(pr_files)
    raw = _call(user_prompt)
    return _parse(raw)


if __name__ == "__main__":
    sample_files = [
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

    result = run_agent1(sample_files)
    print(json.dumps(result, indent=2))

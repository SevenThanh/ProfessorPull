"""
Agent 1 — Diff Summarizer (Qwen2.5-Coder-32B)

Input:  List of changed files from retrieval_log.txt → retrieval["files"]
Output: Structured JSON summary of what changed, for Agent 2 to reason over

Calls the Hugging Face Inference API — no local GPU required.
"""

import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing env var HF_TOKEN")

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL   = "Qwen/Qwen2.5-Coder-32B-Instruct"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
You will be given a list of file diffs from a GitHub pull request.
Your job is to analyze what changed and produce a structured JSON summary.

IMPORTANT:
- Respond ONLY with valid JSON. No explanation, no markdown, no code fences.
- Be specific about what logic changed, not just what lines changed.
- If a change is purely cosmetic (whitespace, comments, text copy), mark it as such.
"""

def _build_user_prompt(pr_files: list[dict]) -> str:
    sections = []

    for f in pr_files:
        filename  = f.get("filename", "unknown")
        status    = f.get("status", "modified")
        additions = f.get("additions", 0)
        deletions = f.get("deletions", 0)
        patch     = f.get("patch", "")

        if not patch:
            continue  # skip binary files or files with no diff

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


# ── API Call ───────────────────────────────────────────────────────────────────

def _call_api(user_prompt: str, retries: int = 2) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"Agent 1: request timed out, retrying (attempt {attempt + 1}/{retries})...")
                continue
            raise

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {response.status_code}: {response.text}"
            )

        return response.json()["choices"][0]["message"]["content"]


# ── Output Parsing ─────────────────────────────────────────────────────────────

def _parse_output(raw: str) -> dict:
    """
    Extract JSON from model output.
    Models sometimes wrap output in markdown fences even when told not to.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{raw[:500]}")

    return json.loads(match.group())


# ── Public Interface ───────────────────────────────────────────────────────────

def run_agent1(pr_files: list[dict]) -> dict:
    """
    Summarize a PR's diffs using Qwen2.5-Coder via HuggingFace API.

    Args:
        pr_files: The `files` list from retrieval_log.txt → retrieval["files"]
                  Each item has: filename, status, additions, deletions, patch

    Returns:
        Structured dict with changed_files, overall_summary, risk, flags.
        Passed directly to Agent 2 as input.
    """
    if not pr_files:
        return {
            "changed_files": [],
            "overall_summary": "No files changed.",
            "overall_change_type": "cosmetic",
            "overall_risk": "low",
            "flags": [],
        }

    user_prompt = _build_user_prompt(pr_files)
    raw_output  = _call_api(user_prompt)
    return _parse_output(raw_output)


# ── Quick local test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Uses the real patch from your retrieval_log.txt
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
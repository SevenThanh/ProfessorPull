import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

OR_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OR_KEY:
    raise RuntimeError("Missing env var OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.3-70b-instruct"

HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/SevenThanh/ProfessorPull",
    "X-Title": "ProfessorPull",
}

SYSTEM_PROMPT = """You are a senior software engineer writing a formal code review comment on GitHub.
You will be given a structured analysis of a pull request including what changed and any issues found.

Your job is to synthesize this into a single, well-formatted GitHub markdown comment.

Guidelines:
- Use a professional and formal tone throughout, but sound human and keep things simplistic enough to understand.
- Be constructive — explain why something is an issue, not just that it is one.
- Use GitHub markdown formatting: headers, bullet points, code blocks where appropriate.
- Do not invent issues that are not in the findings. Only report what you were given.
- If there are no findings, write a clean approval comment.
- End with a clear verdict: approved or changes requested.

Respond with ONLY the markdown comment text. No preamble, no explanation.
"""


def _prompt(agent1_summary, agent2_findings):
    overall_summary = agent1_summary.get("overall_summary", "No summary available.")
    overall_change_type = agent1_summary.get("overall_change_type", "unknown")
    agent1_risk = agent1_summary.get("overall_risk", "low")
    agent1_flags = agent1_summary.get("flags", [])

    findings = agent2_findings.get("findings", [])
    overall_risk = agent2_findings.get("overall_risk", "low")
    risk_reasoning = agent2_findings.get("risk_reasoning", "")
    approved = agent2_findings.get("approved", True)

    changed_files = agent1_summary.get("changed_files", [])
    files_block = "\n".join(
        f"  - {f['filename']} ({f['status']}) — {f['summary']}"
        for f in changed_files
    )

    if findings:
        findings_block = "\n\n".join(
            f"  Finding #{i+1}:\n"
            f"  File: {f['filename']}\n"
            f"  Type: {f['type']}\n"
            f"  Severity: {f['severity']}\n"
            f"  Title: {f['title']}\n"
            f"  Description: {f['description']}\n"
            f"  Suggestion: {f['suggestion']}"
            for i, f in enumerate(findings)
        )
    else:
        findings_block = "  No issues found."

    flags_block = "\n".join(f"  - {flag}" for flag in agent1_flags) if agent1_flags else "  None."

    verdict = "APPROVED" if approved else "CHANGES REQUESTED"

    return f"""Write a formal GitHub PR review comment based on the following analysis.

PULL REQUEST OVERVIEW:
  Summary: {overall_summary}
  Change type: {overall_change_type}
  Risk level: {agent1_risk}

FILES CHANGED:
{files_block}

ADDITIONAL FLAGS FROM INITIAL ANALYSIS:
{flags_block}

BUG & PERFORMANCE FINDINGS:
{findings_block}

OVERALL RISK ASSESSMENT:
  Risk level: {overall_risk}
  Reasoning: {risk_reasoning}

VERDICT: {verdict}

Write the GitHub markdown review comment now.
Format it clearly with sections for: Overview, Changes, Findings (if any), and Verdict.
"""


def _call(user_prompt, retries=2):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"Agent 3: request timed out, retrying (attempt {attempt + 1}/{retries})...")
                continue
            raise
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {r.status_code}: {r.text}")
        return r.json()["choices"][0]["message"]["content"]


def _clean(raw):
    lines = raw.strip().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("##") or stripped.startswith("**"):
            return "\n".join(lines[i:])
    return raw.strip()


def run_agent3(agent1_summary, agent2_findings):
    user_prompt = _prompt(agent1_summary, agent2_findings)
    raw = _call(user_prompt)
    return _clean(raw)


if __name__ == "__main__":
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
        "overall_summary": "This PR updates placeholder text in the About component. No logic or structure was modified.",
        "overall_change_type": "cosmetic",
        "overall_risk": "low",
        "flags": []
    }

    sample_agent2_findings = {
        "findings": [],
        "overall_risk": "low",
        "risk_reasoning": "The change is purely cosmetic — a text string update with no impact on logic, rendering behavior, or performance.",
        "approved": True
    }

    result = run_agent3(sample_agent1_summary, sample_agent2_findings)
    print(result)

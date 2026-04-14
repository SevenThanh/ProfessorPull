"""
Agent 3 — Review Comment Writer (Llama 3.3-70B)

Input:  - agent1_summary: dict output from run_agent1()
        - agent2_findings: dict output from run_agent2()

Output: A formatted markdown string ready to be posted as a GitHub PR review comment.
        This is the final human-facing output of the entire pipeline.

Calls the Hugging Face Inference API — no local GPU required.
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing env var HF_TOKEN")

API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
MODEL   = "meta-llama/Llama-3.3-70B-Instruct"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software engineer writing a formal code review comment on GitHub.
You will be given a structured analysis of a pull request including what changed and any issues found.

Your job is to synthesize this into a single, well-formatted GitHub markdown comment.

Guidelines:
- Use a professional and formal tone throughout.
- Be constructive — explain why something is an issue, not just that it is one.
- Use GitHub markdown formatting: headers, bullet points, code blocks where appropriate.
- Do not invent issues that are not in the findings. Only report what you were given.
- If there are no findings, write a clean approval comment.
- End with a clear verdict: approved or changes requested.

Respond with ONLY the markdown comment text. No preamble, no explanation.
"""

def _build_user_prompt(agent1_summary: dict, agent2_findings: dict) -> str:
    overall_summary     = agent1_summary.get("overall_summary", "No summary available.")
    overall_change_type = agent1_summary.get("overall_change_type", "unknown")
    agent1_risk         = agent1_summary.get("overall_risk", "low")
    agent1_flags        = agent1_summary.get("flags", [])

    findings            = agent2_findings.get("findings", [])
    overall_risk        = agent2_findings.get("overall_risk", "low")
    risk_reasoning      = agent2_findings.get("risk_reasoning", "")
    approved            = agent2_findings.get("approved", True)

    # Format changed files
    changed_files = agent1_summary.get("changed_files", [])
    files_block = "\n".join(
        f"  - {f['filename']} ({f['status']}) — {f['summary']}"
        for f in changed_files
    )

    # Format findings
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

    # Format Agent 1 flags
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


# ── API Call ───────────────────────────────────────────────────────────────────

def _call_api(user_prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.3,   # Slightly higher than Agents 1 & 2 — we want natural writing, not rigid JSON
        "max_tokens": 1024,
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"HuggingFace API error {response.status_code}: {response.text}"
        )

    return response.json()["choices"][0]["message"]["content"]


# ── Light cleanup ──────────────────────────────────────────────────────────────

def _clean_output(raw: str) -> str:
    """
    Strip any accidental preamble the model adds before the markdown.
    e.g. "Here is the review comment:" or "Sure! Here's the formatted review:"
    """
    lines = raw.strip().splitlines()

    # Drop leading lines that don't look like markdown
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("##") or stripped.startswith("**"):
            return "\n".join(lines[i:])

    return raw.strip()


# ── Public Interface ───────────────────────────────────────────────────────────

def run_agent3(agent1_summary: dict, agent2_findings: dict) -> str:
    """
    Write a formatted GitHub PR review comment using Llama 3.3-70B.

    Args:
        agent1_summary:  The dict returned by run_agent1(). Contains changed_files,
                         overall_summary, overall_change_type, overall_risk, flags.
        agent2_findings: The dict returned by run_agent2(). Contains findings,
                         overall_risk, risk_reasoning, approved.

    Returns:
        A markdown string ready to be passed directly to Matt's post_pr_review()
        as the `body` parameter.
    """
    user_prompt = _build_user_prompt(agent1_summary, agent2_findings)
    raw_output  = _call_api(user_prompt)
    return _clean_output(raw_output)


# ── Quick local test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate Agent 1 output
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

    # Simulate Agent 2 output
    sample_agent2_findings = {
        "findings": [],
        "overall_risk": "low",
        "risk_reasoning": "The change is purely cosmetic — a text string update with no impact on logic, rendering behavior, or performance.",
        "approved": True
    }

    result = run_agent3(sample_agent1_summary, sample_agent2_findings)
    print(result)
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


SYSTEM_PROMPT_COMMENT = """You are a senior engineer leaving a single short inline code review comment on a specific change.

You will be given the raw diff plus an analysis from previous steps. The raw diff is ground truth — if the analysis describes something that is not actually present in the diff, ignore that part of the analysis.

Your comment will be scored on three dimensions:
- Correctness: identifies real issues actually present in the diff, without inventing problems. Every claim must be traceable to a line visible in the diff.
- Actionability: a developer can act on it without asking follow-up questions. Concrete, located, fixable — reference exact function or variable names.
- Depth: goes beyond style nitpicks to logic, design, or architectural concerns. Substantive reasoning about code quality, not surface observation.

Write ONLY the comment text. No headers, sections, verdict line, markdown structure, or preamble. 3 to 5 sentences of plain prose.

Structure the comment to cover, in order:
1. The specific issue, referenced by function name, variable name, or exact behavior — not "this change" or "the code".
2. Why it matters — the concrete scenario where it breaks, the assumption it violates, or the downstream effect. This is what makes the comment substantive.
3. A concrete fix — what to change, not "consider changing".

Rules:
- Ground every claim in the raw diff. Do not invent symbols, functions, or behaviors that are not visible in the diff.
- Never use hedging words: "maybe", "perhaps", "possibly", "might want to", "seems like", "I think", "consider".
- Never restate the diff or describe what you're about to say.
- Reference code by name (function, variable, condition), not by location words ("this line", "here", "above").
- If the change is genuinely low-risk, state *what specifically* is safe and *why* in one crisp sentence — still concrete, still substantive.
- Tone: senior engineer typing a focused inline comment. Direct, grounded, no fluff.
"""


def _prompt_comment(agent1_summary, agent2_findings, raw_diff=""):
    overall_summary = agent1_summary.get("overall_summary", "No summary available.")
    overall_change_type = agent1_summary.get("overall_change_type", "unknown")

    findings = agent2_findings.get("findings", [])
    risk_reasoning = agent2_findings.get("risk_reasoning", "")

    changed_files = agent1_summary.get("changed_files", [])
    files_block = "\n".join(
        f"- {f['filename']} ({f['status']}): {f['summary']}"
        for f in changed_files
    ) or "- (no files)"

    if findings:
        findings_block = "\n".join(
            f"- [{f['severity']}] {f['filename']}: {f['title']} — {f['description']} Suggestion: {f['suggestion']}"
            for f in findings
        )
    else:
        findings_block = "- (no issues found)"

    diff_block = raw_diff.strip() or "(diff not provided)"

    return f"""Raw diff (ground truth):
{diff_block}

Analysis from previous steps (use only where it matches the diff):
Summary: {overall_summary}
Change type: {overall_change_type}

Files:
{files_block}

Findings:
{findings_block}

Reasoning: {risk_reasoning}

Now write the single short review comment, grounded in the raw diff above."""


def _clean_comment(raw):
    s = raw.strip()
    s = re.sub(r"^```(?:\w+)?\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    s = re.sub(r"^#{1,6}\s.*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def run_agent3_comment(agent1_summary, agent2_findings, raw_diff=""):
    user_prompt = _prompt_comment(agent1_summary, agent2_findings, raw_diff)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMMENT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.exceptions.Timeout:
            if attempt < 2:
                print(f"Agent 3 comment: timeout, retrying ({attempt + 1}/2)...")
                continue
            raise
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {r.status_code}: {r.text}")
        return _clean_comment(r.json()["choices"][0]["message"]["content"])


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

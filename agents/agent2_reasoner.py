"""
Agent 2 — Bug & Performance Reasoner (DeepSeek-V3)

Input:  - agent1_summary: dict output from run_agent1()
        - context_files: list of relevant files from rag.retrieval.get_context()
          Each item has: path (str), content (str)

Output: Structured findings JSON with bugs, performance issues, and risk scores.
        Passed directly to Agent 3 to write the human review comment.

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
MODEL   = "deepseek-ai/DeepSeek-V3"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software engineer performing a thorough code review.
You will be given:
1. A summary of what changed in a pull request (from a previous analysis step)
2. Relevant files from the codebase for context

Your job is to reason carefully about whether the changes could cause bugs or performance issues.
Be thorough — flag anything suspicious, even if it might be minor.

Focus on:
- Bug detection: logic errors, null/undefined handling, off-by-one errors, broken imports, type mismatches
- Performance issues: unnecessary re-renders, expensive operations in loops, memory leaks, blocking calls

IMPORTANT:
- Respond ONLY with valid JSON. No explanation, no markdown, no code fences.
- If you find no issues, return an empty findings array — do not invent problems.
- Be specific: reference exact filenames and describe the exact concern.
"""

def _build_user_prompt(agent1_summary: dict, relevant_files: list[dict]) -> str:
    # Format Agent 1's summary
    summary_block = json.dumps(agent1_summary, indent=2)

    # Format relevant repo files for context
    file_blocks = []
    for f in relevant_files:
        path    = f.get("path", "unknown")
        content = f.get("content", "")
        if content:
            file_blocks.append(f"FILE: {path}\n```\n{content}\n```")

    files_block = "\n\n".join(file_blocks) if file_blocks else "No additional context files available."

    return f"""Here is the pull request summary from the previous analysis step:

{summary_block}

Here are the relevant files from the codebase for context:

{files_block}

Based on the changes described and the codebase context, identify any bugs or performance issues.

Return this exact JSON structure:
{{
  "findings": [
    {{
      "filename": "path/to/file.ext",
      "type": "bug | performance",
      "severity": "low | medium | high",
      "title": "Short title for this finding.",
      "description": "Detailed explanation of the issue and why it matters.",
      "suggestion": "Concrete suggestion for how to fix or improve it."
    }}
  ],
  "overall_risk": "low | medium | high",
  "risk_reasoning": "One paragraph explaining the overall risk assessment.",
  "approved": true or false
}}

Set approved to true only if there are no high severity findings.
"""


# ── API Call ───────────────────────────────────────────────────────────────────

def _call_api(user_prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,  # Agent 2 needs more tokens than Agent 1 — findings can be verbose
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)

    if response.status_code != 200:
        raise RuntimeError(
            f"HuggingFace API error {response.status_code}: {response.text}"
        )

    return response.json()["choices"][0]["message"]["content"]


# ── Output Parsing ─────────────────────────────────────────────────────────────

def _parse_output(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{raw[:500]}")

    return json.loads(match.group())


# ── Public Interface ───────────────────────────────────────────────────────────

def run_agent2(agent1_summary: dict, context_files: list[dict]) -> dict:
    """
    Analyze a PR for bugs and performance issues using DeepSeek-V3.

    Args:
        agent1_summary: The dict returned by run_agent1(). Contains changed_files,
                        overall_summary, overall_risk, and flags.
        context_files:  Relevant files returned by rag.retrieval.get_context().
                        Each item has: path (str), content (str).

    Returns:
        Structured dict with findings, overall_risk, risk_reasoning, and approved.
        Passed directly to Agent 3 as input.
    """
    if not agent1_summary.get("changed_files"):
        return {
            "findings": [],
            "overall_risk": "low",
            "risk_reasoning": "No files were changed.",
            "approved": True,
        }

    user_prompt = _build_user_prompt(agent1_summary, context_files)
    raw_output  = _call_api(user_prompt)
    return _parse_output(raw_output)


# ── Quick local test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Simulate what Agent 1 would return for the About.jsx change
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

    # Simulate context returned by rag.retrieval.get_context()
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

    result = run_agent2(sample_agent1_summary, sample_context_files)
    print(json.dumps(result, indent=2))
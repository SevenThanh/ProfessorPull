from rag.retrieval import get_context
from agents.agent1_summarizer import run_agent1
from agents.agent2_reasoner import run_agent2
from agents.agent3_reviewer import run_agent3


def run_review(files, repo_name):
    try:
        print("Step 1: Retrieving context...")
        ctx = get_context(files, repo_name)
        print("Step 2: Running Agent 1 (summarizer)...")
        summary = run_agent1(files)
        print("Step 3: Running Agent 2 (reasoner)...")
        findings = run_agent2(summary, ctx, files)
        print("Step 4: Running Agent 3 (reviewer)...")
        md = run_agent3(summary, findings)
        print("Pipeline complete.")
        return md
    except Exception as e:
        print(f"Pipeline error: {e}")
        return f"## Professor Pull Review\n\n⚠️ Pipeline error: {e}"

# TriAgents - Multi-agents PR Reviewer

<img width="9600" height="7200" alt="TriAgents_Research-1" src="https://github.com/user-attachments/assets/6fc643d5-424a-4006-bdbd-5ea42cd541d1" />



An AI-powered GitHub App that automatically reviews pull requests using a three-agent pipeline backed by RAG (Retrieval-Augmented Generation).

When a PR is opened or updated, TriAgents fetches the diff, retrieves semantically relevant context from across the whole repository via Pinecone, runs three specialized AI agents in sequence, and posts a structured code review comment directly to the PR.

The retrieval layer is what lets the reviewer reason beyond the diff. Instead of looking only at the changed lines, TriAgents embeds the full repo and pulls in the most relevant code from *other* files — callers, related helpers, shared types — so the reasoning agent can catch issues that only show up in context (a broken caller, a violated invariant, an API change with downstream impact).

## How It Works

```
PR opened/updated
       │
       ▼
  GitHub Webhook
       │
       ▼
  RAG Retrieval  ──── Pinecone (cross-file semantic search over the repo)
       │
       ▼
  Agent 1: Diff Summarizer (Qwen3-Coder)
  → Produces structured JSON summary of what changed
       │
       ▼
  Agent 2: Bug & Performance Reasoner (DeepSeek-V3.2)
  → Identifies bugs, syntax errors, and performance issues
       │
       ▼
  Agent 3: Review Comment Writer (Llama 3.3 70B)
  → Synthesizes findings into a formatted GitHub PR comment
       │
       ▼
  GitHub PR Review posted
```

All three agents are called via the [OpenRouter](https://openrouter.ai/) API — no local GPU required. Retrieval embeds repo files with OpenAI `text-embedding-3-small` (768-dim) and stores them in a per-repo Pinecone namespace; the first PR for a repo triggers an automatic ingest, and later PRs query that namespace for context. No manual indexing step is required.

## Project Structure

```
ProfessorPull/
├── WebHook/
│   └── server.py             # FastAPI webhook server (GitHub App entrypoint)
├── agents/
│   ├── agent1_summarizer.py  # Diff summarizer — Qwen3-Coder
│   ├── agent2_reasoner.py    # Bug/perf reasoner — DeepSeek-V3.2
│   └── agent3_reviewer.py    # Review writer — Llama 3.3 70B
├── rag/
│   ├── embeddings.py         # OpenAI text-embedding-3-small (768-dim)
│   ├── ingest.py             # Chunk and upsert repo files into Pinecone
│   ├── pinecone_db.py        # Pinecone client (upsert + query)
│   ├── pipeline.py           # Convenience wrapper for the full pipeline
│   └── retrieval.py          # Query Pinecone with patch embeddings
└── eval/
    ├── dataset.py            # Load the Zenodo Comment_Generation benchmark
    ├── adapters.py           # Pipeline configs (full_pp, no_rag, baselines)
    ├── run_eval.py           # Generate reviews for each config
    ├── judge.py              # Score reviews with Claude Haiku 4.5
    └── stats.py              # Paired bootstrap confidence intervals
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# GitHub App credentials
GITHUB_APP_ID=your_app_id
GITHUB_WEBHOOK_SECRET=your_webhook_secret
GITHUB_PRIVATE_KEY_PATH=path/to/your-private-key.pem

# OpenRouter (for all three agents)
OPENROUTER_API_KEY=your_openrouter_key

# OpenAI (for RAG embeddings)
OPENAI_API_KEY=your_openai_key

# Pinecone (for RAG)
PINECONE_API_KEY=your_pinecone_api_key

# Anthropic (only needed to run the evaluation judge)
ANTHROPIC_API_KEY=your_anthropic_key

# Optional tuning
MAX_CONTEXT_FILES=300          # Max repo files to include in context
MAX_CONTEXT_FILE_BYTES=200000  # Max file size to fetch
```

### 3. Create a GitHub App

1. Go to **GitHub → Settings → Developer settings → GitHub Apps → New GitHub App**
2. Set the webhook URL to your server's `/webhook/github` endpoint
3. Set the webhook secret (must match `GITHUB_WEBHOOK_SECRET`)
4. Grant **Pull requests: Read & Write** permissions
5. Subscribe to **Pull request** events
6. Install the app on the repositories you want reviewed

### 4. Run the webhook server

```bash
cd WebHook
uvicorn server:app --host 0.0.0.0 --port 8000
```

The first time a PR arrives for a repo, TriAgents automatically ingests that repo's file tree into Pinecone — no manual ingest step is required. To re-index a repo manually, you can still run:

```bash
python -m rag.ingest path/to/repo_context_log.txt --repo your-repo-name
```

For local development, use a tunneling tool like [ngrok](https://ngrok.com/) to expose the server to GitHub.

## Agents

| Agent | Model | Role | Output |
|-------|-------|------|--------|
| Agent 1 | Qwen3-Coder | Diff Summarizer | Structured JSON: change types, risk levels, flags |
| Agent 2 | DeepSeek-V3.2 | Bug & Perf Reasoner | Structured JSON: findings, severity, suggestions |
| Agent 3 | Llama 3.3 70B | Review Writer | Formatted GitHub markdown PR comment |

Each agent can be tested independently by running its module directly:

```bash
python -m agents.agent1_summarizer
python -m agents.agent2_reasoner
python -m agents.agent3_reviewer
```

## Evaluation

TriAgents is benchmarked on 200 real PRs from the Zenodo Comment_Generation dataset. Each generated review is scored 1–5 on correctness, actionability, and depth by a reference-free Claude Haiku 4.5 judge. The harness supports per-component ablation (e.g. `full_pp` with RAG vs. `no_rag`) and reports paired bootstrap confidence intervals so improvements are tested for significance rather than read off point estimates.

```bash
python -m eval.run_eval --configs full_pp,no_rag --n 200   # generate reviews
python -m eval.judge                                       # score with Claude Haiku 4.5
python -m eval.stats                                       # paired bootstrap CIs
```

## Dependencies

- `fastapi` + `uvicorn` — webhook server
- `httpx` — async GitHub API calls
- `PyJWT[crypto]` — GitHub App JWT authentication
- `requests` — OpenRouter API calls
- `python-dotenv` — environment variable loading
- `openai` — embeddings (RAG)
- `pinecone` — vector database for RAG
- `anthropic` — Claude Haiku judge (evaluation only)

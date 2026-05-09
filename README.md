# TriAgents - iCNS AI Engineering & Science Symposium 

An AI-powered GitHub App that automatically reviews pull requests using a three-agent pipeline backed by RAG (Retrieval-Augmented Generation).

When a PR is opened or updated, ProfessorPull fetches the diff, retrieves semantically relevant context from the repository via Pinecone, runs three specialized AI agents in sequence, and posts a structured code review comment directly to the PR.

## How It Works

```
PR opened/updated
       │
       ▼
  GitHub Webhook
       │
       ▼
  RAG Retrieval  ──── Pinecone (semantic search over repo)
       │
       ▼
  Agent 1: Diff Summarizer (Qwen2.5-Coder-32B)
  → Produces structured JSON summary of what changed
       │
       ▼
  Agent 2: Bug & Performance Reasoner (DeepSeek-V3)
  → Identifies bugs, syntax errors, and performance issues
       │
       ▼
  Agent 3: Review Comment Writer (Llama 3.3-70B)
  → Synthesizes findings into a formatted GitHub PR comment
       │
       ▼
  GitHub PR Review posted
```

All three models are called via the HuggingFace Inference API — no local GPU required.

## Project Structure

```
ProfessorPull/
├── WebHook/
│   └── server.py           # FastAPI webhook server (GitHub App entrypoint)
├── agents/
│   ├── agent1_summarizer.py  # Diff summarizer — Qwen2.5-Coder-32B
│   ├── agent2_reasoner.py    # Bug/perf reasoner — DeepSeek-V3
│   └── agent3_reviewer.py    # Review writer — Llama 3.3-70B
└── rag/
    ├── embeddings.py         # Gemini embedding-001 via Google Generative AI
    ├── ingest.py             # Chunk and upsert repo files into Pinecone
    ├── pinecone_db.py        # Pinecone client (upsert + query)
    ├── pipeline.py           # Convenience wrapper for the full pipeline
    └── retrieval.py          # Query Pinecone with patch embeddings
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

# HuggingFace (for all three agents)
HF_TOKEN=your_huggingface_token

# Google Generative AI (for embeddings)
GEMINI_API_KEY=your_gemini_api_key

# Pinecone (for RAG)
PINECONE_API_KEY=your_pinecone_api_key

# Optional tuning
MAX_CONTEXT_FILES=300        # Max repo files to include in context
MAX_CONTEXT_FILE_BYTES=200000  # Max file size to fetch
```

### 3. Create a GitHub App

1. Go to **GitHub → Settings → Developer settings → GitHub Apps → New GitHub App**
2. Set the webhook URL to your server's `/webhook/github` endpoint
3. Set the webhook secret (must match `GITHUB_WEBHOOK_SECRET`)
4. Grant **Pull requests: Read & Write** permissions
5. Subscribe to **Pull request** events
6. Install the app on the repositories you want reviewed

### 4. Ingest a repository into Pinecone (RAG setup)

Before reviews can use semantic context, ingest a repository's file tree into Pinecone. This uses the `repo_context_log.txt` file saved by the webhook on first run:

```bash
python -m rag.ingest WebHook/logs/pr_1/repo_context_log.txt --repo your-repo-name
```

### 5. Run the webhook server

```bash
cd WebHook
uvicorn server:app --host 0.0.0.0 --port 8000
```

For local development, use a tunneling tool like [ngrok](https://ngrok.com/) to expose the server to GitHub.

## Agents

| Agent | Model | Role | Output |
|-------|-------|------|--------|
| Agent 1 | Qwen2.5-Coder-32B | Diff Summarizer | Structured JSON: change types, risk levels, flags |
| Agent 2 | DeepSeek-V3 | Bug & Perf Reasoner | Structured JSON: findings, severity, suggestions |
| Agent 3 | Llama 3.3-70B | Review Writer | Formatted GitHub markdown PR comment |

Each agent can be tested independently by running its module directly:

```bash
python -m agents.agent1_summarizer
python -m agents.agent2_reasoner
python -m agents.agent3_reviewer
```

## Dependencies

- `fastapi` + `uvicorn` — webhook server
- `httpx` — async GitHub API calls
- `PyJWT[crypto]` — GitHub App JWT authentication
- `requests` — HuggingFace API calls
- `python-dotenv` — environment variable loading
- `google-generativeai` — Gemini embeddings
- `pinecone` — vector database for RAG

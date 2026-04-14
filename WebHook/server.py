import os
import sys
import hmac
import hashlib
import json
import time
import base64
import asyncio
from typing import Any

# Allow imports from the ProfessorPull root (agents/, rag/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import httpx
import jwt
from fastapi import FastAPI, Request, HTTPException

from rag.retrieval import get_context
from agents.agent1_summarizer import run_agent1
from agents.agent2_reasoner import run_agent2
from agents.agent3_reviewer import run_agent3

app = FastAPI()

WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")
if not WEBHOOK_SECRET:
    raise RuntimeError("Missing env var GITHUB_WEBHOOK_SECRET")

WEBHOOK_SECRET_BYTES = WEBHOOK_SECRET.encode("utf-8")
GITHUB_API_BASE = "https://api.github.com"
MAX_CONTEXT_FILES = int(os.environ.get("MAX_CONTEXT_FILES", "300"))
MAX_CONTEXT_FILE_BYTES = int(os.environ.get("MAX_CONTEXT_FILE_BYTES", "200000"))


def verify_github_signature(raw_body: bytes, signature_256: str | None) -> None:
    if not signature_256 or not signature_256.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing/invalid X-Hub-Signature-256")

    received = signature_256.split("=", 1)[1]

    mac = hmac.new(WEBHOOK_SECRET_BYTES, msg=raw_body, digestmod=hashlib.sha256)
    expected = mac.hexdigest()

    if not hmac.compare_digest(expected, received):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing env var {name}")
    return value

def _get_private_key_pem() -> str:
    key_path = os.environ.get("GITHUB_PRIVATE_KEY_PATH")
    if key_path:
        with open(key_path, "r", encoding="utf-8") as f:
            return f.read()

    raise RuntimeError("Missing env var GITHUB_PRIVATE_KEY_PATH")


def make_github_app_jwt() -> str:
    app_id = _get_required_env("GITHUB_APP_ID")
    private_key_pem = _get_private_key_pem()
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 540, "iss": app_id}
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


async def get_installation_token(installation_id: int) -> str:
    app_jwt = make_github_app_jwt()
    url = f"{GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Token exchange failed ({resp.status_code}): {resp.text}",
            )
        data = resp.json()
        token = data.get("token")
        if not token:
            raise HTTPException(status_code=502, detail="Token exchange returned no token")
        return token


async def github_post(path: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{GITHUB_API_BASE}{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"GitHub API POST {path} failed ({resp.status_code}): {resp.text}",
            )
        return resp.json()


async def post_pr_comment(owner: str, repo: str, pr_number: int, token: str, body: str) -> dict[str, Any]:
    """Post a top-level comment on a PR (visible in the Conversation tab)."""
    return await github_post(
        f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
        token,
        {"body": body},
    )


async def post_pr_review(
    owner: str,
    repo: str,
    pr_number: int,
    token: str,
    commit_id: str,
    body: str,
    inline_comments: list[dict[str, Any]] | None = None,
    event: str = "COMMENT",
) -> dict[str, Any]:
    """
    Submit a PR review (visible in the Files Changed tab).

    inline_comments is a list of dicts with keys:
      - path: str          — file path relative to repo root
      - line: int          — line number in the NEW file (use position for diff-relative)
      - body: str          — comment text
    event: "COMMENT" | "APPROVE" | "REQUEST_CHANGES"
    """
    payload: dict[str, Any] = {
        "commit_id": commit_id,
        "body": body,
        "event": event,
    }
    if inline_comments:
        payload["comments"] = [
            {
                "path": c["path"],
                "line": c["line"],
                "body": c["body"],
                "side": c.get("side", "RIGHT"),
            }
            for c in inline_comments
        ]

    return await github_post(
        f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
        token,
        payload,
    )


async def github_get(path: str, token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{GITHUB_API_BASE}{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"GitHub API GET {path} failed ({resp.status_code}): {resp.text}",
            )
        return resp.json()


async def fetch_repo_tree(owner: str, repo: str, ref: str, token: str) -> dict[str, Any]:
    data = await github_get(
        f"/repos/{owner}/{repo}/git/trees/{ref}",
        token,
        params={"recursive": "1"},
    )
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Unexpected response format for repository tree")
    return data


async def fetch_blob(owner: str, repo: str, blob_sha: str, token: str) -> dict[str, Any]:
    data = await github_get(f"/repos/{owner}/{repo}/git/blobs/{blob_sha}", token)
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Unexpected response format for blob")
    return data


def decode_blob_content(blob_data: dict[str, Any]) -> tuple[str | None, str | None]:
    encoding = blob_data.get("encoding")
    content = blob_data.get("content")
    if encoding != "base64" or not isinstance(content, str):
        return None, "unsupported_encoding"

    try:
        decoded_bytes = base64.b64decode(content, validate=False)
    except Exception:
        return None, "invalid_base64"

    try:
        return decoded_bytes.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "binary_or_non_utf8"


async def fetch_repo_context(owner: str, repo: str, ref: str, token: str) -> dict[str, Any]:
    tree_data = await fetch_repo_tree(owner, repo, ref, token)
    tree_entries = tree_data.get("tree", [])
    if not isinstance(tree_entries, list):
        raise HTTPException(status_code=502, detail="Tree payload missing list field")

    blob_entries = [e for e in tree_entries if e.get("type") == "blob"]
    selected_blobs = blob_entries[:MAX_CONTEXT_FILES]

    async def fetch_one(entry: dict[str, Any]) -> dict[str, Any]:
        path = entry.get("path")
        sha = entry.get("sha")
        size = entry.get("size")

        if not path or not sha:
            return {"path": path, "sha": sha, "skipped": "missing_path_or_sha"}

        if isinstance(size, int) and size > MAX_CONTEXT_FILE_BYTES:
            return {"path": path, "sha": sha, "size": size, "skipped": "file_too_large"}

        blob = await fetch_blob(owner, repo, sha, token)
        text, error = decode_blob_content(blob)
        if error:
            return {"path": path, "sha": sha, "size": size, "skipped": error}

        return {
            "path": path,
            "sha": sha,
            "size": size,
            "content": text,
        }

    semaphore = asyncio.Semaphore(10)

    async def bounded_fetch(entry: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await fetch_one(entry)

    files = await asyncio.gather(*(bounded_fetch(entry) for entry in selected_blobs))
    skipped_files = sum(1 for item in files if "skipped" in item)

    return {
        "ref": ref,
        "tree_truncated": bool(tree_data.get("truncated")),
        "total_blob_entries": len(blob_entries),
        "fetched_blob_entries": len(selected_blobs),
        "skipped_files": skipped_files,
        "limits": {
            "max_context_files": MAX_CONTEXT_FILES,
            "max_context_file_bytes": MAX_CONTEXT_FILE_BYTES,
        },
        "files": files,
    }


async def fetch_pr_files(owner: str, repo: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    page = 1
    per_page = 100

    while True:
        data = await github_get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
            token,
            params={"page": page, "per_page": per_page},
        )
        if not isinstance(data, list):
            raise HTTPException(status_code=502, detail="Unexpected response format for PR files")

        files.extend(data)
        if len(data) < per_page:
            break
        page += 1

    return files


def get_next_pr_folder() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("pr_")
    ]
    numbers = []
    for name in existing:
        try:
            numbers.append(int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(numbers, default=0) + 1
    folder = os.path.join(base_dir, f"pr_{next_num}")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_retrieval_summary(delivery_id: str | None, summary: dict[str, Any], folder: str) -> str:
    out_path = os.path.join(folder, "retrieval_log.txt")
    entry = {
        "delivery_id": delivery_id,
        "retrieval": summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, indent=2))
        f.write("\n" + ("=" * 80) + "\n")
    return out_path


def save_repo_context(delivery_id: str | None, context: dict[str, Any], folder: str) -> str:
    out_path = os.path.join(folder, "repo_context_log.txt")
    entry = {
        "delivery_id": delivery_id,
        "repo_context": context,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, indent=2))
        f.write("\n" + ("=" * 80) + "\n")
    return out_path


@app.post("/webhook/github")
async def github_webhook(request: Request):
    raw_body = await request.body()

    # 1) Verify signature (security)
    verify_github_signature(raw_body, request.headers.get("X-Hub-Signature-256"))

    # 2) Identify event type
    event = request.headers.get("X-GitHub-Event")
    delivery_id = request.headers.get("X-GitHub-Delivery")  # useful for idempotency/logging

    # 3) Parse JSON payload
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 4) Handle ping (GitHub sends this when you first save the webhook)
    if event == "ping":
        return {"ok": True, "event": "ping"}

    # 5) Handle pull request events
    if event == "pull_request":
        action = payload.get("action")
        if action not in {"opened", "synchronize", "reopened"}:
            return {"ok": True, "ignored": True, "action": action}

        installation_id = payload.get("installation", {}).get("id")
        pr = payload.get("pull_request", {})
        pr_number = pr.get("number")
        repo = payload.get("repository", {})
        full_name = repo.get("full_name")
        owner = repo.get("owner", {}).get("login")
        repo_name = repo.get("name")

        if not installation_id or not pr_number or not owner or not repo_name:
            raise HTTPException(status_code=400, detail="Missing required pull request payload fields")

        installation_token = await get_installation_token(int(installation_id))
        pr_data = await github_get(f"/repos/{owner}/{repo_name}/pulls/{pr_number}", installation_token)
        pr_files = await fetch_pr_files(owner, repo_name, int(pr_number), installation_token)
        head_sha = pr_data.get("head", {}).get("sha")
        if not head_sha:
            raise HTTPException(status_code=502, detail="Missing PR head SHA")
        repo_context = await fetch_repo_context(owner, repo_name, head_sha, installation_token)

        retrieval_summary = {
            "repo": full_name,
            "pr_number": pr_number,
            "title": pr_data.get("title"),
            "state": pr_data.get("state"),
            "changed_files_count": len(pr_files),
            "head_sha": head_sha,
            "files": [
                {
                    "filename": f.get("filename"),
                    "status": f.get("status"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                    "changes": f.get("changes"),
                    "patch": f.get("patch"),
                    "raw_url": f.get("raw_url"),
                    "blob_url": f.get("blob_url"),
                }
                for f in pr_files
            ],
        }

        print(f"[{delivery_id}] PR event: {full_name} #{pr_number} (installation_id={installation_id}) action={action}")
        print(
            f"[{delivery_id}] Retrieved PR data: changed_files={len(pr_files)} "
            f"title={pr_data.get('title')!r}"
        )
        pr_folder = get_next_pr_folder()
        saved_path = save_retrieval_summary(delivery_id, retrieval_summary, pr_folder)
        context_path = save_repo_context(delivery_id, repo_context, pr_folder)
        print(f"[{delivery_id}] Saved retrieval summary to {saved_path}")
        print(
            f"[{delivery_id}] Saved repo context to {context_path} "
            f"(fetched={repo_context.get('fetched_blob_entries')}, skipped={repo_context.get('skipped_files')})"
        )

        # --- Handoff A: Matt → Johan — semantic context from Pinecone ---
        rag_context = get_context(retrieval_summary["files"], repo_name)
        print(f"[{delivery_id}] RAG returned {len(rag_context)} context chunks")

        # --- Handoff B: Johan + Matt → Jake — run the three-agent pipeline ---
        agent1_summary  = run_agent1(retrieval_summary["files"])
        agent2_findings = run_agent2(agent1_summary, rag_context)
        review_body     = run_agent3(agent1_summary, agent2_findings)

        # --- Handoff C: Jake → Matt — post the final review to the PR ---
        try:
            await post_pr_review(
                owner=owner,
                repo=repo_name,
                pr_number=int(pr_number),
                token=installation_token,
                commit_id=head_sha,
                body=review_body,
                event="COMMENT",
            )
            print(f"[{delivery_id}] Posted review comment to PR #{pr_number}")
        except HTTPException as exc:
            # Don't fail the webhook response just because the comment post failed
            print(f"[{delivery_id}] WARNING: Failed to post review comment: {exc.detail}")

        return {
            "ok": True,
            "retrieval": retrieval_summary,
            "repo_context": {
                "head_sha": repo_context.get("ref"),
                "total_blob_entries": repo_context.get("total_blob_entries"),
                "fetched_blob_entries": repo_context.get("fetched_blob_entries"),
                "skipped_files": repo_context.get("skipped_files"),
                "tree_truncated": repo_context.get("tree_truncated"),
                "limits": repo_context.get("limits"),
            },
        }

    return {"ok": True, "ignored": True, "event": event}

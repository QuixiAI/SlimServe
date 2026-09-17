#!/usr/bin/env python3
"""Review-loop helpers for a SlimServe campaign.

CodeRabbit (via gh) and Astra (via codex).

    review.py astra-preflight [--model gpt-6-astra]
    review.py astra-review --base origin/main --out FILE [--model ...]
    review.py astra-exec --prompt-file F --out FILE [--model ...]
    review.py cr-trigger --pr N
    review.py cr-wait --pr N [--since ISO] [--timeout 1800]
    review.py cr-threads --pr N [--unresolved] [--full]
    review.py cr-reply --pr N --comment-id DBID --body-file F
    review.py cr-resolve --thread-id PRRT_...
    review.py cr-status --pr N

The repo is taken from `git remote get-url origin` of the cwd unless --repo is given.
Every command prints one JSON document on its last line. Python 3.9+, stdlib only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

CR_LOGIN = "coderabbitai"


def sh(
    cmd: list[str], timeout: float | None = None, input_text: str | None = None
) -> subprocess.CompletedProcess:
    # stdin is closed unless we feed it: codex (and gh in some modes) otherwise
    # wait for EOF forever.
    if input_text is None:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, input=input_text
    )


def emit(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, indent=None))


def repo_slug(explicit: str | None) -> str:
    if explicit:
        return explicit
    out = sh(["git", "remote", "get-url", "origin"])
    url = out.stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
    if not m:
        raise SystemExit(
            f"cannot derive owner/repo from origin url {url!r}; pass --repo"
        )
    return f"{m.group(1)}/{m.group(2)}"


def gh_api(
    path: str,
    method: str = "GET",
    fields: dict[str, Any] | None = None,
    paginate: bool = False,
) -> Any:
    cmd = ["gh", "api", path, "-X", method]
    if paginate:
        cmd.append("--paginate")
    if fields:
        cmd += ["--input", "-"]
        out = sh(cmd, timeout=120, input_text=json.dumps(fields))
    else:
        out = sh(cmd, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"gh api {path} failed: {out.stderr.strip()[:400]}")
    text = out.stdout.strip()
    if not text:
        return None
    if paginate:
        # --paginate concatenates JSON arrays; merge them.
        merged: list[Any] = []
        for chunk in re.split(r"\]\s*\[", text):
            chunk = chunk if chunk.startswith("[") else "[" + chunk
            chunk = chunk if chunk.endswith("]") else chunk + "]"
            merged += json.loads(chunk)
        return merged
    return json.loads(text)


def gh_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        cmd += ["-F", f"{k}={v}"] if isinstance(v, int) else ["-f", f"{k}={v}"]
    out = sh(cmd, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"gh graphql failed: {out.stderr.strip()[:400]}")
    return json.loads(out.stdout)


def strip_noise(body: str) -> str:
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"<details>.*?</details>", "", body, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


# ---------------------------------------------------------------------------
# Astra (codex)
# ---------------------------------------------------------------------------


def astra_preflight(args: argparse.Namespace) -> None:
    try:
        out = sh(
            [
                "codex",
                "exec",
                "-s",
                "read-only",
                "--skip-git-repo-check",
                "-c",
                f'model="{args.model}"',
                "Reply with exactly: model-ok",
            ],
            timeout=200,
        )
    except FileNotFoundError:
        emit({"available": False, "detail": "codex CLI not installed"})
        return
    except subprocess.TimeoutExpired:
        emit(
            {
                "available": False,
                "detail": "codex did not answer within 200 s (network or auth prompt?)",
            }
        )
        return
    text = out.stdout + out.stderr
    ok = out.returncode == 0 and "model-ok" in text
    detail = "ok" if ok else re.sub(r"\s+", " ", text.strip())[-300:]
    if not ok and re.search(r"login|auth|token|unauthori[sz]ed", text, re.I):
        detail = (
            "not authenticated: run `codex logout && codex login` on this machine; "
            + detail
        )
    if (
        not ok
        and re.search(r"model", text, re.I)
        and re.search(r"not (found|available)|unknown|invalid", text, re.I)
    ):
        detail = (
            f"model {args.model} not available to this account (try `codex update`); "
            + detail
        )
    emit({"available": ok, "model": args.model, "detail": detail})


def astra_review(args: argparse.Namespace) -> None:
    cmd = ["codex", "review", "--base", args.base, "-c", f'model="{args.model}"']
    if args.title:
        cmd += ["--title", args.title]
    t0 = time.time()
    with open(args.out, "w") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=args.timeout,
        )
    with open(args.out) as fh:
        text = fh.read()
    findings = extract_findings(text)
    with open(args.out + ".findings.md", "w") as fh:
        fh.write(findings["markdown"])
    emit(
        {
            "rc": proc.returncode,
            "elapsed_s": int(time.time() - t0),
            "log": args.out,
            "findings_file": args.out + ".findings.md",
            "p1": findings["p1"],
            "p2": findings["p2"],
            "p3": findings["p3"],
            "total": findings["total"],
        }
    )


def astra_exec(args: argparse.Namespace) -> None:
    with open(args.prompt_file) as fh:
        prompt = fh.read()
    cmd = ["codex", "exec", "-s", "read-only", "-c", f'model="{args.model}"', "-"]
    t0 = time.time()
    with open(args.out, "w") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            input=prompt,
            timeout=args.timeout,
        )
    emit({"rc": proc.returncode, "elapsed_s": int(time.time() - t0), "log": args.out})


def extract_findings(text: str) -> dict[str, Any]:
    idx = text.rfind("Full review comments:")
    block = text[idx:] if idx >= 0 else text[-6000:]
    block = re.sub(r"\nrc=\d+\s*$", "", block)
    counts = {
        lvl: len(re.findall(rf"\[P{lvl[-1]}\]", block)) for lvl in ("p1", "p2", "p3")
    }
    return dict(counts, total=sum(counts.values()), markdown=block.strip() + "\n")


# ---------------------------------------------------------------------------
# CodeRabbit (gh)
# ---------------------------------------------------------------------------


def cr_trigger(args: argparse.Namespace) -> None:
    repo = repo_slug(args.repo)
    out = sh(
        [
            "gh",
            "pr",
            "comment",
            str(args.pr),
            "--repo",
            repo,
            "--body",
            "@coderabbitai review",
        ],
        timeout=60,
    )
    emit(
        {
            "ok": out.returncode == 0,
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "detail": (out.stdout + out.stderr).strip()[-200:],
        }
    )


def head_sha(repo: str, pr: int) -> str:
    return gh_api(f"repos/{repo}/pulls/{pr}")["head"]["sha"]


def cr_reviews(repo: str, pr: int) -> list[dict[str, Any]]:
    revs = gh_api(f"repos/{repo}/pulls/{pr}/reviews", paginate=True) or []
    return [r for r in revs if r.get("user", {}).get("login", "").startswith(CR_LOGIN)]


def cr_issue_comments(repo: str, pr: int) -> list[dict[str, Any]]:
    cs = gh_api(f"repos/{repo}/issues/{pr}/comments", paginate=True) or []
    return [c for c in cs if c.get("user", {}).get("login", "").startswith(CR_LOGIN)]


def cr_wait(args: argparse.Namespace) -> None:
    repo = repo_slug(args.repo)
    since = args.since or dt.datetime.now(dt.timezone.utc).isoformat()
    head = head_sha(repo, args.pr)
    t0 = time.time()
    while True:
        for r in cr_reviews(repo, args.pr):
            if r.get("submitted_at", "") > since and r.get("commit_id") == head:
                emit(
                    {
                        "status": "reviewed",
                        "review_id": r["id"],
                        "submitted_at": r["submitted_at"],
                        "head": head,
                        "body": strip_noise(r.get("body") or "")[:600],
                    }
                )
                return
        for c in cr_issue_comments(repo, args.pr):
            if c.get("created_at", "") > since:
                body = c.get("body", "")
                if re.search(r"rate limit", body, re.I):
                    re.search(
                        r"(\d+)\s*minutes?\s*and\s*(\d+)\s*seconds?|(\d+)\s*minutes?",
                        body,
                    )
                    emit(
                        {
                            "status": "rate_limited",
                            "detail": strip_noise(body)[:300],
                            "retry_after_s": 900,
                        }
                    )
                    sys.exit(3)
                if re.search(
                    r"skipped|base or head changed", body, re.I
                ) and not re.search(r"Review triggered|Review finished", body):
                    emit({"status": "skipped", "detail": strip_noise(body)[:300]})
                    sys.exit(4)
        if time.time() - t0 > args.timeout:
            emit({"status": "timeout", "waited_s": int(time.time() - t0), "head": head})
            sys.exit(5)
        time.sleep(args.poll)


THREADS_QUERY = """
query($o:String!,$r:String!,$n:Int!,$after:String){
  repository(owner:$o,name:$r){ pullRequest(number:$n){
    isDraft headRefOid
    reviewThreads(first:100, after:$after){
      pageInfo{hasNextPage endCursor}
      nodes{ id isResolved isOutdated path line originalLine
        comments(first:20){ nodes{ id databaseId author{login} createdAt body } }
      } } } } }
"""


def fetch_threads(repo: str, pr: int) -> dict[str, Any]:
    owner, name = repo.split("/")
    after = None
    threads: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    while True:
        data = gh_graphql(
            THREADS_QUERY, {"o": owner, "r": name, "n": pr, "after": after or ""}
        )
        prd = data["data"]["repository"]["pullRequest"]
        meta = {"isDraft": prd["isDraft"], "head": prd["headRefOid"]}
        rt = prd["reviewThreads"]
        threads += rt["nodes"]
        if not rt["pageInfo"]["hasNextPage"]:
            break
        after = rt["pageInfo"]["endCursor"]
    return {"meta": meta, "threads": threads}


def cr_threads(args: argparse.Namespace) -> None:
    repo = repo_slug(args.repo)
    data = fetch_threads(repo, args.pr)
    out = []
    for t in data["threads"]:
        cs = t["comments"]["nodes"]
        if not cs:
            continue
        first = cs[0]
        if args.unresolved and t["isResolved"]:
            continue
        if args.since and first["createdAt"] <= args.since:
            continue
        body = first["body"] if args.full else strip_noise(first["body"])
        out.append(
            {
                "thread_id": t["id"],
                "comment_id": first["databaseId"],
                "author": (first.get("author") or {}).get("login"),
                "path": t["path"],
                "line": t["line"] or t["originalLine"],
                "resolved": t["isResolved"],
                "outdated": t["isOutdated"],
                "created_at": first["createdAt"],
                "body": body[:4000] if not args.full else body,
                "replies": [
                    {
                        "author": (c.get("author") or {}).get("login"),
                        "created_at": c["createdAt"],
                        "body": strip_noise(c["body"])[:600],
                    }
                    for c in cs[1:]
                ],
            }
        )
    emit(
        {
            "pr": args.pr,
            "head": data["meta"]["head"],
            "isDraft": data["meta"]["isDraft"],
            "count": len(out),
            "threads": out,
        }
    )


def cr_reply(args: argparse.Namespace) -> None:
    repo = repo_slug(args.repo)
    with open(args.body_file) as fh:
        body = fh.read().strip()
    forbid = [e for e in os.environ.get("CAMPAIGN_PRIVATE_EMAILS", "").split(":") if e]
    for e in forbid:
        if e in body:
            raise SystemExit("refusing to post: body contains a private address")
    attrib = (
        r"co-authored-by|generated (with|by)|(assisted|written|authored|produced|"
        r"drafted|implemented) (by|with) (claude|codex|anthropic|openai|gpt|an? "
        r"(ai|llm|model|agent)\b)|with (the )?(help|assistance) of (claude|codex|"
        r"an? (ai|llm|model|agent))|ai[- ]assisted|automated assistance"
    )
    if re.search(attrib, body, re.I):
        raise SystemExit("refusing to post: body credits a tool or model as author")
    res = gh_api(
        f"repos/{repo}/pulls/{args.pr}/comments/{args.comment_id}/replies",
        "POST",
        {"body": body},
    )
    emit({"ok": True, "reply_id": res.get("id"), "url": res.get("html_url")})


def cr_resolve(args: argparse.Namespace) -> None:
    q = (
        "mutation($id:ID!){ resolveReviewThread(input:{threadId:$id})"
        "{ thread{ id isResolved } } }"
    )
    data = gh_graphql(q, {"id": args.thread_id})
    emit({"ok": True, "thread": data["data"]["resolveReviewThread"]["thread"]})


def cr_status(args: argparse.Namespace) -> None:
    repo = repo_slug(args.repo)
    if not args.pr:
        emit({"error": "no PR number"})
        sys.exit(1)
    data = fetch_threads(repo, args.pr)
    unresolved = [
        t for t in data["threads"] if not t["isResolved"] and t["comments"]["nodes"]
    ]
    revs = cr_reviews(repo, args.pr)
    head = data["meta"]["head"]
    latest = max(revs, key=lambda r: r.get("submitted_at", ""), default=None)
    emit(
        {
            "pr": args.pr,
            "isDraft": data["meta"]["isDraft"],
            "head": head,
            "unresolved": len(unresolved),
            "unresolved_paths": sorted({t["path"] for t in unresolved})[:20],
            "latest_cr_review_at": latest.get("submitted_at") if latest else None,
            "latest_cr_review_head": latest.get("commit_id") if latest else None,
            "head_reviewed": bool(latest and latest.get("commit_id") == head),
        }
    )


def build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repo", default=None, help="owner/name (default: from origin)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("astra-preflight", parents=[common])
    a.add_argument("--model", default="gpt-6-astra")
    a.set_defaults(func=astra_preflight)
    a = sub.add_parser("astra-review", parents=[common])
    a.add_argument("--base", default="origin/main")
    a.add_argument("--out", required=True)
    a.add_argument("--model", default="gpt-6-astra")
    a.add_argument("--title", default=None)
    a.add_argument("--timeout", type=int, default=3600)
    a.set_defaults(func=astra_review)
    a = sub.add_parser("astra-exec", parents=[common])
    a.add_argument("--prompt-file", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--model", default="gpt-6-astra")
    a.add_argument("--timeout", type=int, default=3600)
    a.set_defaults(func=astra_exec)

    c = sub.add_parser("cr-trigger", parents=[common])
    c.add_argument("--pr", type=int, required=True)
    c.set_defaults(func=cr_trigger)
    c = sub.add_parser("cr-wait", parents=[common])
    c.add_argument("--pr", type=int, required=True)
    c.add_argument("--since", default=None)
    c.add_argument("--timeout", type=int, default=1800)
    c.add_argument("--poll", type=int, default=30)
    c.set_defaults(func=cr_wait)
    c = sub.add_parser("cr-threads", parents=[common])
    c.add_argument("--pr", type=int, required=True)
    c.add_argument("--unresolved", action="store_true")
    c.add_argument("--since", default=None)
    c.add_argument("--full", action="store_true")
    c.set_defaults(func=cr_threads)
    c = sub.add_parser("cr-reply", parents=[common])
    c.add_argument("--pr", type=int, required=True)
    c.add_argument("--comment-id", type=int, required=True)
    c.add_argument("--body-file", required=True)
    c.set_defaults(func=cr_reply)
    c = sub.add_parser("cr-resolve", parents=[common])
    c.add_argument("--thread-id", required=True)
    c.set_defaults(func=cr_resolve)
    c = sub.add_parser("cr-status", parents=[common])
    c.add_argument("--pr", type=int, required=True)
    c.set_defaults(func=cr_status)
    return p


if __name__ == "__main__":
    ns = build().parse_args()
    ns.func(ns)

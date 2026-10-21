"""Shared helpers: load the question set for one system, ordered so each follow-up runs in its parent's thread."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.getenv("SEGMENTATION_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(HERE, "results")


def load_items(system: str, bank: str = "questions.json") -> list[dict]:
    items = json.load(open(os.path.join(HERE, bank), encoding="utf-8"))["items"]
    return [it for it in items if it["sets"] in ("both", system)]


def threads(items: list[dict]) -> list[list[dict]]:
    """[[root, follow-up, ...], ...] in question-bank order; a thread is keyed by its root's id."""
    by_parent = {it["parent"]: it for it in items if it["parent"]}
    out = []
    for it in items:
        if it["parent"]:
            continue
        chain = [it]
        while chain[-1]["id"] in by_parent:
            chain.append(by_parent[chain[-1]["id"]])
        out.append(chain)
    assert sum(map(len, out)) == len(items)
    return out


def done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    return {json.loads(line)["id"] for line in open(path, encoding="utf-8") if line.strip()}


def repo_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""

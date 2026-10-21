"""Ask Cortex Analyst its 250 questions against the segmentation semantic model, and run the SQL it returns.

    <repo>\\.venv\\Scripts\\python run_analyst.py           # resumes: skips threads already answered

Each answer is rendered the way the Cortex Analyst Streamlit quickstart shows it: the
interpretation text, any suggestions, and the result table. The raw rows are kept for grading.
About 0.067 credits per question (one Analyst message each), plus a little warehouse time.
Writes results/analyst_answers.jsonl.
"""
import argparse
import json
import math
import os
import sys
import time

import requests

from common_eval import REPO, RESULTS, done_ids, load_items, repo_commit, threads

sys.path.insert(0, REPO)
from common import get_session  # noqa: E402

STAGE = "@CUSTOMER_SEGMENTATION.ANALYST.SEMANTIC_MODELS"
YAML = os.getenv("ANALYST_YAML", "segmentation_r7.yaml")
MAX_ROWS = 200


class Analyst:
    def __init__(self):
        self.session = get_session("ANALYST")
        self.conn = self.session._conn._conn
        self.url = f"https://{self.conn.host}/api/v2/cortex/analyst/message"

    def ask(self, messages):
        body = {"messages": messages, "semantic_model_file": f"{STAGE}/{YAML}", "stream": False}
        for attempt in range(4):
            r = requests.post(self.url, json=body, timeout=180, headers={
                "Authorization": f'Snowflake Token="{self.conn.rest.token}"',
                "Content-Type": "application/json", "Accept": "application/json"})
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(5 * (attempt + 1))
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def run_sql(self, sql):
        sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--")).strip().rstrip(";")
        return self.session.sql(sql).to_pandas().head(MAX_ROWS)  # run as written: wrapping can drop ORDER BY


def cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v if isinstance(v, (int, float, str, bool)) else str(v)


def md_table(df):
    fmt = lambda v: "" if v is None else (f"{v:,.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))  # noqa: E731
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(fmt(cell(v)) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def render(content, df=None, err=None):
    parts = []
    for c in content:
        if c["type"] == "text":
            parts.append(c["text"])
        elif c["type"] == "suggestions":
            parts.append("Suggestions: " + " | ".join(c.get("suggestions", [])))
    if err:
        parts.append(f"SQL error: {err}")
    elif df is not None:
        parts.append(md_table(df) if len(df) else "(no rows)")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analyst_answers.jsonl")
    ap.add_argument("--limit", type=int, help="stop after this many new threads (smoke test)")
    ap.add_argument("--bank", default="questions.json", help="question bank (questions_v2.json is the held-out set)")
    ap.add_argument("--ids", help="comma-separated question IDs: only the threads holding them (a targeted check)")
    args = ap.parse_args()
    wanted = set(args.ids.split(",")) if args.ids else None
    started = 0
    path = os.path.join(RESULTS, args.out)
    done = done_ids(path)
    an = Analyst()
    os.makedirs(os.path.join(os.path.dirname(RESULTS), "semantic_model"), exist_ok=True)
    an.session.file.get(f"{STAGE}/{YAML}", os.path.join(os.path.dirname(RESULTS), "semantic_model"))
    print("semantic model:", YAML, "| already done:", len(done), flush=True)
    with open(path, "a", encoding="utf-8") as f:
        for chain in threads(load_items("analyst", args.bank)):
            if all(it["id"] in done for it in chain):
                continue
            if wanted and not wanted & {it["id"] for it in chain}:
                continue
            if args.limit is not None and started >= args.limit:
                break
            started += 1
            history = []
            for it in chain:
                msgs = history + [{"role": "user", "content": [{"type": "text", "text": it["q"]}]}]
                rec = {"id": it["id"], "type": it["type"], "q": it["q"], "parent": it["parent"], "semantic_model": YAML}
                t0 = time.time()
                try:
                    r = an.ask(msgs)
                    content = r["message"]["content"]
                    sql = next((c["statement"] for c in content if c["type"] == "sql"), None)
                    df = err = None
                    if sql:
                        try:
                            df = an.run_sql(sql)
                        except Exception as e:
                            err = repr(e)[:300]
                    meta = r.get("response_metadata", {})
                    rec.update(sql=sql, answer=render(content, df, err), sql_error=err,
                               text=" ".join(c["text"] for c in content if c["type"] == "text"),
                               suggestions=[s for c in content if c["type"] == "suggestions" for s in c.get("suggestions", [])],
                               columns=list(df.columns) if df is not None else None,
                               rows=[{k: cell(v) for k, v in row.items()} for row in df.head(50).to_dict("records")]
                               if df is not None else None,
                               category=meta.get("question_category"), models=meta.get("model_names"),
                               request_id=r.get("request_id"))
                    history = msgs + [{"role": "analyst", "content": [
                        {k: v for k, v in c.items() if k in ("type", "text", "statement", "suggestions")} for c in content]}]
                except Exception as e:
                    rec["error"] = repr(e)[:500]
                    history = msgs + [{"role": "analyst", "content": [{"type": "text", "text": "(error)"}]}]
                rec["secs"] = round(time.time() - t0, 1)
                if it["id"] not in done:  # a re-asked root keeps its first answer
                    f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
                    f.flush()
                print(f"{it['id']:5} {rec['secs']:5.1f}s {str(rec.get('category')):16} "
                      f"{'SQL' if rec.get('sql') else 'no-sql'} {('ERR ' + rec['error'][:80]) if 'error' in rec else ''}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

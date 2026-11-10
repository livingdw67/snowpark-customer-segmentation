"""Ask the Genie space the 175 held-out data questions, the ones the Assistant and Cortex Analyst answered.

    python setup_genie.py                  # once: frozen tables and the space
    python run_genie.py --limit 3          # smoke test: the first 3 new threads
    python run_genie.py                    # resumes: skips threads already answered

Writes results/v2_genie_answers.jsonl in the format grade.py reads (system name v2_genie). A follow-up is sent in
its parent's conversation. Free Edition has one daily compute quota, shared with the public demos, so the run stops
at the first sign of a limit and threads are spaced out; use --limit to spread it over days. A question that got no
answer at all (the warehouse didn't start in time, or the connection failed) isn't saved as Genie's answer: it goes
to results/v2_genie_transport_errors.jsonl and is asked again on the next run.
Needs databricks-sdk.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from datetime import timedelta

from databricks.sdk import WorkspaceClient

from common_eval import RESULTS, done_ids, load_items, threads

BANK = "questions_v2.json"
OUT = "v2_genie_answers.jsonl"
NO_ANSWER_LOG = "v2_genie_transport_errors.jsonl"
SPACE_TITLE = "Segment Analyst rematch (frozen eval data)"
TIMEOUT = timedelta(seconds=180)
MAX_ROWS = 50
LIMIT_SIGNS = re.compile(r"quota|rate.?limit|limit exceeded|too many requests|\b429\b|free daily|hit your", re.I)
NUMERIC = {"BYTE", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE", "DECIMAL"}


def no_answer(error: Exception) -> bool:
    """Nothing came back to grade: a timeout, a dropped connection, or a warehouse that never started."""
    return isinstance(error, OSError) or "PENDING_WAREHOUSE" in repr(error)


def value(raw, type_name: str):
    if raw is None:
        return None
    if type_name in NUMERIC:
        number = float(raw)
        return int(number) if number.is_integer() and type_name not in ("FLOAT", "DOUBLE", "DECIMAL") else number
    return raw


def md_table(columns: list[str], rows: list[dict]) -> str:
    def fmt(v):
        if v is None:
            return ""
        return f"{v:,.4f}".rstrip("0").rstrip(".") if isinstance(v, float) and not math.isnan(v) else str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def query_result(w: WorkspaceClient, space_id: str, message, attachment_id: str) -> tuple[list[str], list[dict]]:
    response = w.genie.get_message_attachment_query_result(space_id, message.conversation_id, message.message_id,
                                                           attachment_id).statement_response
    schema = response.manifest.schema.columns
    names = [c.name for c in schema]
    kinds = [getattr(c.type_name, "value", str(c.type_name)).upper() for c in schema]
    data = (response.result.data_array if response.result else None) or []
    return names, [{n: value(v, k) for n, v, k in zip(names, row, kinds)} for row in data[:MAX_ROWS]]


def record(w: WorkspaceClient, space_id: str, message) -> dict:
    """Genie's reply, rendered as Analyst's replies were: its words, then the result table."""
    texts, sql, columns, rows, error, suggestions, parts = [], None, None, None, None, [], []
    for a in message.attachments or []:
        if a.text and a.text.content:
            texts.append(a.text.content)
            parts.append(a.text.content)
        if a.suggested_questions and a.suggested_questions.questions:
            suggestions += a.suggested_questions.questions
        if a.query and a.query.query:
            sql = a.query.query
            if a.query.description:
                texts.append(a.query.description)
                parts.append(a.query.description)
            try:
                columns, rows = query_result(w, space_id, message, a.attachment_id)
                parts.append(md_table(columns, rows) if rows else "(no rows)")
            except Exception as e:  # noqa: BLE001  the answer is still graded, as a failed query
                error = repr(e)[:300]
                parts.append(f"SQL error: {error}")
    if suggestions:
        parts.append("Suggestions: " + " | ".join(suggestions))
    if message.error:
        error = error or str(message.error)[:300]
    return dict(sql=sql, answer="\n\n".join(parts), sql_error=error, text=" ".join(texts), suggestions=suggestions,
                columns=columns, rows=rows, status=getattr(message.status, "value", str(message.status)),
                request_id=message.message_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="free")
    parser.add_argument("--limit", type=int, help="stop after this many new threads")
    parser.add_argument("--pause", type=float, default=15, help="seconds between questions")
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)
    space_id = next(s.space_id for s in w.genie.list_spaces().spaces if s.title == SPACE_TITLE)
    path = os.path.join(RESULTS, OUT)
    done = done_ids(path)
    items = load_items("genie", BANK)
    print(f"space {space_id} | {len(items)} questions | already done: {len(done)}", flush=True)
    started = failures = 0
    with open(path, "a", encoding="utf-8") as f:
        for chain in threads(items):
            if all(it["id"] in done for it in chain):
                continue
            if args.limit is not None and started >= args.limit:
                break
            started += 1
            conversation = None
            for it in chain:
                rec = {"id": it["id"], "type": it["type"], "q": it["q"], "parent": it["parent"], "space": space_id}
                t0 = time.time()
                try:
                    if conversation is None:
                        message = w.genie.start_conversation_and_wait(space_id, it["q"], timeout=TIMEOUT)
                        conversation = message.conversation_id
                    else:
                        message = w.genie.create_message_and_wait(space_id, conversation, it["q"], timeout=TIMEOUT)
                    rec.update(record(w, space_id, message))
                    failures = 0
                except Exception as e:  # noqa: BLE001
                    rec["error"] = repr(e)[:500]
                    rec["no_answer"] = no_answer(e)
                    failures += 1
                rec["secs"] = round(time.time() - t0, 1)
                print(f"{it['id']:6} {rec['secs']:6.1f}s {'SQL' if rec.get('sql') else 'no-sql':6} "
                      f"{str(rec.get('status')):12} {('ERR ' + rec['error'][:100]) if 'error' in rec else ''}", flush=True)
                if "error" in rec and (LIMIT_SIGNS.search(rec["error"]) or failures >= 3):
                    sys.exit(f"stopping: {rec['error'][:300]}\nThat question wasn't saved, so a rerun will ask it again.")
                if rec.get("no_answer"):  # logged, not saved, so the next run asks it again
                    with open(os.path.join(RESULTS, NO_ANSWER_LOG), "a", encoding="utf-8") as log:
                        log.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
                elif it["id"] not in done:  # a re-asked root keeps its first answer
                    f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
                    f.flush()
                time.sleep(args.pause)
    print("done")


if __name__ == "__main__":
    main()

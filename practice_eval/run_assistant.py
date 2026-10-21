"""Ask the Segment Assistant its 250 questions in its default offline setup (DuckDB + Ollama qwen2.5:7b).

    <repo>\\.venv\\Scripts\\python run_assistant.py            # resumes: skips threads already answered
    OLLAMA_MODEL=qwen2.5:14b ... run_assistant.py --out x.jsonl  # another model
    ... run_assistant.py --items extra --out assistant_extra_answers.jsonl  # Analyst-only data questions

Writes results/assistant_answers.jsonl, one line per question.
"""
import argparse
import json
import os
import sys
import time

from common_eval import REPO, RESULTS, done_ids, load_items, repo_commit, threads

sys.path.insert(0, REPO)
os.environ.setdefault("ASSISTANT_MODE", "local")
os.environ.setdefault("ASSISTANT_LLM", "ollama")

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from assistant.data import open_data  # noqa: E402
from assistant.graph import build_graph, turn_input  # noqa: E402
from assistant.llm import describe, make_llm, warm_up  # noqa: E402


class Usage(BaseCallbackHandler):
    def __init__(self):
        self.calls = self.inp = self.out = 0

    def on_llm_end(self, response, **kw):
        for gens in response.generations:
            for g in gens:
                self.calls += 1
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.inp += um.get("input_tokens", 0) or 0
                self.out += um.get("output_tokens", 0) or 0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # answers can contain emoji
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assistant_answers.jsonl")
    ap.add_argument("--items", choices=["assistant", "extra"], default="assistant",
                    help="assistant: its own 250; extra: the data questions only Analyst's set has, for more paired data")
    ap.add_argument("--limit", type=int, help="stop after this many new threads (smoke test)")
    ap.add_argument("--bank", default="questions.json", help="question bank (questions_v2.json is the held-out set)")
    args = ap.parse_args()
    started = 0
    path = os.path.join(RESULTS, args.out)
    done = done_ids(path)
    llm = make_llm()
    warm_up(llm)
    graph = build_graph(llm, open_data())
    model = describe(llm)
    commit = repo_commit()
    print("model:", model, "| already done:", len(done), flush=True)
    with open(path, "a", encoding="utf-8") as f:
        items = load_items("assistant", args.bank) if args.items == "assistant" else [
            it for it in load_items("analyst", args.bank)
            if it["sets"] == "analyst" and it["type"] not in ("docs", "writing", "off_topic")]
        for chain in threads(items):
            if all(it["id"] in done for it in chain):
                continue
            if args.limit is not None and started >= args.limit:
                break
            started += 1
            # A partly answered thread is re-asked from its root, so the follow-up sees real context.
            tid = f"practice-{chain[0]['id']}-{int(time.time())}"
            for it in chain:
                u, t0 = Usage(), time.time()
                rec = {"id": it["id"], "type": it["type"], "q": it["q"], "parent": it["parent"], "model": model,
                       "commit": commit}
                try:
                    out = graph.invoke(turn_input(it["q"]), {"configurable": {"thread_id": tid}, "callbacks": [u]})
                    rec.update(answer=out["messages"][-1].content, intent=out.get("intent"), declined=out.get("declined"),
                               plan=out.get("plan"), check=out.get("check"),
                               rows=(out.get("rows") or [])[:30] if isinstance(out.get("rows"), list) else None,
                               adjust=[s["detail"] for s in out.get("trace", [])
                                       if s["title"].startswith(("Adjusted", "Rewrite", "Reused", "Ignored"))])
                except Exception as e:  # keep going; an error is graded as a failure
                    rec["error"] = repr(e)[:500]
                rec.update(secs=round(time.time() - t0, 1), llm_calls=u.calls, in_tok=u.inp, out_tok=u.out)
                if it["id"] not in done:
                    f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
                    f.flush()
                print(f"{it['id']:5} {rec['secs']:5.1f}s {rec.get('intent') or 'ERR':13} "
                      f"{(rec.get('answer') or rec.get('error', ''))[:90]!r}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

"""Build the power-run bank (questions_v3.json): n new data questions from questions_v3.py, checked on
both data copies and frozen with a hash before any system answers them.

    <repo>\.venv\Scripts\python build_questions_v3.py <n>
    <repo>\.venv\Scripts\python build_questions_v3.py <n> --extend   # questions_v3b.json: n more, IDs Y...

The extension uses its own seed, never repeats a v3 question, and tries every template v3 didn't use
before reusing one.

A template whose truth differs between the CSVs (the Assistant's copy) and the live ANALYST views
(Analyst's copy) in which segments it names is left out and the bank is rebuilt, so no question
depends on rounding.
"""
import datetime
import hashlib
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.getenv("SEGMENTATION_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import questions as v1  # noqa: E402
import questions_v3 as v3  # noqa: E402
from common import get_session  # noqa: E402


def disagree(a, b):
    if a.get("kind") == "answer":
        return sorted(a["codes"]) != sorted(b["codes"]) or len(a["values"]) != len(b["values"])
    if a.get("kind") == "ambiguous":
        return [sorted(r["codes"]) for r in a["readings"]] != [sorted(r["codes"]) for r in b["readings"]]
    return False


V3_LEFT_OUT = {"('eval', 'outcome_guided_grid_size', 'PERSISTENCE', None)", "('eval', 'two_axis_value', 'RULE_FIDELITY', None)"}


def main(n, extend=False):
    csv_d = v1.load_csv(REPO)
    s = get_session()
    live_d = v1.load_live(s)
    s.close()
    seen = [it["q"] for it in v1.ITEMS] + [it["q"] for it in json.load(open(os.path.join(HERE, "questions_v2.json"),
                                                                                encoding="utf-8"))["items"]]
    used = frozenset()
    if extend:  # rebuild v3 exactly, to know which templates it used; never repeat one of its questions
        frozen = json.load(open(os.path.join(HERE, "questions_v3.json"), encoding="utf-8"))["items"]
        again = v3.build(len(frozen), csv_d, seen, frozenset(V3_LEFT_OUT))[0]
        assert [it["q"] for it in again] == [it["q"] for it in frozen], "v3 rebuild differs from the frozen bank"
        used = frozenset(it["key"] for it in again)
        seen = seen + [it["q"] for it in frozen]
    name, seed, prefix = ("questions_v3b", v3.SEED + 1, "Y") if extend else ("questions_v3", v3.SEED, "X")
    exclude = set()
    while True:
        items, pools = v3.build(n, csv_d, seen, frozenset(exclude), seed, prefix, used)
        csv_t, live_t = v3.evaluate(items, csv_d), v3.evaluate(items, live_d)
        bad = {it["key"] for it in items if disagree(csv_t[it["id"]], live_t[it["id"]])}
        nan = {it["key"] for it in items if any(x != x for x in csv_t[it["id"]].get("values", []))}
        if not bad | nan:
            break
        print("left out (copies disagree or NaN):", sorted(bad | nan))
        exclude |= {k for k in bad | nan if not k.startswith("fu:")}
        if any(k.startswith("fu:") for k in bad | nan):
            exclude |= {k[3:] for k in bad | nan if k.startswith("fu:")}
    print("pools", pools)
    print("types", dict(Counter(it["type"] for it in items)))
    print("styles", dict(Counter(it["style"] for it in items)))
    out = []
    for it in items:
        rec = {k: v for k, v in it.items() if k not in ("truth", "key")}
        rec["truth_assistant"], rec["truth_analyst"] = csv_t[it["id"]], live_t[it["id"]]
        out.append(rec)
    body = json.dumps({"note": f"power-run bank {name[10:]}: {n} generated data questions, frozen before any "
                               "system answered them", "items": out}, indent=1, ensure_ascii=False, default=float)
    with open(os.path.join(HERE, f"{name}.json"), "w", encoding="utf-8") as f:
        f.write(body)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with open(os.path.join(HERE, f"{name}.frozen.txt"), "w", encoding="utf-8") as f:
        f.write(f"sha256 {digest}\nfrozen {datetime.datetime.now(datetime.timezone.utc).isoformat()}\nitems {len(out)}\n"
                f"seed {seed}\nbefore any system answered them\n")
    print(f"wrote {name}.json", len(out), "items; sha256", digest[:16])


if __name__ == "__main__":
    main(int(sys.argv[1]), "--extend" in sys.argv)

"""Evaluate every truth on the committed CSVs and on the live ANALYST views, check the set, write questions.json.

    <repo>\\.venv\\Scripts\\python build_questions.py
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.getenv("SEGMENTATION_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from composition import COUNTS, STYLE_TARGET  # noqa: E402
from questions import ITEMS, evaluate, in_set, load_csv, load_live  # noqa: E402
from common import get_session  # noqa: E402


def main():
    ids = [it["id"] for it in ITEMS]
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1]
    by_id = {it["id"]: it for it in ITEMS}

    for system, want in COUNTS.items():
        got = Counter(it["type"] for it in ITEMS if in_set(it, system))
        assert dict(got) == want, (system, dict(got), want)
        for it in ITEMS:
            if in_set(it, system) and it["parent"]:
                p = by_id[it["parent"]]
                assert in_set(p, system), (system, it["id"], "parent not in set")
                assert ids.index(p["id"]) < ids.index(it["id"]) or p["type"] != "followup"
        n = sum(want.values())
        styles = Counter(it["style"] for it in ITEMS if in_set(it, system))
        print(system, n, "tests;", {s: f"{styles[s] / n:.0%} (target {t:.0%})" for s, t in STYLE_TARGET.items()})
    parents = Counter(it["parent"] for it in ITEMS if it["parent"])
    assert max(parents.values()) == 1, "a question has two follow-ups"

    csv_truth = evaluate(load_csv(REPO))
    session = get_session()
    live_truth = evaluate(load_live(session))
    session.close()

    diffs = []
    for i in ids:
        a, b = csv_truth[i], live_truth[i]
        if a.get("codes") != b.get("codes") or a.get("words") != b.get("words"):
            diffs.append(f"{i}: codes/words differ  csv={a.get('codes')} {a.get('words')}  live={b.get('codes')} {b.get('words')}")
        for x, y in zip(a.get("values", []), b.get("values", [])):
            if abs(x - y) > max(0.01 * abs(x), 1e-9):
                diffs.append(f"{i}: value differs >1%  csv={x}  live={y}")
    print(f"\n{len(diffs)} items where CSV and live truths differ materially:")
    print("\n".join(diffs) or "  none")

    out = []
    for it in ITEMS:
        rec = {k: v for k, v in it.items() if k != "truth"}
        rec["truth_assistant"], rec["truth_analyst"] = csv_truth[it["id"]], live_truth[it["id"]]
        out.append(rec)
    with open(os.path.join(HERE, "questions.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "truth_assistant from results/*.csv, truth_analyst from CUSTOMER_SEGMENTATION.ANALYST views",
                   "items": out}, f, indent=1, ensure_ascii=False, default=float)
    print(f"\nwrote questions.json: {len(out)} unique questions")


if __name__ == "__main__":
    main()

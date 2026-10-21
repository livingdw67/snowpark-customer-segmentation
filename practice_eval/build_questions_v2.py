"""Build the held-out set (questions_v2.json), check it, and freeze it with a hash before any fix is made.

    <repo>\\.venv\\Scripts\\python build_questions_v2.py
"""
import datetime
import difflib
import hashlib
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.getenv("SEGMENTATION_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from composition import COUNTS, STYLE_TARGET  # noqa: E402
import questions as v1  # noqa: E402
from questions_v2 import ITEMS, evaluate  # noqa: E402
from common import get_session  # noqa: E402


def main():
    ids = [it["id"] for it in ITEMS]
    assert len(ids) == len(set(ids))
    got = Counter(it["type"] for it in ITEMS)
    assert dict(got) == COUNTS["assistant"], (dict(got), COUNTS["assistant"])
    by_id = {it["id"]: it for it in ITEMS}
    for it in ITEMS:
        if it["parent"]:
            assert by_id[it["parent"]]["type"] != "followup" and ids.index(it["parent"]) != ids.index(it["id"])
    assert max(Counter(it["parent"] for it in ITEMS if it["parent"]).values()) == 1
    styles = Counter(it["style"] for it in ITEMS)
    print({s: f"{styles[s] / len(ITEMS):.0%} (target {t:.0%})" for s, t in STYLE_TARGET.items()})
    print("analyst gets", sum(it["sets"] == "both" for it in ITEMS), "data questions")

    old = [it["q"].lower() for it in v1.ITEMS]
    close, repeats = [], [it["id"] for it in ITEMS if it["q"].lower() in old]
    assert not repeats, ("repeats of earlier questions", repeats)
    for it in ITEMS:
        best = max(old, key=lambda o: difflib.SequenceMatcher(None, it["q"].lower(), o).ratio())
        r = difflib.SequenceMatcher(None, it["q"].lower(), best).ratio()
        if r > 0.85:
            close.append((it["id"], round(r, 2), it["q"], best))
    print(f"{len(close)} questions within 0.85 similarity of an earlier one:")
    for c in close:
        print("  ", c)

    csv_truth = evaluate(v1.load_csv(REPO))
    s = get_session()
    live_truth = evaluate(v1.load_live(s))
    s.close()
    out = []
    for it in ITEMS:
        rec = {k: v for k, v in it.items() if k != "truth"}
        rec["truth_assistant"], rec["truth_analyst"] = csv_truth[it["id"]], live_truth[it["id"]]
        out.append(rec)
    body = json.dumps({"note": "held-out set v2: written and frozen before the semantic-layer fixes",
                       "items": out}, indent=1, ensure_ascii=False, default=float)
    path = os.path.join(HERE, "questions_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with open(os.path.join(HERE, "questions_v2.frozen.txt"), "w", encoding="utf-8") as f:
        f.write(f"sha256 {digest}\nfrozen {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
                f"items {len(out)}\nbefore any change to assistant/ on branch practice-eval-fixes\n")
    print("wrote questions_v2.json", len(out), "items; sha256", digest[:16])


if __name__ == "__main__":
    main()

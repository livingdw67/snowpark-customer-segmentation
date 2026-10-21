"""Blind grading of the held-out set: old code vs fixed code, without knowing which wrote which.

    python blind_v2.py sheet                 # results/v2_blind_sheet.jsonl + v2_blind_key.json (letters shuffled, seed 2026)
    python blind_v2.py show <a> <b> [width]  # questions a..b-1 with lettered answers, no versions shown
    python blind_v2.py unblind               # results/v2_blind_review.txt ("ID|letter|verdict|note") -> grades_v2_before.csv, grades_v2_after.csv

Identical answers from both versions get one letter, so they are graded once and identically.
"""
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ARMS = {"before": "v2_assistant_before.jsonl", "after": "v2_assistant_after.jsonl"}
VERDICTS = {"correct", "wrong number", "wrong claim", "unhelpful", "declined-but-answerable",
            "answered-instead-of-declining", "misrouted", "unsafe", "error"}


def load(name):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RES, name), encoding="utf-8") if l.strip()}


def items():
    return {it["id"]: it for it in json.load(open(os.path.join(HERE, "questions_v2.json"), encoding="utf-8"))["items"]}


def sheet():
    rows = {arm: load(f) for arm, f in ARMS.items()}
    qs = items()
    rng = random.Random(2026)
    out, key = [], {}
    for i in qs:
        texts = {}
        for arm in ARMS:
            a = rows[arm][i]
            texts.setdefault(a.get("answer") or ("ERROR: " + a.get("error", "")), []).append(arm)
        order = list(texts)
        rng.shuffle(order)
        letters = {}
        for letter, text in zip("AB", order):
            letters[letter] = text
            key[f"{i}|{letter}"] = texts[text]
        out.append({"id": i, "answers": letters})
    with open(os.path.join(RES, "v2_blind_sheet.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    json.dump(key, open(os.path.join(RES, "v2_blind_key.json"), "w"))
    print(len(out), "questions;", sum(len(r["answers"]) for r in out), "distinct answers;",
          sum(len(r["answers"]) == 2 for r in out), "questions where the two versions differ")


def show(a, b, width=900):
    sys.stdout.reconfigure(encoding="utf-8")
    qs = items()
    sh = [json.loads(l) for l in open(os.path.join(RES, "v2_blind_sheet.jsonl"), encoding="utf-8")]
    by = {r["id"]: r for r in sh}
    # The machine first pass per letter, as a hint: it is the same check for both versions, so it gives
    # nothing away about which wrote which.
    key = json.load(open(os.path.join(RES, "v2_blind_key.json")))
    autos = {arm: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RES, f"auto_v2_{arm}.jsonl"))}
             for arm in ARMS}
    for r in sh[a:b]:
        it = qs[r["id"]]
        print(f"=== {r['id']} [{it['type']}/{it['style']}]")
        if it["parent"]:
            p = by[it["parent"]]["answers"]
            first = next(iter(p.values()))
            print(f"  EARLIER Q: {qs[it['parent']]['q']}")
            if len(p) == 1:
                print(f"  EARLIER A: {first[:260]!r}")
            else:
                for L, t in p.items():
                    print(f"  EARLIER {L}: {t[:200]!r}")
        print(f"Q: {it['q']}\nTRUTH: {it['truth_assistant']['truth']}")
        for L, t in r["answers"].items():
            au = autos[key[f"{r['id']}|{L}"][0]][r["id"]]
            print(f"--- {L} (auto {au['auto']}: {au['detail'][:80]}): "
                  + (t if len(t) <= width else t[:width] + f" ...[+{len(t) - width}]"))
        print()


def unblind():
    key = json.load(open(os.path.join(RES, "v2_blind_key.json")))
    qs = items()
    grades = {}
    for line in open(os.path.join(RES, "v2_blind_review.txt"), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            i, L, v, note = (line.rstrip("\n").split("|", 3) + [""])[:4]
            assert v in VERDICTS, (i, v)
            grades[(i, L)] = (v, note)
    missing = [k for k in key if tuple(k.split("|")) not in grades]
    assert not missing, f"ungraded: {missing[:10]} ({len(missing)})"
    per = {arm: {} for arm in ARMS}
    for k, arms in key.items():
        i, L = k.split("|")
        for arm in arms:
            per[arm][i] = grades[(i, L)]
    for arm, g in per.items():
        with open(os.path.join(RES, f"grades_v2_{arm}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "type", "style", "sets", "parent", "question", "verdict", "correct", "note"])
            for i in qs:
                v, note = g[i]
                it = qs[i]
                w.writerow([i, it["type"], it["style"], it["sets"], it["parent"] or "", it["q"], v, int(v == "correct"), note])
        print(arm, sum(v == "correct" for v, _ in g.values()), "/", len(g))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sheet":
        sheet()
    elif cmd == "show":
        show(int(sys.argv[2]), int(sys.argv[3]), *(int(x) for x in sys.argv[4:5]))
    else:
        unblind()

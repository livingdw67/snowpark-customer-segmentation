"""Statistics for the Segment Assistant fixes, on the held-out set frozen before the fixes were written.

    <repo>\\.venv\\Scripts\\python stats_v2.py   # needs grades_v2_before.csv and grades_v2_after.csv
                                                 # (blind_v2.py unblind) and grades_v2_analyst.csv

Writes results/dashboard/v2_*.json. Tests, and the question each answers:

- McNemar (exact, mid-p), old code vs fixed code on all 250 held-out questions: did the fixes help on
  questions they were not written against? Paired, because both versions answered the same questions.
  Checks: follow-ups grouped with the question before them (clustered McNemar, Durkalski et al. 2003),
  first questions only, data questions only, and the machine first pass instead of the blind read.
- Newcombe's interval (method 10) for the paired gain.
- The same McNemar for the fixed Assistant vs Cortex Analyst on the 175 held-out data questions, and
  for the old Assistant vs Analyst alongside it.
- Holm's adjustment across those three comparisons: three tests are three chances of a fluke.
- The largest Analyst lead the data still allow (the Newcombe bound): how far "no significant
  difference" goes towards "as good as".
- Per-type exact McNemar on the gain (descriptive: most types have few discordant questions).
  The mix is the same for both versions, so the gain splits additively by type: each type's Shapley
  (and Owen) value is just its share of the questions times its change in accuracy.
- The same comparisons on the original questions the fixes were written against (tuned.py), when graded:
  descriptive only, since fixes tuned on those questions are bound to look good there.
"""
import csv
import json
import math
import os
import re
from collections import defaultdict

from composition import REPORT_GROUP
from stats import LABELS, OUT, RES, mcnemar, newcombe_paired, r4, wilson, write

HERE = os.path.dirname(os.path.abspath(__file__))


def grades(name):
    rows = list(csv.DictReader(open(os.path.join(RES, f"grades_{name}.csv"), encoding="utf-8")))
    for r in rows:
        r["correct"] = int(r["correct"])
        r["group"] = REPORT_GROUP[r["type"]]
    return {r["id"]: r for r in rows}


def paired(x, y, ids):
    """a both right, b only x right, c only y right, d both wrong."""
    a = sum(x[i]["correct"] and y[i]["correct"] for i in ids)
    b = sum(x[i]["correct"] and not y[i]["correct"] for i in ids)
    c = sum(y[i]["correct"] and not x[i]["correct"] for i in ids)
    return a, b, c, len(ids) - a - b - c


def compare(label, x, y, ids, x_name, y_name):
    a, b, c, d = paired(x, y, ids)
    n = len(ids)
    exact, mid, _, _ = mcnemar(b, c)
    (lo, hi), phi = newcombe_paired(a, b, c, d)
    xl, xh = wilson(a + b, n)
    yl, yh = wilson(a + c, n)
    return {"comparison": label, "x": x_name, "y": y_name, "n": n, "both_right": a, "x_only": b, "y_only": c,
            "both_wrong": d, "discordant": b + c, "x_correct": a + b, "y_correct": a + c,
            "one_in": round(1 / exact) if exact > 0 else None, "x_accuracy": r4((a + b) / n), "x_lo": r4(xl), "x_hi": r4(xh),
            "y_accuracy": r4((a + c) / n), "y_lo": r4(yl), "y_hi": r4(yh), "diff": r4((c - b) / n),
            "diff_lo": r4(lo), "diff_hi": r4(hi), "phi": r4(phi), "exact_p": r4(exact), "mid_p": r4(mid)}


def holm(rows):
    order = sorted(range(len(rows)), key=lambda k: rows[k]["exact_p"])
    m, running = len(rows), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[k]["exact_p"]))
        rows[k]["holm_p"] = r4(running)


def main():
    old, new, an = grades("v2_before"), grades("v2_after"), grades("v2_analyst")
    qs = {it["id"]: it for it in json.load(open(os.path.join(HERE, "questions_v2.json"), encoding="utf-8"))["items"]}
    every = list(qs)
    data = [i for i in every if i in an]
    assert len(every) == 250 and len(data) == 175

    tests = [compare("Old vs fixed Assistant, all 250 held-out questions", old, new, every, "Old code", "Fixed code"),
             compare("Fixed Assistant vs Cortex Analyst, 175 held-out data questions", new, an, data,
                     "Fixed Assistant", "Cortex Analyst"),
             compare("Old Assistant vs Cortex Analyst, 175 held-out data questions", old, an, data,
                     "Old Assistant", "Cortex Analyst")]
    holm(tests)
    for t in tests:  # the largest lead for y the interval still allows, in points
        t["largest_y_lead"] = r4(max(t["diff_hi"], 0.0))
        t["largest_x_lead"] = r4(max(-t["diff_lo"], 0.0))
    write("v2_mcnemar", tests)

    # --- assumption checks on the main comparison (old vs fixed)
    def root(i):
        while qs[i]["parent"]:
            i = qs[i]["parent"]
        return i
    clusters = defaultdict(lambda: [0, 0])
    for i in every:
        clusters[root(i)][0] += old[i]["correct"] and not new[i]["correct"]
        clusters[root(i)][1] += new[i]["correct"] and not old[i]["correct"]
    num = sum(bk - ck for bk, ck in clusters.values()) ** 2
    den = sum((bk - ck) ** 2 for bk, ck in clusters.values())
    durk = num / den if den else 0.0
    first = [i for i in every if qs[i]["type"] != "followup"]
    autos = {}
    for arm in ("before", "after"):
        path = os.path.join(RES, f"auto_v2_{arm}.jsonl")
        if os.path.exists(path):
            autos[arm] = {json.loads(l)["id"]: json.loads(l)["auto"] for l in open(path, encoding="utf-8")}

    def row(check, unit, ids_or_n, b, c, p, method, **extra):
        return {"check": check, "unit": unit, "n": ids_or_n, "old_only": b, "fixed_only": c, "p": r4(p),
                "method": method, **extra}
    _, b, c, _ = paired(old, new, every)
    checks = [row(f"All {len(every)} questions (main result)", "question", len(every), b, c, mcnemar(b, c)[0],
                  "exact binomial"),
              row("Follow-ups grouped with the question before them", "conversation", len(clusters), b, c,
                  math.erfc(math.sqrt(durk / 2)) if durk > 0 else 1.0, "clustered McNemar (Durkalski 2003)",
                  statistic=r4(durk))]
    _, fb, fc, _ = paired(old, new, first)
    checks.append(row("First questions only (follow-ups dropped)", "question", len(first), fb, fc,
                      mcnemar(fb, fc)[0], "exact binomial"))
    _, db, dc, _ = paired(old, new, data)
    checks.append(row("Data questions only", "question", len(data), db, dc, mcnemar(db, dc)[0], "exact binomial"))
    rest = [i for i in every if i not in an]
    _, rb, rc, _ = paired(old, new, rest)
    checks.append(row("Docs, writing and off-topic only", "question", len(rest), rb, rc, mcnemar(rb, rc)[0],
                      "exact binomial"))
    if len(autos) == 2:
        mb = sum(autos["before"][i] == "pass" and autos["after"][i] != "pass" for i in data)
        mc = sum(autos["after"][i] == "pass" and autos["before"][i] != "pass" for i in data)
        checks.append(row("Graded by the machine first pass instead of the blind read", "question", len(data), mb,
                          mc, mcnemar(mb, mc)[0], "exact binomial"))
    write("v2_checks", checks)

    # --- the same checks on fixed Assistant vs Analyst, plus one for Analyst's FLOAT-equality failures:
    # its SQL compares a FLOAT column to its own MAX with "=", which returns no rows at full precision.
    # Rounding those columns in the views is a one-line fix on Analyst's side, so count them right.
    def vs_row(check, unit, n, b, c, p, method):
        return {"check": check, "unit": unit, "n": n, "assistant_only": b, "analyst_only": c, "p": r4(p),
                "method": method}
    _, b, c, _ = paired(new, an, data)
    vs = [vs_row(f"All {len(data)} data questions (main result)", "question", len(data), b, c, mcnemar(b, c)[0],
                 "exact binomial")]
    cl = defaultdict(lambda: [0, 0])
    for i in data:
        cl[root(i)][0] += new[i]["correct"] and not an[i]["correct"]
        cl[root(i)][1] += an[i]["correct"] and not new[i]["correct"]
    num = sum(bk - ck for bk, ck in cl.values()) ** 2
    den = sum((bk - ck) ** 2 for bk, ck in cl.values())
    st = num / den if den else 0.0
    vs.append(vs_row("Follow-ups grouped with the question before them", "conversation", len(cl), b, c,
                     math.erfc(math.sqrt(st / 2)) if st > 0 else 1.0, "clustered McNemar (Durkalski 2003)"))
    fd = [i for i in data if qs[i]["type"] != "followup"]
    _, fb2, fc2, _ = paired(new, an, fd)
    vs.append(vs_row("First questions only (follow-ups dropped)", "question", len(fd), fb2, fc2, mcnemar(fb2, fc2)[0],
                     "exact binomial"))
    an_auto = {json.loads(l)["id"]: json.loads(l)["auto"] for l in open(os.path.join(RES, "auto_v2_analyst.jsonl"))}
    if "after" in autos:
        mb2 = sum(autos["after"][i] == "pass" and an_auto[i] != "pass" for i in data)
        mc2 = sum(an_auto[i] == "pass" and autos["after"][i] != "pass" for i in data)
        vs.append(vs_row("Graded by the machine first pass instead of the read", "question", len(data), mb2, mc2,
                         mcnemar(mb2, mc2)[0], "exact binomial"))
    float_bug = {i for i in data if re.search(r"float", an[i]["note"], re.I)}
    an_fixed = {i: {**an[i], "correct": 1 if i in float_bug else an[i]["correct"]} for i in data}
    _, fb3, fc3, _ = paired(new, an_fixed, data)
    vs.append(vs_row(f"Analyst's {len(float_bug)} FLOAT-equality failures counted right", "question", len(data), fb3,
                     fc3, mcnemar(fb3, fc3)[0], "exact binomial"))
    vs[-1]["counted_right"] = len(float_bug)
    write("v2_vs_analyst_checks", vs)

    # --- by type: where the gain came from (additive, so each type's Shapley value is its own term)
    types = []
    for g in dict.fromkeys(REPORT_GROUP.values()):
        ids = [i for i in every if old[i]["group"] == g]
        a_, b_, c_, _ = paired(old, new, ids)
        an_ids = [i for i in ids if i in an]
        types.append({"group": g, "label": LABELS[g], "n": len(ids), "old_correct": a_ + b_, "fixed_correct": a_ + c_,
                      "old_accuracy": r4((a_ + b_) / len(ids)), "fixed_accuracy": r4((a_ + c_) / len(ids)),
                      "fixed": c_, "broke": b_, "exact_p": r4(mcnemar(b_, c_)[0]),
                      "contribution": r4((c_ - b_) / len(every)),
                      "analyst_n": len(an_ids),
                      "analyst_accuracy": r4(sum(an[i]["correct"] for i in an_ids) / len(an_ids)) if an_ids else None})
    write("v2_types", types)

    moved = []
    for i in every:
        if old[i]["correct"] != new[i]["correct"]:
            loser = old[i] if new[i]["correct"] else new[i]
            moved.append({"id": i, "type": LABELS[old[i]["group"]], "question": qs[i]["q"],
                          "change": "fixed" if new[i]["correct"] else "broke",
                          "wrong_verdict": loser["verdict"], "note": loser["note"][:160]})
    write("v2_changes", moved)

    # --- the fixed code on the original questions its fixes were written against (tuned.py): descriptive only
    tuned = []
    if all(os.path.exists(os.path.join(RES, f"grades_{s}.csv")) for s in ("tuned", "tuned_extra")):
        A, B, E = grades("assistant"), grades("analyst"), grades("assistant_extra")
        T, TE = grades("tuned"), grades("tuned_extra")
        shared = [i for i in A if i in B and A[i]["group"] != "non_data"]
        paired_ids = shared + [i for i in E if i in B]
        old_p, new_p = {**{i: A[i] for i in shared}, **E}, {**{i: T[i] for i in shared}, **TE}
        tuned = [compare("Old vs fixed Assistant, the original 250 (tuned on)", A, T, list(A), "Old code", "Fixed code"),
                 compare(f"Fixed Assistant vs Cortex Analyst, the original {len(paired_ids)} shared data questions (tuned on)",
                         new_p, B, paired_ids, "Fixed Assistant", "Cortex Analyst"),
                 compare(f"Old Assistant vs Cortex Analyst, the original {len(paired_ids)} shared data questions",
                         old_p, B, paired_ids, "Old Assistant", "Cortex Analyst")]
        write("v2_tuned", tuned)

    for t in tests + tuned:
        print(f"{t['comparison']}: {t['x']} {t['x_accuracy']:.1%} vs {t['y']} {t['y_accuracy']:.1%} | "
              f"{t['x_only']} vs {t['y_only']} discordant | exact p {t['exact_p']} (Holm {t.get('holm_p', '-')}) | "
              f"diff {t['diff']:+.1%} [{t['diff_lo']:+.1%}, {t['diff_hi']:+.1%}]")
    for ch in checks:
        print(f"  {ch['check']}: {ch['old_only']} vs {ch['fixed_only']}, p {ch['p']}")
    for ch in vs:
        print(f"  vs Analyst, {ch['check']}: {ch['assistant_only']} vs {ch['analyst_only']}, p {ch['p']}")


if __name__ == "__main__":
    main()

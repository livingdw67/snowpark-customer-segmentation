"""Data for the power page (power_app/rounds.json): each paired comparison of the Assistant and Analyst so far.

    python power_page_data.py

Per round: the 2x2 counts, accuracies, Analyst-minus-Assistant difference with Newcombe (method 10)
intervals at 95% and 90% (the 90% one is the equivalence test's), and the exact McNemar p-value. The
power run is added once grades_v3_* and grades_v3b_* exist, with its robustness checks, its result by
question type (Holm-adjusted) and its achieved power, all from stats_power.py's results/dashboard/power_*.json.
"""
import csv
import json
import math
import os

from stats import mcnemar

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WHY = {
    "Follow-ups grouped with the question before them":
        "A follow-up leans on the question before it, so the two aren't independent. Counting each conversation as "
        "one unit (clustered McNemar, Durkalski 2003) checks the p-value isn't flattered by that.",
    "First questions only (follow-ups dropped)": "Every question stands alone, with no conversation to lean on.",
    "Hand-written held-out questions only":
        "The 175 written by hand before any fix. Only 27 split decisions: too few to tell the systems apart, "
        "which is why the run needed 1,400.",
    "Generated questions only": "The 1,225 template questions on their own, in the same mix of question types.",
    "Graded by the machine first pass instead of the read":
        "Swaps the read for the automatic first-pass check, which misses errors both ways (it passed any eta-squared "
        "figure the Assistant gave). The gap shrinks, so the result rests on reading the answers.",
    "Assistant's unaudited passes counted right (the plan as written)":
        "The plan read every Analyst pass (its audit failed the 2% rule) but only a 10% audit of the Assistant's. "
        "Reading the rest found 17 more Assistant errors; counted right, as the plan alone would have, Analyst "
        "still leads.",
}


def grades(name):
    path = os.path.join(RES, f"grades_{name}.csv")
    if not os.path.exists(path):
        return None
    return {r["id"]: int(r["correct"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def wilson(k, n, z):
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def newcombe(a, b, c, d, z):
    """Interval for p2 - p1 (Analyst minus Assistant): a both right, b Assistant only, c Analyst only."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    num = a * d - b * c
    phi = (max(num - n / 2, 0) if num > 0 else num) / den if den else 0.0
    theta = p1 - p2
    delta = math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    eps = math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return -theta - eps, -theta + delta


def holm(ps):
    """Holm's step-down adjustment: keeps the chance of any false 'different' at 5% across all the types."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def round_row(label, short, assistant, analyst, ids, note):
    a = sum(assistant[i] and analyst[i] for i in ids)
    b = sum(assistant[i] and not analyst[i] for i in ids)
    c = sum(analyst[i] and not assistant[i] for i in ids)
    d = len(ids) - a - b - c
    n = len(ids)
    lo95, hi95 = newcombe(a, b, c, d, 1.959964)
    lo90, hi90 = newcombe(a, b, c, d, 1.644854)
    return dict(label=label, short=short, note=note, n=n, both_right=a, assistant_only=b, analyst_only=c,
                both_wrong=d, assistant_accuracy=(a + b) / n, analyst_accuracy=(a + c) / n, diff=(c - b) / n,
                lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90, exact_p=mcnemar(b, c)[0])


def main():
    rounds = []
    A, E, B = grades("assistant"), grades("assistant_extra"), grades("analyst")
    shared = [i for i in A if i in B and i[:2] not in ("DC", "WR", "OT")]
    old = {**{i: A[i] for i in shared}, **E}
    rounds.append(round_row("First round", "Round 1", old, B, shared + [i for i in E if i in B],
                            "Original Assistant vs Analyst (semantic model r7), the first practice sets"))
    before, after, r7, r8 = grades("v2_before"), grades("v2_after"), grades("v2_analyst"), grades("v2_analyst_r8")
    held = list(r7)
    rounds.append(round_row("Held-out, neither fixed", "Held-out A", before, r7, held,
                            "New questions written before any fix"))
    rounds.append(round_row("Held-out, Assistant fixed", "Held-out B", after, r7, held,
                            "Same questions; the Assistant's fixes in"))
    rounds.append(round_row("Held-out, both fixed", "Held-out C", after, r8, held,
                            "Same questions; Analyst's FLOAT ranking bug fixed too (r8)"))
    parts = [grades(x) for x in ("v3_assistant", "v3_analyst", "v3b_assistant", "v3b_analyst")]
    if all(parts):
        asst = {**after, **parts[0], **parts[2]}
        anl = {**r8, **parts[1], **parts[3]}
        run = round_row("Power run", "Power run", asst, anl, held + list(parts[1]) + list(parts[3]),
                        "1,400 questions: the held-out 175 plus 1,225 generated")
        dash = os.path.join(RES, "dashboard")
        checks = json.load(open(os.path.join(dash, "power_checks.json"), encoding="utf-8"))
        primary = checks[0]
        assert (primary["assistant_only"], primary["analyst_only"]) == (run["assistant_only"], run["analyst_only"])
        run.update(split_share=primary["split_share"], connor_power=primary["connor_power"],
                   exact_power=primary["exact_power"])
        run["checks"] = [dict(check=c["check"], n=c["n"], assistant_only=c["assistant_only"],
                              analyst_only=c["analyst_only"], exact_p=c["exact_p"], why=WHY[c["check"]])
                         for c in checks[1:]]
        types = json.load(open(os.path.join(dash, "power_types.json"), encoding="utf-8"))
        for t, adj in zip(types, holm([t["exact_p"] for t in types])):
            t["holm_p"] = adj
        run["types"] = [{k: t[k] for k in ("type", "n", "assistant_accuracy", "analyst_accuracy", "assistant_only",
                                           "analyst_only", "diff", "lo95", "hi95", "exact_p", "holm_p")} for t in types]
        run["kappa"] = json.load(open(os.path.join(dash, "power_kappa.json"), encoding="utf-8"))
        rounds.append(run)
    out = os.path.join(HERE, "power_app", "rounds.json")
    json.dump(rounds, open(out, "w", encoding="utf-8"), indent=1)
    for r in rounds:
        print(f"{r['short']:10} n={r['n']:5} asst {r['assistant_accuracy']:.1%} analyst {r['analyst_accuracy']:.1%} "
              f"diff {r['diff']*100:+.1f} [{r['lo95']*100:+.1f}, {r['hi95']*100:+.1f}] p={r['exact_p']:.3f} "
              f"split {r['assistant_only']}/{r['analyst_only']}")


if __name__ == "__main__":
    main()

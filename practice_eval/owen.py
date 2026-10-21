"""Scores, the paired comparison, and Owen values for the gap between the two headline scores.

    python owen.py            # reads results/grades_<system>.csv, writes results/summary.json

Why Owen values here
--------------------
Each system's headline score is its accuracy on a test set built to the mix it would see in
practice, so it is a weighted sum over question types: score = sum_g w_g * a_g. The two
systems were tested on different mixes (the assistant gets writing and documentation
questions; Analyst gets almost only data questions), so the gap between their scores mixes
two things: how often each type is asked (w) and how well it is answered (a).

The game has one union per question type and two players per union, "mix" and "accuracy",
the same structure semantic-rca uses to split credit between a segment's rate and volume:

    v(S) = sum_g  w_g(S) * (a_g(S) - c)

where w_g(S) is Analyst's weight if (g, mix) is in S and the assistant's otherwise, likewise
a_g(S), and c is a fixed reference accuracy (the mean of the two headline scores). Because both
weight vectors sum to 1, v(all) - v(none) is the score gap for any c; c only decides which
types get mix credit: shifting weight towards a type counts against a system only if that type
is answered worse than the reference.

Owen values: credit is shared first between unions (types), then within each union between its
two players. v is additive across unions, so each type's union total is its own term, and the
within-union split is the symmetric (Shapley) midpoint for a product. The exact computation
below enumerates coalitions and is checked against that closed form.
"""
import csv
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np

from composition import REPORT_GROUP

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GROUPS = ["lookup", "total_share", "ranking", "comparison", "breakdown", "filter", "model_eval", "followup",
          "ambiguous", "unanswerable", "non_data"]
A, B = "assistant", "analyst"
DATA = {"lookup", "total_share", "ranking", "comparison", "breakdown", "filter", "model_eval", "followup",
        "ambiguous", "unanswerable"}


def load(system):
    rows = list(csv.DictReader(open(os.path.join(RES, f"grades_{system}.csv"), encoding="utf-8")))
    for r in rows:
        r["group"] = REPORT_GROUP[r["type"]]
        r["correct"] = int(r["correct"])
    return rows


def stats(rows):
    n, k = defaultdict(int), defaultdict(int)
    for r in rows:
        n[r["group"]] += 1
        k[r["group"]] += r["correct"]
    total = sum(n.values())
    return {g: (n[g] / total, k[g] / n[g] if n[g] else float("nan"), n[g]) for g in GROUPS}


def v_of(S, wA, aA, wB, aB, c):
    return sum((wB[g] if (g, "mix") in S else wA[g]) * ((aB[g] if (g, "acc") in S else aA[g]) - c) for g in GROUPS)


def owen_exact(wA, aA, wB, aB, c):
    unions = [[(g, "mix"), (g, "acc")] for g in GROUPS]
    m = len(unions)
    phi = {}
    for k, U in enumerate(unions):
        others = [j for j in range(m) if j != k]
        for i in U:
            rest = [p for p in U if p != i]
            total = 0.0
            for r in range(m):
                wr = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
                for R in itertools.combinations(others, r):
                    Q = {p for j in R for p in unions[j]}
                    for t in range(len(rest) + 1):
                        wt = math.factorial(t) * math.factorial(len(U) - t - 1) / math.factorial(len(U))
                        for T in itertools.combinations(rest, t):
                            base = Q | set(T)
                            total += wr * wt * (v_of(base | {i}, wA, aA, wB, aB, c) - v_of(base, wA, aA, wB, aB, c))
            phi[i] = total
    return phi


def owen_closed(wA, aA, wB, aB, c):
    out = {}
    for g in GROUPS:
        out[(g, "mix")] = (wB[g] - wA[g]) * ((aA[g] + aB[g]) / 2 - c)
        out[(g, "acc")] = (aB[g] - aA[g]) * (wA[g] + wB[g]) / 2
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ra, rb = load(A), load(B)
    sa, sb = stats(ra), stats(rb)
    wA, aA = {g: sa[g][0] for g in GROUPS}, {g: sa[g][1] for g in GROUPS}
    wB, aB = {g: sb[g][0] for g in GROUPS}, {g: sb[g][1] for g in GROUPS}
    scoreA, scoreB = sum(r["correct"] for r in ra) / len(ra), sum(r["correct"] for r in rb) / len(rb)
    c = (scoreA + scoreB) / 2
    exact, closed = owen_exact(wA, aA, wB, aB, c), owen_closed(wA, aA, wB, aB, c)
    assert all(abs(exact[p] - closed[p]) < 1e-12 for p in closed), "Owen enumeration disagrees with closed form"
    assert abs(sum(exact.values()) - (scoreB - scoreA)) < 1e-12

    # Bootstrap, resampling question ids within each type so shared questions stay paired.
    by_id = {A: {r["id"]: r for r in ra}, B: {r["id"]: r for r in rb}}
    strata = defaultdict(set)
    for sys_rows in (ra, rb):
        for r in sys_rows:
            strata[r["type"]].add(r["id"])
    strata = {t: sorted(ids) for t, ids in strata.items()}
    rng = np.random.default_rng(2026)
    boot = []
    for _ in range(4000):
        pick = {t: rng.choice(ids, len(ids)) for t, ids in strata.items()}
        res = {}
        for s in (A, B):
            rows = [by_id[s][i] for t in pick for i in pick[t] if i in by_id[s]]
            st = stats(rows)
            res[s] = (st, sum(r["correct"] for r in rows) / len(rows))
        cc = (res[A][1] + res[B][1]) / 2
        ow = owen_closed({g: res[A][0][g][0] for g in GROUPS}, {g: res[A][0][g][1] for g in GROUPS},
                         {g: res[B][0][g][0] for g in GROUPS}, {g: res[B][0][g][1] for g in GROUPS}, cc)
        boot.append((res[A][1], res[B][1], ow))
    ci = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]  # noqa: E731

    # Paired comparison on the shared data questions.
    shared = [i for i in by_id[A] if i in by_id[B] and REPORT_GROUP[by_id[A][i]["type"]] in DATA]
    both = sum(by_id[A][i]["correct"] and by_id[B][i]["correct"] for i in shared)
    a_only = sum(by_id[A][i]["correct"] and not by_id[B][i]["correct"] for i in shared)
    b_only = sum(by_id[B][i]["correct"] and not by_id[A][i]["correct"] for i in shared)
    neither = len(shared) - both - a_only - b_only

    # Sensitivity: perturb each system's practice weights (Dirichlet, ~30% relative SD on a 14% weight).
    kappa = 60
    sens = []
    for _ in range(4000):
        pa = rng.dirichlet([kappa * wA[g] for g in GROUPS])
        pb = rng.dirichlet([kappa * wB[g] for g in GROUPS])
        sens.append((sum(p * aA[g] for p, g in zip(pa, GROUPS)), sum(p * aB[g] for p, g in zip(pb, GROUPS))))
    sens = np.array(sens)

    groups_out = []
    for g in GROUPS:
        groups_out.append({
            "group": g, "assistant": {"n": sa[g][2], "weight": wA[g], "accuracy": aA[g]},
            "analyst": {"n": sb[g][2], "weight": wB[g], "accuracy": aB[g]},
            "owen_mix": exact[(g, "mix")], "owen_mix_ci": ci([b[2][(g, "mix")] for b in boot]),
            "owen_accuracy": exact[(g, "acc")], "owen_accuracy_ci": ci([b[2][(g, "acc")] for b in boot]),
            "owen_union": exact[(g, "mix")] + exact[(g, "acc")],
            "owen_union_ci": ci([b[2][(g, "mix")] + b[2][(g, "acc")] for b in boot])})
    by_style = {s: {st: [sum(r["correct"] for r in rows if r["style"] == st), sum(r["style"] == st for r in rows)]
                    for st in ("terse", "casual", "formal", "typo")} for s, rows in ((A, ra), (B, rb))}
    verdicts = {s: dict(sorted(((v, sum(r["verdict"] == v for r in rows)) for v in {r["verdict"] for r in rows}),
                               key=lambda x: -x[1])) for s, rows in ((A, ra), (B, rb))}
    summary = {
        "scores": {A: scoreA, B: scoreB, "gap_analyst_minus_assistant": scoreB - scoreA, "reference_c": c,
                   "ci95": {A: ci([b[0] for b in boot]), B: ci([b[1] for b in boot]),
                            "gap": ci([b[1] - b[0] for b in boot])}},
        "owen_totals": {"mix": sum(exact[(g, "mix")] for g in GROUPS),
                        "accuracy": sum(exact[(g, "acc")] for g in GROUPS),
                        "mix_ci": ci([sum(b[2][(g, "mix")] for g in GROUPS) for b in boot]),
                        "accuracy_ci": ci([sum(b[2][(g, "acc")] for g in GROUPS) for b in boot])},
        "groups": groups_out,
        "paired_shared_data": {"n": len(shared), "both": both, "assistant_only": a_only, "analyst_only": b_only,
                               "neither": neither, "assistant_acc": (both + a_only) / len(shared),
                               "analyst_acc": (both + b_only) / len(shared), "mcnemar_p": mcnemar_exact(a_only, b_only)},
        "weight_sensitivity": {"kappa": kappa,
                               A: [float(np.percentile(sens[:, 0], 5)), float(np.percentile(sens[:, 0], 95))],
                               B: [float(np.percentile(sens[:, 1], 5)), float(np.percentile(sens[:, 1], 95))],
                               "gap": [float(np.percentile(sens[:, 1] - sens[:, 0], 5)),
                                       float(np.percentile(sens[:, 1] - sens[:, 0], 95))],
                               "p_analyst_ahead": float(np.mean(sens[:, 1] > sens[:, 0])),
                               "gap_draws": [round(float(x), 5) for x in sens[:, 1] - sens[:, 0]]},
        "by_style": by_style, "verdicts": verdicts,
    }
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

    print(f"Assistant {scoreA:.1%}  Analyst {scoreB:.1%}  gap {scoreB - scoreA:+.1%}  (c = {c:.3f})")
    print(f"Owen totals: mix {summary['owen_totals']['mix']:+.1%}  accuracy {summary['owen_totals']['accuracy']:+.1%}")
    print(f"{'type':13} {'asst n/w/acc':>20} {'analyst n/w/acc':>20} {'mix':>8} {'accuracy':>9} {'union':>8}")
    for o in groups_out:
        print(f"{o['group']:13} {o['assistant']['n']:4} {o['assistant']['weight']:5.1%} {o['assistant']['accuracy']:6.1%}"
              f"   {o['analyst']['n']:4} {o['analyst']['weight']:5.1%} {o['analyst']['accuracy']:6.1%}"
              f" {o['owen_mix']:+8.1%} {o['owen_accuracy']:+9.1%} {o['owen_union']:+8.1%}")
    p = summary["paired_shared_data"]
    print(f"Shared data questions: {p['n']}; assistant {p['assistant_acc']:.1%}, analyst {p['analyst_acc']:.1%}; "
          f"assistant-only right {p['assistant_only']}, analyst-only right {p['analyst_only']}, McNemar p={p['mcnemar_p']:.3g}")
    print("weight sensitivity (5-95%):", {k: v for k, v in summary["weight_sensitivity"].items() if k != "gap_draws"})


if __name__ == "__main__":
    main()

"""The power run's analysis, exactly as POWER_RUN_PLAN.md set it out before the answers existed.

    python stats_power.py      # needs grades for v2_after, v2_analyst_r8, v3_*, v3b_*

Paired questions: the 175 held-out data questions plus 1,225 generated ones (questions_v3 and v3b).
Fixed Segment Assistant (98bcfbe) vs Cortex Analyst (segmentation_r8.yaml).

- Primary: exact two-sided McNemar test; Analyst minus Assistant with Newcombe's 95% interval.
- Secondary: equivalence by two one-sided tests at a ±3.4-point margin (the 90% interval inside it).
- Checks: clustered McNemar by conversation (Durkalski 2003), first questions only, machine first-pass
  grades, hand-written vs generated questions, by type, and Cohen's kappa of first pass vs read.
- Achieved power at the observed share of split decisions, by Connor's formula and the exact test.
Writes results/dashboard/power_*.json.
"""
import csv
import json
import math
import os
from collections import defaultdict
from statistics import NormalDist

from power_page_data import newcombe
from stats import mcnemar, r4, wilson, write

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MARGIN = 0.034
Z = NormalDist()


def grades(name):
    return {r["id"]: r for r in csv.DictReader(open(os.path.join(RES, f"grades_{name}.csv"), encoding="utf-8"))}


def bank(name):
    return {it["id"]: it for it in json.load(open(os.path.join(HERE, name), encoding="utf-8"))["items"]}


def autos(name):
    return {json.loads(l)["id"]: json.loads(l)["auto"] for l in open(os.path.join(RES, f"auto_{name}.jsonl"), encoding="utf-8")}


def sig(x):
    """p-values to three significant figures, so the tiny ones don't round to zero."""
    return float(f"{x:.3g}")


def split(x, y, ids):
    a = sum(x[i] and y[i] for i in ids)
    b = sum(x[i] and not y[i] for i in ids)
    c = sum(y[i] and not x[i] for i in ids)
    return a, b, c, len(ids) - a - b - c


def connor_power(n, effect, psi, alpha=0.05):
    za = Z.inv_cdf(1 - alpha / 2)
    return Z.cdf((math.sqrt(n) * effect - za * math.sqrt(psi)) / math.sqrt(psi - effect**2))


def exact_power(n, effect, psi, alpha=0.05):
    """Exact McNemar power: average the exact binomial test's rejection chance over d ~ Bin(n, psi)."""
    theta = 0.5 + effect / (2 * psi)
    lf = [0.0]
    for k in range(1, n + 1):
        lf.append(lf[-1] + math.log(k))
    choose = lambda m, k: lf[m] - lf[k] - lf[m - k]  # noqa: E731
    total = 0.0
    for d in range(6, n + 1):
        wd = math.exp(choose(n, d) + d * math.log(psi) + (n - d) * math.log(1 - psi))
        if wd < 1e-12:
            continue
        null = [math.exp(choose(d, k) + d * math.log(0.5)) for k in range(d + 1)]
        lower, acc = [], 0.0
        for p in null:
            acc += p
            lower.append(acc)
        upper = [1 - (lower[k - 1] if k else 0.0) for k in range(d + 1)]
        rej = sum(math.exp(choose(d, k) + k * math.log(theta) + (d - k) * math.log(1 - theta))
                  for k in range(d + 1) if min(1, 2 * min(lower[k], upper[k])) <= alpha)
        total += wd * rej
    return total


def main():
    qs = {**bank("questions_v2.json"), **bank("questions_v3.json"), **bank("questions_v3b.json")}
    A = {**grades("v2_after"), **grades("v3_assistant"), **grades("v3b_assistant")}
    B = {**grades("v2_analyst_r8"), **grades("v3_analyst"), **grades("v3b_analyst")}
    ids = [i for i in B if i in A]
    assert len(ids) == 1400, len(ids)
    a_ok = {i: int(A[i]["correct"]) for i in ids}
    b_ok = {i: int(B[i]["correct"]) for i in ids}

    def compare(label, sub, unit="question", x=a_ok, y=b_ok):
        a, b, c, d = split(x, y, sub)
        n = len(sub)
        lo95, hi95 = newcombe(a, b, c, d, 1.959964)
        lo90, hi90 = newcombe(a, b, c, d, 1.644854)
        return dict(check=label, unit=unit, n=n, both_right=a, assistant_only=b, analyst_only=c, both_wrong=d,
                    assistant_accuracy=r4((a + b) / n), analyst_accuracy=r4((a + c) / n), diff=r4((c - b) / n),
                    lo95=r4(lo95), hi95=r4(hi95), lo90=r4(lo90), hi90=r4(hi90), exact_p=sig(mcnemar(b, c)[0]),
                    mid_p=sig(mcnemar(b, c)[1]), equivalent=bool(lo90 > -MARGIN and hi90 < MARGIN))

    main_row = compare("All 1,400 paired questions (primary)", ids)
    psi = (main_row["assistant_only"] + main_row["analyst_only"]) / 1400
    main_row.update(split_share=r4(psi), connor_power=r4(connor_power(1400, MARGIN, psi)),
                    exact_power=r4(exact_power(1400, MARGIN, psi)))
    for k, (lo, hi) in (("assistant", wilson(main_row["both_right"] + main_row["assistant_only"], 1400)),
                        ("analyst", wilson(main_row["both_right"] + main_row["analyst_only"], 1400))):
        main_row[f"{k}_lo"], main_row[f"{k}_hi"] = r4(lo), r4(hi)

    def root(i):
        while qs[i]["parent"]:
            i = qs[i]["parent"]
        return i
    cl = defaultdict(lambda: [0, 0])
    for i in ids:
        cl[root(i)][0] += a_ok[i] and not b_ok[i]
        cl[root(i)][1] += b_ok[i] and not a_ok[i]
    num = sum(x - y for x, y in cl.values()) ** 2
    den = sum((x - y) ** 2 for x, y in cl.values())
    durk = num / den if den else 0.0
    checks = [main_row,
              dict(check="Follow-ups grouped with the question before them", unit="conversation", n=len(cl),
                   assistant_only=main_row["assistant_only"], analyst_only=main_row["analyst_only"],
                   exact_p=sig(math.erfc(math.sqrt(durk / 2)) if durk > 0 else 1.0), statistic=r4(durk),
                   method="clustered McNemar (Durkalski 2003)"),
              compare("First questions only (follow-ups dropped)", [i for i in ids if qs[i]["type"] != "followup"]),
              compare("Hand-written held-out questions only", [i for i in ids if i.startswith("H")]),
              compare("Generated questions only", [i for i in ids if not i.startswith("H")])]
    au_a = {**autos("v2_after"), **autos("v3_assistant"), **autos("v3b_assistant")}
    au_b = {**autos("v2_analyst_r8"), **autos("v3_analyst"), **autos("v3b_analyst")}
    checks.append(compare("Graded by the machine first pass instead of the read", ids,
                          x={i: int(au_a[i] == "pass") for i in ids}, y={i: int(au_b[i] == "pass") for i in ids}))
    # The plan read every Analyst pass (its audit failed the 2% rule) but only audited the Assistant's;
    # all of them were read anyway. This is the Assistant graded as the plan alone would have.
    a_plan = dict(a_ok)
    for s in ("v3_assistant", "v3b_assistant"):
        read = json.load(open(os.path.join(RES, f"power_read_{s}.txt"), encoding="utf-8"))
        skip = set(read["must"]) | set(read["audit"])
        a_plan.update({i: 1 for i, v in autos(s).items() if v == "pass" and i not in skip})
    checks.append(compare("Assistant's unaudited passes counted right (the plan as written)", ids, x=a_plan))
    write("power_checks", checks)

    types = []
    for t in dict.fromkeys(qs[i]["type"] for i in ids):
        sub = [i for i in ids if qs[i]["type"] == t]
        r = compare(t, sub)
        r["type"] = r.pop("check")
        types.append(r)
    write("power_types", types)

    # first pass vs the read, on every answer that was read and the first pass could check
    kappas = []
    for name, au, G in (("assistant", au_a, A), ("analyst", au_b, B)):
        pairs = [(au[i] == "pass", G[i]["verdict"] == "correct") for i in ids
                 if au[i] in ("pass", "fail") and "first pass (not read" not in G[i]["note"] and "same answer" not in G[i]["note"]]
        n = len(pairs)
        po = sum(x == y for x, y in pairs) / n
        px, py = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
        pe = px * py + (1 - px) * (1 - py)
        kappas.append(dict(system=name, n=n, agreement=r4(po), kappa=r4((po - pe) / (1 - pe))))
    write("power_kappa", kappas)

    m = main_row
    print(f"Assistant {m['assistant_accuracy']:.1%} vs Analyst {m['analyst_accuracy']:.1%} on 1,400 | "
          f"split {m['assistant_only']} / {m['analyst_only']} | exact p {m['exact_p']} | "
          f"Analyst − Assistant {m['diff'] * 100:+.1f} pts, 95% [{m['lo95'] * 100:+.1f}, {m['hi95'] * 100:+.1f}], "
          f"90% [{m['lo90'] * 100:+.1f}, {m['hi90'] * 100:+.1f}] -> equivalent at ±3.4: {m['equivalent']}")
    print(f"split share {psi:.1%}; power at 3.4 pts: Connor {m['connor_power']:.1%}, exact {m['exact_power']:.1%}")
    for c in checks[1:]:
        print(f"  {c['check']}: {c['assistant_only']} vs {c['analyst_only']}, p {c['exact_p']}")
    for t in types:
        print(f"  {t['type']:13} n={t['n']:4} asst {t['assistant_accuracy']:.1%} analyst {t['analyst_accuracy']:.1%} p {t['exact_p']}")
    for k in kappas:
        print(f"  kappa {k['system']}: {k['kappa']} (agreement {k['agreement']:.1%}, n={k['n']})")


if __name__ == "__main__":
    main()

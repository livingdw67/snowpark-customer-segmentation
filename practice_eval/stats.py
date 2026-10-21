"""Statistical tests on the graded answers, written as the datasets behind the results page.

    <repo>\\.venv\\Scripts\\python stats.py      # needs results/grades_*.csv and results/summary.json (owen.py)

Writes results/dashboard/*.json (arrays of row objects). Tests, and the question each answers:

- Wilson intervals: how sure is each score, given 250 questions (and far fewer per type)?
- McNemar (exact, mid-p, chi-square): on the 175 questions both systems got, is either more accurate?
  The two answers to one question are paired, which a two-proportion test would ignore.
  Assumption checks: dependence inside follow-up threads (clustered McNemar, Durkalski et al. 2003,
  and first-turn-only), and grader bias (the same test on the machine first-pass grades).
- Newcombe's interval (method 10) for the paired difference in accuracy.
- Cohen's kappa: does the machine first pass agree with the full read beyond chance?
- Fisher's exact: are follow-ups really answered worse than first questions?
- Chi-square with a permutation p: does wording (terse, casual, formal, typos) change accuracy?
- Shapley vs Owen: Monte Carlo Shapley values over all 22 players, to show they equal the Owen values
  here, and the same comparison on a renormalised game where they don't.
"""
import csv
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np

import owen as ow
from composition import REPORT_GROUP

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "dashboard")
DATA_GROUPS = {"lookup", "total_share", "ranking", "comparison", "breakdown", "filter", "model_eval", "followup",
               "ambiguous", "unanswerable"}
FIRST_TURN = {"lookup", "total_share", "ranking", "comparison", "breakdown", "filter", "model_eval"}
LABELS = {"lookup": "Lookup", "total_share": "Totals and shares", "ranking": "Ranking", "comparison": "Comparison",
          "breakdown": "Breakdown", "filter": "Threshold filter", "model_eval": "Backtests, k, cleaning",
          "followup": "Follow-up", "ambiguous": "Ambiguous", "unanswerable": "Unanswerable",
          "non_data": "Docs, writing, off-topic"}
rng = np.random.default_rng(2026)


# ----------------------------------------------------------------------------- helpers
def load_grades(system):
    rows = list(csv.DictReader(open(os.path.join(RES, f"grades_{system}.csv"), encoding="utf-8")))
    for r in rows:
        r["correct"] = int(r["correct"])
        r["group"] = REPORT_GROUP[r["type"]]
    return {r["id"]: r for r in rows}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def binom_pmf(k, n, p=0.5):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def chi2_sf_1df(x):
    return math.erfc(math.sqrt(x / 2)) if x > 0 else 1.0


def mcnemar(b, c):
    """Exact two-sided, mid-p and continuity-corrected chi-square for discordant counts b, c."""
    n = b + c
    lo = min(b, c)
    exact = min(1.0, 2 * sum(binom_pmf(i, n) for i in range(lo + 1)))
    mid = min(1.0, 2 * (sum(binom_pmf(i, n) for i in range(lo)) + 0.5 * binom_pmf(lo, n)))
    chi2 = (abs(b - c) - 1) ** 2 / n if n else 0.0
    return exact, mid, chi2, chi2_sf_1df(chi2)


def newcombe_paired(a, b, c, d):
    """95% CI for p2 - p1 (column 2 minus column 1) with paired data: Newcombe 1998, method 10.
    a both right, b only system 1 right, c only system 2 right, d both wrong."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    num = a * d - b * c
    phi = (max(num - n / 2, 0) if num > 0 else num) / denom if denom else 0.0
    theta = p1 - p2
    delta = math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    eps = math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return (-theta - eps, -theta + delta), phi


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]] and the sample odds ratio."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def prob(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p_obs = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    p = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))
    oddsr = (a * d) / (b * c) if b * c else float("inf")
    return min(1.0, p), oddsr


def chi2_stat(table):
    t = np.asarray(table, float)
    exp = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    return float(((t - exp) ** 2 / exp).sum()), exp


def cohen_kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = float(np.mean(x == y))
    pe = float(np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y)))
    return (po - pe) / (1 - pe), po, pe


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    clean = [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()} for r in rows]
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=1, ensure_ascii=False)


def r4(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 5)


# ----------------------------------------------------------------------------- main
def main():
    A, B = load_grades("assistant"), load_grades("analyst")
    summary = json.load(open(os.path.join(RES, "summary.json")))
    qs = {it["id"]: it for it in json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))["items"]}

    # --- scores with Wilson intervals (and the bootstrap interval from owen.py)
    shared = [i for i in A if i in B and A[i]["group"] in DATA_GROUPS]  # the 175 in both practice sets
    E = load_grades("assistant_extra")  # the Assistant on the data questions only Analyst's set had
    AP = {**{i: A[i] for i in shared}, **E}  # the Assistant's answer to every data question both answered
    paired = shared + [i for i in E if i in B]
    scores = []
    for key, G, label in (("assistant", A, "Segment Assistant"), ("analyst", B, "Cortex Analyst")):
        k, n = sum(r["correct"] for r in G.values()), len(G)
        lo, hi = wilson(k, n)
        Gp = AP if key == "assistant" else G
        ks = sum(Gp[i]["correct"] for i in paired)
        slo, shi = wilson(ks, len(paired))
        blo, bhi = summary["scores"]["ci95"][key]
        scores.append({"system": key, "label": label, "n": n, "correct": k, "accuracy": r4(k / n),
                       "wilson_lo": r4(lo), "wilson_hi": r4(hi), "boot_lo": r4(blo), "boot_hi": r4(bhi),
                       "shared_n": len(paired), "shared_correct": ks, "shared_accuracy": r4(ks / len(paired)),
                       "shared_lo": r4(slo), "shared_hi": r4(shi)})
    write("scores", scores)

    # --- per-type table with Owen values and Wilson intervals
    types = []
    for g in summary["groups"]:
        name = g["group"]
        ka = sum(r["correct"] for r in A.values() if r["group"] == name)
        kb = sum(r["correct"] for r in B.values() if r["group"] == name)
        la, ha = wilson(ka, g["assistant"]["n"])
        lb, hb = wilson(kb, g["analyst"]["n"])
        types.append({"group": name, "label": LABELS[name],
                      "assistant_n": g["assistant"]["n"], "assistant_correct": ka,
                      "assistant_weight": r4(g["assistant"]["weight"]), "assistant_accuracy": r4(g["assistant"]["accuracy"]),
                      "assistant_lo": r4(la), "assistant_hi": r4(ha),
                      "analyst_n": g["analyst"]["n"], "analyst_correct": kb,
                      "analyst_weight": r4(g["analyst"]["weight"]), "analyst_accuracy": r4(g["analyst"]["accuracy"]),
                      "analyst_lo": r4(lb), "analyst_hi": r4(hb),
                      "mix": r4(g["owen_mix"]), "mix_lo": r4(g["owen_mix_ci"][0]), "mix_hi": r4(g["owen_mix_ci"][1]),
                      "accuracy": r4(g["owen_accuracy"]), "accuracy_lo": r4(g["owen_accuracy_ci"][0]),
                      "accuracy_hi": r4(g["owen_accuracy_ci"][1]),
                      "total": r4(g["owen_union"]), "total_lo": r4(g["owen_union_ci"][0]),
                      "total_hi": r4(g["owen_union_ci"][1])})

    # --- Shapley vs Owen: Monte Carlo Shapley over all 22 players of the same game
    wA = {t["group"]: t["assistant_weight"] for t in types}
    aA = {t["group"]: t["assistant_accuracy"] for t in types}
    wB = {t["group"]: t["analyst_weight"] for t in types}
    aB = {t["group"]: t["analyst_accuracy"] for t in types}
    c = summary["scores"]["reference_c"]
    players = [(g, p) for g in ow.GROUPS for p in ("mix", "acc")]

    def v_add(S):
        return ow.v_of(S, wA, aA, wB, aB, c)

    def v_renorm(S):  # weights renormalised to sum to 1 inside every coalition: types now interact
        w = {g: (wB[g] if (g, "mix") in S else wA[g]) for g in ow.GROUPS}
        a = {g: (aB[g] if (g, "acc") in S else aA[g]) for g in ow.GROUPS}
        return sum(w[g] * a[g] for g in ow.GROUPS) / sum(w.values())

    def shapley_mc(v, perms=20000):
        phi = defaultdict(float)
        for _ in range(perms):
            order = rng.permutation(len(players))
            S, prev = set(), v(set())
            for j in order:
                S.add(players[j])
                cur = v(S)
                phi[players[j]] += cur - prev
                prev = cur
        return {p: phi[p] / perms for p in players}

    def owen_exact_game(v):
        unions = [[(g, "mix"), (g, "acc")] for g in ow.GROUPS]
        m, out = len(unions), {}
        for k, U in enumerate(unions):
            others = [j for j in range(m) if j != k]
            for i in U:
                rest = [p for p in U if p != i]
                tot = 0.0
                for r in range(m):
                    wr = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
                    for R in itertools.combinations(others, r):
                        Q = {p for j in R for p in unions[j]}
                        for t in range(len(rest) + 1):
                            wt = math.factorial(t) * math.factorial(len(U) - t - 1) / math.factorial(len(U))
                            for T in itertools.combinations(rest, t):
                                base = Q | set(T)
                                tot += wr * wt * (v(base | {i}) - v(base))
                out[i] = tot
        return out

    sh_add = shapley_mc(v_add)
    ow_add = owen_exact_game(v_add)
    sh_ren = shapley_mc(v_renorm)
    ow_ren = owen_exact_game(v_renorm)
    diff_add = max(abs(sh_add[p] - ow_add[p]) for p in players)
    diff_ren_union = max(abs((sh_ren[(g, "mix")] + sh_ren[(g, "acc")]) - (ow_ren[(g, "mix")] + ow_ren[(g, "acc")]))
                         for g in ow.GROUPS)
    diff_ren_player = max(abs(sh_ren[p] - ow_ren[p]) for p in players)
    for t in types:
        g = t["group"]
        t["shapley_mix"] = r4(sh_add[(g, "mix")])
        t["shapley_accuracy"] = r4(sh_add[(g, "acc")])
    write("types", types)

    # --- McNemar on the shared data questions
    a_ = sum(AP[i]["correct"] and B[i]["correct"] for i in paired)
    b_ = sum(AP[i]["correct"] and not B[i]["correct"] for i in paired)  # only the assistant right
    c_ = sum(B[i]["correct"] and not AP[i]["correct"] for i in paired)  # only Analyst right
    d_ = len(paired) - a_ - b_ - c_
    exact, mid, chi2, chi2p = mcnemar(b_, c_)
    (dlo, dhi), phi = newcombe_paired(a_, b_, c_, d_)
    nd = b_ + c_
    need = next(k for k in range(nd, -1, -1) if mcnemar(nd - k, k)[0] >= 0.05) + 1  # wins needed out of nd
    theta = max(b_, c_) / nd
    z_a, z_b = 1.959964, 0.841621
    plan_disc = math.ceil((z_a * 0.5 + z_b * math.sqrt(theta * (1 - theta))) ** 2 / (theta - 0.5) ** 2)
    plan_q = math.ceil(plan_disc / (nd / len(paired)))
    write("mcnemar", [{"n": len(paired), "n_core": len(shared), "n_extra": len(paired) - len(shared), "both_right": a_, "assistant_only": b_, "analyst_only": c_, "both_wrong": d_,
                       "discordant": nd, "exact_p": r4(exact), "mid_p": r4(mid), "chi2_cc": r4(chi2),
                       "chi2_cc_p": r4(chi2p), "assistant_shared_accuracy": r4((a_ + b_) / len(paired)),
                       "analyst_shared_accuracy": r4((a_ + c_) / len(paired)),
                       "diff": r4((c_ - b_) / len(paired)), "diff_lo": r4(dlo), "diff_hi": r4(dhi), "phi": r4(phi),
                       "wins_needed": need, "losses_allowed": nd - need, "analyst_share_of_discordant": r4(theta),
                       "plan_discordant": plan_disc, "plan_questions": plan_q,
                       "discordance_rate": r4(nd / len(paired))}])
    write("binomial", [{"k": k, "probability": r4(binom_pmf(k, nd)),
                        "region": "observed" if k == c_ else ("as extreme" if (k <= min(b_, c_) or k >= max(b_, c_)) else "less extreme")}
                       for k in range(nd + 1)])

    # --- assumption checks: dependence (threads), first turns only, grader (machine first pass)
    def root(i):
        while qs[i]["parent"]:
            i = qs[i]["parent"]
        return i
    clusters = defaultdict(lambda: [0, 0])
    for i in paired:
        clusters[root(i)][0] += AP[i]["correct"] and not B[i]["correct"]
        clusters[root(i)][1] += B[i]["correct"] and not AP[i]["correct"]
    num = sum(bk - ck for bk, ck in clusters.values()) ** 2
    den = sum((bk - ck) ** 2 for bk, ck in clusters.values())
    durk = num / den if den else 0.0
    multi = sum(1 for i in {root(i) for i in paired} if sum(root(j) == i for j in paired) > 1)

    first = [i for i in paired if AP[i]["group"] != "followup"]
    fb = sum(AP[i]["correct"] and not B[i]["correct"] for i in first)
    fc = sum(B[i]["correct"] and not AP[i]["correct"] for i in first)

    autos = {s: {json.loads(l)["id"]: json.loads(l)["auto"] for l in open(os.path.join(RES, f"auto_{s}.jsonl"))}
             for s in ("assistant", "analyst")}
    autos_ap = {**autos["assistant"], **{json.loads(l)["id"]: json.loads(l)["auto"]
                                         for l in open(os.path.join(RES, "auto_assistant_extra.jsonl"))}}
    mb = sum(autos_ap[i] == "pass" and autos["analyst"][i] != "pass" for i in paired)
    mc = sum(autos["analyst"][i] == "pass" and autos_ap[i] != "pass" for i in paired)

    kb = sum(A[i]["correct"] and not B[i]["correct"] for i in shared)
    kc = sum(B[i]["correct"] and not A[i]["correct"] for i in shared)
    checks = [
        {"check": f"All {len(paired)} questions both answered (main result)", "unit": "question", "n": len(paired),
         "assistant_only": b_, "analyst_only": c_, "p": r4(exact), "method": "exact binomial"},
        {"check": "Follow-ups grouped with the question before them", "unit": "conversation",
         "n": len({root(i) for i in paired}), "assistant_only": b_, "analyst_only": c_, "p": r4(chi2_sf_1df(durk)),
         "method": "clustered McNemar (Durkalski 2003)", "statistic": r4(durk), "multi_turn_threads": multi},
        {"check": "First questions only (follow-ups dropped)", "unit": "question", "n": len(first),
         "assistant_only": fb, "analyst_only": fc, "p": r4(mcnemar(fb, fc)[0]), "method": "exact binomial"},
        {"check": "Graded by the machine first pass instead of the read", "unit": "question", "n": len(paired),
         "assistant_only": mb, "analyst_only": mc, "p": r4(mcnemar(mb, mc)[0]), "method": "exact binomial"},
        {"check": "Only the questions in both practice sets", "unit": "question", "n": len(shared),
         "assistant_only": kb, "analyst_only": kc, "p": r4(mcnemar(kb, kc)[0]), "method": "exact binomial"},
    ]
    write("checks", checks)

    # discordant questions
    disc = []
    for i in paired:
        if AP[i]["correct"] != B[i]["correct"]:
            loser = B[i] if AP[i]["correct"] else AP[i]
            disc.append({"id": i, "type": LABELS[AP[i]["group"]], "question": qs[i]["q"],
                         "right": "Assistant" if AP[i]["correct"] else "Analyst",
                         "other_verdict": loser["verdict"], "note": loser["note"][:160]})
    write("discordant", disc)

    # --- other tests
    tests = []
    # grader agreement: machine first pass vs full read, on answers the first pass can check
    for s in ("assistant", "analyst"):
        G = A if s == "assistant" else B
        ids = [i for i in G if autos[s][i] in ("pass", "fail")]
        x = [autos[s][i] == "pass" for i in ids]
        y = [bool(G[i]["correct"]) for i in ids]
        k, po, pe = cohen_kappa(x, y)
        boots = []
        for _ in range(4000):
            idx = rng.integers(0, len(ids), len(ids))
            kb_, _, _ = cohen_kappa(np.asarray(x)[idx], np.asarray(y)[idx])
            boots.append(kb_)
        tests.append({"test": "kappa", "system": s, "n": len(ids), "statistic": r4(k), "agreement": r4(po),
                      "chance_agreement": r4(pe), "lo": r4(np.percentile(boots, 2.5)), "hi": r4(np.percentile(boots, 97.5)),
                      "first_pass_pass_read_wrong": sum(a and not b for a, b in zip(x, y)),
                      "first_pass_fail_read_right": sum(b and not a for a, b in zip(x, y))})
    # follow-ups vs first questions
    for s in ("assistant", "analyst"):
        G = A if s == "assistant" else B
        f1 = [G[i]["correct"] for i in G if G[i]["group"] in FIRST_TURN]
        f2 = [G[i]["correct"] for i in G if G[i]["group"] == "followup"]
        p, orr = fisher_exact(sum(f1), len(f1) - sum(f1), sum(f2), len(f2) - sum(f2))
        tests.append({"test": "followup", "system": s, "first_n": len(f1), "first_correct": sum(f1),
                      "first_accuracy": r4(sum(f1) / len(f1)), "followup_n": len(f2), "followup_correct": sum(f2),
                      "followup_accuracy": r4(sum(f2) / len(f2)), "statistic": r4(orr), "p": r4(p)})
    # wording
    wording = []
    for s in ("assistant", "analyst"):
        G = A if s == "assistant" else B
        styles = ["terse", "casual", "formal", "typo"]
        lab = np.array([styles.index(r["style"]) for r in G.values()])
        cor = np.array([r["correct"] for r in G.values()])
        table = [[int(((lab == j) & (cor == 1)).sum()), int(((lab == j) & (cor == 0)).sum())] for j in range(4)]
        stat, exp = chi2_stat(table)
        perm = 0
        reps = 20000
        for _ in range(reps):
            sh = rng.permutation(cor)
            t2 = [[int(((lab == j) & (sh == 1)).sum()), int(((lab == j) & (sh == 0)).sum())] for j in range(4)]
            perm += chi2_stat(t2)[0] >= stat - 1e-12
        min_exp = float(exp.min())
        tests.append({"test": "wording", "system": s, "statistic": r4(stat), "p": r4((perm + 1) / (reps + 1)),
                      "p_chi2": r4(math.exp(-stat / 2) * (1 + stat / 2)),  # chi-square sf, 3 df
                      "min_expected": r4(min_exp), "cells_below_5": int((exp < 5).sum()), "permutations": reps})
        for j, st in enumerate(styles):
            n = table[j][0] + table[j][1]
            wlo, whi = wilson(table[j][0], n)
            wording.append({"system": s, "style": st, "n": n, "correct": table[j][0], "accuracy": r4(table[j][0] / n),
                            "lo": r4(wlo), "hi": r4(whi)})
    write("tests", tests)
    write("wording", wording)

    # --- Owen summary, Shapley check, weight sensitivity
    sens = summary["weight_sensitivity"]
    draws = np.array(sens["gap_draws"])
    edges = np.arange(np.floor(draws.min() * 200) / 200, draws.max() + 0.005, 0.005)
    hist, _ = np.histogram(draws, edges)
    write("sensitivity", [{"bin_start": r4(lo), "bin_end": r4(hi), "draws": int(n)}
                          for lo, hi, n in zip(edges[:-1], edges[1:], hist)])
    st = summary["owen_totals"]
    write("owen", [{"gap": r4(summary["scores"]["gap_analyst_minus_assistant"]),
                    "gap_lo": r4(summary["scores"]["ci95"]["gap"][0]), "gap_hi": r4(summary["scores"]["ci95"]["gap"][1]),
                    "reference_c": r4(c), "mix": r4(st["mix"]), "mix_lo": r4(st["mix_ci"][0]), "mix_hi": r4(st["mix_ci"][1]),
                    "accuracy": r4(st["accuracy"]), "accuracy_lo": r4(st["accuracy_ci"][0]),
                    "accuracy_hi": r4(st["accuracy_ci"][1]), "players": len(players), "unions": len(ow.GROUPS),
                    "shapley_owen_max_diff": r4(diff_add), "shapley_permutations": 20000,
                    "renorm_union_max_diff": r4(diff_ren_union), "renorm_player_max_diff": r4(diff_ren_player),
                    "sens_draws": len(draws), "sens_gap_p5": r4(np.percentile(draws, 5)),
                    "sens_gap_p95": r4(np.percentile(draws, 95)), "sens_share_ahead": r4(np.mean(draws > 0)),
                    "bootstrap_reps": 4000}])

    print(f"McNemar: a={a_} b={b_} c={c_} d={d_}; exact p={exact:.4f}, mid-p={mid:.4f}, chi2cc={chi2:.3f} p={chi2p:.4f}")
    print(f"paired diff {(c_ - b_) / len(paired):+.4f} CI [{dlo:+.4f}, {dhi:+.4f}] phi={phi:.3f}; "
          f"wins needed {need}/{nd}; plan {plan_disc} discordant ~ {plan_q} questions")
    for ch in checks:
        print("check:", ch)
    for t in tests:
        print("test:", t)
    print(f"Shapley vs Owen (additive game): max diff {diff_add:.5f}; renormalised game: union {diff_ren_union:.5f}, "
          f"player {diff_ren_player:.5f}")


if __name__ == "__main__":
    main()

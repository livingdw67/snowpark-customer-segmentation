"""Grading for the power run, by the rules in POWER_RUN_PLAN.md.

    python power_grade.py plan <system>             # results/power_read_<system>.txt: the answers to read
    python power_grade.py show <system> <n> [width] # the next n answers on that list without a verdict
    python power_grade.py fill <system>             # after reading: unread first-pass passes become correct

Read in full: every answer the first pass fails or can't check, every ambiguous and unanswerable answer,
and a random 10% of the remaining first-pass passes (seeded per system). If more than 2% of a system's
audited passes turn out wrong, fill refuses and every pass has to be read.
"""
import json
import math
import os
import random
import sys

import grade

RES = grade.RES


def autos(system):
    return {x["id"]: x["auto"] for x in grade.load(os.path.join(RES, f"auto_{system}.jsonl"))}


def plan(system):
    qs, au = grade.questions(system), autos(system)
    must = [i for i in au if au[i] != "pass" or qs[i]["type"] in ("ambiguous", "unanswerable")]
    passes = sorted(i for i in au if i not in must)
    audit = sorted(random.Random(f"2026-{system}").sample(passes, math.ceil(0.1 * len(passes))))
    with open(os.path.join(RES, f"power_read_{system}.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"must": must, "audit": audit}))
    print(f"{system}: {len(au)} answers; read {len(must)} (fails, can't-check, ambiguous, unanswerable) "
          f"+ audit {len(audit)} of {len(passes)} passes")


def read_list(system):
    return json.load(open(os.path.join(RES, f"power_read_{system}.txt"), encoding="utf-8"))


def show(system, n, width=900):
    sys.stdout.reconfigure(encoding="utf-8")
    rl = read_list(system)
    grade.sheet(system, 0, n, width, todo=True, only=set(rl["must"]) | set(rl["audit"]))


def verdicts_of(system):
    out = {}
    path = os.path.join(RES, f"review_{system}.txt")
    for line in open(path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            i, v = line.split("|")[:2]
            out[i.strip()] = v
    return out


def fill(system):
    """The 2% rule is per system, so the audit pools every bank that system answered (v3 and v3b)."""
    rl, au = read_list(system), autos(system)
    path = os.path.join(RES, f"review_{system}.txt")
    verdicts = verdicts_of(system)
    missing = [i for i in rl["must"] + rl["audit"] if i not in verdicts]
    assert not missing, f"still to read: {len(missing)}, e.g. {missing[:5]}"
    engine = grade.ENGINE[system]
    pooled, wrong = 0, []
    for s in [s for s in grade.SYSTEMS if grade.ENGINE[s] == engine and s.startswith(("v3_", "v3b_"))]:
        other = read_list(s)
        v = verdicts_of(s)
        assert all(i in v for i in other["audit"]), f"{s}: audit not finished, so the pooled check must wait"
        pooled += len(other["audit"])
        wrong += [i for i in other["audit"] if v[i] != "correct"]
    rate = len(wrong) / pooled
    print(f"{engine}, pooled over its banks: audit found {len(wrong)} of {pooled} passes wrong ({rate:.1%}) {wrong}")
    if rate > 0.02:
        sys.exit("over 2%: the plan says read every pass for this system")
    rest = [i for i in au if i not in verdicts]
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(f"{i}|correct|first pass (not read; pooled audit {len(wrong)}/{pooled} wrong)\n" for i in rest)
    print(f"filled {len(rest)} unread passes as correct")


def unread_passes(system):
    """Passes not yet given a verdict (or only filled in as an unread pass)."""
    v = {}
    for line in open(os.path.join(RES, f"review_{system}.txt"), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            i, verdict, note = (line.rstrip("\n").split("|", 2) + [""])[:3]
            v[i.strip()] = note
    au = autos(system)
    return [i for i in au if i not in v or "first pass (not read" in v[i]]


def compact(system, a, b, width=260):
    """Unread passes in a compact view: question, truth, start of the answer."""
    sys.stdout.reconfigure(encoding="utf-8")
    qs = grade.questions(system)
    answers = {x["id"]: x for x in grade.load(grade.answers_path(system))}
    eng = grade.ENGINE[system]
    for i in unread_passes(system)[a:b]:
        raw = (answers[i].get("answer") or "").replace("This is our interpretation of your question:\n\n", "")
        lines = [l for l in raw.splitlines() if l.strip()]
        rows = [l for l in lines if l.startswith("|") and not set(l) <= set("|-: ")]
        if eng == "analyst" and rows:  # interpretation, then the table body (header dropped)
            head = " ".join(l for l in lines if not l.startswith("|"))[:110]
            body = head + " || " + " / ".join(" ".join(c.strip() for c in r.strip("|").split("|")) for r in rows[1:])
        else:
            body = " ".join(raw.split())
        print(f"{i} | {qs[i]['q']} || T: {qs[i][f'truth_{eng}']['truth'][:100]} || A: {body[:width]}")


def mark(system, n, exceptions=""):
    """Record the first n unread passes as read and correct, except IDs given as "ID=verdict=note;..."."""
    exc = {}
    for part in [x for x in exceptions.split(";") if x.strip()]:
        i, v, note = (part.split("=", 2) + [""])[:3]
        assert v in grade.VERDICTS, v
        exc[i.strip()] = (v, note)
    ids = unread_passes(system)[:n]
    with open(os.path.join(RES, f"review_{system}.txt"), "a", encoding="utf-8") as f:
        for i in ids:
            v, note = exc.get(i, ("correct", "pass read in the full read"))
            f.write(f"{i}|{v}|{note}\n")
    missing = set(exc) - set(ids)
    assert not missing, f"exceptions not in this batch: {missing}"
    print(f"{system}: marked {len(ids)} ({len(exc)} not correct); {len(unread_passes(system))} unread passes left")


if __name__ == "__main__":
    cmd, system = sys.argv[1], sys.argv[2]
    if cmd == "plan":
        plan(system)
    elif cmd == "compact":
        compact(system, int(sys.argv[3]), int(sys.argv[4]), *(int(x) for x in sys.argv[5:6]))
    elif cmd == "mark":
        mark(system, int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
    elif cmd == "count":
        print(system, len(unread_passes(system)), "unread passes")
    elif cmd == "show":
        show(system, int(sys.argv[3]), *(int(x) for x in sys.argv[4:5]))
    else:
        fill(system)

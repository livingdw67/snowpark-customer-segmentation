"""Grade the answers: a deterministic first pass, then a full read of every answer.

    python grade.py auto                       # first pass -> results/auto_<system>.jsonl
    python grade.py sheet <system> <a> <b>     # print answers a..b-1 for reading, with truth and first-pass result
    python grade.py merge                      # results/review_<system>.txt -> results/grades_<system>.csv

The first pass checks figures (rounding-aware, independent of the assistant's own figure
checker), segment codes, yes/no words and decline wording. It is only a hint: every answer is
read and given a verdict in results/review_<system>.txt, one line per question:

    ID|verdict|note

Verdicts: correct, wrong number, wrong claim, unhelpful, declined-but-answerable,
answered-instead-of-declining, misrouted, unsafe, error.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SYSTEMS = ("assistant", "analyst", "assistant_extra", "v2_analyst", "v2_before", "v2_after", "tuned", "tuned_extra",
           "v2_analyst_r8", "v3_assistant", "v3_analyst", "v3b_assistant", "v3b_analyst")
ENGINE = {"assistant": "assistant", "analyst": "analyst", "assistant_extra": "assistant", "v2_analyst": "analyst",
          "v2_before": "assistant", "v2_after": "assistant", "tuned": "assistant",
          "tuned_extra": "assistant", "v2_analyst_r8": "analyst", "v3_assistant": "assistant",
          "v3_analyst": "analyst", "v3b_assistant": "assistant", "v3b_analyst": "analyst"}  # whose truths and answer format
# Held-out runs of the old and fixed code, and the fixed code on the original questions it was tuned on
ANSWERS = {"v2_before": "v2_assistant_before.jsonl", "v2_after": "v2_assistant_after.jsonl",
           "tuned": "tuned_assistant_answers.jsonl", "tuned_extra": "tuned_assistant_extra_answers.jsonl"}
VERDICTS = {"correct", "wrong number", "wrong claim", "unhelpful", "declined-but-answerable",
            "answered-instead-of-declining", "misrouted", "unsafe", "error"}
TIERS = {"C1": "Top spenders", "C2": "Core", "C3": "Occasional", "C4": "Low value"}
STYLES = {"B1": "Bulk buyers", "B2": "Holiday buyers", "B3": "Small-basket buyers"}
NAMES = {f"{t}-{b}": f"{tn} / {bn}" for t, tn in TIERS.items() for b, bn in STYLES.items()}

NUM = re.compile(r"(?<![\w.])(?P<cur>[£$€])?\s?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
                 r"\s?(?P<suf>%|k\b|K\b|m\b|M\b|bn\b|million\b|thousand\b|p\b)?")
SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "m": 1e6, "M": 1e6, "million": 1e6, "bn": 1e9, "p": 0.01}
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12}
DECLINE = re.compile(
    r"not (available|in the data|held|included|tracked|recorded|something)|doesn'?t (hold|have|contain|include|track)|"
    r"do(es)? not (hold|have|contain|include|track)|don'?t have|no (data|information|column|field|record|details?)\b|"
    r"can'?t|cannot|unable|isn'?t available|not able|outside (the|my)|only (has|holds|covers|includes|contains)|"
    r"no (customer|product|age|gender|country)|not possible|I'?m sorry|unfortunately|not supported", re.I)
CLARIFY = re.compile(r"do you mean|did you mean|could you (clarify|specify)|can you (clarify|specify)|please (clarify|specify)|"
                     r"which (one|metric|measure|do you)|what do you mean|could mean|can mean|depends on", re.I)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def answers_path(system):
    return os.path.join(RES, ANSWERS.get(system, f"{system}_answers.jsonl"))


def questions(system=""):
    bank = ("questions_v3b.json" if system.startswith("v3b_") else "questions_v3.json" if system.startswith("v3_") else "questions_v2.json" if system.startswith("v2_")
            else "questions.json")
    return {it["id"]: it for it in json.load(open(os.path.join(HERE, bank), encoding="utf-8"))["items"]}


def figures(text):
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)            # dates
    text = re.sub(r"\b[CcBb]\d(?:-[Bb]\d)?\b|\b[Qq][1-4]\b|\btop[- ]?20\b|\bk\s?=\s?\d\b", " ", text)  # codes, labels
    out = []
    for m in NUM.finditer(text):
        dec = m.group("dec") or ""
        value = float(m.group("int").replace(",", "") + dec)
        suf = m.group("suf") or ""
        scale = SCALE.get(suf, 1.0)
        places = len(dec) - 1 if dec else 0
        out.append((value * scale, 0.5 * 10 ** -places * scale * 1.02, suf == "%"))
    for w, n in WORDNUM.items():
        if re.search(rf"\b{w}\b", text, re.I):
            out.append((float(n), 0.0, False))
    return out


def cells(rows):
    out = []
    for row in rows or []:
        for v in row.values():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                d = next((d for d in range(7) if abs(v - round(v, d)) < 1e-9), 7)
                out.append((float(v), 0.5 * 10 ** -d * 1.02 if d < 4 else 0.0))
    return out


def found(v, figs, cs):
    for x, tol, pct in figs:
        if abs(x - v) <= max(tol, 0.005 * abs(v)) or (pct and abs(x - 100 * v) <= max(tol, 0.005 * abs(100 * v))):
            return True
    for x, tol in cs:
        if abs(x - v) <= max(tol, 0.005 * abs(v), 1e-9) or abs(x - 100 * v) <= max(tol, 0.005 * abs(100 * v)):
            return True
    return False


def codes_in(text):
    got = {c.upper().replace("B", "-B", 1) if "-" not in c else c.upper()
           for c in re.findall(r"\b[Cc]\d-?[Bb]\d\b", text)}
    low = text.lower()
    return got | {c for c, n in NAMES.items() if n.lower() in low}


def auto(it, ans, system):
    system = ENGINE[system]
    t = it[f"truth_{system}"]
    text = ans.get("answer") or ""
    if ans.get("error") or not text:
        return {"auto": "error", "detail": ans.get("error", "empty answer")[:120]}
    figs, cs = figures(text), cells(ans.get("rows") if system == "analyst" else None)
    declined = bool(DECLINE.search(text)) or (system == "assistant" and ans.get("declined"))
    no_sql = system == "analyst" and not ans.get("sql")
    kind = t["kind"]
    if kind == "answer":
        hits = [found(v, figs, cs) for v in t["values"]]
        frac = sum(hits) / len(hits) if hits else 1.0
        codes = codes_in(text)
        miss_codes = [c for c in t["codes"] if c not in codes]
        words_ok = not t["words"] or any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in t["words"])
        absent = [a for a in t["absent"] if a.lower() in text.lower()]
        ok = frac >= t["min_frac"] and not miss_codes and words_ok and not absent and not no_sql
        detail = f"values {sum(hits)}/{len(hits)}" + (f" missing codes {miss_codes}" if miss_codes else "") + \
                 ("" if words_ok else f" no word {t['words']}") + (f" absent-found {absent}" if absent else "") + \
                 (" NO SQL" if no_sql else "") + (" decline-words" if declined else "")
        return {"auto": "pass" if ok else "fail", "detail": detail}
    if kind == "ambiguous":
        matched = [r["label"] for r in t["readings"] if r["values"] and all(found(v, figs, cs) for v in r["values"])]
        clar = bool(CLARIFY.search(text)) or "suggestions:" in text.lower()
        return {"auto": "pass" if matched or clar else "fail",
                "detail": f"readings {matched}" + (" clarifies" if clar else "") + (" NO SQL" if no_sql else "")}
    if kind == "decline":
        return {"auto": "pass" if declined or no_sql else "fail",
                "detail": ("declines" if declined else "no decline words") + (" no SQL" if no_sql else " has SQL")}
    if kind == "writing":
        notes = []
        if "max_chars" in t and len(text) > t["max_chars"]:
            notes.append(f"{len(text)} chars > {t['max_chars']}")
        if "max_words" in t and len(text.split()) > t["max_words"]:
            notes.append(f"{len(text.split())} words > {t['max_words']}")
        if "count" in t:
            items = [l for l in text.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+", l)]
            if items and len(items) != t["count"]:
                notes.append(f"{len(items)} list items, asked {t['count']}")
        return {"auto": "check", "detail": "; ".join(notes) or f"{len(text.split())} words"}
    return {"auto": "check", "detail": ("declines" if declined else "") + (" no SQL" if no_sql else "")}


def run_auto():
    for system in SYSTEMS:
        path = answers_path(system)
        if not os.path.exists(path):
            continue
        qs = questions(system)
        out = []
        for a in load(path):
            r = auto(qs[a["id"]], a, system)
            out.append({"id": a["id"], **r})
        with open(os.path.join(RES, f"auto_{system}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
        from collections import Counter
        print(system, len(out), dict(Counter(r["auto"] for r in out)))


def reviewed(system):
    path = os.path.join(RES, f"review_{system}.txt")
    if not os.path.exists(path):
        return set()
    return {l.split("|")[0].strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")}


def sheet(system, a, b, width=1400, todo=False, only=None):
    qs = questions(system)
    answers = {x["id"]: x for x in load(answers_path(system))}
    autos = {x["id"]: x for x in load(os.path.join(RES, f"auto_{system}.jsonl"))}
    order = [i for i in qs if i in answers and (only is None or i in only)]
    if todo:  # answered but not yet given a verdict
        done = reviewed(system)
        order = [i for i in order if i not in done]
    for i in order[a:b]:
        it, ans, au = qs[i], answers[i], autos[i]
        print(f"=== {i} [{it['type']}/{it['style']}] auto={au['auto']} ({au['detail']})")
        if it["parent"]:
            p = answers.get(it["parent"], {})
            print(f"  EARLIER Q: {qs[it['parent']]['q']}\n  EARLIER A: {(p.get('answer') or p.get('error', ''))[:300]!r}")
        print(f"Q: {it['q']}\nTRUTH: {it[f'truth_{ENGINE[system]}']['truth']}")
        if ENGINE[system] == "assistant" and (ans.get("adjust") or ans.get("intent") != "analytical"):
            print(f"  intent={ans.get('intent')} declined={ans.get('declined')} adjust={ans.get('adjust')}")
        if ENGINE[system] == "analyst" and system != "analyst" and ans.get("sql_error"):
            print(f"  sql_error={ans.get('sql_error')}")
        if system == "analyst":
            print(f"  category={ans.get('category')} sql_error={ans.get('sql_error')}")
        body = ans.get("answer") or ("ERROR: " + ans.get("error", ""))
        print("A: " + (body if len(body) <= width else body[:width] + f" ...[+{len(body) - width} chars]"))
        print()


def merge():
    for system in SYSTEMS:
        qs = questions(system)
        path = os.path.join(RES, f"review_{system}.txt")
        if not os.path.exists(path):
            continue
        answers = {x["id"]: x for x in load(answers_path(system))}
        autos = {x["id"]: x for x in load(os.path.join(RES, f"auto_{system}.jsonl"))}
        grades = {}
        for line in open(path, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                i, verdict, note = (line.rstrip("\n").split("|", 2) + [""])[:3]
                assert verdict in VERDICTS, (system, i, verdict)
                grades[i.strip()] = (verdict, note.strip())
        missing = [i for i in answers if i not in grades]
        assert not missing, f"{system}: ungraded {missing[:10]} ({len(missing)})"
        with open(os.path.join(RES, f"grades_{system}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "type", "style", "sets", "parent", "question", "verdict", "correct", "auto", "note"])
            for i in [i for i in qs if i in answers]:
                it, (v, note) = qs[i], grades[i]
                w.writerow([i, it["type"], it["style"], it["sets"], it["parent"] or "", it["q"], v, int(v == "correct"),
                            autos[i]["auto"], note])
        agree = [(autos[i]["auto"], grades[i][0]) for i in answers if autos[i]["auto"] in ("pass", "fail")]
        both = sum((a == "pass") == (g == "correct") for a, g in agree)
        print(f"{system}: {sum(g[0] == 'correct' for g in grades.values())}/{len(grades)} correct; "
              f"first pass agreed with the read on {both}/{len(agree)} auto-checkable answers")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "auto":
        run_auto()
    elif cmd == "sheet":
        sys.stdout.reconfigure(encoding="utf-8")
        sheet(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), *(int(x) for x in sys.argv[5:6]))
    elif cmd == "todo":  # python grade.py todo <system> <n> [width]
        sys.stdout.reconfigure(encoding="utf-8")
        sheet(sys.argv[2], 0, int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1400, todo=True)
    elif cmd == "merge":
        merge()

"""The fixed code on the original questions, the ones its fixes were written against ("tuned on").

    python tuned.py carry [system]  # review_<system>.txt: every answer identical to the old run's keeps its verdict
    python grade.py todo tuned 40        # then read the answers that changed, appending ID|verdict|note
    python grade.py todo tuned_extra 40
    python grade.py merge                # -> grades_tuned.csv, grades_tuned_extra.csv

Not blind: the old verdicts were given before the fixes, and the changed answers are read knowing they are new.
These numbers show how much of the gain is fitting the questions the fixes came from; the held-out set
(stats_v2.py) is the fair test.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PAIRS = {"tuned": ("tuned_assistant_answers.jsonl", "assistant_answers.jsonl", "review_assistant.txt"),
         "tuned_extra": ("tuned_assistant_extra_answers.jsonl", "assistant_extra_answers.jsonl",
                         "review_assistant_extra.txt"),
         # Analyst with semantic model r8 on the held-out set: same carry-over, against its r7 answers
         "v2_analyst_r8": ("v2_analyst_r8_answers.jsonl", "v2_analyst_answers.jsonl", "review_v2_analyst.txt")}


def load(name):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(RES, name), encoding="utf-8") if l.strip()}


def reviews(name):
    out = {}
    for line in open(os.path.join(RES, name), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            i, v, note = (line.rstrip("\n").split("|", 2) + [""])[:3]
            out[i.strip()] = (v, note.strip())
    return out


def carry(only=None):
    for system, (new_f, old_f, rev_f) in PAIRS.items():
        if only and system != only:
            continue
        new, old, rev = load(new_f), load(old_f), reviews(rev_f)
        path = os.path.join(RES, f"review_{system}.txt")
        done = set(reviews(f"review_{system}.txt")) if os.path.exists(path) else set()
        same = [i for i in new if i not in done and new[i].get("answer") == old[i].get("answer")]
        with open(path, "a", encoding="utf-8") as f:
            if not done:
                f.write(f"# {system}: answers identical to {old_f} keep its verdict; changed ones are read.\n")
            for i in same:
                v, note = rev[i]
                f.write(f"{i}|{v}|same answer as before{': ' + note if note else ''}\n")
        changed = [i for i in new if i not in done and i not in same]
        print(f"{system}: {len(new)} answers, {len(same)} unchanged (verdict carried), {len(changed)} changed to read")


if __name__ == "__main__":
    {"carry": carry}[sys.argv[1]](*sys.argv[2:3])

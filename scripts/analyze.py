#!/usr/bin/env python3
"""Story-quality report for generated packages.

Measures the things we iterate on, beyond the hard gates:
  - repetition        duplicated sentences (the #1 template failure mode)
  - citation tells    exhibit-language that announces the puzzle
  - clue delivery     how each clue lands: inside speech/attribution, inside
                      a narrative join, fused with a distractor, or bare
  - clue burial       share of clues sitting below scene-opening paragraphs
  - lexical variety   distinct-sentence ratio

Usage: python3 scripts/analyze.py runs/one runs/two ...
"""
import json
import os
import re
import sys
from collections import Counter

TELLS = ["a note mentioned", "in passing.", "wait among the files",
         "went through the season's papers", "reading it out",
         "There it was, plain", "— THE RECORD —", "figure out"]

ATTR = re.compile(r"\b(allowed that|was to be believed|put it plainly|"
                  r"said nothing for a while|had to ask \w+ twice|"
                  r"said,|said that|remarked|admitted|muttered|"
                  r"finally said it)\b")
JOIN = re.compile(r"^(And though|Even the newest|By then everyone|"
                  r"The talk kept|Sooner or later|Which was how|"
                  r"It went unchallenged|Still,)")


def sentences(text):
    out = []
    for para in text.split("\n\n"):
        for s in re.split(r"(?<=[.!?])\s+", para.strip()):
            if len(s) > 15:
                out.append(s.strip())
    return out


def sentence_of(text, start):
    beg = text.rfind("\n\n", 0, start) + 2
    seg = text[beg:]
    best = None
    for m in re.finditer(r"[.!?](\s|$)", seg[:start - beg + 400]):
        if m.end() + beg <= start:
            best = beg + m.end()
    s = seg[(best or 0) - beg:]
    end = re.search(r"[.!?](\s|$)", s)
    return s[:end.end()].strip() if end else s.strip()


def analyze(d):
    r = {"dir": d}
    try:
        v = json.load(open(os.path.join(d, "verification.json")))
        r["gates"] = all(g["pass"] and g["roundtrip"]["pass"]
                         for g in v["realization"].values())
        r["polished"] = any(g.get("polished") for g in v["realization"].values())
    except FileNotFoundError:
        r["gates"] = None
    try:
        m = json.load(open(os.path.join(d, "realization_map.json")))
    except FileNotFoundError:
        m = None
    text = open(os.path.join(d, "story.md")).read() \
        if os.path.exists(os.path.join(d, "story.md")) else \
        (open(os.path.join(d, "challenge.md")).read()
         if os.path.exists(os.path.join(d, "challenge.md")) else "")
    r["genre"] = json.load(open(os.path.join(d, "hidden_formal.json"))) \
        ["config"].get("genre", "?")
    sents = sentences(text)
    r["paragraphs"] = len([p for p in text.split("\n\n") if p.strip()])
    r["sentences"] = len(sents)
    r["dup_sentences"] = sum(c - 1 for c in Counter(sents).values() if c > 1)
    low = text.lower()
    r["tells"] = sum(low.count(t.lower()) for t in TELLS)
    if m:
        hidden = json.load(open(os.path.join(d, "hidden_formal.json")))
        distractors = set(hidden["distractors"])
        clue_ids = [ref for ref in m["rendered"]
                    if ref.startswith("E")]
        delivery = Counter()
        buried = 0
        for ref in clue_ids:
            s, e = m["spans"][ref]
            sent = sentence_of(text, s)
            if ATTR.search(sent):
                delivery["speech"] += 1
            elif ", though " in sent:
                delivery["fused"] += 1
            elif JOIN.match(sent):
                delivery["join"] += 1
            else:
                delivery["other"] += 1
            if text[:s].count("\n\n") >= 1 and \
                    text[:s].split("\n\n")[-1].strip()[:80] != sent[:80]:
                buried += 1
        n = max(1, len(clue_ids))
        r["delivery"] = {k: round(v / n, 2) for k, v in delivery.items()}
        # first-clue depth: how many paragraphs precede the first clue
        first = min(m["spans"][ref][0] for ref in clue_ids)
        r["first_clue_depth"] = text[:first].count("\n\n")
    try:
        sk = json.load(open(os.path.join(d, "skeleton.json")))
        r["scenes"] = sk["pacing"]["params"]["n_scenes"]
        r["burial"] = sk["pacing"]["params"].get("burial")
    except FileNotFoundError:
        pass
    # monotony: repeated attribution verbs and runs of same-verb scenic
    # sentences are the template modes' failure modes
    attr_counts = Counter(ATTR.findall(text))
    r["attr_repeats"] = sum(c - 1 for c in attr_counts.values() if c > 1)
    verbs = [m.group(0) for m in re.finditer(
        r"\b(?:put right|left standing|carried down|set to rights|"
        r"gave up on|argued over|turned out|tripped over|"
        r"kept half an eye on|made room for)\b", text)]
    run = best = 0
    for i, v in enumerate(verbs):
        run = run + 1 if i and v == verbs[i - 1] else 1
        best = max(best, run)
    r["max_verb_run"] = best
    r["scenic_verbs"] = len(verbs)
    return r

def main():
    rows = [analyze(d) for d in sys.argv[1:]]
    cols = ["dir", "genre", "gates", "polished", "scenes", "burial",
            "paragraphs", "sentences", "dup_sentences", "tells",
            "delivery", "first_clue_depth", "attr_repeats",
            "max_verb_run", "scenic_verbs"]
    for row in rows:
        print(" | ".join(f"{c}={row.get(c)}" for c in cols if c in row))
    print(f"\n{len(rows)} runs")


if __name__ == "__main__":
    main()

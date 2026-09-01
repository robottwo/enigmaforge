"""Test battery: genre packs — seeded selection, per-pack gate health,
no cross-pack leakage, no real-world names reintroduced."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enigmaforge.genres import PACKS, pick_genre, get_pack
from enigmaforge.narrative import compile_narrative
from enigmaforge.story import build_skeleton, compile_story
from enigmaforge.verify import verify_realization, verify_roundtrip

CFG = dict(n_variables=8, n_constraints=10, dependency_depth=3, domain_size=4,
           n_people=4, n_bridges=2, n_distractors=2, n_objective_stages=2,
           narrative_tokens=600)

def _world(seed, **kw):
    from enigmaforge.generator import generate_world
    from enigmaforge.populate import (populate_evidence, populate_bridges,
                                      populate_objectives)
    cfg = dict(CFG); cfg.update(kw)
    w = generate_world(cfg, seed)
    populate_evidence(w, seed)
    populate_bridges(w, seed)
    populate_objectives(w, seed)
    return w

def test_genre_is_seeded_axis():
    assert pick_genre(7) == pick_genre(7), "same seed, same genre"
    seen = {pick_genre(s) for s in range(20)}
    assert len(seen) >= 3, f"expected variety across seeds, got {seen}"

def test_every_pack_passes_all_surface_gates():
    for name in sorted(PACKS):
        w = _world(3, genre=name)
        for r in (compile_narrative(w, 100),
                  compile_story(w, build_skeleton(w, 3), 900)):
            g = verify_realization(w, r)
            assert g["pass"], (name, g["issues"])
            rt = verify_roundtrip(w, r)
            assert rt["pass"], (name, rt)

def test_pack_nouns_unique_and_unknown_genre_rejected():
    for name, pack in PACKS.items():
        assert len(set(pack.nouns)) == len(pack.nouns), name
        assert len(pack.nouns) >= 30, f"{name} too small for medium tier"
    w = _world(1)
    w.config["genre"] = "spacewestern"
    try:
        get_pack(w)
        assert False, "unknown genre must raise"
    except ValueError:
        pass

MARITIME_TELLS = ["harbor", "quay", "chandlery", "mooring", "tide"]

def test_no_cross_pack_leakage():
    for name in ("manor", "hotel", "theater", "observatory"):
        w = _world(5, genre=name)
        r = compile_story(w, build_skeleton(w, 5), 900)
        low = r.text.lower()
        hits = [t for t in MARITIME_TELLS if t in low]
        assert not hits, f"{name} leaked maritime flavor: {hits}"

REAL_WORLD_TELLS = ["George Washington", "Neil Armstrong", "Shakespeare",
                    "Titanic", "penicillin", "the Nile", "Everest"]

def test_no_real_world_names_any_pack():
    for name in sorted(PACKS):
        w = _world(2, genre=name)
        r = compile_story(w, build_skeleton(w, 2), 700)
        for tell in REAL_WORLD_TELLS:
            assert tell not in r.text, f"{name}: {tell!r} leaked"

def test_genre_recorded_in_package():
    from enigmaforge.pipeline import build
    import json, tempfile
    w = build("small", seed=4, config_overrides={"mode": "story",
                                                 "genre": "theater"})
    assert w.config["genre"] == "theater"
    from enigmaforge.pipeline import package
    with tempfile.TemporaryDirectory() as td:
        package(w, td)
        hidden = json.load(open(os.path.join(td, "hidden_formal.json")))
        assert hidden["config"]["genre"] == "theater"

CITATION_TELLS = ["a note mentioned", "in passing.", "wait among the files",
                  "went through the season's papers", "reading it out",
                  "There it was, plain"]

def test_story_mode_has_no_citation_frame():
    """Story surfaces must deliver clues as events, not citations: no
    'a note mentioned X', no reading out, no exhibit-label connectives."""
    for name in sorted(PACKS):
        w = _world(9, genre=name)
        r = compile_story(w, build_skeleton(w, 9), 1100)
        for tell in CITATION_TELLS:
            assert tell not in r.text, f"{name}: citation tell {tell!r}"


def test_burial_buries_clues_and_keeps_gates():
    """burial>0: scenic paragraphs precede/follow clue paragraphs, clues
    sink below the scene opening, and every gate still passes."""
    from enigmaforge.verify import verify_realization, verify_roundtrip
    w = _world(6, genre="manor", burial=3)
    sk = build_skeleton(w, 6)
    r = compile_story(w, sk, 900)
    assert verify_realization(w, r)["pass"]
    assert verify_roundtrip(w, r)["pass"]
    # find the first clue-bearing scene with an opening depth and prove the
    # clue is not in that scene's first paragraph
    clue_units = [u for u in w.evidence if not u.is_distractor]
    buried = 0
    for sc in sk.scenes:
        if sc.depth_open and any(b.kind == "clue" for b in sc.beats):
            paras = r.text.split("\n\n")
            first_clue = next(b for b in sc.beats if b.kind == "clue")
            span = r.spans[first_clue.ref]
            # paragraphs before the clue span >= the scene's opening depth:
            # the clue sits under its burial paragraphs
            assert r.text[:span[0]].count("\n\n") >= sc.depth_open, \
                "clue not buried under its opening paragraphs"
            buried += 1
    assert buried > 0, "burial=3 should bury at least one scene"
    assert sk.pacing["params"]["burial"] == 3

def test_high_burial_adds_story_scenes():
    saw_empty = False
    for seed in range(10):
        w = _world(seed, genre="theater", burial=3)
        sk = build_skeleton(w, seed)
        if any(not s.beats for s in sk.scenes):
            saw_empty = True
            r = compile_story(w, sk, 900)
            assert verify_realization(w, r)["pass"]
            assert verify_roundtrip(w, r)["pass"]
            break
    assert saw_empty, "burial=3 should sometimes insert clue-free scenes"

def test_burial_zero_keeps_clues_high():
    w = _world(6, genre="manor", burial=0)
    sk = build_skeleton(w, 6)
    assert all(s.depth_open == 0 and s.depth_close == 0 for s in sk.scenes)
    assert all(s.beats for s in sk.scenes), "no story-only scenes at burial 0"


def test_scenic_sentences_do_not_repeat():
    """The scenic event grammar exists so burial paragraphs stop repeating:
    across a burial-3 story, near-zero duplicated sentences, and cast-event
    sentences are present."""
    from collections import Counter
    import re as _re
    w = _world(6, genre="manor", burial=3)
    r = compile_story(w, build_skeleton(w, 6), 900)
    sents = [s.strip() for p in r.text.split("\n\n") for s in p.split(". ")
             if len(s.strip()) > 15]
    dups = [(s, c) for s, c in Counter(sents).items() if c > 1]
    assert len(dups) <= 1, f"repeating fragments are back: {dups[:4]}"
    assert _re.search(r"\b(gave up on|put right|argued over|"
                      r"kept half an eye on) the\b", r.text), \
        "no cast event sentences in scenic text"

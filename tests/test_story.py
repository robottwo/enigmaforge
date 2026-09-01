"""Test battery: realization contract, story skeleton + pacing, surface
faithfulness gates, extraction round-trip, renderer rejection loop, and the
discovery scoring dimensions."""
import sys, os, re, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enigmaforge.narrative import Realization, compile_narrative
from enigmaforge.story import (build_skeleton, compile_story,
                               compile_story_verified, template_scene_renderer,
                               RenderContractError)
from enigmaforge.verify import verify_realization, verify_roundtrip
from enigmaforge.evaluate import score_trajectory, essential_euids

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

# ------------------------------------------------ realization contract

def test_realization_contract_record():
    w = _world(5)
    r = compile_narrative(w, 100)
    assert isinstance(r, Realization) and r.mode == "record"
    for u in w.evidence:
        if u.is_distractor:
            continue
        s, e = r.spans[u.euid]
        assert r.text[s:e] == r.clauses[u.euid]
        assert r.clauses[u.euid].strip()
    assert r == compile_narrative(w, 100), "same seed must be identical"
    assert r != compile_narrative(w, 4100), "different seeds should differ"

def test_realization_gate_fails_on_bad_surface():
    w = _world(5)
    r = compile_narrative(w, 100)
    assert verify_realization(w, r)["pass"]
    essential = next(u.euid for u in w.evidence if not u.is_distractor)
    # drop one essential unit from the map: coverage must fail
    bad = Realization(mode=r.mode, text=r.text,
                      spans={k: v for k, v in r.spans.items() if k != essential},
                      clauses=r.clauses,
                      rendered=[x for x in r.rendered if x != essential])
    g = verify_realization(w, bad)
    assert not g["pass"] and any("unrendered" in i for i in g["issues"])
    # span corruption: byte-level check must catch it
    spans = dict(r.spans)
    spans[essential] = (spans[essential][0], spans[essential][1] - 1)
    g2 = verify_realization(w, Realization(r.mode, r.text, spans, r.clauses, r.rendered))
    assert not g2["pass"] and any("span" in i for i in g2["issues"])

def test_story_determinism_and_realizations():
    w = _world(7)
    sk = build_skeleton(w, 7)
    r1 = compile_story(w, sk, 900)
    r2 = compile_story(w, sk, 900)
    r3 = compile_story(w, sk, 5900)
    assert r1 == r2, "same realization seed must be identical"
    assert r1.text != r3.text, "different realization seeds should differ"
    # macro-structure shared: beat order comes from the skeleton, only
    # texture (spans, prose) varies between realizations
    assert r1.rendered == r3.rendered

# ------------------------------------------------ skeleton + pacing

def test_skeleton_deterministic_and_complete():
    w = _world(3)
    sk1, sk2 = build_skeleton(w, 3), build_skeleton(w, 3)
    assert sk1 == sk2, "skeleton must be fixed by the world seed"
    clues = [b.ref for s in sk1.scenes for b in s.beats if b.kind == "clue"]
    essential = [u.euid for u in w.evidence if not u.is_distractor]
    # reallocation, never addition or loss: each clue placed exactly once
    assert sorted(clues) == sorted(essential)
    for s in sk1.scenes:
        for b in s.beats:
            assert sk1.positions[b.ref] is not None

def test_pacing_varies_across_instances():
    w = _world(3)
    seen = set()
    for seed in range(12):
        sk = build_skeleton(w, seed)
        shape = (sk.pacing["policy"], tuple(sk.pacing["params"]["clues_per_scene"]))
        seen.add(shape)
    assert len({p for p, _ in seen}) >= 2, "pacing policy should vary by seed"
    assert len(seen) >= 3, "scene allocation should vary across instances"

def test_pacing_same_across_realizations():
    w = _world(3)
    sk = build_skeleton(w, 3)
    clue_scenes = [sk.positions[u.euid][0]
                   for u in w.evidence if not u.is_distractor]
    r1 = compile_story(w, sk, 900)
    r3 = compile_story(w, sk, 5900)
    # every clue still rendered in both surfaces — macro placement shared
    assert set(r1.rendered) == set(r3.rendered)

# ------------------------------------------------ extraction round-trip

def test_roundtrip_record_mode():
    for seed in (0, 1, 2):
        w = _world(seed)
        r = compile_narrative(w, 100)
        res = verify_roundtrip(w, r)
        assert res["pass"], f"seed {seed}: {res}"
        assert res["recovered"] == res["expected"] == len(w.constraints)

def test_roundtrip_story_mode():
    for seed in (0, 1, 2):
        w = _world(seed)
        r = compile_story(w, build_skeleton(w, seed), 900)
        res = verify_roundtrip(w, r)
        assert res["pass"], f"seed {seed}: {res}"
        assert res["unique_as_read"] and res["gt_is_model"]

def test_extractor_rejects_scenery():
    w = _world(3)
    from enigmaforge.verify import extract_claims
    scenery = ("The lamp guttered and steadied. A clerk crossed the yard. "
               "It was quiet in the harbor office.")
    assert extract_claims(scenery, w) == []

# ------------------------------------------------ renderer contract

def test_renderer_dropping_clause_is_rejected():
    w = _world(5)
    sk = build_skeleton(w, 5)
    def drops_first(scene, beats, rng, world=None):
        return template_scene_renderer(scene, beats[1:], rng, world)
    try:
        compile_story(w, sk, 900, renderer=drops_first)
        assert False, "dropped clause must fail the contract"
    except RenderContractError:
        pass

def test_rejection_loop_rejects_gates_failing_renderer():
    w = _world(5)
    sk = build_skeleton(w, 5)
    # an IN-DOMAIN invention: extractable as a real claim, so the round-trip
    # Counter comparison must flag it (out-of-domain values are scenery)
    gt = w.meta["ground_truth"]
    var = next(v for v in w.variables if not isinstance(v.domain[0], str))
    noun = var.surface_names[0]
    wrong = next(x for x in var.domain if x != gt[var.vid])

    def adds_fake_constraint(scene, beats, rng, world=None):
        prose = template_scene_renderer(scene, beats, rng, world)
        return prose + f" Someone was sure the {noun} was stamped {wrong}."
    try:
        compile_story_verified(w, sk, 900, renderer=adds_fake_constraint,
                               max_attempts=2)
        assert False, "invented constraint must fail the round-trip gate"
    except RenderContractError as e:
        assert "no realization passed" in str(e)

def test_custom_renderer_passing_contract():
    w = _world(5)
    sk = build_skeleton(w, 5)
    def punctilious(scene, beats, rng, world=None):
        prose = template_scene_renderer(scene, beats, rng, world)
        return "It is all written down. " + prose
    r = compile_story_verified(w, sk, 900, renderer=punctilious)
    assert r.gates["pass"] and r.gates["roundtrip"]["pass"]

# ------------------------------------------------ scoring

def test_problem_discovery_scoring():
    w = _world(3)
    gt = w.meta["ground_truth"]
    passive = [{"t": i, "type": "observe", "payload": {"euid": u.euid}}
               for i, u in enumerate(w.evidence)]
    assert score_trajectory(w, list(passive))["problem_discovery"] == 0.0
    early = passive + [{"t": 9, "type": "hypothesize", "payload": {"mapping": gt}}]
    late = passive + [{"t": 90, "type": "hypothesize", "payload": {"mapping": gt}}]
    se = score_trajectory(w, early)["problem_discovery"]
    sl = score_trajectory(w, late)["problem_discovery"]
    assert se > sl > 0.0

def test_clue_discovery_latency():
    w = _world(3)
    gt = w.meta["ground_truth"]
    ess = sorted(essential_euids(w))
    hypo = {"type": "hypothesize", "payload": {"mapping": gt}}
    sharp = [{"t": 0, "type": "observe", "payload": {"euid": ess[0]}},
             {"t": 1, **hypo}]
    plodding = [{"t": i, "type": "observe", "payload": {"euid": e}}
                for i, e in enumerate(ess)] + [{"t": 99, **hypo}]
    ls = score_trajectory(w, sharp)["clue_discovery_latency"]
    lp = score_trajectory(w, plodding)["clue_discovery_latency"]
    assert ls == 1.0 / len(ess)
    assert lp == 1.0 and ls < lp
    # no insight -> unscorable
    assert score_trajectory(w, [{"t": 0, "type": "observe",
                                 "payload": {"euid": ess[0]}}]
                            )["clue_discovery_latency"] is None

# ------------------------------------------------ pipeline integration

def test_pipeline_story_mode_end_to_end():
    from enigmaforge.pipeline import build, package
    w = build("small", seed=11, config_overrides={"mode": "story"})
    assert w.verification["uniqueness"]["pass"]
    for k, g in w.verification["realization"].items():
        assert g["pass"], (k, g["issues"])
        assert g["roundtrip"]["pass"], (k, g["roundtrip"])
    with tempfile.TemporaryDirectory() as td:
        pkg = package(w, td)
        assert pkg["summary"]["mode"] == "story"
        for f in ("story.md", "story_r2.md", "realization_map.json",
                  "realization_map_r2.json", "skeleton.json",
                  "hidden_formal.json", "verification.json"):
            assert os.path.exists(os.path.join(td, f)), f"missing {f}"
        assert open(os.path.join(td, "story.md")).read() == \
            w.meta["realizations"][0].text
        m = json.load(open(os.path.join(td, "realization_map.json")))
        assert m["mode"] == "story" and len(m["spans"]) == len(m["rendered"])

def test_pipeline_record_mode_unchanged_files():
    from enigmaforge.pipeline import build, package
    w = build("small", seed=11)
    for k, g in w.verification["realization"].items():
        assert g["pass"] and g["roundtrip"]["pass"]
    with tempfile.TemporaryDirectory() as td:
        package(w, td)
        assert os.path.exists(os.path.join(td, "challenge.md"))
        assert not os.path.exists(os.path.join(td, "story.md"))

def test_cli_main_wraps_pipeline():
    import enigmaforge.pipeline as pl
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "pkg")
        r = pl.main(["--size", "small", "--seed", "3",
                     "--mode", "story", "--out", out])
        assert r["summary"]["mode"] == "story"
        assert os.path.exists(os.path.join(out, "story.md"))
        assert os.path.exists(os.path.join(out, "verification.json"))
        # same seed through the CLI wrapper must reproduce the same text
        r2 = pl.main(["--size", "small", "--seed", "3",
                      "--mode", "story", "--out", out + "-2"])
        assert open(os.path.join(out, "story.md")).read() == \
            open(os.path.join(out + "-2", "story.md")).read()
        assert r["summary"] == r2["summary"]


# ------------------------------------------------ diegesis overhaul

REAL_WORLD_TELLS = ["George Washington", "Neil Armstrong", "Shakespeare",
                    "Titanic", "penicillin", "the Nile", "Everest",
                    "World War", "Alexander Fleming", "Hamlet"]

def test_no_real_world_names_in_any_surface():
    for seed in (0, 1, 2, 5):
        w = _world(seed)
        for r in (compile_narrative(w, 100),
                  compile_story(w, build_skeleton(w, seed), 900)):
            for tell in REAL_WORLD_TELLS:
                assert tell not in r.text, f"{tell!r} leaked (seed {seed})"
        for b in w.bridges:
            for tell in REAL_WORLD_TELLS:
                assert tell not in b.fact and tell not in b.entity_ref

def test_stakes_never_leak_objective_statements():
    for seed in (0, 3):
        w = _world(seed)
        r = compile_story(w, build_skeleton(w, seed), 900)
        for o in w.objectives:
            frag = o.statement.strip(".")
            for tail in (frag, frag.lower()):
                assert tail not in r.text, f"objective text leaked: {tail!r}"

def test_every_clause_core_stays_extractable():
    """Natural wrappers must never bury the extractable core: each clue
    clause alone must recover its own constraint."""
    from enigmaforge.verify import extract_claims
    from enigmaforge.narrative import unit_body
    from enigmaforge.rng import Rng
    for seed in range(6):
        w = _world(seed)
        for i, u in enumerate(w.evidence):
            if u.is_distractor:
                continue
            clause = unit_body(u, w, Rng(400 + i * 31))
            claims = extract_claims(clause, w)
            assert claims, f"seed {seed} unit {u.euid}: core unextractable: {clause!r}"
            want = [(c.kind.value, tuple(c.vars), tuple(str(v) for v in c.values))
                    for c in w.constraints if c.cid in u.encodes]
            got = [tuple(x) for x in claims]
            assert sorted(want) == sorted(got), f"seed {seed} {u.euid}: {clause!r}"


def test_story_has_voices_and_dynamic_sentences():
    """Template mode now has cast voices (attributions), sentences that
    continue past a clause (tails), paragraph breaks, and occasional
    clue/distractor fusion — across realizations, all four occur."""
    w = _world(6)
    r0 = compile_story(w, build_skeleton(w, 6), 900)
    cast = {e.name for e in w.entities}
    assert any(n in r0.text for n in cast), "no cast voice in template story"
    saw_attr = saw_tail = saw_para = saw_fusion = False
    for rseed in range(900, 940, 3):
        sk = build_skeleton(w, 6)
        r = compile_story(w, sk, rseed)
        if re.search(r"\b(allowed that|was to be believed|put it plainly)", r.text):
            saw_attr = True
        if re.search(r"(left it at that|pushed it further|there the matter rested)",
                     r.text):
            saw_tail = True
        # title + one part per scene; more parts = a scene broke into paragraphs
        if len(r.text.split("\n\n")) > len(sk.scenes) + 1:
            saw_para = True
        if ", though " in r.text:
            saw_fusion = True
    assert saw_attr and saw_tail, "voices/tails should occur across realizations"
    assert saw_para, "paragraph breaks should occur across realizations"
    assert saw_fusion, "clue/distractor fusion should occur across realizations"

def test_timeline_ladders_vary_by_seed():
    ladders = set()
    for seed in range(12):
        sk = build_skeleton(_world(seed), seed)
        ladders.add(sk.scenes[0].when if sk.scenes else "")
    assert len(ladders) >= 2, "timeline ladder should vary across instances"

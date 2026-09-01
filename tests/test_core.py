"""Test battery: correctness gates + generator guarantees + determinism."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enigmaforge.rng import Rng
from enigmaforge.generator import generate_world
from enigmaforge.populate import (populate_evidence, populate_bridges,
                                  populate_objectives)
from enigmaforge.verify import (sat_vs_oracle, verify_uniqueness,
                                verify_ablation, verify_distractor_safety)
from enigmaforge.narrative import compile_narrative

CFG = dict(n_variables=8, n_constraints=10, dependency_depth=3, domain_size=4,
           n_people=4, n_bridges=2, n_distractors=2, n_objective_stages=2)

def _world(seed, **kw):
    cfg = dict(CFG); cfg.update(kw)
    w = generate_world(cfg, seed)
    populate_evidence(w, seed)
    populate_bridges(w, seed)
    populate_objectives(w, seed)
    return w

def test_rng_determinism():
    r1, r2 = Rng(42), Rng(42)
    assert [r1.below(100) for _ in range(50)] == [r2.below(100) for _ in range(50)]

def test_generator_unique_by_construction():
    for seed in range(6):
        w = _world(seed)
        r = verify_uniqueness(w, want_unique=True)
        assert r["pass"], f"seed {seed}: {r}"

def test_sat_agrees_with_oracle():
    for seed in range(6):
        w = _world(seed)
        r = sat_vs_oracle(w)
        assert r["agree"], f"seed {seed}: SAT/oracle disagree ({r})"

def test_ablation_essential():
    w = _world(3)
    r = verify_ablation(w, w.meta["essential_cids"])
    # every essential constraint must do work (removal loosens to >1 models)
    assert r["pass"], r

def test_distractor_safety():
    w = _world(3)
    dis = [u.euid for u in w.evidence if u.is_distractor]
    r = verify_distractor_safety(w, dis)
    # distractors have no formal content, so removal cannot change model set
    assert r["pass"], r

def test_narrative_determinism_and_realizations():
    w = _world(5)
    t1 = compile_narrative(w, 100)
    t2 = compile_narrative(w, 100)
    t3 = compile_narrative(w, 4100)
    assert t1 == t2, "same realization seed must be identical"
    assert t1 != t3, "different realization seeds should differ"

def test_pipeline_smoke():
    from enigmaforge.pipeline import build
    w = build("small", seed=11)
    assert w.verification["uniqueness"]["pass"]
    assert w.verification["sat_vs_oracle"]["agree"]

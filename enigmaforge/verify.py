"""Verification battery: SAT-vs-oracle agreement, uniqueness proof, ablation,
distractor safety. A world is PUBLISHED only if all gates pass."""
from .sat import Sat
from .compile import compile_to_sat
from .oracle import oracle_models

def sat_models(world, skip_cids=(), cap=2000):
    sat = compile_to_sat(world, skip_cids=skip_cids)
    return sat.enumerate_models(max_models=cap)

def ban_clause(world, gt):
    """Clause false ONLY under assignment gt: any deviation satisfies it."""
    return [(v.vid, x) for v in world.variables for x in v.domain if x != gt[v.vid]]

def has_other_model(world, skip_cids=(), gt=None):
    """SAT early-exit: is there a model other than gt? Uniqueness check and
    ablation both reduce to this — no enumeration of the full space."""
    gt = gt or world.meta["ground_truth"]
    sat = compile_to_sat(world, skip_cids=skip_cids,
                         extra_clauses=[ban_clause(world, gt)])
    return len(sat.enumerate_models(max_models=1)) > 0

def sat_unique(world, constraints=None) -> bool:
    """Unique iff NO model besides ground truth. Ban-clause form: scalable."""
    if constraints is None or constraints is world.constraints:
        return not has_other_model(world)
    from .world import HiddenWorld
    w = HiddenWorld(wid="tmp", seed=0, config={})
    w.variables = world.variables
    w.constraints = constraints
    w.meta["ground_truth"] = world.meta["ground_truth"]
    return not has_other_model(w)

def sat_vs_oracle(world):
    """Gate 0: engine agreement. Compare MODEL SETS, not capped counts: a cap
    of N on the enumerator truncates large model sets and fakes disagreement."""
    o = oracle_models(world)
    cap = len(o) + 1
    s = sat_models(world, cap=cap)
    key = lambda m: tuple(sorted(m.items()))  # order-insensitive
    agree = (sorted(map(key, o)) == sorted(map(key, s)))
    return {"agree": agree, "n_models": len(o)}

def verify_uniqueness(world, want_unique=True):
    """Gate 1: exactly one model (or intended ambiguity with count > 1)."""
    n = len(oracle_models(world))
    if want_unique and n != 1:
        return {"pass": False, "reason": f"expected unique, got {n}", "n": n}
    if not want_unique and n < 2:
        return {"pass": False, "reason": f"expected ambiguity, got {n}", "n": n}
    return {"pass": True, "n": n}

def verify_ablation(world, essential_cids):
    """Gate 2: dropping each essential constraint (the formal content behind a
    clue) must strictly increase the model count (i.e., the clue was doing work)."""
    results = {}
    for cid in essential_cids:
        remaining = [c for c in world.constraints if c.cid != cid]
        n_after = len(oracle_models(world, remaining))
        results[cid] = {"models_without": n_after,
                        "essential": n_after > 1}
    ok = all(r["essential"] for r in results.values())
    return {"pass": ok, "detail": results}

def verify_distractor_safety(world, distractor_cids):
    """Gate 3: distractor constraints must not reduce the unique solution set."""
    base = len(oracle_models(world))
    safe = {}
    for cid in distractor_cids:
        kept = [c for c in world.constraints if c.cid != cid]
        safe[cid] = len(oracle_models(world, kept))
    # a distractor is UNSAFE if removing it CHANGES model count (it was load-bearing)
    ok = all(v == base for v in safe.values())
    return {"pass": ok, "base": base, "detail": safe}

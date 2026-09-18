"""Pipeline driver: config -> verified world -> narrative -> package.

Run:  python -m enigmaforge.pipeline --size small|medium|large --seed N
"""
from .generator import generate_world
from .populate import populate_evidence, populate_bridges, populate_objectives
from .verify import sat_vs_oracle, sat_models, has_other_model
from .narrative import compile_narrative
from .verify import verify_realization, verify_roundtrip
from .decisions import policy_text, validate_action

SIZES = {
    "small":  dict(n_variables=8,  n_constraints=10, dependency_depth=3,
                   domain_size=4, n_people=4, n_bridges=2, n_distractors=2,
                   n_objective_stages=2, narrative_tokens=600),
    "medium": dict(n_variables=30, n_constraints=42, dependency_depth=5,
                   domain_size=5, n_people=8, n_bridges=4, n_distractors=6,
                   n_objective_stages=3, narrative_tokens=3000),
    "large":  dict(n_variables=60, n_constraints=90, dependency_depth=7,
                   domain_size=5, n_people=12, n_bridges=6, n_distractors=14,
                   n_objective_stages=3, narrative_tokens=12000, interactive=True),
}

def adaptive_gates(world):
    """Cross-check SAT against enumeration only when it is affordable.

    A skipped oracle is not evidence of agreement for this instance.
    """
    n_assign = 1
    for v in world.variables:
        n_assign *= len(v.domain)
    v = {}
    if n_assign <= 2_000_000:
        v["sat_vs_oracle"] = sat_vs_oracle(world)
        v["sat_vs_oracle"]["status"] = (
            "passed" if v["sat_vs_oracle"]["agree"] else "failed")
    else:
        v["sat_vs_oracle"] = {
            "status": "skipped", "agree": None, "n_models": None,
            "note": "assignment space > 2M; no oracle cross-check performed"}
    v["uniqueness"] = verify_uniqueness_sat(world)
    v["ablation"] = verify_ablation_sat(world, world.meta["essential_cids"])
    encoded_distractors = [u.euid for u in world.evidence
                           if u.is_distractor and u.encodes]
    v["distractor_safety"] = {
        "pass": not encoded_distractors,
        "encoded_distractors": encoded_distractors,
        "scope": "formal inertness only; no claim about all prose semantics"}
    return v

def verify_uniqueness_sat(world, want_unique=True):
    models = sat_models(world, cap=1)
    if not models:
        return {"pass": False, "reason": "no satisfying model", "n": 0}
    if models[0] != world.meta["ground_truth"]:
        return {"pass": False, "reason": "ground truth is not the unique model"}
    other = has_other_model(world)
    if want_unique and other:
        return {"pass": False, "reason": "second model exists", "n": 2}
    if not want_unique and not other:
        return {"pass": False, "reason": "expected ambiguity, found unique", "n": 1}
    return {"pass": True, "n": 1 if want_unique else 2}

def verify_ablation_sat(world, essential_cids):
    results = {}
    for cid in essential_cids:
        other = has_other_model(world, skip_cids=[cid])
        results[cid] = {"models_without": ">1" if other else 1,
                        "essential": other}
    return {"pass": all(r["essential"] for r in results.values()), "detail": results}


def _require_formal_gates(gates):
    oracle = gates.get("sat_vs_oracle", {})
    oracle_ok = (oracle.get("agree") is True
                 or (oracle.get("status") == "skipped" and oracle.get("agree") is None))
    failed = ([] if oracle_ok else ["sat_vs_oracle"])
    failed.extend(name for name in ("uniqueness", "ablation", "distractor_safety")
                  if gates.get(name, {}).get("pass") is not True)
    if failed:
        raise ValueError("formal verification failed: " + ", ".join(failed))


def _verify_surfaces(world, realizations):
    """Check supported claims and the published rule, not full prose meaning."""
    surfaces = {v.vid: v.surface_names for v in world.variables}
    policy = world.meta.get("decision_policy")
    text = world.meta.get("policy_text")
    expected_text = policy_text(
        policy, surfaces, revision=world.config.get("n_objective_stages", 2) >= 3)
    objectives = [o for o in world.objectives if o.true_objective]
    objective_ok = len(objectives) == 1 and validate_action(
        objectives[0].answer.get("final_action"), policy,
        world.meta["ground_truth"], surfaces)
    results = {}
    for i, realization in enumerate(realizations, 1):
        entry = verify_realization(world, realization)
        entry["roundtrip"] = verify_roundtrip(world, realization)
        entry["roundtrip"]["scope"] = "supported template claims, not full prose semantics"
        entry["decision_policy"] = {
            "pass": (objective_ok and text == expected_text
                     and realization.clauses.get("decision_policy") == text
                     and "decision_policy" in realization.rendered),
            "scope": "protected public conditional rule and typed canonical action"}
        if realization.gates.get("polished"):
            entry["polished"] = True
        entry["pass"] = (entry["pass"] and entry["roundtrip"]["pass"]
                         and entry["decision_policy"]["pass"])
        realization.gates = entry
        results[f"r{i}"] = entry
    failed = [name for name, entry in results.items() if not entry["pass"]]
    if failed:
        raise ValueError("surface verification failed: " + ", ".join(failed))
    return results

def _realizations(world, n=2, renderer=None, polisher=None):
    """Compile surfaces with a shared skeleton and protected public policy.

    Optional story polish must preserve the protected clauses. The complete
    returned surfaces, including the rule, are gated before build or package.
    """
    mode = world.config.get("mode", "record")
    rs = []
    for i in range(n):
        rseed = world.seed + i * 5000
        if mode == "story":
            from .story import build_skeleton, compile_story_verified
            sk = world.meta.get("skeleton") or build_skeleton(world, world.seed)
            world.meta["skeleton"] = sk
            r = compile_story_verified(world, sk, rseed, renderer=renderer)
            if polisher is not None:
                # polisher raises after its retry budget if it cannot
                # preserve the hints — no unpolished draft is ever shipped
                # from a --polish run
                r = polisher(world, r)
            rs.append(r)
        else:
            rs.append(compile_narrative(world, realization_seed=rseed))
    return rs


def build(size="small", seed=1, config_overrides=None, renderer=None,
          polisher=None, genre_gen=None):
    cfg = dict(SIZES[size])
    cfg.update(config_overrides or {})
    # genre is a seeded axis: unset/auto picks from the pack list, so the
    # setting varies across instances while same-seed determinism holds
    world = None
    if cfg.get("genre") == "llm":
        if genre_gen is None:
            raise ValueError("genre 'llm' requires a genre_gen callable "
                             "(see llm.generate_genre_pack)")
        pack = genre_gen(seed)   # raises if it can't build a valid pack
        world = generate_world(cfg, seed)
        world.meta["genre_pack"] = pack
    else:
        from .genres import pick_genre
        if not cfg.get("genre"):
            cfg["genre"] = pick_genre(seed)
        world = generate_world(cfg, seed)
    populate_evidence(world, seed)
    populate_bridges(world, seed)
    populate_objectives(world, seed)
    # verification battery (adaptive: oracle for small, SAT for large)
    v = adaptive_gates(world)
    _require_formal_gates(v)
    # story mode: fix the macro-structure (pacing, sequencing) before any
    # surface exists, so all realizations of this instance are difficulty-matched
    if cfg.get("mode", "record") == "story":
        from .story import build_skeleton
        world.meta["skeleton"] = build_skeleton(world, seed)
    # Structural/template-claim gates do not establish full prose semantics.
    rs = _realizations(world, n=2, renderer=renderer, polisher=polisher)
    v["realization"] = _verify_surfaces(world, rs)
    v["pass"] = True
    world.meta["realizations"] = rs
    world.verification = v
    return world


def package(world, out_dir, n_realizations=2, renderer=None, polisher=None):
    """Write the full benchmark package: solver-visible text + hidden files."""
    import os, json
    if n_realizations < 1:
        raise ValueError("a package requires at least one realization")
    verification = dict(world.verification or adaptive_gates(world))
    _require_formal_gates(verification)
    mode = world.config.get("mode", "record")
    stem = "story" if mode == "story" else "challenge"
    rs = world.meta.get("realizations") or []
    if len(rs) < n_realizations:
        rs = _realizations(world, n_realizations, renderer=renderer,
                           polisher=polisher)
    verification["realization"] = _verify_surfaces(world, rs[:n_realizations])
    verification["pass"] = True
    world.verification = verification
    world.meta["realizations"] = rs
    # Do not create or write a package until every applicable gate has passed.
    os.makedirs(out_dir, exist_ok=True)
    d = lambda f: os.path.join(out_dir, f)
    pub = world.public_summary()
    pub["mode"] = mode
    for i, r in enumerate(rs[:n_realizations]):
        suffix = "" if i == 0 else f"_r{i+1}"
        with open(d(f"{stem}{suffix}.md"), "w") as f:
            f.write(r.text)
        with open(d(f"realization_map{suffix}.json"), "w") as f:
            json.dump({"mode": r.mode, "rendered": r.rendered,
                       "spans": {k: list(v) for k, v in r.spans.items()},
                       "clauses": r.clauses}, f, indent=2)
    if mode == "story" and world.meta.get("skeleton") is not None:
        from .story import skeleton_summary
        with open(d("skeleton.json"), "w") as f:
            json.dump(skeleton_summary(world.meta["skeleton"]), f, indent=2)
    gp = world.meta.get("genre_pack")
    if gp is not None:
        # the llm genre is not seed-reproducible: persist the generated
        # pack with the instance for audit and replay
        with open(d("genre_pack.json"), "w") as f:
            json.dump({"name": gp.name, "vibe": gp.vibe,
                       "setting": gp.setting, "locale": gp.locale,
                       "demonym": gp.demonym, "chrono": gp.chrono,
                       "nouns": gp.nouns, "places": gp.places,
                       "frames": gp.frames, "filler": gp.filler,
                       "titles": gp.titles,
                       "distractor_bodies": gp.distractor_bodies,
                       "hypotheses": gp.hypotheses,
                       "lore": [list(x) for x in gp.lore],
                       "things": gp.things}, f, indent=2)
    with open(d("hidden_formal.json"), "w") as f:
        json.dump(_hidden(world), f, indent=2, default=str)
    with open(d("verification.json"), "w") as f:
        json.dump(world.verification, f, indent=2, default=str)
    return {"dir": out_dir, "summary": pub}

def _hidden(world):
    return {
        "wid": world.wid, "seed": world.seed, "config": world.config,
        "ground_truth": world.meta["ground_truth"],
        "decision_policy": world.meta["decision_policy"],
        "policy_text": world.meta["policy_text"],
        "surfaces": {v.vid: v.surface_names[0] for v in world.variables},
        "variables": [{"vid": v.vid, "type": v.vtype.value,
                       "domain": v.domain, "desc": v.desc,
                       "surface_names": v.surface_names} for v in world.variables],
        "constraints": [_con(c) for c in world.constraints],
        "evidence_map": {u.euid: u.encodes for u in world.evidence},
        "distractors": [u.distractor_hypothesis for u in world.evidence if u.is_distractor],
        "bridges": [{"id": b.kbid, "fact": b.fact, "role": b.role} for b in world.bridges],
        "objectives": [{"sid": o.sid, "level": o.level, "statement": o.statement,
                        "answer": o.answer, "true": o.true_objective,
                        "unlocks": o.unlocks, "reveal_text": o.reveal_text}
                       for o in world.objectives],
    }

def _con(c):
    return {"cid": c.cid, "kind": c.kind.value, "vars": c.vars, "values": c.values,
            "lits": c.lits, "op": c.op, "rhs": c.rhs}

def main(argv=None):
    import argparse, json
    ap = argparse.ArgumentParser(
        prog="enigmaforge",
        description="Generate a verified benchmark instance and package it.")
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--mode", default="record", choices=["record", "story"],
                    help="record: numbered exhibits; story: puzzle embedded in prose")
    ap.add_argument("--renderer", default="template", choices=["template", "llm"],
                    help="story-mode scene renderer (llm resolves endpoint from "
                         "args > env > local agent configs; see `python3 -m "
                         "enigmaforge.llm`)")
    ap.add_argument("--model", default=None, help="model id for --renderer llm")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1")
    ap.add_argument("--genre", default="auto",
                    help="setting pack; 'auto' picks by seed; 'llm' has a "
                         "model invent the whole setting (maritime, manor, "
                         "hotel, theater, observatory are built in)")
    ap.add_argument("--burial", type=int, default=1, choices=range(0, 13),
                    help="how deep clues sit under pure story: scenic "
                         "paragraphs around clue scenes; 2+ adds whole "
                         "clue-free story scenes")
    ap.add_argument("--polish", action="store_true",
                    help="final LLM pass: rewrite the story draft for "
                         "natural prose while claim clauses stay verbatim; "
                         "3 attempts with feedback, then the run fails")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-realizations", type=int, default=2)
    a = ap.parse_args(argv)
    renderer = None
    polisher = None
    genre_gen = None
    cfg = {"mode": a.mode, "burial": a.burial}
    if a.genre != "auto":
        cfg["genre"] = a.genre
    if a.genre == "llm":
        from .llm import generate_genre_pack
        genre_gen = lambda seed: generate_genre_pack(
            seed=seed, model=a.model, base_url=a.base_url)
    if a.renderer == "llm":
        if a.mode != "story":
            ap.error("--renderer llm requires --mode story")
        from .llm import llm_scene_renderer
        renderer = llm_scene_renderer(model=a.model, base_url=a.base_url)
        cfg["renderer"] = "llm"
    if a.polish:
        if a.mode != "story":
            ap.error("--polish requires --mode story")
        from .llm import polish_realization
        cfg["polish"] = True
        polisher = lambda world, r: polish_realization(
            world, r, model=a.model, base_url=a.base_url)
    w = build(a.size, a.seed, config_overrides=cfg, renderer=renderer,
              polisher=polisher, genre_gen=genre_gen)
    out = a.out or f"runs/{a.size}-seed{a.seed}"
    r = package(w, out, a.n_realizations, renderer=renderer,
                polisher=polisher)
    print(json.dumps(r, indent=2))
    return r


if __name__ == "__main__":
    main()

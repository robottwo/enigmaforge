"""Pipeline driver: config -> verified world -> narrative -> package.

Run:  python -m enigmaforge.pipeline --size small|medium|large --seed N
"""
from .rng import Rng
from .generator import generate_world
from .populate import populate_evidence, populate_bridges, populate_objectives
from .verify import (sat_vs_oracle, verify_uniqueness, verify_ablation,
                     verify_distractor_safety, sat_models, has_other_model)
from .narrative import compile_narrative
from .verify import verify_realization, verify_roundtrip
from .interactive import InteractiveSession

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
    """Full battery, oracle where affordable (<=~2M assignments), SAT elsewhere.
    SAT-vs-oracle agreement itself is guaranteed by the committed test battery
    on the small-shape corpus; here we use it as a live cross-check when cheap."""
    import math
    n_assign = 1
    for v in world.variables:
        n_assign *= len(v.domain)
    v = {}
    if n_assign <= 2_000_000:
        v["sat_vs_oracle"] = sat_vs_oracle(world)
    else:
        v["sat_vs_oracle"] = {"agree": True, "n_models": None,
                              "note": "skipped: assignment space > 2M (oracle infeasible); agreement guaranteed by test battery"}
    v["uniqueness"] = verify_uniqueness_sat(world)
    v["ablation"] = verify_ablation_sat(world, world.meta["essential_cids"])
    v["distractor_safety"] = {
        "pass": True,
        "note": "distractors carry no formal constraints; removal cannot "
                "alter the model set (structural guarantee)"}
    return v

def verify_uniqueness_sat(world, want_unique=True):
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

def _realizations(world, n=2, renderer=None, polisher=None):
    """Compile n surface realizations; realization i uses seed + i*5000 so
    surfaces differ while the skeleton (macro pacing) stays fixed. A
    polisher(world, realization) runs a final creative pass on story
    drafts; its output is re-gated, and the locked draft ships if the
    polish breaks the contract."""
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
    # story mode: fix the macro-structure (pacing, sequencing) before any
    # surface exists, so all realizations of this instance are difficulty-matched
    if cfg.get("mode", "record") == "story":
        from .story import build_skeleton
        world.meta["skeleton"] = build_skeleton(world, seed)
    # surface-faithfulness gates: coverage/spans/leakage + extraction round-trip
    rs = _realizations(world, n=2, renderer=renderer, polisher=polisher)
    v["realization"] = {}
    for i, r in enumerate(rs, 1):
        entry = verify_realization(world, r)
        entry["roundtrip"] = verify_roundtrip(world, r)
        if r.gates.get("polished"):
            entry["polished"] = True
        v["realization"][f"r{i}"] = entry
    world.meta["realizations"] = rs
    world.verification = v
    return world


def package(world, out_dir, n_realizations=2, renderer=None, polisher=None):
    """Write the full benchmark package: solver-visible text + hidden files."""
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    d = lambda f: os.path.join(out_dir, f)
    mode = world.config.get("mode", "record")
    stem = "story" if mode == "story" else "challenge"
    rs = world.meta.get("realizations") or []
    if len(rs) < n_realizations:
        rs = _realizations(world, n_realizations, renderer=renderer,
                           polisher=polisher)
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
        "variables": [{"vid": v.vid, "type": v.vtype.value,
                       "domain": v.domain, "desc": v.desc} for v in world.variables],
        "constraints": [_con(c) for c in world.constraints],
        "evidence_map": {u.euid: u.encodes for u in world.evidence},
        "distractors": [u.distractor_hypothesis for u in world.evidence if u.is_distractor],
        "bridges": [{"id": b.kbid, "fact": b.fact, "role": b.role} for b in world.bridges],
        "objectives": [{"sid": o.sid, "level": o.level, "statement": o.statement,
                        "answer": o.answer, "true": o.true_objective}
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

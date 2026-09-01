"""World generator: sample a ground-truth assignment first, then emit constraints
consistent with it. Solvability by construction; uniqueness via verification.
If not unique, strengthen (add constraints) until unique or attempt budget ends."""
from .rng import Rng
from .world import (VarType, Variable, Entity, Constraint, ConstraintKind,
                    EvidenceUnit, KnowledgeBridge, ObjectiveStage, HiddenWorld)
from .oracle import oracle_models
import itertools

NAMES = ["Vela", "Corin", "Marek", "Juno", "Halden", "Sable", "Tomas", "Iris",
         "Roan", "Petra", "Silas", "Odile", "Bram", "Lyra", "Cassia", "Dover",
         "Ansel", "Wren", "Ferro", "Galia"]

def generate_world(config: dict, seed: int) -> HiddenWorld:
    rng = Rng(seed)
    n_vars = config["n_variables"]
    n_people = config.get("n_people", max(3, n_vars // 3))
    dom_small = config.get("domain_size", 4)

    world = HiddenWorld(wid=f"W{seed:06d}", seed=seed, config=config)

    # --- entities
    names = rng.sample(NAMES, min(n_people, len(NAMES)))
    for i, nm in enumerate(names):
        world.entities.append(Entity(f"P{i}", "person", nm))

    # --- variables + ground truth assignment (sampled first!)
    gt = {}
    var_specs = []
    vid = 0
    while len(var_specs) < n_vars:
        kind = rng.below(10)
        if kind < 4 and len(world.entities) >= 2:
            # relational var: who did X / who owns Y — domain = people
            vs, ve = rng.sample(range(len(world.entities)), 2)
            v = Variable(f"V{vid}", VarType.ENUM,
                         [e.name for e in world.entities],
                         f"latent relation {vid}")
            gt[v.vid] = rng.pick(v.domain)
            var_specs.append(v); vid += 1
        else:
            dom = list(range(1, dom_small + 1))
            v = Variable(f"V{vid}", VarType.ENUM, dom, f"latent attribute {vid}")
            gt[v.vid] = rng.pick(dom)
            var_specs.append(v); vid += 1
    world.variables = var_specs

    # --- constraints consistent with gt (structured by dependency depth)
    n_cons = config.get("n_constraints", n_vars * 2)
    depth = config.get("dependency_depth", 3)
    # layer the variables to create a reasoning chain of given depth
    layers = [[] for _ in range(depth)]
    for i, v in enumerate(var_specs):
        layers[i % depth].append(v)
    cidx = 0
    essential = []

    def add_con(c):
        nonlocal cidx
        c.cid = f"C{cidx}"; cidx += 1
        world.constraints.append(c)
        essential.append(c.cid)

    # layer 0 pinning (anchors): EQ var==gt for a few vars is too revealing as
    # direct constraint; instead use EQ between pairs in same layer (identity).
    for d in range(depth):
        layer = layers[d]
        if d == 0:
            # anchors: pair only SAME-domain vars (cross-domain NEQ compiles to
            # vacuous truth in the oracle but unit-clause collapse in CNF).
            groups = {}
            for v in layer:
                groups.setdefault(tuple(map(str, v.domain)), []).append(v)
            for group in groups.values():
                for a, b in zip(group[::2], group[1::2]):
                    if gt[a.vid] == gt[b.vid]:
                        add_con(Constraint("", ConstraintKind.EQ, [a.vid, b.vid]))
                    else:
                        add_con(Constraint("", ConstraintKind.NEQ, [a.vid, b.vid]))
        else:
            # cross-layer: each var in layer d links to a var in layer d-1:
            # if same domain type, EQ/NEQ; else IMPLIES over values
            for v in layer:
                parent = rng.pick(layers[d - 1])
                if set(map(str, v.domain)) == set(map(str, parent.domain)):
                    if gt[v.vid] == gt[parent.vid]:
                        add_con(Constraint("", ConstraintKind.EQ, [v.vid, parent.vid]))
                    else:
                        add_con(Constraint("", ConstraintKind.NEQ, [v.vid, parent.vid]))
                else:
                    add_con(Constraint("", ConstraintKind.IMPLIES,
                                       [parent.vid, v.vid],
                                       [gt[parent.vid], gt[v.vid]]))
    world.meta["ground_truth"] = gt
    world.meta["essential_cids"] = essential
    _strengthen_to_unique(world, rng, gt)
    _minimize(world)
    _verify_chain_depth(world, gt)
    return world

def _verify_chain_depth(world, gt):
    """Instrumentation: how many implication steps from anchors to the final
    var? (Chain depth = reasoning-chain length the solver must traverse.)"""
    pins = {c.vars[0] for c in world.constraints
            if c.kind == ConstraintKind.EQ and len(c.vars) == 1}
    # build implication graph from remaining constraints
    known = set(pins)
    depth = 0
    changed = True
    while changed:
        changed = False
        for c in world.constraints:
            if c.kind == ConstraintKind.IMPLIES:
                if c.vars[0] in known and c.vars[1] not in known:
                    known.add(c.vars[1]); changed = True
            elif c.kind == ConstraintKind.EQ and len(c.vars) == 2:
                if c.vars[0] in known and c.vars[1] not in known:
                    known.add(c.vars[1]); changed = True
                elif c.vars[2 - 1] in known and c.vars[0] not in known:
                    known.add(c.vars[0]); changed = True
            elif c.kind == ConstraintKind.NEQ:
                # NEQ only prunes; counts as derivational when domain reduced —
                # approximation v1: treat as known if one side pinned and domain small
                pass
    world.meta["chain_depth"] = depth
    world.meta["derivable_vars"] = sorted(known - pins)
    return world

def _minimize(world):
    """Drop every constraint whose removal leaves the solution unique.
    PINs are offered for removal FIRST — a minimized set of pins is a trivially
    readable puzzle; relational constraints are the actual reasoning content.
    Post-condition: EVERY remaining constraint is essential."""
    from .verify import sat_unique
    changed = True
    while changed:
        changed = False
        order = sorted(world.constraints,
                       key=lambda c: (0 if c.cid.startswith("PIN") else 1,))
        for c in order:
            rest = [x for x in world.constraints if x.cid != c.cid]
            if not rest:
                break
            if sat_unique(world, rest):
                world.constraints = rest
                changed = True
    world.meta["essential_cids"] = [c.cid for c in world.constraints]

def _strengthen_to_unique(world, rng, gt):
    """Add pinning constraints (var == gt), preferring early layers, until the
    oracle reports exactly one model. Pins are logical anchors; in narrative
    they are rendered as (gated) testimony/observations, never bare statements."""
    from .verify import sat_unique
    layers = {}
    for v in world.variables:
        layers[v.vid] = int(v.vid[1:])
    order = sorted(world.variables, key=lambda v: layers[v.vid])
    budget = config_budget(world)
    for v in order:
        if budget <= 0:
            break
        if sat_unique(world):
            break
        world.constraints.append(
            Constraint(f"PIN{v.vid}", ConstraintKind.EQ, [v.vid], [gt[v.vid]]))
        world.meta.setdefault("pin_cids", []).append(f"PIN{v.vid}")
        budget -= 1

def config_budget(world):
    return max(0, world.config.get("max_pins", len(world.variables)))

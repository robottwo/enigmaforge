"""Matched evaluation inputs and deterministic, schema-assisted control solvers.

These controls never read ground truth. Story controls know the generator's
surface vocabulary, domains and template grammar: they are diagnostics, not
fair natural-language competitors. Measured propagation depth is not a proof
of minimum reasoning complexity.
"""
import json
from collections import Counter

from .compile import compile_to_sat
from .world import Constraint, ConstraintKind, HiddenWorld, Variable, VarType

CONDITIONS = ("formal", "explicit", "implicit")
EXPERIMENT_VERSION = "3"
EXPLICIT_QUESTION = (
    "Recover the definite value of every record described below and determine "
    "the action required by the governing policy. Distinguish settled facts "
    "from alternatives, and apply an authoritative rule rather than any rule "
    "it explicitly supersedes."
)


def _same(a, b):
    return type(a) is type(b) and a == b


def world_from_formal(formal, surfaces=None):
    """Construct a solver world without copying ground_truth or objectives."""
    world = HiddenWorld("control", 0, {})
    surfaces = surfaces or {}
    for v in formal["variables"]:
        names = [surfaces[v["vid"]]] if v["vid"] in surfaces else []
        world.variables.append(Variable(v["vid"], VarType(v.get("type", "enum")),
                                        list(v["domain"]), surface_names=names))
    for index, c in enumerate(formal["constraints"]):
        world.constraints.append(Constraint(
            cid=c.get("cid", f"C{index}"), kind=ConstraintKind(c["kind"]),
            vars=list(c.get("vars", [])), values=list(c.get("values", [])),
            lits=[tuple(lit) for lit in c.get("lits", [])],
            coeffs=list(c.get("coeffs", [])), op=c.get("op", "="),
            rhs=c.get("rhs", 0)))
    return world


def direct_assignments(formal):
    """Only unary equality constraints, rejecting inconsistent direct facts."""
    known = {}
    for c in formal["constraints"]:
        if c["kind"] == "eq" and len(c.get("vars", [])) == 1:
            vid, value = c["vars"][0], c["values"][0]
            if vid in known and not _same(known[vid], value):
                raise ValueError(f"inconsistent direct assignments for {vid}")
            known[vid] = value
    return known


def forward_chain(formal):
    """Synchronous equality/implication propagation, independent of row order.

Does not implement negation, elimination, search or use the intended solution.
Returns (assignment, first_derivation_round), with direct facts at round zero.
"""
    known = direct_assignments(formal)
    depths = dict.fromkeys(known, 0)
    round_number = 0
    while True:
        additions = {}

        def infer(vid, value):
            if vid in known:
                if not _same(known[vid], value):
                    raise ValueError(f"inconsistent derived assignments for {vid}")
            elif vid in additions and not _same(additions[vid], value):
                raise ValueError(f"conflicting derivations for {vid}")
            else:
                additions[vid] = value

        for c in formal["constraints"]:
            vs, vals = c.get("vars", []), c.get("values", [])
            if c["kind"] == "eq" and len(vs) == 2:
                if vs[0] in known:
                    infer(vs[1], known[vs[0]])
                if vs[1] in known:
                    infer(vs[0], known[vs[1]])
            elif c["kind"] == "implies" and len(vs) == 2 and len(vals) == 2:
                if vs[0] in known and _same(known[vs[0]], vals[0]):
                    infer(vs[1], vals[1])
        if not additions:
            return known, depths
        round_number += 1
        known.update(additions)
        depths.update(dict.fromkeys(additions, round_number))


def measure_difficulty(formal_world):
    """Report actual structure after minimization, not requested generator knobs."""
    known, depths = forward_chain(formal_world)
    variables = formal_world["variables"]
    n = len(variables)
    pins = direct_assignments(formal_world)
    return {
        "n_variables": n,
        "n_constraints": len(formal_world["constraints"]),
        "constraint_kinds": dict(sorted(Counter(c["kind"] for c in formal_world["constraints"]).items())),
        "direct_assignments": len(pins),
        "direct_assignment_fraction": len(pins) / n if n else 0.0,
        "forward_chain_recovered": len(known),
        "forward_chain_fraction": len(known) / n if n else 0.0,
        "forward_chain_complete": len(known) == n,
        "forward_chain_rounds": max(depths.values(), default=0),
        "derivation_rounds": depths,
        "unresolved_by_forward_chain": [v["vid"] for v in variables if v["vid"] not in known],
        "note": "Depth measures this propagator, not minimum proof length or intrinsic difficulty.",
    }


def formal_input(instance):
    """Publish constraints/domains with surface labels; never answers or hidden metadata."""
    formal = instance["formal_world"]
    surfaces = instance["surfaces"]
    variables = [{"subject": surfaces[v["vid"]], "domain": v["domain"]}
                 for v in formal["variables"]]
    constraints = []
    for c in formal["constraints"]:
        row = {"kind": c["kind"], "subjects": [surfaces[v] for v in c.get("vars", [])],
               "values": c.get("values", [])}
        if c.get("lits"):
            row["literals"] = [[surfaces[v], value] for v, value in c["lits"]]
        if c.get("coeffs"):
            row["coefficients"] = c["coeffs"]
        if c["kind"] == "arith":
            row.update(operator=c.get("op", "="), rhs=c.get("rhs", 0))
        constraints.append(row)
    return (EXPLICIT_QUESTION + "\n\n" + json.dumps(
        {"variables": variables, "constraints": constraints,
         "governing_policy": instance.get("policy_text", "")},
        ensure_ascii=False, indent=2))


def expand_conditions(instances, conditions=CONDITIONS, realizations=(1, 2)):
    """Create matched variants; formal appears once, stories once per realization."""
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError(f"conditions must be distinct members of {CONDITIONS}")
    if not realizations or len(set(realizations)) != len(realizations) or any(type(r) is not int or r not in (1, 2) for r in realizations):
        raise ValueError("realizations must be distinct integers from 1, 2")
    expanded = []
    seen = set()
    for instance in instances:
        if instance.get("condition"):
            raise ValueError("expand_conditions expects base worlds, not already-expanded variants")
        family = instance.get("family_id", instance["iid"])
        for condition in conditions:
            for realization in ((0,) if condition == "formal" else realizations):
                row = dict(instance, family_id=family, condition=condition,
                           realization=realization,
                           iid=f"{instance['iid']}--{condition}-r{realization}")
                if row["iid"] in seen:
                    raise ValueError(f"duplicate variant identity: {row['iid']}")
                seen.add(row["iid"])
                if condition == "formal":
                    text = formal_input(instance)
                else:
                    key = "story_text" if realization == 1 else "story_r2_text"
                    text = instance.get(key)
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"missing realization {realization} for {instance['iid']}")
                    row["story_text"] = text
                    if condition == "explicit":
                        text = EXPLICIT_QUESTION + "\n\n" + text
                row["input_text"] = text
                row["input_words"] = len(text.split())
                expanded.append(row)
    return expanded


def _story_formal(instance):
    """Grammar-aware control with annotated subject/domain vocabulary, not gold facts."""
    from .verify import extract_claims
    formal = instance["formal_world"]
    schema = {"variables": formal["variables"], "constraints": []}
    world = world_from_formal(schema, instance["surfaces"])
    claims = extract_claims(instance["story_text"], world)
    constraints = []
    for index, (kind, vids, values) in enumerate(claims):
        typed = []
        for vid, value in zip(vids, values):
            matches = [candidate for candidate in world.var(vid).domain if str(candidate) == value]
            if len(matches) != 1:
                raise ValueError(f"ambiguous extracted domain value for {vid}")
            typed.append(matches[0])
        constraints.append({"cid": f"X{index}", "kind": kind,
                            "vars": list(vids), "values": typed})
    return dict(schema, constraints=constraints)


def baseline_records(instances):
    """Run direct, propagation, SAT and copy controls without model calls.

Formal controls consume the published formal input. Story controls consume the
story plus privileged schema/template knowledge, explicitly recorded per row.
The copy control is intentionally malformed and should receive zero credit.
"""
    from .decisions import expected_action
    records = []
    for instance in instances:
        formal = (instance["formal_world"] if instance.get("condition") == "formal"
                  else _story_formal(instance))
        propagated, _ = forward_chain(formal)
        models = compile_to_sat(world_from_formal(formal)).enumerate_models(max_models=2)
        # Never choose an arbitrary model as a set of definitely established facts.
        solved = models[0] if len(models) == 1 else {}
        for method, assignment in (("direct-facts", direct_assignments(formal)),
                                   ("forward-chain", propagated),
                                   ("constraint-solver", solved)):
            action = None
            policy = instance.get("decision_policy")
            if policy and all(vid in assignment for vid in (policy["target_vid"], policy["gate_vid"])):
                action = expected_action(policy, assignment, instance["surfaces"])
            text = json.dumps({"observations": [],
                               "fixed_facts": [{"subject": instance["surfaces"][vid], "value": value}
                                               for vid, value in sorted(assignment.items())],
                               "final_action": action})
            records.append({"provider": f"baseline:{method}", "iid": instance["iid"],
                            "text": text, "status": "answered", "error": None,
                            "seconds": None, "cost": None, "tokens": None,
                            "baseline_access": ("public_formal" if instance.get("condition") == "formal"
                                                else "story_with_privileged_schema_and_template_grammar")})
        records.append({"provider": "baseline:story-copy", "iid": instance["iid"],
                        "text": json.dumps({"observations": ["Copied input, not a solution."],
                                            "fixed_facts": [instance.get("input_text", instance["story_text"])],
                                            "final_action": instance.get("input_text", instance["story_text"])}),
                        "status": "answered", "error": None, "seconds": None,
                        "cost": None, "tokens": None, "baseline_access": "public_input"})
    return records

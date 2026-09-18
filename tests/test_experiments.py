"""Controls for measured deduction and matched, answer-blind evaluation inputs."""
import json

import pytest

from enigmaforge.compile import compile_to_sat
from enigmaforge.experiments import (
    baseline_records, expand_conditions, forward_chain, measure_difficulty,
    world_from_formal,
)


def _formal():
    return {
        "variables": [{"vid": v, "type": "enum", "domain": [1, 2]} for v in ("A", "B", "C", "D")],
        "constraints": [
            {"kind": "eq", "vars": ["A"], "values": [1]},
            {"kind": "implies", "vars": ["A", "B"], "values": [1, 2]},
            {"kind": "eq", "vars": ["B", "C"], "values": []},
            {"kind": "implies", "vars": ["C", "D"], "values": [1, 2]},
        ],
    }


def _instance():
    return {"iid": "world-1", "formal_world": _formal(),
            "surfaces": {"A": "first record", "B": "second record", "C": "third record", "D": "fourth record"},
            "ground_truth": {"A": "HIDDEN_SENTINEL"},
            "story_text": "The first telling.", "story_r2_text": "An alternate telling.",
            "policy_text": "The posted policy applies.", "decision_policy": None}


def test_propagation_depth_is_synchronous_and_checks_antecedent():
    formal = _formal()
    assignment, depths = forward_chain(formal)
    assert assignment == {"A": 1, "B": 2, "C": 2}
    assert depths == {"A": 0, "B": 1, "C": 2}
    formal["constraints"].reverse()
    assert forward_chain(formal) == (assignment, depths)
    measured = measure_difficulty(formal)
    assert measured["forward_chain_rounds"] == 2
    assert measured["unresolved_by_forward_chain"] == ["D"]
    assert measured["direct_assignment_fraction"] == .25


def test_conflicting_derivations_fail_instead_of_overwriting():
    formal = _formal()
    formal["constraints"].append({"kind": "implies", "vars": ["A", "B"], "values": [1, 1]})
    with pytest.raises(ValueError, match="conflicting derivations"):
        forward_chain(formal)


def test_conditions_share_world_but_not_question_or_realization():
    instance = _instance()
    variants = expand_conditions([instance])
    assert {(i["condition"], i["realization"]) for i in variants} == {
        ("formal", 0), ("explicit", 1), ("explicit", 2), ("implicit", 1), ("implicit", 2)}
    assert {i["family_id"] for i in variants} == {"world-1"}
    assert all("HIDDEN_SENTINEL" not in i["input_text"] for i in variants)
    formal = next(i for i in variants if i["condition"] == "formal")
    assert "first record" in formal["input_text"]
    for realization in (1, 2):
        pair = {i["condition"]: i for i in variants if i["realization"] == realization}
        assert pair["explicit"]["input_text"].endswith(pair["implicit"]["input_text"])
    assert next(i for i in variants if i["condition"] == "implicit" and i["realization"] == 2)["input_text"] == "An alternate telling."


def test_missing_realization_cannot_silently_reuse_first_story():
    instance = _instance()
    del instance["story_r2_text"]
    with pytest.raises(ValueError, match="missing realization 2"):
        expand_conditions([instance])


def test_control_solver_does_not_read_gold_or_choose_ambiguous_solution():
    variants = expand_conditions([_instance()], conditions=("formal",))
    rows = {r["provider"]: json.loads(r["text"]) for r in baseline_records(variants)}
    assert rows["baseline:direct-facts"]["fixed_facts"] == [{"subject": "first record", "value": 1}]
    assert rows["baseline:forward-chain"]["fixed_facts"] == [
        {"subject": "first record", "value": 1},
        {"subject": "second record", "value": 2},
        {"subject": "third record", "value": 2}]
    assert rows["baseline:constraint-solver"]["fixed_facts"] == []
    assert all("HIDDEN_SENTINEL" not in json.dumps(row) for row in rows.values())


def test_all_different_solver_enforces_distinct_values():
    formal = {"variables": [{"vid": v, "type": "enum", "domain": [1, 2, 3]} for v in ("A", "B", "C")],
              "constraints": [{"kind": "alldiff", "vars": ["A", "B", "C"], "values": []}]}
    models = compile_to_sat(world_from_formal(formal)).enumerate_models()
    assert {tuple(m[v] for v in ("A", "B", "C")) for m in models} == {
        (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}

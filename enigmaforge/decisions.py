"""Conditional decision policies and exact, typed action validation.

Policies name formal variables internally. Their public rendering substitutes
surface labels and supplies a rule, never the variables' resolved values.
"""
import json


def _labels(surfaces, vid):
    labels = surfaces[vid]
    if isinstance(labels, str):
        labels = [labels]
    if not isinstance(labels, (list, tuple)) or not labels or any(
            not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError(f"invalid surface labels for {vid}")
    return labels


def _same_value(left, right):
    return type(left) is type(right) and left == right


def _subject_key(subject):
    return " ".join(subject.casefold().split())


def _check_policy(policy):
    fields = {"version", "target_vid", "gate_vid", "gate_value",
              "match_operation", "otherwise_operation"}
    if not isinstance(policy, dict) or set(policy) != fields:
        raise ValueError("decision policy must contain exactly the version-1 fields")
    if type(policy["version"]) is not int or policy["version"] != 1:
        raise ValueError("unsupported decision policy version")
    if type(policy["gate_value"]) not in (str, int, bool):
        raise ValueError("decision gate value must be a string, integer, or boolean")
    for field in ("target_vid", "gate_vid", "match_operation", "otherwise_operation"):
        if not isinstance(policy[field], str) or not policy[field].strip():
            raise ValueError(f"invalid decision policy {field}")
    if policy["match_operation"] == policy["otherwise_operation"]:
        raise ValueError("decision policy branches must have different operations")


def expected_action(policy, ground_truth, surfaces):
    """Return the conditional action, or None for a legacy world without policy.

    Surface values may be a canonical label or a list containing that label
    followed by explicitly declared aliases. Internal variable IDs are not aliases.
    """
    if policy is None:
        return None
    _check_policy(policy)
    target = policy["target_vid"]
    value = ground_truth[target]
    gate = ground_truth[policy["gate_vid"]]
    if type(value) not in (str, int, bool) or type(gate) not in (str, int, bool):
        raise ValueError("decision assignments must be strings, integers, or booleans")
    operation = (policy["match_operation"]
                 if _same_value(gate, policy["gate_value"])
                 else policy["otherwise_operation"])
    return {"operation": operation, "subject": _labels(surfaces, target)[0],
            "value": value}


def validate_action(action, policy, ground_truth, surfaces):
    """Validate all three action fields; no lexical overlap or verb synonyms."""
    if policy is None or not isinstance(action, dict):
        return False
    if set(action) != {"operation", "subject", "value"}:
        return False
    if not isinstance(action["operation"], str) or not isinstance(action["subject"], str):
        return False
    expected = expected_action(policy, ground_truth, surfaces)
    return (action["operation"] == expected["operation"]
            and _same_value(action["value"], expected["value"])
            and _subject_key(action["subject"]) in {
                _subject_key(label) for label in _labels(surfaces, policy["target_vid"])})


def policy_text(policy, surfaces, *, revision=False):
    """Render the complete rule without consulting a hidden assignment.

    A revision explicitly reverses the provisional operations in both branches;
    applying the superseded rule therefore cannot accidentally be correct.
    """
    _check_policy(policy)
    target = json.dumps(_labels(surfaces, policy["target_vid"])[0], ensure_ascii=False)
    gate = json.dumps(_labels(surfaces, policy["gate_vid"])[0], ensure_ascii=False)
    value = json.dumps(policy["gate_value"], ensure_ascii=False)
    match = json.dumps(policy["match_operation"], ensure_ascii=False)
    otherwise = json.dumps(policy["otherwise_operation"], ensure_ascii=False)
    action_fields = (f"The action's subject is {target}, and its value must be that "
                     "subject's resolved value, not the gate's value. ")
    condition = f"If the resolved value of {gate} equals {value}, "
    authoritative = (condition + f"the operation is {match}; otherwise the operation "
                     f"is {otherwise}. These operation names are exact. "
                     "String, integer, and boolean values remain distinct.")
    if not revision:
        return "The standing instruction governs the final action. " + action_fields + authoritative
    provisional = ("The initial instruction was explicitly provisional. " + action_fields
                   + condition + f"the operation was {otherwise}; otherwise the operation "
                   f"was {match}.")
    return (provisional + "\n\nThe later authoritative instruction supersedes the initial "
            "instruction in full; the provisional operations must not be used. "
            + action_fields + authoritative)

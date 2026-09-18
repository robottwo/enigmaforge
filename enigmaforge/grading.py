"""Versioned, deterministic grading of declared assignments (never raw prose)."""
import json

GRADER_VERSION = "3"
SOLVER_PROMPT = """Read the supplied material and return ONLY one JSON object, without
markdown or surrounding prose, with exactly these keys:
"observations": an array of zero to six nonempty strings describing your inferences;
"fixed_facts": an array of objects with exactly "subject" and "value". Use the
subject's surface label and its exact established value (string, integer, boolean).
Use null to abstain. Assert only your concluded assignments, not quotations,
possibilities, negations, or copied source claims. Do not assert competing values.
"final_action": null if no decision policy is given, otherwise an object with
exactly "operation", "subject", and "value" following the supplied decision policy.
The action value must be a string, integer, or boolean. Do not include extra keys."""


def strict_json(text):
    """Parse a whole JSON document, rejecting duplicate keys and non-JSON numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def normalize_subject(value):
    return " ".join(value.casefold().split())


def scalar(value, nullable=False):
    return type(value) in (str, int, bool) or (nullable and value is None)


def typed_value(value):
    return (type(value).__name__, value)


def subject_lookup(surfaces, aliases=None):
    """Surface lists explicitly declare aliases; ambiguous labels never resolve."""
    lookup = {}
    for vid, labels in surfaces.items():
        labels = [labels] if isinstance(labels, str) else list(labels)
        extra = (aliases or {}).get(vid, [])
        labels += [extra] if isinstance(extra, str) else list(extra)
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("surface labels must be nonempty strings")
            label = normalize_subject(label)
            if label in lookup and lookup[label] != vid:
                raise ValueError(f"ambiguous surface label: {label}")
            lookup[label] = vid
    return lookup


def parse_response(text):
    response = strict_json(text)
    if not isinstance(response, dict) or set(response) != {
            "observations", "fixed_facts", "final_action"}:
        raise ValueError("response must contain exactly observations, fixed_facts, final_action")
    observations = response["observations"]
    if not isinstance(observations, list) or len(observations) > 6 or any(
            not isinstance(o, str) or not o.strip() for o in observations):
        raise ValueError("observations must contain zero to six nonempty strings")
    facts = response["fixed_facts"]
    if not isinstance(facts, list):
        raise ValueError("fixed_facts must be an array")
    for fact in facts:
        if (not isinstance(fact, dict) or set(fact) != {"subject", "value"}
                or not isinstance(fact["subject"], str) or not fact["subject"].strip()
                or not scalar(fact["value"], nullable=True)):
            raise ValueError("each fixed fact must be a subject/value assignment")
    action = response["final_action"]
    if action is not None and (
            not isinstance(action, dict) or set(action) != {"operation", "subject", "value"}
            or any(not isinstance(action[k], str) or not action[k].strip()
                   for k in ("operation", "subject"))
            or not scalar(action["value"])):
        raise ValueError("final_action must be null or an operation/subject/value object")
    return response


def _fact_metrics(facts, instance):
    gt = instance.get("ground_truth") or {}
    lookup = subject_lookup(instance.get("surfaces") or {}, instance.get("aliases"))
    claims = {}
    for fact in facts:
        if fact["value"] is None:
            continue
        label = normalize_subject(fact["subject"])
        # A tagged identity prevents an unknown surface from impersonating a vid.
        subject = ("known", lookup[label]) if label in lookup else ("unknown", label)
        claims.setdefault(subject, set()).add(typed_value(fact["value"]))
    predicted = sum(map(len, claims.values()))
    correct = sum(1 for vid, value in gt.items()
                  if claims.get(("known", vid)) == {typed_value(value)})
    precision = correct / predicted if predicted else 0.0
    recall = correct / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = bool(gt) and correct == len(gt) and predicted == len(gt)
    return {"fact_precision": precision, "fact_recall": recall, "fact_f1": f1,
            "exact_world": exact}


def grade_instance(instance, record, extraction=None):
    """Grade strict v3 output, or explicitly supplied blind/human legacy extraction.

    Extraction is never inferred from prose. Flagged extractions are unavailable
    pending adjudication, not optimistically scored. Legacy decisions are absent.
    """
    graded = {"provider": record.get("provider"), "iid": record.get("iid", instance["iid"]),
              "grader_version": GRADER_VERSION, "status": "missing", "error": record.get("error"),
              "score": None, "components": {"compliance": None, "comprehension": None,
                                             "decisions": None},
              "metrics": {k: None for k in ("fact_precision", "fact_recall", "fact_f1",
                                            "exact_world", "decision_correct", "task_success")},
              "decision": None, "seconds": record.get("seconds"),
              "cost": record.get("cost"), "tokens": record.get("tokens"),
              "finish_reason": record.get("finish_reason")}
    status = record.get("status")
    if status == "content_filter" or "content_filter" in str(record.get("error") or ""):
        graded["status"] = "content_filter"
        return graded
    if status == "transport_error" or record.get("error"):
        # A length finish with no visible text is the model exhausting its
        # completion budget on reasoning — a model/config failure mode worth
        # its own status, not an endpoint transport failure.
        if record.get("finish_reason") == "length" and not (record.get("text") or "").strip():
            graded.update(status="invalid_response",
                          error=str(record.get("error")), score=0.0)
            graded["components"].update(compliance=0.0, comprehension=0.0)
            graded["metrics"].update(fact_precision=0.0, fact_recall=0.0,
                                     fact_f1=0.0, exact_world=False,
                                     task_success=False)
            if instance.get("decision_policy") is not None:
                graded["components"]["decisions"] = 0.0
                graded["metrics"]["decision_correct"] = False
            return graded
        graded["status"] = "transport_error"
        return graded
    raw = record.get("text")
    if status == "missing" or (raw is None and status not in ("empty", "refusal", "invalid_response")):
        return graded
    parsed = None
    try:
        if not isinstance(raw, str):
            raise ValueError("response has no visible text")
        parsed = parse_response(raw)
        graded["components"]["compliance"] = 1.0
    except (ValueError, TypeError) as exc:
        graded["components"]["compliance"] = 0.0
        if extraction is None:
            graded.update(status="invalid_response", error=str(exc), score=0.0)
            graded["components"]["comprehension"] = 0.0
            graded["metrics"].update(fact_precision=0.0, fact_recall=0.0, fact_f1=0.0,
                                     exact_world=False, task_success=False)
            if instance.get("decision_policy") is not None:
                graded["components"]["decisions"] = 0.0
                graded["metrics"]["decision_correct"] = False
            return graded
    if extraction is not None:
        from .judge import validate_extraction
        envelope = extraction if isinstance(extraction, dict) else {}
        if "status" in envelope and envelope["status"] not in ("ok", "adjudicated"):
            graded["status"] = ("adjudication_required" if envelope["status"] == "adjudication_required"
                                else "judge_error")
            graded["error"] = envelope.get("error") or envelope["status"]
            graded["extraction"] = envelope
            return graded
        try:
            extracted = envelope.get("extraction", envelope)
            flags = validate_extraction(extracted, raw)
            flags += list(envelope.get("review_flags") or [])
        except (ValueError, TypeError) as exc:
            graded.update(status="judge_error", error=str(exc))
            return graded
        graded["extraction"] = envelope
        if flags:
            graded.update(status="adjudication_required", error="; ".join(sorted(set(flags))))
            return graded
        facts = [claim for claim in extracted["claims"] if claim["stance"] == "assertion"]
    else:
        facts = parsed["fixed_facts"]
    graded["metrics"].update(_fact_metrics(facts, instance))
    graded.update(status="answered", error=None, score=graded["metrics"]["fact_f1"])
    graded["components"]["comprehension"] = graded["metrics"]["fact_recall"]
    if parsed is not None:
        graded["decision"] = parsed["final_action"]
    if instance.get("decision_policy") is not None:
        from .decisions import validate_action
        correct = validate_action(graded["decision"], instance["decision_policy"],
                                  instance["ground_truth"], instance["surfaces"])
        graded["metrics"]["decision_correct"] = correct
        graded["components"]["decisions"] = float(correct)
        graded["metrics"]["task_success"] = graded["metrics"]["exact_world"] and correct
    else:
        graded["metrics"]["task_success"] = graded["metrics"]["exact_world"]
    return graded

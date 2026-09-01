"""Trajectory evaluator: scores from the hidden formal representation.
v1 dimensions: objective identification, evidence efficiency, hypothesis
quality, insight latency, calibration. Computed mechanically where possible."""
from .world import HiddenWorld

def score_trajectory(world, trajectory):
    """trajectory = list of events, each {t, type, payload}.
    Types: observe(euid), hypothesize(latent_vars, mapping), answer(stage, answer),
    revise(stage, new_answer), done."""
    events = trajectory
    essential = set(world.meta.get("essential_cids", []))
    n_ev = len(world.evidence)
    dims = {
        "final_outcome": 0.0, "objective_identification": 0.0,
        "evidence_efficiency": 0.0, "redundant_investigation": 0.0,
        "hypothesis_quality": 0./1, "insight_latency": 0.0,
        "objective_revision_accuracy": 0.0, "calibration": None,
    }
    gt = world.meta.get("ground_truth", {})
    true_stage = [o for o in world.objectives if o.true_objective]
    true_stage = true_stage[0] if true_stage else None
    first_hypo_correct_vars = None
    hypo_events = [e for e in events if e["type"] == "hypothesize"]
    answer_events = [e for e in events if e["type"] == "answer"]
    observe_events = [e for e in events if e["type"] in ("observe", "inspect")]

    # final outcome
    if true_stage is not None:
        finals = [e for e in answer_events if e["payload"].get("stage") == true_stage.sid]
        if finals:
            best = max(finals, key=lambda e: e["t"])
            fa = finals[-1]["payload"].get("answer", {})
            dims["final_outcome"] = _match(fa.get("_all_", fa), gt)

    # objective identification: did solver answer the TRUE stage at all
    dims["objective_identification"] = float(any(
        e["payload"].get("stage") == true_stage.sid for e in answer_events)
        if true_stage else 0.0)

    # evidence efficiency: fraction of observations that were essential
    obs = [e["payload"].get("euid") for e in observe_events]
    ess_obs = sum(1 for o in obs if o in essential_euids(world))
    dims["evidence_efficiency"] = ess_obs / len(obs) if obs else 0.0
    dims["redundant_investigation"] = len(obs) - len(set(obs))

    # insight latency: t of first hypothesis containing >=50% of gt vars
    for e in hypo_events:
        m = e["payload"].get("mapping", {})
        if m and _match(m, gt) >= 0.5:
            dims["insight_latency"] = 1.0 / (1.0 + e["t"])
            break
    # hypothesis quality: mean match of all hypotheses over time
    if hypo_events:
        dims["hypothesis_quality"] = sum(
            _match(e["payload"].get("mapping", {}), gt) for e in hypo_events
        ) / len(hypo_events)
    # objective revision accuracy
    revise_events = [e for e in events if e["type"] == "hypothesize" and
                     e["payload"].get("abandoned", False)]
    if revise_events and hypo_events:
        kept = [e for e in hypo_events if not e["payload"].get("abandoned", False)]
        dims["objective_revision_accuracy"] = len(kept) / len(hypo_events)
    return dims

def _match(answer, gt):
    if not isinstance(answer, dict):
        return 0.0
    keys = [k for k in gt if k in answer]
    if not keys:
        return 0.0
    return sum(1 for k in keys if answer[k] == gt[k]) / len(keys)

def essential_euids(world):
    return {u.euid for u in world.evidence if not u.is_distractor}

"""Evidence, bridges, objectives, distractors — built from the verified HFW."""
from .rng import Rng
from .genres import get_pack
from .narrative import assign_surfaces
from .world import EvidenceUnit, KnowledgeBridge, ObjectiveStage
from .decisions import expected_action, policy_text

CHANNELS = ["letter", "receipt", "logbook", "dialogue", "marginalia",
            "photo_caption", "chronology", "omission_note", "rule_text"]

def populate_evidence(world, seed, distractor_hypotheses=()):
    rng = Rng(seed + 7717)
    assign_surfaces(world, seed)
    # one evidence unit per constraint (essential), shuffled across channels
    for c in world.constraints:
        ch = rng.pick(CHANNELS)
        speaker = rng.pick(world.entities).eid if ch in ("dialogue", "letter") else None
        world.evidence.append(EvidenceUnit(
            euid=f"E{c.cid}", channel=ch, encodes=[c.cid], speaker=speaker,
            surface={"tone": rng.pick(["neutral", "warm", "brisk", "evasive"])}))
    hyps = list(distractor_hypotheses) or get_pack(world).hypotheses
    n_dis = world.config.get("n_distractors", max(2, len(world.constraints) // 4))
    for i in range(n_dis):
        ch = rng.pick(CHANNELS)
        world.evidence.append(EvidenceUnit(
            euid=f"D{i}", channel=ch, encodes=[], is_distractor=True,
            distractor_hypothesis=rng.pick(hyps),
            surface={"tone": rng.pick(["neutral", "warm", "brisk"])}))
    rng.shuffle(world.evidence)
    return world

# Bridges are optional in-world lore, not external knowledge requirements.
# They carry no formal content and play no role in the solution or decision.
def populate_bridges(world, seed):
    rng = Rng(seed + 991)
    lore = get_pack(world).lore
    n = world.config.get("n_bridges", 2)
    picks = rng.sample(lore, min(n, len(lore)))
    for i, (fact, ref) in enumerate(picks):
        world.bridges.append(KnowledgeBridge(
            kbid=f"K{i}", fact=fact, entity_ref=ref,
            role="lore"))
    return world

def populate_objectives(world, seed):
    """Publish a conditional rule and derive its answer from the resolved world.

    Rule selection never consults the ground truth: the target, gate, and
    comparison value come from the public variable domains. Multi-stage
    objectives require replacing an explicitly provisional, opposite rule.
    """
    rng = Rng(seed + 5501)
    if len(world.variables) < 2:
        raise ValueError("a conditional objective requires distinct target and gate variables")
    target, gate = rng.sample(world.variables, 2)
    policy = {
        "version": 1,
        "target_vid": target.vid,
        "gate_vid": gate.vid,
        "gate_value": rng.pick(gate.domain),
        "match_operation": "register",
        "otherwise_operation": "hold",
    }
    surfaces = {v.vid: v.surface_names for v in world.variables}
    revision = world.config.get("n_objective_stages", 2) >= 3
    text = policy_text(policy, surfaces, revision=revision)
    world.meta["decision_policy"] = policy
    world.meta["policy_text"] = text
    gt = world.meta["ground_truth"]
    action = expected_action(policy, gt, surfaces)
    stages = [ObjectiveStage(
        sid="S0", level=0,
        statement="Recover the target and gate values needed by the instruction.",
        answer={target.vid: gt[target.vid], gate.vid: gt[gate.vid]},
        unlocks="S1",
        reveal_text=text)]
    if revision:
        provisional = dict(policy, match_operation=policy["otherwise_operation"],
                           otherwise_operation=policy["match_operation"])
        stages.append(ObjectiveStage(
            sid="S1", level=1,
            statement="Identify the action under the initial provisional instruction.",
            answer={"final_action": expected_action(provisional, gt, surfaces)},
            unlocks="S2",
            reveal_text="The later authoritative instruction supersedes the provisional rule in full."))
    stages.append(ObjectiveStage(
        sid=f"S{len(stages)}", level=len(stages),
        statement="Apply the authoritative instruction to the recovered target and gate values.",
        answer={"final_action": action}, true_objective=True))
    world.objectives = stages
    return world

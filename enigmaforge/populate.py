"""Evidence, bridges, objectives, distractors — built from the verified HFW."""
from .rng import Rng
from .genres import get_pack
from .narrative import assign_surfaces
from .world import EvidenceUnit, KnowledgeBridge, ObjectiveStage, Constraint

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

# Bridges are in-world lore from the active genre pack. Real-world knowledge
# grounding (facts the SOLVER must supply) stays on the roadmap; these are
# diegetic texture only, carrying no formal content.
def populate_bridges(world, seed):
    rng = Rng(seed + 991)
    lore = get_pack(world).lore
    n = world.config.get("n_bridges", 2)
    picks = rng.sample(lore, min(n, len(lore)))
    roles = ["essential"] + ["confirmatory", "seductive"] * 4
    for i, (fact, ref) in enumerate(picks):
        world.bridges.append(KnowledgeBridge(
            kbid=f"K{i}", fact=fact, entity_ref=ref,
            role=roles[i % len(roles)] if i else "essential"))
    return world

def populate_objectives(world, seed):
    """Staged objectives. Level 0 = apparent (characters' belief), final =
    true objective whose answer is derived from the ground truth model."""
    gt = world.meta["ground_truth"]
    stages = []
    stages.append(ObjectiveStage(
        sid="S0", level=0, statement="Identify the origin of the disruption.",
        answer={}, unlocks="S1",
        reveal_text="With the origin named, the sealed registry becomes legible."))
    if world.config.get("n_objective_stages", 2) >= 3:
        stages.append(ObjectiveStage(
            sid="S1", level=1,
            statement="Recover the registry key (an attribute of the origin).",
            answer={"_key": None}, unlocks="S2",
            reveal_text="The key opens the annex ledger — which contradicts the ledger already relied upon."))
        stages.append(ObjectiveStage(
            sid="S2", level=2,
            statement="Determine the correct final action given the invalidated assumption.",
            answer={"final_action": "discard primary ledger; act on annex"},
            true_objective=True))
    else:
        stages.append(ObjectiveStage(
            sid="S1", level=1,
            statement="Determine the correct final action.",
            answer={"final_action": "act on the corrected record"},
            true_objective=True))
    world.objectives = stages
    return world

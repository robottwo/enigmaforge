"""Narrative compiler v1: evidence units -> heterogeneous prose fragments.
The same EvidenceUnit can render differently per realization seed (paraphrase
jitter), so one hidden instance yields multiple non-isomorphic surfaces."""
from .rng import Rng
from .world import ConstraintKind

SYN = {  # one latent concept, many surface expressions
    "reliable": ["trusted", "vindicated", "accurate", "her reputation survived",
                 "I would believe her", "never once wrong"],
    "late": ["delayed", "behind schedule", "not on time", "running behind"],
}

CHANNEL_TEMPLATES = {
    "letter": [
        "{speaker} wrote later that {body}.",
        "In a letter found behind the {obj}: '{body}, and you know it.'",
        "The correspondence is blunt: {body}. No signature followed.",
    ],
    "receipt": [
        "A receipt, water-stained: {body}. The ink had run at the total.",
        "RECEIPT — {body}. Stamped twice, as if the clerk doubted himself.",
    ],
    "logbook": [
        "Logbook, hour 14: {body}.",
        "The duty log records, in a steadier hand than the rest: {body}.",
    ],
    "dialogue": [
        "'{body},' {speaker} said, not looking up.",
        "'{body}.' It was said once, flat, and not repeated.",
        "When pressed, {speaker} allowed only that {body}.",
    ],
    "marginalia": [
        "In the margin, pencil, pressed hard: {body}.",
        "Someone had underlined a passage twice and written: {body}.",
    ],
    "photo_caption": [
        "Photograph, undated. On the reverse, pencil: {body}.",
        "A print with a caption scratched into the border: {body}.",
    ],
    "chronology": [
        "{body}. This is fixed by the tide tables.",
        "The sequence is not disputed: {body}.",
    ],
    "omission_note": [
        "The page for that week is missing. What survives implies {body}.",
        "Nothing in the record states {body} — the absence is itself the record.",
    ],
    "channel_fallback": ["{body}."],
}

SURFACE_NOUNS = [
    "harbor manifest", "crate mark", "tide-table entry", "broker's stamp",
    "consignment tag", "watch rotation", "chandlery invoice", "ballast slip",
    "pilot's note", "quarantine chalk", "mooring receipt", "cargo tally",
]

def assign_surfaces(world, seed):
    """Deterministic surface noun per variable, fixed at build time so
    rendering is order-independent."""
    from .rng import Rng
    r = Rng(seed + 555)
    nouns = list(SURFACE_NOUNS)
    i = 0
    for v in world.variables:
        if not v.surface_names:
            v.surface_names = [nouns[i % len(nouns)]]
            i += 1

def _surface(v, rng):
    return v.surface_names[0] if v.surface_names else "record"

def constraint_phrase(c, world, rng):
    """Turn a constraint into a naturalistic clause (never 'X = Y')."""
    k = c.kind
    if k == ConstraintKind.EQ and len(c.vars) == 1:
        v = world.var(c.vars[0])
        val = c.values[0]
        if isinstance(val, str):
            return rng.pick([
                f"the {_surface(v, rng)} carried {val}'s mark",
                f"the {_surface(v, rng)} was signed out under {val}",
                f"no one disputed that the {_surface(v, rng)} pointed to {val}"])
        return rng.pick([
            f"the {_surface(v, rng)} was stamped {val}",
            f"the {_surface(v, rng)} read {val}"])
    if k == ConstraintKind.EQ:
        a, b = (world.var(x) for x in c.vars)
        return rng.pick([
            f"whatever the {_surface(a, rng)} showed, the {_surface(b, rng)} matched it",
            f"the {_surface(a, rng)} and the {_surface(b, rng)} agreed"])
    if k == ConstraintKind.NEQ:
        a, b = (world.var(x) for x in c.vars)
        return rng.pick([
            f"the {_surface(a, rng)} and the {_surface(b, rng)} did not agree",
            f"whatever else was true, the {_surface(a, rng)} never matched the {_surface(b, rng)}"])
    if k == ConstraintKind.IMPLIES:
        a, b = (world.var(x) for x in c.vars)
        return (f"whenever the {_surface(a, rng)} read {c.values[0]}, "
                f"the {_surface(b, rng)} read {c.values[1]}")
    if k == ConstraintKind.ALLDIFF:
        vs = None
        return f"no two of {', '.join(_surface(world.var(x), rng) for x in c.vars)} coincided"
    return "the rule was honored to the letter"

def render_unit(u, world, rseed):
    rng = Rng(rseed)
    if u.is_distractor:
        body = rng.pick([
            "the matter of the unpaid harbor fee surfaced again",
            "an inventory discrepancy of three barrels was noted",
            "a second signature on the deed had been discussed",
            "the vintage of the wine did not match the year of the dinner",
            "the dog barked at nobody in particular that night"])
        body += f" — {u.distractor_hypothesis}"
    else:
        body = " and ".join(constraint_phrase(world.constraint(cid), world, rng)
                            for cid in u.encodes)
    tpl = rng.pick(CHANNEL_TEMPLATES.get(u.channel, CHANNEL_TEMPLATES["channel_fallback"]))
    spk = _speaker_name(world, u.speaker)
    return tpl.format(speaker=spk, body=body, obj=rng.pick(["strongbox", "drawer", "frame"]))

def _speaker_name(world, eid):
    for e in world.entities:
        if e.eid == eid:
            return e.name
    return "the correspondent"

def compile_narrative(world, realization_seed, include_bridges=True):
    """Full solver-visible challenge text (one surface realization)."""
    rng = Rng(realization_seed)
    out = []
    out.append("You have been given the complete record of an unusual sequence "
               "of events. Determine what the record ultimately requires you to "
               "figure out. Then figure it out.")
    out.append("")
    out.append("— THE RECORD —")
    for i, u in enumerate(world.evidence):
        if u.is_distractor and not world.config.get("include_distractors", True):
            continue
        out.append(f"({i+1}) {render_unit(u, world, realization_seed + i * 31)}")
    if include_bridges and world.bridges:
        out.append("")
        out.append("— MARGINAL REFERENCES —")
        for b in world.bridges:
            if b.role == "seductive":
                out.append(f"({b.kbid}) A note mentions {b.entity_ref} in passing.")
            else:
                out.append(f"({b.kbid}) A note mentions {b.entity_ref}.")
    return "\n".join(out)

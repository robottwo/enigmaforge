"""Narrative compiler v1: evidence units -> heterogeneous prose fragments.
The same EvidenceUnit can render differently per realization seed (paraphrase
jitter), so one hidden instance yields multiple non-isomorphic surfaces.

Contract: a compiler returns a Realization — text plus a span map locating
every rendered evidence unit's clause VERBATIM. The faithfulness gates in
verify.py consume the spans; nothing downstream trusts prose alone.
All flavor (nouns, wrappers, distractors) comes from the active GenrePack."""
from dataclasses import dataclass, field

from .genres import get_pack
from .rng import Rng
from .world import ConstraintKind


@dataclass
class Realization:
    """One solver-visible surface.

    spans:   ref (euid/kbid) -> (start, end) char offsets into text
    clauses: ref -> the exact clause embedded at that span
    rendered: refs in text order. text[s:e] == clauses[ref], byte for byte."""
    mode: str
    text: str
    spans: dict
    clauses: dict
    rendered: list
    gates: dict = field(default_factory=dict)


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
        "{body}. This is fixed by the {chrono}.",
        "The sequence is not disputed: {body}.",
    ],
    "omission_note": [
        "The page for that week is missing. What survives implies {body}.",
        "The record is silent everywhere else; what survives comes to this: {body}.",
    ],
    "channel_fallback": ["{body}."],
}

_SERIES = ["", "second ", "third "]


def assign_surfaces(world, seed):
    """Deterministic surface noun per variable from the active genre pack,
    fixed at build time so rendering is order-independent. Nouns are unique
    within the pack; series prefixes disambiguate past the pack's list."""
    nouns = get_pack(world).nouns
    i = 0
    for v in world.variables:
        if not v.surface_names:
            v.surface_names = [f"{_SERIES[i // len(nouns)]}"
                               f"{nouns[i % len(nouns)]}"]
            i += 1


def _surface(v, rng):
    return v.surface_names[0] if v.surface_names else "record"


def constraint_phrase(c, world, rng):
    """Turn a constraint into a naturalistic clause (never 'X = Y').

    Each clause = optional wrapper + EXTRACTABLE CORE. The core is what the
    template extractor inverts (and what gates protect); wrappers are
    natural lead-ins/tails that make the clause testimony or memory rather
    than a database row. Wrappers must never contain words the extractor's
    patterns match on (agreed/matched/read/stamped/signed out/carried/
    pointed to/disputed/whenever/coincided)."""
    k = c.kind
    if k == ConstraintKind.EQ and len(c.vars) == 1:
        v = world.var(c.vars[0])
        val = c.values[0]
        if isinstance(val, str):
            core = rng.pick([
                f"the {_surface(v, rng)} carried {val}'s mark",
                f"the {_surface(v, rng)} was signed out under {val}",
                f"no one disputed that the {_surface(v, rng)} pointed to {val}"])
        else:
            core = rng.pick([
                f"the {_surface(v, rng)} was stamped {val}",
                f"the {_surface(v, rng)} read {val}"])
        return _pick_wrapper(_wrappers_for(get_pack(world)), rng).format(core=core)
    if k == ConstraintKind.EQ:
        a, b = (world.var(x) for x in c.vars)
        core = rng.pick([
            f"whatever the {_surface(a, rng)} showed, the {_surface(b, rng)} matched it",
            f"the {_surface(a, rng)} and the {_surface(b, rng)} agreed"])
        return _pick_wrapper(_wrappers_for(get_pack(world)), rng).format(core=core)
    if k == ConstraintKind.NEQ:
        a, b = (world.var(x) for x in c.vars)
        core = rng.pick([
            f"the {_surface(a, rng)} and the {_surface(b, rng)} did not agree",
            f"whatever else was true, the {_surface(a, rng)} never matched the {_surface(b, rng)}"])
        return _pick_wrapper(_wrappers_for(get_pack(world)), rng).format(core=core)
    if k == ConstraintKind.IMPLIES:
        a, b = (world.var(x) for x in c.vars)
        core = (f"whenever the {_surface(a, rng)} read {c.values[0]}, "
                f"the {_surface(b, rng)} read {c.values[1]}")
        return _pick_wrapper(_implies_wrappers_for(get_pack(world)), rng).format(core=core)
    if k == ConstraintKind.ALLDIFF:
        vs = None
        return f"no two of {', '.join(_surface(world.var(x), rng) for x in c.vars)} coincided"
    return "the rule was honored to the letter"


_USED_WRAPPERS = set()   # wrapper lead-ins used in the current realization


def reset_realization_texture():
    """Per-realization no-repeat state for clause wrappers: the wrappers
    survive polish verbatim (they carry the clauses), so their framing
    phrases must not repeat across a story."""
    _USED_WRAPPERS.clear()


def _pick_wrapper(candidates, rng):
    fresh = [w for w in candidates if w not in _USED_WRAPPERS]
    pick = rng.pick(fresh if fresh else candidates)
    _USED_WRAPPERS.add(pick)
    return pick


def _wrappers_for(pack):
    """Eventive wrappers: the clause lands as something that HAPPENED — an
    errand, an argument, a habit — never as someone citing a record."""
    loc = pack.locale[4:] if pack.locale.startswith("the ") else pack.locale
    return [
        "{core}",
        "{core}",
        f"it came out at the {loc}, the way such things do — {{core}}",
        "whoever went and checked came back with the same answer: {core}",
        "it took an argument and half the evening, but everyone finally "
        "allowed that {core}",
        "{core}, and that was the end of the discussion",
        "the talk kept circling back to the same stubborn fact: {core}",
        f"old {pack.demonym} hands still traded stories about the season "
        f"when {{core}}",
    ]


def _implies_wrappers_for(pack):
    return [
        "{core}",
        "{core}",
        "it held like a habit: {core}",
        "people honestly set their clocks by it — {core}",
        "it was proverbial by midseason: {core}",
        "{core}, season after season, without a single exception",
    ]


def unit_body(u, world, rng):
    """The claim clause of a unit — the exact text any realization must
    embed verbatim (gates and the extractor both key off this string)."""
    pack = get_pack(world)
    if u.is_distractor:
        body = rng.pick(pack.distractor_bodies)
        return body + f" — {u.distractor_hypothesis}"
    return " and ".join(constraint_phrase(world.constraint(cid), world, rng)
                        for cid in u.encodes)


def channel_fragment(u, world, body, rng):
    """Wrap an already-computed clause body in its channel template. The
    body must be computed exactly once per unit (wrapper picks are
    realization-stateful), which is why render_unit and compile_narrative
    both route through here."""
    tpl = rng.pick(CHANNEL_TEMPLATES.get(u.channel, CHANNEL_TEMPLATES["channel_fallback"]))
    spk = _speaker_name(world, u.speaker)
    return tpl.format(speaker=spk, body=body,
                      obj=rng.pick(["strongbox", "drawer", "frame"]),
                      chrono=get_pack(world).chrono)


def render_unit(u, world, rseed):
    rng = Rng(rseed)
    body = unit_body(u, world, rng)
    return channel_fragment(u, world, body, rng)


def _speaker_name(world, eid):
    for e in world.entities:
        if e.eid == eid:
            return e.name
    return "the correspondent"


def compile_narrative(world, realization_seed, include_bridges=True):
    """Record-mode realization: the numbered exhibit list (one surface)."""
    reset_realization_texture()
    parts = []
    entries = []  # (ref, clause, line_index)
    parts.append("You have been given the complete record of an unusual sequence "
                 "of events. Determine what the record ultimately requires you to "
                 "figure out. Then figure it out.")
    parts.append("")
    parts.append("— THE RECORD —")
    for i, u in enumerate(world.evidence):
        if u.is_distractor and not world.config.get("include_distractors", True):
            continue
        rng = Rng(realization_seed + i * 31)
        body = unit_body(u, world, rng)
        frag = channel_fragment(u, world, body, rng)
        parts.append(f"({i+1}) {frag}")
        entries.append((u.euid, body, len(parts) - 1))
    if include_bridges and world.bridges:
        parts.append("")
        parts.append("— MARGINAL REFERENCES —")
        for b in world.bridges:
            clause = (f"A note mentions {b.entity_ref} in passing."
                      if b.role == "seductive" else f"A note mentions {b.entity_ref}.")
            parts.append(f"({b.kbid}) {clause}")
            entries.append((b.kbid, clause, len(parts) - 1))
    text = "\n".join(parts)
    spans, clauses, rendered = {}, {}, []
    offs, acc = [], 0
    for p in parts:
        offs.append(acc)
        acc += len(p) + 1
    for ref, clause, li in entries:
        start = offs[li] + parts[li].find(clause)
        spans[ref] = (start, start + len(clause))
        clauses[ref] = clause
        rendered.append(ref)
    return Realization(mode="record", text=text, spans=spans,
                       clauses=clauses, rendered=rendered)

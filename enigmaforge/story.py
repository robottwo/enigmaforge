"""Story realization: embed the hidden world in prose where the puzzle form
is not announced — no exhibit list, no instruction, no frame.

Macro-structure (scene allocation, clue sequencing, pacing policy, stake and
distractor placement, length budget) is fixed by the WORLD seed, so all
realizations of one instance stay difficulty-matched. The realization seed
controls only micro-pacing and scene texture.

Contract: every clue-bearing beat's clause is embedded VERBATIM in the scene
prose (connectives keep clauses from ever being sentence-initial), so spans
locate it exactly and the template extractor in verify.py can round-trip the
text back to the formal model. Custom (e.g. LLM) renderers plug in behind the
same contract and are rejection-sampled against the gates."""
import dataclasses as _dc
from dataclasses import dataclass, field
from .rng import Rng
from .narrative import Realization, unit_body


class RenderContractError(Exception):
    """Renderer violated the verbatim-embedding contract, or no attempt
    passed the faithfulness gates within the attempt budget."""


@dataclass
class Beat:
    ref: str    # euid / kbid / "stake_open" / "stake_close"
    kind: str   # clue | distractor | bridge | stake_open | stake_close


@dataclass
class Scene:
    sid: str
    when: str = ""                                # timeline label (macro pacing)
    filler: int = 0                               # scenic sentences (macro pacing)
    depth_open: int = 0                           # scenic paragraphs before the clue paragraph
    depth_close: int = 0                          # scenic paragraphs after it
    deck: list = field(default_factory=list)      # exclusive scenic sentences
    beats: list = field(default_factory=list)


@dataclass
class StorySkeleton:
    scenes: list = field(default_factory=list)
    pacing: dict = field(default_factory=dict)      # {"policy", "params"}
    positions: dict = field(default_factory=dict)   # ref -> [scene_idx, beat_idx]


PACING_POLICIES = ("uniform", "bursty", "longrange")



def build_skeleton(world, seed):
    """Deterministic macro-structure for story mode. Pacing policy, timeline
    ladder, and all placements come from the world seed — never from the
    realization seed."""
    from .genres import get_pack
    rng = Rng(seed + 3331)
    ladders = get_pack(world).timelines
    ladder = ladders[rng.below(len(ladders))]

    def when(i):
        return ladder[min(i, len(ladder) - 1)]

    inc_dis = world.config.get("include_distractors", True)
    clues = [u.euid for u in world.evidence if not u.is_distractor]
    distractors = [u.euid for u in world.evidence if u.is_distractor and inc_dis]
    bridges = [b.kbid for b in world.bridges]

    policy = rng.pick(PACING_POLICIES) if len(clues) >= 4 else "uniform"
    ordered = _order_clues(world, clues, policy, rng)
    groups = _allocate(ordered, policy, rng)
    scenes = [Scene(sid=f"S{i}", when=when(i),
                    beats=[Beat(c, "clue") for c in g])
              for i, g in enumerate(groups)]


    # distractors: adjacency policy — near a clue (local confusion) or loose
    for d in distractors:
        if scenes:
            sc = rng.pick(scenes)
            if rng.chance(0.6):
                sc.beats.insert(rng.below(len(sc.beats) + 1), Beat(d, "distractor"))
            else:
                sc.beats.append(Beat(d, "distractor"))
    for kb in bridges:
        if scenes:
            sc = rng.pick(scenes)
            sc.beats.insert(rng.below(len(sc.beats) + 1), Beat(kb, "bridge"))
    # stakes frame the story without ever stating a task
    if scenes:
        scenes[0].beats.insert(0, Beat("stake_open", "stake_open"))
        scenes[-1].beats.append(Beat("stake_close", "stake_close"))

    # burial: how deep clues sit under pure story. A measured, seeded axis
    # (config 'burial', 0-3): per-scene scenic paragraphs before/after the
    # clue paragraph, and at higher settings whole clue-free story scenes.
    burial = world.config.get("burial", 1)
    if burial > 0:
        for sc in scenes:
            if any(b.kind == "clue" for b in sc.beats):
                sc.depth_open = rng.range(0, burial + 1)
                sc.depth_close = rng.range(0, burial + 1)
        if burial >= 2 and len(scenes) > 2:
            n_extra = rng.range(0, burial)  # pure story scenes, seeded spots
            for _ in range(n_extra):
                pos = rng.range(1, len(scenes) - 1)
                scenes.insert(pos, Scene(sid=f"B{pos}", when=when(pos),
                                         filler=rng.range(2, 5)))

    sk = StorySkeleton(scenes=scenes, pacing={
        "policy": policy,
        "params": {"n_scenes": len(scenes), "burial": burial,
                   "clues_per_scene": [sum(1 for b in s.beats if b.kind == "clue")
                                       for s in scenes]}})
    _budget_filler(world, sk, rng)
    sk.positions = {b.ref: [si, bi] for si, s in enumerate(scenes)
                    for bi, b in enumerate(s.beats)}
    return sk


def _order_clues(world, clues, policy, rng):
    order = rng.shuffle(clues)
    if policy != "longrange" or len(order) < 4:
        return order
    # maximize separation of clues that share variables: greedy, least overlap
    # with the recent window — dependent clues land far apart on purpose
    overlap = _overlap_map(world)
    window, out, remaining = [], [], list(order)
    cur = max(remaining, key=lambda c: len(overlap.get(c, ())))
    while remaining:
        remaining.remove(cur)
        out.append(cur)
        window = (window + [cur])[-3:]
        if remaining:
            cur = min(remaining,
                      key=lambda c: sum(1 for w in window if c in overlap.get(w, ())))
    return out


def _overlap_map(world):
    def vars_of(u):
        return {v for cid in u.encodes for v in world.constraint(cid).vars}
    vs = {u.euid: vars_of(u) for u in world.evidence if not u.is_distractor}
    return {a: {b for b, w in vs.items() if b != a and s & w}
            for a, s in vs.items()}


def _allocate(ordered, policy, rng):
    groups = []
    if policy == "uniform":
        groups = [[c] for c in ordered]
    elif policy == "bursty":
        i = 0
        while i < len(ordered):
            k = min(rng.pick([1, 1, 2, 2, 3]) if len(ordered) - i > 1 else 1,
                    len(ordered) - i)
            groups.append(ordered[i:i + k])
            i += k
        if len(groups) > 2 and rng.chance(0.3):
            groups.insert(rng.below(len(groups) - 1), [])   # pure scenic scene
    else:  # longrange
        groups = [[c] for c in ordered]
    return groups or [[]]


def _budget_filler(world, sk, rng):
    """Wire narrative_tokens (finally): scenic filler scales so the estimated
    token count lands near the configured budget. Macro pacing — same for
    every realization of the instance."""
    target = world.config.get("narrative_tokens", 600)
    idx = {u.euid: i for i, u in enumerate(world.evidence)}
    clue_words = 0
    for u in world.evidence:
        if u.is_distractor and not world.config.get("include_distractors", True):
            continue
        clue_words += len(unit_body(u, world, Rng(world.seed + idx[u.euid] * 31)).split())
    frame_words = 14 * len(sk.scenes) + 20
    spare = max(0.0, target / 1.3 - clue_words - frame_words)
    per_scene = int(spare // max(1, len(sk.scenes)) // 7)
    hi = min(4, per_scene + 1)
    lo = max(0, hi - 2)
    for s in sk.scenes:
        s.filler = rng.range(lo, hi) if hi > lo else hi


def skeleton_summary(sk):
    return {"pacing": sk.pacing,
            "scenes": [{"sid": s.sid, "when": s.when, "filler": s.filler,
                        "beats": [{"ref": b.ref, "kind": b.kind} for b in s.beats]}
                       for s in sk.scenes],
            "positions": {k: list(v) for k, v in sk.positions.items()}}

# ---------------------------------------------------------------- rendering

def _texture(world):
    """Active genre pack's texture (world may be None for bare renderers)."""
    if world is None:
        from .genres import PACKS
        return PACKS["maritime"]
    from .genres import get_pack
    return get_pack(world)


# Genre-neutral narrative joins: clauses ride inside running sentences as
# things said or admitted, not introduced as exhibits.
CONNECTIVES = [
    "And though nobody said so,", "Even the newest hand knew that",
    "By then everyone had heard that", "The talk kept coming back to it:",
    "Sooner or later everyone admitted that", "Which was how the season went —",
    "It went unchallenged that", "Still, the season being what it was,",
]


def _scenic_decks(world, skeleton, realization_seed):
    """Serially pre-build each scene's scenic sentences for one realization:
    authored filler sentences first (each used once), then grammar events.
    compile_story hands each scene its own exclusive deck, so parallel
    rendering is deterministic — no shared mutable state."""
    pack = _texture(world)
    cast = [e.name for e in (world.entities if world else [])][:6] or ["the clerk"]
    rng = Rng(realization_seed + 520257)
    used_fillers, decks = set(), {}
    for scene in skeleton.scenes:
        slots = (3 * scene.depth_open + 3 * scene.depth_close
                 + scene.filler + max(2, len(scene.beats) // 2) + 4)
        deck = []
        unused = [f for f in pack.filler if f not in used_fillers]
        last_action = None
        for _ in range(slots):
            if unused and rng.chance(0.3):
                sent = unused.pop(0)
                used_fillers.add(sent)
                deck.append(sent)
                continue
            if pack.things:
                for _try in range(8):
                    action = rng.pick(pack.actions)
                    thing = rng.pick(pack.things)
                    from .genres import ACTION_CLASSES, _thing_class
                    if _thing_class(thing) not in ACTION_CLASSES.get(
                            action, ("movable", "fixture", "surface")):
                        continue
                    if action == last_action:
                        continue
                    last_action = action
                    deck.append(f"{rng.pick(cast)} {action} {thing} "
                                f"{rng.pick(pack.details)}.")
                    break
                else:
                    deck.append(rng.pick(pack.filler))
            else:
                deck.append(rng.pick(pack.filler))
        decks[scene.sid] = deck
    return decks


def template_scene_renderer(scene, beat_clauses, rng, world=None):
    """Default renderer. Dynamics that break the one-clause-one-sentence
    meter: dialogue attributions give clauses voices, trailing tails keep
    clauses from always ending their sentence, adjacent clue/distractor
    pairs sometimes fuse into one sentence, and longer scenes split into
    paragraphs. Scenic sentences come from the scene's EXCLUSIVE deck
    (pre-built serially by compile_story), so parallel rendering is
    deterministic. All flavor comes from the genre pack; nothing here can
    be mistaken for a claim by the extractor."""
    pack = _texture(world)
    cast = [e.name for e in (world.entities if world else [])][:6] or ["the clerk"]
    focal = rng.pick(cast)
    sentences = [rng.pick(pack.frames).format(place=rng.pick(pack.places),
                                             name=focal)]
    joins = rng.shuffle(CONNECTIVES)
    attributions = rng.shuffle(ATTRIBUTIONS)
    tails = rng.shuffle(TAILS)
    deck = list(getattr(scene, "deck", None) or rng.shuffle(pack.filler))
    di = ci = ai = ti = 0
    last_action = [None]

    def scenic_sentence():
        nonlocal di
        if di < len(deck):
            sent = deck[di]
            di += 1
            return sent
        return rng.pick(pack.filler)

    def scenic_paragraph():
        n = rng.range(2, 4)
        return " ".join(scenic_sentence() for _ in range(n))

    paras = [scenic_paragraph() for _ in range(getattr(scene, "depth_open", 0))]
    kinds = {u.euid: ("distractor" if u.is_distractor else "clue")
             for u in (world.evidence if world else [])}
    beats = list(beat_clauses)
    i = 0
    while i < len(beats):
        ref, clause = beats[i]
        if rng.chance(0.35):
            sentences.append(scenic_sentence())
        # fuse an adjacent clue/distractor pair into one sentence — clue and
        # noise become harder to tell apart on the surface
        if (i + 1 < len(beats) and rng.chance(0.35)
                and {kinds.get(ref), kinds.get(beats[i + 1][0])}
                == {"clue", "distractor"}):
            c2 = beats[i + 1][1]
            sentences.append(f"{joins[ci % len(joins)]} {clause}, though {c2}.")
            ci += 1
            i += 2
            continue
        if rng.chance(0.5) and cast:
            speaker = rng.pick(cast)
            sentences.append(attributions[ai % len(attributions)]
                             .format(name=speaker, clause=clause) + ".")
            ai += 1
        elif rng.chance(0.3):
            sentences.append(f"{joins[ci % len(joins)]} {clause}"
                             f"{tails[ti % len(tails)]}.")
            ci += 1
            ti += 1
        else:
            sentences.append(f"{joins[ci % len(joins)]} {clause}.")
            ci += 1
        i += 1
    for _ in range(scene.filler):
        sentences.append(scenic_sentence())
    # the clue-bearing core may breathe into two paragraphs (micro-pacing)
    if len(sentences) >= 5 and rng.chance(0.5):
        cut = rng.range(2, len(sentences) - 1)
        paras.extend([" ".join(sentences[:cut]), " ".join(sentences[cut:])])
    else:
        paras.append(" ".join(sentences))
    paras.extend(scenic_paragraph()
                 for _ in range(getattr(scene, "depth_close", 0)))
    return "\n\n".join(p for p in paras if p)


ATTRIBUTIONS = [
    "{name} allowed that {clause}",
    "It was {name} who finally said it: {clause}",
    "If {name} was to be believed, {clause}",
    "{name} put it plainly: {clause}",
    "{name} said nothing for a while, and then: {clause}",
    "Nobody had to ask {name} twice: {clause}",
]
TAILS = [
    " — and left it at that",
    ", and no one pushed it further",
    ", for whatever that was worth",
    " — or so it was repeated",
    ", and there the matter rested",
]


def _stake_clause(world, rng, opening):
    """Stakes in story-voice only, flavored by the active genre pack. The
    formal objective statements (hidden) never surface: printing 'the origin
    of the disruption' is a puzzle tell. Opening plants unease; closing
    raises it to consequence."""
    from .genres import get_pack
    setting = get_pack(world).setting
    if opening:
        return rng.pick([
            "the season had gone wrong in a way nobody could point to, "
            "and everyone felt it",
            f"something had unsettled the ordinary working of {setting}, "
            "and no one would say what",
            "it had been a strange stretch of weeks, the kind people "
            "stop talking about"])
    return rng.pick([
        "pointing at it was no longer the point; what mattered now was "
        "what to do",
        "by then it was no longer a matter of curiosity, but of what "
        "must be done",
        "whatever was wrong had to be answered with something more "
        "than words"])


def _beat_clause(world, beat, ev_index, br_index, realization_seed):
    if beat.kind == "stake_open":
        return _stake_clause(world, Rng(realization_seed + 7), opening=True)
    if beat.kind == "stake_close":
        return _stake_clause(world, Rng(realization_seed + 13), opening=False)
    if beat.kind == "bridge":
        b = world.bridges[br_index[beat.ref]]
        rng = Rng(realization_seed + 101 + br_index[beat.ref] * 17)
        return rng.pick([
            f"the old hands still argued about {b.entity_ref}",
            f"{b.entity_ref} came up when the talk went quiet",
            f"no one had brought up {b.entity_ref} in years, and then "
            f"someone did"])
    u = world.evidence[ev_index[beat.ref]]
    # same per-unit seeding scheme as record mode: clause text is stable for
    # a given realization seed regardless of where the skeleton places it
    return unit_body(u, world, Rng(realization_seed + ev_index[beat.ref] * 31))

def compile_story(world, skeleton, realization_seed, renderer=None,
                  max_workers=3):
    """One story realization. renderer(scene, beat_clauses, rng, world) must
    return prose embedding every clause verbatim; spans are located by
    search, so a renderer that drops or rewrites a clause fails loudly.
    Scenes are independent (own beats, own seeded rng) and render in
    parallel — thread scheduling cannot affect output, and gates run after
    assembly. max_workers=1 restores fully serial rendering."""
    renderer = renderer or template_scene_renderer
    # optional two-phase renderers draft a shared plan first (premise,
    # character sheet, per-scene settings) so independently rendered scenes
    # still tell one story; called once, before any scene dispatch
    from .narrative import reset_realization_texture
    reset_realization_texture()
    prepare = getattr(renderer, "prepare", None)
    if prepare is not None:
        prepare(world, skeleton)
    ev_index = {u.euid: i for i, u in enumerate(world.evidence)}
    br_index = {b.kbid: i for i, b in enumerate(world.bridges)}
    from .genres import get_pack
    trng = Rng(realization_seed + 3)
    if world.bridges and trng.chance(0.4):
        ref = trng.pick(world.bridges).entity_ref
        title = ref[0].upper() + ref[1:]
    else:
        title = trng.pick(get_pack(world).titles)
    beat_sets = [
        [(b.ref, _beat_clause(world, b, ev_index, br_index, realization_seed))
         for b in scene.beats]
        for scene in skeleton.scenes]
    # scenic decks: each scene gets an EXCLUSIVE, serially pre-built list of
    # scenic sentences. Parallel renderers only consume their own deck, so
    # output is deterministic regardless of thread scheduling.
    deck_by_sid = _scenic_decks(world, skeleton, realization_seed)
    rendered_scenes = [_dc.replace(s) for s in skeleton.scenes]
    for s in rendered_scenes:
        s.deck = deck_by_sid[s.sid]

    def render_scene(si, scene, bs):
        """Render one scene; endpoint-transient failures retry in place so a
        single slow call cannot abort a whole realization."""
        err = None
        for try_ in range(3):
            try:
                return renderer(scene, bs,
                                Rng(realization_seed + (si + 1) * 7919), world)
            except RenderContractError:
                raise                      # contract violations are not transient
            except RuntimeError as e:      # endpoint chain exhausted
                err = e
        raise RuntimeError(f"scene {scene.sid} failed after 3 tries: {err}")

    if max_workers <= 1 or len(skeleton.scenes) < 2:
        scene_blocks = [
            (bs, render_scene(si, scene, bs))
            for si, (scene, bs) in enumerate(zip(rendered_scenes, beat_sets))]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(
                max_workers=min(max_workers, len(skeleton.scenes))) as ex:
            futures = [
                ex.submit(render_scene, si, scene, bs)
                for si, (scene, bs) in enumerate(zip(rendered_scenes, beat_sets))]
            # .result() in scene order: deterministic assembly, first
            # failure (if any) reported for the earliest scene
            scene_blocks = [(bs, fut.result())
                            for bs, fut in zip(beat_sets, futures)]
    text = title + "\n\n" + "\n\n".join(p for _, p in scene_blocks)

    spans, clauses, rendered = {}, {}, []
    off = len(title) + 2
    for beats, prose in scene_blocks:
        for ref, clause in beats:
            pos = prose.find(clause)
            if pos < 0:
                raise RenderContractError(
                    f"renderer dropped or rewrote the clause for {ref!r}")
            spans[ref] = (off + pos, off + pos + len(clause))
            clauses[ref] = clause
            rendered.append(ref)
        off += len(prose) + 2
    return Realization(mode="story", text=text, spans=spans,
                       clauses=clauses, rendered=rendered)


def compile_story_verified(world, skeleton, realization_seed, renderer=None,
                           max_attempts=3, max_workers=3):
    """Rejection-sampled story compile: render -> structural gate + extraction
    round-trip; retry with a fresh surface seed until both pass. The default
    template renderer passes by construction; the loop exists for custom
    (LLM) renderers."""
    from .verify import verify_realization, verify_roundtrip
    attempts = []
    for i in range(max_attempts):
        rseed = realization_seed + i * 104729
        try:
            r = compile_story(world, skeleton, rseed, renderer=renderer,
                              max_workers=max_workers)
        except RenderContractError as e:
            attempts.append({"attempt": i, "error": str(e)})
            continue
        gates = verify_realization(world, r)
        gates["roundtrip"] = verify_roundtrip(world, r)
        if gates["pass"] and gates["roundtrip"]["pass"]:
            r.gates = gates
            return r
        attempts.append({"attempt": i, "gates": gates})
    raise RenderContractError(
        f"no realization passed the gates in {max_attempts} attempts: {attempts}")

"""Genre packs: setting, lexicon, and texture per genre.

The harbor was v1's only skin — baked into surface nouns, distractors, lore,
places, filler, titles, stakes, and the LLM prompts. Packs make the setting
a seeded axis instead: `genre` is resolved at build time (seeded pick when
unset/auto, explicit config/CLI to pin), so instances vary in setting while
same-seed determinism holds.

Every pack must keep the two invariants the gates depend on:
- surface nouns are unique within the pack (noun -> vid must invert);
- flavor text never contains words the claim-extractor matches on
  (agreed/matched/read <val>/stamped/signed out/carried/pointed to/
  disputed/whenever/coincided/no two).
"""
from .rng import Rng


class GenrePack:
    def __init__(self, name, vibe, setting, locale, demonym, chrono,
                 nouns, places, frames, filler, titles,
                 distractor_bodies, hypotheses, lore, timelines=None,
                 things=None, actions=None, details=None):
        assert len(set(nouns)) == len(nouns), f"{name}: duplicate nouns"
        import re as _re0
        _noun_rx = _re0.compile(r"^[a-z'’\-]+(?: [a-z'’\-]+){0,2}$")
        for n in nouns:
            assert _noun_rx.match(n), \
                f"{name}: noun {n!r} must be 1-3 lowercase words (the " \
                f"extractor's noun shape) — otherwise round-trips break"
        _CLAIM_WORDS = ("agreed", "matched", "stamped", "signed out",
                        "pointed to", "disputed", "coincided", "whenever",
                        "no two")
        for pool_name, pool in (("distractor", distractor_bodies),
                                ("hypothesis", hypotheses),
                                ("scenic thing", things or []),
                                ("scenic action", actions or []),
                                ("scenic detail", details or []),
                                ("filler", filler),
                                ("title", titles),
                                ("place", places)):
            for text in pool:
                hit = [n for n in nouns if n in text]
                assert not hit, f"{name}: {pool_name} {text!r} references " \
                                f"variable surface(s) {hit} — scenery must " \
                                f"carry no formal content"
                bad = [w for w in _CLAIM_WORDS if w in text]
                assert not bad, f"{name}: {pool_name} {text!r} contains " \
                                f"extractor word(s) {bad}"
        import re as _re
        for frame in frames:
            slots = set(_re.findall(r"\{(\w+)\}", frame))
            assert slots <= {"place", "name"}, \
                f"{name}: frame {frame!r} uses unknown slot(s) {slots}"
        for fact, ref in lore:
            hit = [n for n in nouns if n in ref]
            assert not hit, f"{name}: lore ref {ref!r} references " \
                            f"variable surface(s) {hit}"
        self.name = name
        self.vibe = vibe                  # one-liner for LLM prompts
        self.setting = setting            # "the harbor" / "the valley" ...
        self.locale = locale              # where people talk: quay, lobby...
        self.demonym = demonym            # "harbor hands", "valley hands"
        self.chrono = chrono              # what fixes a sequence
        self.nouns = nouns
        self.places = places
        self.frames = frames
        self.filler = filler
        self.titles = titles
        self.distractor_bodies = distractor_bodies
        self.hypotheses = hypotheses
        self.lore = lore                  # (sentence, short_ref) pairs
        self.timelines = timelines or DEFAULT_LADDERS
        # scenic event grammar: cast x action x thing x detail gives a
        # combinatorial sentence space so burial paragraphs stop repeating
        self.things = things or []
        self.actions = actions or DEFAULT_ACTIONS
        self.details = details or DEFAULT_DETAILS


# Shared timeline ladders: which one an instance uses is a seeded macro
# choice in build_skeleton, so the story's sense of time varies too.
DEFAULT_LADDERS = [
    ["on the first evening", "the next morning", "that same afternoon",
     "two days later", "by the end of the week", "the following week",
     "late that month", "at the month's end"],
    ["in the first week of the season", "the following week", "mid-season",
     "the week after that", "toward the close of the season",
     "in the last week", "after the season ended", "long after"],
    ["before the first storm", "after the first storm", "between storms",
     "during the long wet spell", "at the first hard frost",
     "after the frost", "on the first fine day", "at the thaw"],
]

# Scenic event grammar (shared actions/details; things are per-pack).

# Semantic validity for the scenic event grammar: things are classified by
# keyword into movable / fixture / surface, and each action lists the
# classes it can take. Without this, combinatorial grammar produces
# absurdities ("made room for the gravel path", "carried down the stove").
_SURFACE_WORDS = ("path", "tiles", "stones", "passage", "corridor", "yard",
                  "flags", "reach", "mud", "walk")
_MOVABLE_WORDS = ("kettle", "lamp", "tin", "box", "boots", "umbrellas",
                  "mittens", "rope", "hook", "cushions", "cart", "basket",
                  "curtain", "chair", "pull", "cushion", "coats", "rack")


def _thing_class(thing):
    t = thing.lower()
    if any(w in t for w in _SURFACE_WORDS):
        return "surface"
    if any(w in t for w in _MOVABLE_WORDS):
        return "movable"
    return "fixture"


ACTION_CLASSES = {
    "put right": ("movable", "fixture"),
    "left standing": ("movable", "fixture"),
    "carried down": ("movable",),
    "set to rights": ("movable", "fixture"),
    "gave up on": ("movable", "fixture", "surface"),
    "argued over": ("movable", "fixture", "surface"),
    "turned out": ("movable", "fixture"),
    "tripped over": ("movable", "surface"),
    "kept half an eye on": ("movable", "fixture", "surface"),
    "made room for": ("movable",),
}
# cast(4-6) x actions(10) x things(12) x details(10) ~= 5k sentences/pack,
# so burial paragraphs stop repeating the same ten filler lines.
DEFAULT_ACTIONS = [
    "put right", "left standing", "carried down", "set to rights",
    "gave up on", "argued over", "turned out",
    "tripped over", "kept half an eye on", "made room for",
]
DEFAULT_DETAILS = [
    "twice before supper", "without much hope",
    "for the third time that week", "and said nothing about it",
    "as if it mattered", "before the light went", "out of habit",
    "and thought better of it", "with more patience than it deserved",
    "long after it stopped being useful",
]


_MARITIME = GenrePack(
    name="maritime", vibe="small-harbor", setting="the harbor",
    locale="quay", demonym="harbor", chrono="tide tables",
    nouns=[
        "harbor manifest", "crate mark", "tide-table entry", "broker's stamp",
        "consignment tag", "watch rotation", "chandlery invoice", "ballast slip",
        "pilot's note", "quarantine chalk", "mooring receipt", "cargo tally",
        "warehouse ledger", "lighthouse log", "customs seal", "tug order",
        "berthing chart", "dock pass", "ferry ticket", "signal flag",
        "storm glass", "hawser register", "stevedore roster", "wharfage book",
        "cargo manifest", "ballast receipt", "harbor master's file",
        "chandler's daybook", "ship's articles", "tide ledger", "muster roll",
        "slop chest list", "graving dock schedule", "bunkering chit",
        "draught survey sheet", "gangway log", "lamp oil account",
        "rope walk invoice", "sailmaker's bill", "cooperage note",
        "rigger's receipt", "boatman's ticket", "night watch list",
        "fog horn record", "anchorage permit", "quarantine flag log",
        "fresh water chit", "coal measure book", "timber tally",
        "grain probe record", "mail bag manifest", "passenger list",
        "crew agreement paper", "wage account sheet", "salvage claim",
        "wreck register entry", "prize court filing", "admiralty notice",
        "port doctor's line", "victualing bill", "chronometer rate book",
        "compass deviation card", "sounding lead mark", "night order book",
    ],
    places=["chandlery", "counting house", "harbor office", "warehouse loft",
            "commission house", "sail loft"],
    frames=[
        "In the back room of the {place}, {name} sat out the bad hour with the stove drawn close.",
        "The {place} smelled of tar and old paper. {name} had an hour before the tide turned.",
        "It was quiet in the {place}. Outside, the cranes had stopped for the day.",
        "{name} kept the {place} open late that week, and the evenings ran long.",
        "Rain set in over the {place}, and there was nothing to do but talk.",
    ],
    filler=[
        "The lamp guttered and steadied.",
        "Somewhere below, a winch clattered and stopped.",
        "Gulls held the roof slates against the wind.",
        "The tea went cold in its pot.",
        "A clerk crossed the yard with a lantern.",
        "The tide, far out, left the mud shining.",
        "Frost was forming on the inside of the glass.",
        "The clock in the hall struck the quarter hour.",
        "Someone was singing, three doors down.",
        "The cat from the sail loft walked the sill and left.",
    ],
    titles=["What the Winter Kept", "A Season of Papers", "The Quiet Ledger",
            "Orders in Duplicate", "The Long Quarantine", "Ink and Salt Water"],
    distractor_bodies=[
        "the matter of the unpaid harbor fee surfaced again",
        "an inventory discrepancy of three barrels was noted",
        "a second signature on the deed had been discussed",
        "the vintage of the wine did not match the year of the dinner",
        "the dog barked at nobody in particular that night"],
    hypotheses=[
        "the oldest letter is the forgery", "the harbor fee was never paid",
        "the second partner acted alone", "the ledger was altered after the fire"],
    lore=[
        ("the winter the harbor froze solid and no boat left the quay", "the frozen winter"),
        ("the year the foundry delivered the wrong bell to the tower", "the wrong bell"),
        ("the summer the lighthouse oil was cut with paraffin", "the paraffin summer"),
        ("the season the herring never came in", "the missing herring"),
        ("the autumn the rail company bought the north slip", "the rail company's slip"),
        ("the storm that took the outer marker", "the lost marker"),
        ("the year the tide tables were reprinted with errors", "the bad tide tables"),
        ("the decade the pilots kept their own registry", "the pilots' registry"),
    ],
    things=["the stove", "the kettle", "the lamp", "the shutters",
           "the yard", "the quay stones", "the wet rope", "the oilskins by the door",
           "the boat hook", "the biscuit tin", "the boot scraper", "the wood box"]
)

_MANOR = GenrePack(
    name="manor", vibe="English country-house", setting="the valley",
    locale="village", demonym="valley", chrono="parish calendar",
    nouns=[
        "parish register", "game book", "seed ledger", "dairy slate",
        "tenant roll", "wool account", "carrier's waybill", "mill ledger",
        "churchwarden's book", "tithe receipt", "kennel book", "stable diary",
        "forge receipt", "gardeners' roster", "harvest tally", "apiary log",
        "well record", "ferry toll book", "letter book", "household account",
        "dressmaker's bill", "gun-room ledger", "linen inventory", "cellar book",
        "ice-house record", "clock-winding book", "grain chit", "cheese room slate",
        "orchard tally", "fence-riding roster", "dovecote ledger", "gate book",
    ],
    places=["parish office", "estate office", "mill room", "tithe barn",
            "village shop", "kitchen garden"],
    frames=[
        "In the back room of the {place}, {name} sat out the bad hour with the stove drawn close.",
        "The {place} smelled of wax and apples. {name} had an hour before the light went.",
        "It was quiet in the {place}. Outside, the carts had stopped for the day.",
        "{name} kept the {place} open late that week, and the evenings ran long.",
        "Rain set in over the {place}, and there was nothing to do but talk.",
    ],
    filler=[
        "Rooks argued in the elms and settled.",
        "The dogs shifted by the fire and sighed.",
        "The church bell counted the quarter, far off.",
        "Rain beaded on the yew hedge.",
        "Wax and cold apples hung in the air.",
        "A trap went by on the turnpike, wheel-rings ringing.",
        "The kettle murmured and went unanswered.",
        "Frost ferned the pane from the lower corner.",
        "Someone was practising scales in the big house.",
        "The yard cat crossed the flags and vanished.",
    ],
    titles=["The Long Account", "What the Parish Kept", "A Winter of Entries",
            "The Estate Book", "Ledger and Yew", "The Quiet Hundred"],
    distractor_bodies=[
        "the matter of the missing ewe surfaced again",
        "a discrepancy in the estate totals was noted",
        "a stranger on the carrier's cart had been discussed",
        "the sermon that Sunday was never repeated",
        "the lantern in the chapel porch swung at nobody that night"],
    hypotheses=[
        "the oldest letter is the forgery", "the gamekeeper altered the book",
        "the second partner acted alone", "the estate ledger was redrawn after the fire"],
    lore=[
        ("the winter the lake froze from bank to bank", "the frozen lake"),
        ("the year the toll road came through the lower field", "the toll road"),
        ("the summer the blight took the hops", "the blighted hops"),
        ("the season the hunt was cancelled", "the cancelled hunt"),
        ("the autumn the big house changed hands", "the changing of hands"),
        ("the storm that took the folly off the ridge", "the lost folly"),
        ("the year the bells were recast", "the recast bells"),
        ("the decade the estate kept its own poor book", "the estate's poor book"),
    ],
    things=["the stove", "the kettle", "the lamp", "the shutters",
           "the gravel path", "the wet boots", "the coat rack", "the biscuit tin",
           "the boot-room door", "the wood box", "the dogs' basket", "the bell pull"]
)

_HOTEL = GenrePack(
    name="hotel", vibe="grand mountain-hotel between the wars",
    setting="the hotel", locale="lobby", demonym="mountain",
    chrono="guest ledger",
    nouns=[
        "guest register", "lift log", "patrol book", "kitchen chit",
        "bar ledger", "baggage tag", "left-luggage ticket", "switchboard log",
        "housekeeping sheet", "boot-room roster", "funicular ticket",
        "terrace menu", "cellar card", "linen tally", "wax-bench log",
        "cashier's drop slip", "night auditor sheet", "valet ticket",
        "garage log", "shuttle manifest", "cable log", "weather sheet",
        "rescue cache list", "sauna book", "laundry chit", "florist bill",
        "phone billing sheet", "staff meal roster", "supply tram manifest",
        "ice-room log", "orchestra programme", "ski-school roster",
    ],
    places=["lobby", "boot room", "reading room",
            "kitchen corridor", "funicular office", "bar"],
    frames=[
        "In the back room of the {place}, {name} sat out the bad hour with the stove drawn close.",
        "The {place} smelled of boot wax and coffee. {name} had an hour before the last cable down.",
        "It was quiet in the {place}. Outside, the lifts had stopped for the day.",
        "{name} kept the {place} open late that week, and the evenings ran long.",
        "Snow set in over the {place}, and there was nothing to do but talk.",
    ],
    filler=[
        "The radiator ticked and went silent.",
        "The cable hummed faintly in its channel.",
        "Boot powder hung in the cold of the corridor.",
        "A storm slid off the ridge and took the light with it.",
        "The piano upstairs worked through the same eight bars.",
        "Snow squeaked under someone's boots and stopped.",
        "The great clock over the desk lost a second and found it.",
        "Frost ferned the porthole by the stairs.",
        "Someone was drying mittens at the stove, humming.",
        "The lobby cat claimed the window seat and stayed.",
    ],
    titles=["The Season's Ledger", "Above the Tree Line", "The Long Season",
            "The Winter Register", "Storm Days", "The Last Cable Down"],
    distractor_bodies=[
        "the matter of the missing deposit surfaced again",
        "a discrepancy behind the bar totals was noted",
        "a second name in the register had been discussed",
        "the vintage that season did not match the cellar order",
        "the porter's dog barked at nobody that night"],
    hypotheses=[
        "the oldest letter is the forgery", "the deposit was never banked",
        "the second guest acted alone", "the register was altered after the fire"],
    lore=[
        ("the winter the road closed six weeks early", "the early closing"),
        ("the year the telegraph line finally came up", "the telegraph winter"),
        ("the season the north face was barred to guests", "the barred face"),
        ("the summer the lake would not warm", "the cold lake"),
        ("the week the funicular stuck mid-span", "the stuck car"),
        ("the year the kitchen changed hands twice", "the two kitchens"),
        ("the storm that took the weather station off the shoulder", "the lost station"),
        ("the decade the guides kept their own book", "the guides' book"),
    ],
    things=["the stove", "the kettle", "the lamp", "the lobby tiles",
           "the wet umbrellas", "the boot rack", "the biscuit tin", "the linen cart",
           "the stair rail", "the wood box", "the gramophone", "the terrace cushions"]
)

_THEATER = GenrePack(
    name="theater", vibe="touring theatre company",
    setting="the company", locale="stage door", demonym="company",
    chrono="call board",
    nouns=[
        "prompt book", "call sheet", "box-office tally", "prop list",
        "costume plot", "fly schedule", "rehearsal log", "understudy roster",
        "lighting plot", "stage manager's log", "ticket manifest",
        "seating chart", "wardrobe ledger", "wig log", "green-room register",
        "load-in manifest", "freight bill", "rail warrant",
        "per diem sheet", "band librarian's card", "publicity still log",
        "hotel chit", "advance booking sheet", "half-hour call sheet",
        "fitting note", "shoe register", "armory check-out", "laundry bundle chit",
        "orchestra payroll slip", "house manager's sheet", "crew roster",
    ],
    places=["wings", "green room", "property room",
            "wardrobe loft", "stage-door alcove", "band room"],
    frames=[
        "In the back room of the {place}, {name} sat out the bad hour with the stove drawn close.",
        "The {place} smelled of greasepaint and dust. {name} had an hour before the half-hour call.",
        "It was quiet in the {place}. The house had emptied an hour since.",
        "{name} kept the {place} open late that week, and the evenings ran long.",
        "Rain set in on the stage-door step, and there was nothing to do but talk.",
    ],
    filler=[
        "Rosin creaked underfoot somewhere above.",
        "The safety curtain shed its dust in the stillness.",
        "Scales drifted down from a room off the passage.",
        "A steam iron hissed in wardrobe and stopped.",
        "The ghost light hummed on the bare boards.",
        "Freight lifts boomed twice in the dock and went quiet.",
        "Rain found the apron through somewhere and tapped.",
        "A quick-change rack rolled past, unattended.",
        "Frost ferned the dressing-room glass.",
        "The house cat walked the orchestra rail and left.",
    ],
    titles=["The Company Book", "Dark Nights", "The Long Run",
            "Understudy", "Call Time", "The Quiet House"],
    distractor_bodies=[
        "the matter of the missing advance surfaced again",
        "a discrepancy in the house money was noted",
        "a second name on the payroll had been discussed",
        "the opening night was moved and no one would say why",
        "the understudy's dog barked at nobody that night"],
    hypotheses=[
        "the oldest letter is the forgery", "the treasurer skimmed the house",
        "the second partner acted alone", "the house script was altered after the fire"],
    lore=[
        ("the winter the tour lost a week to the floods", "the flooded tour"),
        ("the year the old house was pulled down", "the lost house"),
        ("the season the lead walked out mid-run", "the walkout"),
        ("the summer the company played to empty houses", "the empty summer"),
        ("the autumn the scenery truck went off the ridge road", "the lost truck"),
        ("the storm that closed the ferry and stranded the cast", "the stranded week"),
        ("the year the rights were disputed", "the disputed rights"),
        ("the decade the company kept its own books", "the company's books"),
    ],
    things=["the stove", "the kettle", "the lamp", "the passage",
           "the wet coats", "the prop shelf", "the biscuit tin", "the rehearsal piano",
           "the stair rail", "the wood box", "the quick-change rack", "the house seats"]
)

_OBSERVATORY = GenrePack(
    name="observatory", vibe="remote mountain research station",
    setting="the station", locale="mess", demonym="station",
    chrono="observing schedule",
    nouns=[
        "instrument log", "seeing report", "plate register", "dome log",
        "cryostat sheet", "power schedule", "mess roster", "visitor book",
        "fuel manifest", "snow-cat log", "radio log", "met sheet",
        "time-allocation sheet", "filter log", "tape index", "aurora watch",
        "generator book", "drop manifest", "chamber log", "galley roster",
        "still log", "battery sheet", "road report", "flare alert sheet",
        "ionosonde tape", "theodolite book", "sonde card", "horizon camera log",
        "drive maintenance card", "shutter counter", "darkroom register",
    ],
    places=["control room", "dome floor", "mess",
            "machine shop", "bunk corridor", "weather porch"],
    frames=[
        "In the back room of the {place}, {name} sat out the bad hour with the stove drawn close.",
        "The {place} smelled of machine oil and coffee. {name} had an hour before the drive wound down.",
        "It was quiet in the {place}. The day crew had gone to their bunks.",
        "{name} kept the {place} running late that week, and the nights ran together.",
        "Storm light set in over the {place}, and there was nothing to do but talk.",
    ],
    filler=[
        "Wind pressed on the dome skin and let go.",
        "The drive motors hummed down through the floor.",
        "Coffee went cold on the console, again.",
        "Stars held steady in the shutter slit for a while.",
        "Frost ferned the porthole from the corners.",
        "The kettle in the mess clicked and knocked.",
        "A marmot whistled somewhere below the ridge.",
        "The strip-chart pen drifted through its slow figures.",
        "Someone was tuning a receiver three doors down.",
        "The station cat claimed the warm amplifier and stayed.",
    ],
    titles=["Seeing Bad, Seeing Fair", "The Night Shift", "Plate Glass",
            "The Quiet Signal", "Weather Days", "The Long Dark"],
    distractor_bodies=[
        "the matter of the missing crate surfaced again",
        "a shortfall in the drum count was noted",
        "a second signature on the requisition had been discussed",
        "the seeing that night was blamed on the stove, of all things",
        "the station dog barked at nobody that night"],
    hypotheses=[
        "the oldest letter is the forgery", "the fuel was siphoned on the road",
        "the second partner acted alone", "the log was altered after the fire"],
    lore=[
        ("the winter the road was shut for two months", "the closed road"),
        ("the year the main mirror was re-aluminized", "the re-silvering"),
        ("the season the radio went quiet", "the quiet frequency"),
        ("the summer the meltwater took the lower trail", "the lost trail"),
        ("the week the supply drop landed on the wrong shoulder", "the wrong drop"),
        ("the year the cook left mid-winter", "the cook's winter"),
        ("the storm that iced the anemometer solid", "the iced anemometer"),
        ("the decade the night assistants kept their own log", "the assistants' log"),
    ],
    things=["the stove", "the kettle", "the lamp", "the corridor",
           "the wet mittens", "the boot rack", "the biscuit tin", "the console chair",
           "the stair rail", "the wood box", "the hot-plate", "the bunk curtain"]
)

PACKS = {p.name: p for p in
         (_MARITIME, _MANOR, _HOTEL, _THEATER, _OBSERVATORY)}


def pick_genre(seed):
    return Rng(seed + 61).pick(sorted(PACKS))


def get_pack(world):
    """Active pack for a world. A runtime-generated pack (genre 'llm')
    rides in world.meta['genre_pack']; otherwise the genre is resolved
    concretely at build time and looked up in the static registry."""
    meta = getattr(world, "meta", None)
    if meta and meta.get("genre_pack") is not None:
        return meta["genre_pack"]
    name = (world.config.get("genre") if hasattr(world, "config") else None) \
        or "maritime"
    if name not in PACKS:
        raise ValueError(f"unknown genre {name!r}; known: {sorted(PACKS)}")
    return PACKS[name]

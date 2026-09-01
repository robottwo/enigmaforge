"""Verification battery: SAT-vs-oracle agreement, uniqueness proof, ablation,
distractor safety. A world is PUBLISHED only if all gates pass."""
from .sat import Sat
from .compile import compile_to_sat
from .oracle import oracle_models
import re
from collections import Counter
from .world import Constraint, ConstraintKind, HiddenWorld

def sat_models(world, skip_cids=(), cap=2000):
    sat = compile_to_sat(world, skip_cids=skip_cids)
    return sat.enumerate_models(max_models=cap)

def ban_clause(world, gt):
    """Clause false ONLY under assignment gt: any deviation satisfies it."""
    return [(v.vid, x) for v in world.variables for x in v.domain if x != gt[v.vid]]

def has_other_model(world, skip_cids=(), gt=None):
    """SAT early-exit: is there a model other than gt? Uniqueness check and
    ablation both reduce to this — no enumeration of the full space."""
    gt = gt or world.meta["ground_truth"]
    sat = compile_to_sat(world, skip_cids=skip_cids,
                         extra_clauses=[ban_clause(world, gt)])
    return len(sat.enumerate_models(max_models=1)) > 0

def sat_unique(world, constraints=None) -> bool:
    """Unique iff NO model besides ground truth. Ban-clause form: scalable."""
    if constraints is None or constraints is world.constraints:
        return not has_other_model(world)
    from .world import HiddenWorld
    w = HiddenWorld(wid="tmp", seed=0, config={})
    w.variables = world.variables
    w.constraints = constraints
    w.meta["ground_truth"] = world.meta["ground_truth"]
    return not has_other_model(w)

def sat_vs_oracle(world):
    """Gate 0: engine agreement. Compare MODEL SETS, not capped counts: a cap
    of N on the enumerator truncates large model sets and fakes disagreement."""
    o = oracle_models(world)
    cap = len(o) + 1
    s = sat_models(world, cap=cap)
    key = lambda m: tuple(sorted(m.items()))  # order-insensitive
    agree = (sorted(map(key, o)) == sorted(map(key, s)))
    return {"agree": agree, "n_models": len(o)}

def verify_uniqueness(world, want_unique=True):
    """Gate 1: exactly one model (or intended ambiguity with count > 1)."""
    n = len(oracle_models(world))
    if want_unique and n != 1:
        return {"pass": False, "reason": f"expected unique, got {n}", "n": n}
    if not want_unique and n < 2:
        return {"pass": False, "reason": f"expected ambiguity, got {n}", "n": n}
    return {"pass": True, "n": n}

def verify_ablation(world, essential_cids):
    """Gate 2: dropping each essential constraint (the formal content behind a
    clue) must strictly increase the model count (i.e., the clue was doing work)."""
    results = {}
    for cid in essential_cids:
        remaining = [c for c in world.constraints if c.cid != cid]
        n_after = len(oracle_models(world, remaining))
        results[cid] = {"models_without": n_after,
                        "essential": n_after > 1}
    ok = all(r["essential"] for r in results.values())
    return {"pass": ok, "detail": results}

def verify_distractor_safety(world, distractor_cids):
    """Gate 3: distractor constraints must not reduce the unique solution set."""
    base = len(oracle_models(world))
    safe = {}
    for cid in distractor_cids:
        kept = [c for c in world.constraints if c.cid != cid]
        safe[cid] = len(oracle_models(world, kept))
    # a distractor is UNSAFE if removing it CHANGES model count (it was load-bearing)
    ok = all(v == base for v in safe.values())
    return {"pass": ok, "base": base, "detail": safe}

# ------------------------------------------------------------------
# Surface-faithfulness gates: the narrative is verified, not trusted.
# ------------------------------------------------------------------

def verify_realization(world, realization):
    """Structural gate on one realization: coverage (every essential unit
    rendered), verbatim spans (text[s:e] == clause, byte for byte), distractor
    inertness (distractor clauses reference no variable surface), no formal
    identifiers leaked, and — in story mode — no puzzle frame."""
    issues = []
    text = realization.text
    nouns = {n for v in world.variables for n in v.surface_names}
    rendered = set(realization.rendered)
    for u in world.evidence:
        if u.is_distractor:
            continue
        if u.euid not in rendered:
            issues.append(f"unrendered essential unit {u.euid}")
    for ref in realization.rendered:
        clause = realization.clauses.get(ref, "")
        span = realization.spans.get(ref)
        if span is None or not clause.strip():
            issues.append(f"missing span or clause for {ref}")
            continue
        s, e = span
        if text[s:e] != clause:
            issues.append(f"span does not match clause for {ref}")
    for u in world.evidence:
        if u.is_distractor and u.euid in rendered:
            hit = [n for n in nouns if n in realization.clauses[u.euid]]
            if hit:
                issues.append(f"distractor {u.euid} references variable surface {hit}")
    if re.search(r"\b(?:V\d+|C\d+|PIN\d+|W\d+)\b", text):
        issues.append("formal identifier leaked into narrative")
    if realization.mode == "story":
        low = text.lower()
        if re.search(r"^\s*\(\d+\)", text, re.M):
            issues.append("numbered-list frame in story mode")
        for phrase in ("— the record —", "marginal references",
                       "figure out", "determine what"):
            if phrase in low:
                issues.append(f"puzzle-frame phrase leaked: {phrase!r}")
    return {"pass": not issues, "issues": issues}


_N = r"[a-z'’\-]+(?: [a-z'’\-]+){0,2}"        # surface noun, 1-3 words
_NAME = r"[A-Z][a-z]+"
# Inverse of narrative.constraint_phrase, in priority order: an IMPLIES
# clause contains inner "read <v>" spans, so it must consume before EQ.
_CLAIM_PATTERNS = [
    ("implies", re.compile(rf"whenever the ({_N}) read (\d+|[A-Za-z]+), the ({_N}) read (\d+|[A-Za-z]+)")),
    ("eq1", re.compile(rf"the ({_N}) was stamped (\d+)")),
    ("eq1", re.compile(rf"the ({_N}) read (\d+)")),
    ("eq1", re.compile(rf"the ({_N}) carried ({_NAME})'s mark")),
    ("eq1", re.compile(rf"the ({_N}) was signed out under ({_NAME})")),
    ("eq1", re.compile(rf"no one disputed that the ({_N}) pointed to ({_NAME})")),
    ("eq2", re.compile(rf"whatever the ({_N}) showed, the ({_N}) matched it")),
    ("eq2", re.compile(rf"the ({_N}) and the ({_N}) agreed")),
    ("neq", re.compile(rf"the ({_N}) and the ({_N}) did not agree")),
    ("neq", re.compile(rf"whatever else was true, the ({_N}) never matched the ({_N})")),
    ("alldiff", re.compile(rf"no two of (.+?) coincided")),
]


def _norm_claim(kind, vids, values):
    return (kind, tuple(vids), tuple(str(v) for v in values))


def _norm_constraint(c):
    return _norm_claim(c.kind.value, c.vars, c.values)


def _val(tok, var):
    v = int(tok) if tok.isdigit() else tok
    return v if v in var.domain else None


def extract_claims(text, world):
    """Template extractor: prose -> formal claims. Matches that do not
    resolve to a known (surface noun, domain value) pair are scenery and are
    skipped silently — coverage is judged by set equality in the round-trip,
    so a corrupted real clause still fails the gate as 'missing'."""
    noun2vid = {n: v.vid for v in world.variables for n in v.surface_names}
    claims, consumed = [], []
    for kind, rx in _CLAIM_PATTERNS:
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < ce and cs < e for cs, ce in consumed):
                continue
            consumed.append((s, e))
            g = m.groups()
            if kind == "implies":
                a, b = noun2vid.get(g[0]), noun2vid.get(g[2])
                if not a or not b or a == b:
                    continue
                va = _val(g[1], world.var(a))
                vb = _val(g[3], world.var(b))
                if va is None or vb is None:
                    continue
                claims.append(_norm_claim("implies", [a, b], [va, vb]))
            elif kind == "eq1":
                vid = noun2vid.get(g[0])
                if vid:
                    val = _val(g[1], world.var(vid))
                    if val is not None:
                        claims.append(_norm_claim("eq", [vid], [val]))
            elif kind == "eq2":
                a, b = noun2vid.get(g[0]), noun2vid.get(g[1])
                if a and b and a != b:
                    claims.append(_norm_claim("eq", [a, b], []))
            elif kind == "neq":
                a, b = noun2vid.get(g[0]), noun2vid.get(g[1])
                if a and b and a != b:
                    claims.append(_norm_claim("neq", [a, b], []))
            elif kind == "alldiff":
                nouns = [n.strip() for n in re.split(r", and |, | and ", g[0])]
                vids = [noun2vid[n] for n in nouns if n in noun2vid]
                if len(vids) == len(nouns) and len(vids) >= 2:
                    claims.append(_norm_claim("alldiff", vids, []))
    return claims


def verify_roundtrip(world, realization):
    """Extraction round-trip: the prose ALONE must reconstruct exactly the
    formal model (same constraint multiset) and remain uniquely solvable with
    the same ground truth. SAT-based, so it scales to every tier. An LLM
    extractor can be swapped in for the same comparison — this gate is the
    contract the v2 LLM narrative compiler must survive."""
    claims = extract_claims(realization.text, world)
    want = Counter(_norm_constraint(c) for c in world.constraints)
    got = Counter(claims)
    constraints = []
    for i, (kind, vids, values) in enumerate(claims):
        vals = [int(v) if v.isdigit() else v for v in values]
        constraints.append(Constraint(f"X{i}", ConstraintKind(kind),
                                      list(vids), vals))
    w = HiddenWorld(wid="tmp", seed=0, config={})
    w.variables = world.variables
    w.constraints = constraints
    w.meta["ground_truth"] = world.meta["ground_truth"]
    models = sat_models(w, cap=2)
    unique = not has_other_model(w)
    gt = world.meta["ground_truth"]
    gt_ok = bool(models) and models[0] == gt
    return {"pass": got == want and unique and gt_ok,
            "recovered": len(claims), "expected": sum(want.values()),
            "unique_as_read": unique, "gt_is_model": gt_ok}

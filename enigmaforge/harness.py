"""Benchmark harness: generate a cohort of story instances, run each against
a configurable list of LLM providers, grade every (provider, instance) pair
mechanically from the hidden formal world, and emit results JSON, an
aggregated leaderboard, and a self-contained dark-theme HTML report.

Run:
    python3 -m enigmaforge.harness --providers providers.json --instances 6

Providers file — a JSON list; each entry:
    {"name": "glm-flash",           # required, unique
     "model": "glm-5.3-flash",      # optional; omit to let the endpoint
                                    #   auto-pick its newest model
     "base_url": "https://...",     # optional; omit for env/agent-config
                                    #   autodiscovery (llm.resolve_llm_config)
     "api_key": "sk-...",           # optional literal key, or:
     "api_key_env": "MY_KEY_VAR"}   # optional env var holding the key

Grading is mechanical, computed from the hidden representation (no LLM
judge); per-instance score = 0.5*facts + 0.3*action + 0.2*compliance —
facts is the only per-instance-discriminative component, since
populate_objectives hardcodes the canonical final-action strings.

The run is resumable at both stages: existing instance dirs are never
rebuilt and existing response files are never re-called.
"""
import difflib
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime

from . import pipeline
from .genres import PACKS, pick_genre
from .llm import chat_completion
from .rng import Rng

WEIGHTS = {"facts": 0.5, "action": 0.3, "compliance": 0.2}

# Universal solver prompt. Deliberately never hints that a puzzle exists:
# the benchmark measures discovery + recovery, not puzzle-format compliance.
SOLVER_PROMPT = """You are an archivist reviewing a recovered text. Read it closely, then reply
with ONLY a JSON object (no markdown fences) with exactly these keys:
"observations": array of 3-6 notable inferences you can defend from the text,
"fixed_facts": array of strings "<subject> = <value>" — anything the text
establishes as definitely so,
"final_action": one sentence — given all of the above, what should be done next?"""


def _note(msg):
    print(f"[enigmaforge.harness] {msg}", file=sys.stderr)


# ------------------------------------------------------------- providers

def load_providers(path):
    """Parse the providers JSON file into chat_completion kwargs per provider.

    Raises ValueError on missing file, malformed JSON, a non-list body, a
    missing/duplicate name, or 'api_key' alongside 'api_key_env'."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"providers file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"providers file is not valid JSON: {path}: {e}")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"providers file must contain a non-empty JSON list: {path}")
    out, seen = [], set()
    for i, p in enumerate(raw):
        if not isinstance(p, dict) or not p.get("name"):
            raise ValueError(f"provider #{i}: missing required 'name'")
        name = p["name"]
        if name in seen:
            raise ValueError(f"duplicate provider name: {name!r}")
        seen.add(name)
        if p.get("api_key") is not None and p.get("api_key_env"):
            raise ValueError(f"provider {name!r}: 'api_key' and 'api_key_env' "
                             "are mutually exclusive")
        kw = {}
        for key in ("model", "base_url", "api_key"):
            if p.get(key) is not None:
                kw[key] = p[key]
        if "api_key" not in kw and p.get("api_key_env"):
            env_val = os.environ.get(p["api_key_env"])
            if env_val:
                kw["api_key"] = env_val
        out.append({"name": name, "kwargs": kw})
    return out


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "provider"


# ------------------------------------------------------------- generation

def _tagged_seed(seed, tag):
    """Stable per-(seed, tag) derived seed (hash(), unlike sha256, is salted)."""
    return int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)


def build_specs(n_instances, sizes, genre, burial_min, burial_max, seed_base):
    """Cohort plan: instance i gets seed = seed_base + i*17, the size at
    position i mod len(sizes), an explicit genre or pick_genre(seed), and a
    burial depth drawn from a (seed, 'burial')-derived rng."""
    sizes = list(sizes) or ["small"]
    specs = []
    for i in range(n_instances):
        seed = seed_base + i * 17
        g = genre if genre and genre != "auto" else pick_genre(seed)
        b = Rng(_tagged_seed(seed, "burial")).range(burial_min, burial_max)
        specs.append({"iid": f"inst-{i:03d}-{g}", "index": i, "genre": g,
                      "seed": seed, "size": sizes[i % len(sizes)], "burial": b})
    return specs


def _instance_dir(out_dir, iid):
    return os.path.join(out_dir, "instances", iid)


def generate_cohort(out_dir, specs=None):
    """Build and package every spec into <out>/instances/<iid>/, writing the
    grading payload instance.json. Existing instance dirs are skipped (build
    is deterministic but slow). specs=None scans the dir instead — resumable
    cohorts and --skip-generate."""
    inst_root = os.path.join(out_dir, "instances")
    os.makedirs(inst_root, exist_ok=True)
    if specs is None:
        found = []
        for iid in sorted(os.listdir(inst_root)):
            meta = os.path.join(inst_root, iid, "instance.json")
            if os.path.isfile(meta):
                with open(meta) as f:
                    found.append(json.load(f))
        return found
    instances = []
    for spec in specs:
        d = _instance_dir(out_dir, spec["iid"])
        meta = os.path.join(d, "instance.json")
        if os.path.isfile(meta):
            with open(meta) as f:
                instances.append(json.load(f))
            continue
        world = pipeline.build(spec["size"], spec["seed"], config_overrides={
            "mode": "story", "genre": spec["genre"], "burial": spec["burial"]})
        pipeline.package(world, d)
        with open(os.path.join(d, "story.md")) as f:
            story_text = f.read()
        true_obj = next(o for o in world.objectives if o.true_objective)
        inst = {"iid": spec["iid"], "genre": spec["genre"], "seed": spec["seed"],
                "size": spec["size"], "burial": spec["burial"],
                "story_path": os.path.join(d, "story.md"),
                "story_text": story_text,
                "ground_truth": world.meta["ground_truth"],
                "surfaces": {v.vid: v.surface_names[0] for v in world.variables},
                "final_action": true_obj.answer["final_action"],
                "n_constraints": len(world.constraints)}
        with open(meta, "w") as f:
            json.dump(inst, f, indent=2)
        instances.append(inst)
    return instances


# ------------------------------------------------------------- solving

def _response_path(out_dir, provider_name, iid):
    return os.path.join(out_dir, "responses", f"{_slug(provider_name)}--{iid}.txt")


def run_solvers(instances, providers, out_dir, timeout, call=True):
    """One chat call per (provider, instance); response + wall time + error
    are persisted under responses/ so reruns skip existing files (crash-safe
    resume). Endpoint errors are recorded, never abort the run. With
    call=False (--grade-only) only existing files are loaded."""
    os.makedirs(os.path.join(out_dir, "responses"), exist_ok=True)
    records = []
    for prov in providers:
        for inst in instances:
            path = _response_path(out_dir, prov["name"], inst["iid"])
            if os.path.isfile(path):
                with open(path) as f:
                    records.append({"provider": prov["name"],
                                    "iid": inst["iid"], **json.load(f)})
                continue
            if not call:
                continue
            t0 = time.time()
            try:
                text = chat_completion(
                    [{"role": "system", "content": SOLVER_PROMPT},
                     {"role": "user", "content": inst["story_text"]}],
                    timeout=timeout, **prov["kwargs"])
                rec = {"text": text, "seconds": round(time.time() - t0, 2),
                       "error": None}
                _note(f"{prov['name']} x {inst['iid']}: ok "
                      f"({rec['seconds']}s, {len(text)} chars)")
            except Exception as e:  # endpoint failure: record, keep going
                rec = {"text": None, "seconds": round(time.time() - t0, 2),
                       "error": str(e)[:500]}
                _note(f"{prov['name']} x {inst['iid']}: ERROR {rec['error']}")
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
            records.append({"provider": prov["name"], "iid": inst["iid"], **rec})
    return records


# ------------------------------------------------------------- grading

def _parse_json_block(text):
    """First {...} block, fences tolerated — same shape as llm._parse_plan."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def _normalize(s):
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _fact_recovered(item, surface, value):
    low = item.lower()
    if surface.lower() not in low:
        return False
    if isinstance(value, int) and not isinstance(value, bool):
        return re.search(rf"\b{value}\b", item) is not None
    return str(value).lower() in low


def grade_instance(instance, record):
    """Mechanical rubric from the hidden formal world. Error records get
    None components and None score."""
    comps = {"compliance": None, "facts": None, "action": None}
    graded = {"provider": record["provider"], "iid": record["iid"],
              "components": comps, "score": None,
              "seconds": record.get("seconds"), "error": record.get("error")}
    if record.get("error") or not isinstance(record.get("text"), str):
        return graded
    raw = record["text"]
    parsed = _parse_json_block(raw)

    comps["compliance"] = 1.0 if (isinstance(parsed, dict) and all(
        parsed.get(k) for k in ("observations", "fixed_facts", "final_action"))
        ) else 0.0

    gt = instance.get("ground_truth") or {}
    surfaces = instance.get("surfaces") or {}
    if isinstance(parsed, dict) and isinstance(parsed.get("fixed_facts"), list):
        pool = [str(x) for x in parsed["fixed_facts"]]
    else:  # unparseable: scan the whole raw response with the same rule
        pool = [raw]
    recovered = sum(1 for vid, value in gt.items()
                    if vid in surfaces
                    and any(_fact_recovered(it, surfaces[vid], value)
                            for it in pool))
    comps["facts"] = recovered / len(gt) if gt else 0.0

    resp_action = parsed.get("final_action") if isinstance(parsed, dict) else None
    if not isinstance(resp_action, str):
        resp_action = raw
    ratio = difflib.SequenceMatcher(None, _normalize(resp_action),
                                    _normalize(instance.get("final_action"))).ratio()
    comps["action"] = 1.0 if ratio >= 0.6 else (0.5 if ratio >= 0.4 else 0.0)

    graded["score"] = round(sum(WEIGHTS[k] * comps[k] for k in WEIGHTS), 6)
    return graded


# ------------------------------------------------------------- aggregation

def aggregate(graded, instances, providers):
    """Per-provider composite (mean of non-None scores), ranked descending;
    None composites (no scored instances) rank last."""
    inst_by_iid = {i["iid"]: i for i in instances}
    rows = []
    for prov in providers:
        pname = prov["name"]
        rs = [g for g in graded if g["provider"] == pname]
        scored = [g for g in rs if g["score"] is not None]

        def mean(key, _scored=scored):
            vals = [g["components"][key] for g in _scored]
            return round(sum(vals) / len(vals), 4) if vals else None

        composite = (round(sum(g["score"] for g in scored) / len(scored), 4)
                     if scored else None)
        rows.append({"provider": pname, "composite": composite,
                     "facts": mean("facts"), "action": mean("action"),
                     "compliance": mean("compliance"),
                     "n_scored": len(scored),
                     "n_errors": sum(1 for g in rs if g["error"]),
                     "total_seconds": round(sum(g.get("seconds") or 0 for g in rs), 2)})
    rows.sort(key=lambda r: (r["composite"] is None, -(r["composite"] or 0)))
    return {"generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_instances": len(instances),
            "leaderboard": rows,
            "per_instance": {iid: inst_by_iid[iid]["final_action"]
                             for iid in sorted(inst_by_iid)},
            "runs": [{k: i[k] for k in ("iid", "genre", "size", "seed", "burial")}
                     for i in sorted(instances, key=lambda x: x["iid"])],
            "records": sorted(graded, key=lambda g: (g["provider"], g["iid"]))}


# ------------------------------------------------------------- HTML report

_CSS = """
body { background:#12151c; color:#e6e9ef; margin:0; padding:2rem;
       font:15px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif; }
main { max-width:1200px; margin-inline:auto; }
h1 { font-size:1.5rem; margin-bottom:.2rem; }
h2 { font-size:1.05rem; margin-top:2.4rem; color:#9fb3d1;
     text-transform:uppercase; letter-spacing:.08em; }
table { border-collapse:collapse; width:100%; font-size:.92rem; }
th, td { padding:.35rem .6rem; border-bottom:1px solid #232a38;
         text-align:left; white-space:nowrap; }
th { color:#9fb3d1; font-weight:600; }
.bar-row { display:flex; align-items:center; gap:.8rem; margin:.45rem 0; }
.bar-name { width:170px; text-align:right; color:#c7d1e0; font-size:.9rem;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar-track { flex:1; background:#1a2030; border-radius:4px; height:22px; }
.bar-fill { height:100%; border-radius:4px; display:flex; overflow:hidden;
            min-width:2px; }
.seg-facts { background:#4f9cf9; } .seg-action { background:#f9a94f; }
.seg-compliance { background:#58c98b; }
.bar-score { width:120px; font-variant-numeric:tabular-nums; }
.legend { display:flex; gap:1.2rem; margin:.6rem 0 1rem; font-size:.85rem;
          color:#9fb3d1; }
.dot { display:inline-block; width:10px; height:10px; border-radius:2px;
       margin-right:.35rem; }
.mini { display:flex; align-items:center; gap:.8rem; margin:.25rem 0;
        font-size:.88rem; }
.mini .bar-track { height:12px; max-width:460px; }
details { margin:.3rem 0; } summary { cursor:pointer; color:#c7d1e0; }
code { color:#b7c4d8; }
.muted { color:#8b96a8; } .err-cell { color:#e26d6d; }
"""

_SEGMENTS = (("facts", "seg-facts"), ("action", "seg-action"),
             ("compliance", "seg-compliance"))


def render_html(agg, out_path):
    """Single self-contained report: inline CSS only, zero external assets,
    no JS. Bars are pure CSS; segments are the weighted score contributions."""
    esc = html.escape

    def num(x, nd=2):
        return "&mdash;" if x is None else f"{x:.{nd}f}"

    rows = agg["leaderboard"]
    top = max((r["composite"] for r in rows if r["composite"] is not None),
              default=0.0) or 1.0

    chart = []
    for r in rows:
        name = f'<div class="bar-name" title="{esc(r["provider"])}">{esc(r["provider"])}</div>'
        c = r["composite"]
        if c is None:
            chart.append(
                f'<div class="bar-row">{name}'
                f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:100%;background:#2a3142;"></div></div>'
                f'<div class="bar-score muted">no scored instances '
                f'({r["n_errors"]} errors)</div></div>')
            continue
        segs = "".join(
            f'<div class="{cls}" style="width:'
            f'{100.0 * WEIGHTS[key] * (r[key] or 0.0) / c if c else 0.0:.1f}%" '
            f'title="{key}={num(r[key])}"></div>'
            for key, cls in _SEGMENTS)
        chart.append(
            f'<div class="bar-row">{name}'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{100.0 * c / top:.1f}%">{segs}</div></div>'
            f'<div class="bar-score"><b>{c:.3f}</b></div></div>')

    comp_rows = []
    for r in rows:
        for key, cls in _SEGMENTS:
            v = r[key]
            comp_rows.append(
                f'<div class="mini"><div class="bar-name" style="width:250px">'
                f'{esc(r["provider"])} &middot; {key}</div>'
                f'<div class="bar-track"><div class="bar-fill {cls}" '
                f'style="width:{100.0 * (v or 0.0):.1f}%"></div></div>'
                f'<div class="bar-score muted">{num(v)}</div></div>')

    iids = [r["iid"] for r in agg["runs"]]
    by_pair = {(g["provider"], g["iid"]): g for g in agg["records"]}
    head = "".join(f"<th>{esc(i)}</th>" for i in iids)
    body_rows = []
    for r in rows:
        cells = []
        for iid in iids:
            g = by_pair.get((r["provider"], iid))
            if g is None:
                cells.append('<td class="muted">&mdash;</td>')
            elif g["score"] is None:
                tip = esc(f"error: {g['error'] or 'unknown'}")
                cells.append(f'<td class="err-cell" title="{tip}">err</td>')
            else:
                c = g["components"]
                tip = esc(f"facts={c['facts']:.2f} action={c['action']:.2f} "
                          f"compliance={c['compliance']:.2f} "
                          f"seconds={g['seconds']}")
                light = 8 + 20 * g["score"]
                cells.append(f'<td style="background:hsl(145,60%,{light:.0f}%)" '
                             f'title="{tip}">{g["score"]:.2f}</td>')
        body_rows.append(
            f"<tr><td>{esc(r['provider'])}</td>{''.join(cells)}"
            f"<td><b>{num(r['composite'], 3)}</b></td>"
            f'<td class="muted">{r["total_seconds"]:.1f}s</td></tr>')

    actions = "".join(
        f'<details><summary>{esc(iid)}</summary><code>{esc(a)}</code></details>'
        for iid, a in sorted(agg["per_instance"].items()))
    run_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            esc(r["iid"]), esc(r["genre"]), esc(r["size"]), r["seed"], r["burial"])
        for r in agg["runs"])

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EnigmaForge harness report</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>EnigmaForge benchmark</h1>
<p class="muted">generated {esc(agg["generated_at"])} &middot; {agg["n_instances"]} instances
&middot; mechanical grading from the hidden formal world
(facts 0.5 / action 0.3 / compliance 0.2)</p>

<h2>Leaderboard</h2>
<div class="legend">
<span><span class="dot seg-facts"></span>facts &times;0.5</span>
<span><span class="dot seg-action"></span>action &times;0.3</span>
<span><span class="dot seg-compliance"></span>compliance &times;0.2</span>
</div>
{''.join(chart)}

<h2>Components</h2>
{''.join(comp_rows)}

<h2>Detailed results</h2>
<table>
<thead><tr><th>provider</th>{head}<th>composite</th><th>seconds</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
<h3>Canonical final actions</h3>
{actions}

<h2>Runs</h2>
<table>
<thead><tr><th>instance</th><th>genre</th><th>size</th><th>seed</th><th>burial</th></tr></thead>
<tbody>{run_rows}</tbody>
</table>
</main>
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(doc)


# ------------------------------------------------------------- CLI

def _print_leaderboard(agg):
    print("\n== Leaderboard ==")
    hdr = (f"{'provider':<24} {'composite':>9} {'facts':>6} {'action':>7} "
           f"{'compl':>6} {'scored':>6} {'errors':>6} {'seconds':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in agg["leaderboard"]:
        f3 = lambda x: "\u2014" if x is None else f"{x:.3f}"
        print(f"{r['provider']:<24} {f3(r['composite']):>9} {f3(r['facts']):>6} "
              f"{f3(r['action']):>7} {f3(r['compliance']):>6} {r['n_scored']:>6} "
              f"{r['n_errors']:>6} {r['total_seconds']:>8.1f}")
    print()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="enigmaforge.harness",
        description="Generate a verified puzzle-story cohort, run it against "
                    "a list of LLM providers, grade mechanically, and emit "
                    "results.json + report.html.")
    ap.add_argument("--providers", required=True,
                    help="JSON list of {name, model?, base_url?, api_key?|api_key_env?}")
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--sizes", default="small",
                    help="comma list, cycled across instances (e.g. small,medium)")
    ap.add_argument("--genre", default="auto",
                    choices=["auto"] + sorted(PACKS))
    ap.add_argument("--burial-min", type=int, default=1)
    ap.add_argument("--burial-max", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="instance i uses seed seed_base + i*17")
    ap.add_argument("--out", default=None,
                    help="output dir (default runs/harness-<timestamp>)")
    ap.add_argument("--timeout", type=int, default=500,
                    help="per-call LLM timeout in seconds")
    ap.add_argument("--skip-generate", action="store_true",
                    help="scan existing instances/ instead of building")
    ap.add_argument("--grade-only", action="store_true",
                    help="reuse existing responses; re-run grading + reports only")
    a = ap.parse_args(argv)

    def fail(msg):
        print(f"[enigmaforge.harness] {msg}", file=sys.stderr)
        return 1

    try:
        providers = load_providers(a.providers)
    except ValueError as e:
        return fail(str(e))
    sizes = [s.strip() for s in a.sizes.split(",") if s.strip()]
    bad = [s for s in sizes if s not in pipeline.SIZES]
    if bad:
        return fail(f"unknown size(s) {bad}; choose from {sorted(pipeline.SIZES)}")
    if not (0 <= a.burial_min <= a.burial_max <= 3):
        return fail("burial depth must satisfy 0 <= min <= max <= 3")
    if a.instances < 1:
        return fail("--instances must be >= 1")
    out = a.out or f"runs/harness-{datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(out, exist_ok=True)

    specs = None
    if not (a.skip_generate or a.grade_only):  # grade-only: grade what exists
        specs = build_specs(a.instances, sizes, a.genre,
                            a.burial_min, a.burial_max, a.seed_base)
    instances = generate_cohort(out, specs)
    if not instances:
        return fail(f"no instances found under {out}/instances/")
    records = run_solvers(instances, providers, out, timeout=a.timeout,
                          call=not a.grade_only)
    by_iid = {i["iid"]: i for i in instances}
    graded = [grade_instance(by_iid[r["iid"]], r) for r in records]

    agg = aggregate(graded, instances, providers)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(agg, f, indent=2)
    detailed = dict(agg)
    detailed["raw_responses"] = {
        f"{r['provider']}--{r['iid']}": {"text": r.get("text"),
                                         "seconds": r.get("seconds"),
                                         "error": r.get("error")}
        for r in records}
    with open(os.path.join(out, "results-detailed.json"), "w") as f:
        json.dump(detailed, f, indent=2)
    render_html(agg, os.path.join(out, "report.html"))
    _print_leaderboard(agg)
    print(f"results: {out}/results.json  {out}/results-detailed.json  "
          f"{out}/report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())

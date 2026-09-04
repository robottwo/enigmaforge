"""Benchmark harness: generate a cohort of story instances, run each against
a configurable list of LLM providers, grade every (provider, instance) pair
mechanically from the hidden formal world, and emit results JSON, an
aggregated leaderboard, and a self-contained dark-theme HTML report.

Run:
    python3 -m enigmaforge.harness --providers benchmark.json

Benchmark config — either a bare JSON list of providers (the cohort then
comes from CLI flags) or an object with optional scenarios:

    {"provider_defaults": {        # optional: named default groups
        "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                       "api_key_env": "MY_KEY_VAR"}},
     "providers": [
        {"model": "moonshotai/kimi-k3",     # name optional: defaults to the
         "defaults": "openrouter"},         #   model's last path segment
        {"name": "glm-flash",               # required, unique (if given)
         "model": "glm-5.3-flash",          # omit -> endpoint auto-picks newest
         "base_url": "https://...",         # per-entry values beat the group
         "api_key": "sk-..."}],             # api_key / api_key_env: exclusive
     "scenarios": [                    # optional; cohort definition(s)
        {"name": "core",               # required, unique; names the iids
         "instances": 6,               # stories (per level, if levels given)
         "renderer": "llm",            # optional: LLM-rendered scene prose
         "sizes": ["small", "medium"], # cycled; exclusive with 'levels'
         "genre": "auto",              # or a pack name; auto = pick_genre(seed)
         "burial_min": 1, "burial_max": 2,   # clue-burial depth, 0-12
         "seed_base": 1000,            # story i uses seed seed_base + i*17
         "levels": [                   # optional difficulty ladder, ordered
           {"size": "small", "burial": 0,    # easiest -> hardest; 'instances'
            "overrides": {"n_variables": 4}},#   counts stories PER LEVEL;
           {"size": "medium"}]}]}            # overrides tune any generator knob

Grading is mechanical, computed from the hidden representation (no LLM
judge); per-instance score = 0.66*decisions + 0.24*comprehension + 0.10*compliance —
comprehension is the only per-instance-discriminative component (recovery
of the hidden world); decisions are LLM-authored per story from the
resolved origin, so their canonical text varies with the genre pack.

The run is resumable at both stages: existing instance dirs are never
rebuilt and existing response files are never re-called.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
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
from .llm import (chat_completion, llm_scene_renderer, generate_genre_pack,
                  polish_realization)
from .rng import Rng

WEIGHTS = {"comprehension": 0.24, "decisions": 0.66, "compliance": 0.10}

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

def _read_json_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"config file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"config file is not valid JSON: {path}: {e}")


_PROVIDER_FIELDS = ("name", "model", "base_url", "api_key", "api_key_env",
                    "defaults", "track_cost", "max_tokens")


def _providers_from_list(raw, path, defaults=None):
    """Validate a provider list into chat_completion kwargs per entry.

    'defaults' is the config's provider_defaults map: named groups of
    {model, base_url, api_key, api_key_env} that entries reference with
    "defaults": "<group>". Per-entry fields beat inherited values. An
    entry may omit 'name' — it then defaults to the last path segment of
    the effective model."""
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError(f"'provider_defaults' must be an object of named "
                         f"default groups: {path}")
    for gname, group in defaults.items():
        if not isinstance(group, dict):
            raise ValueError(f"provider_defaults.{gname} must be an object: {path}")
        bad = set(group) - {"model", "base_url", "api_key", "api_key_env",
                            "track_cost", "max_tokens"}
        if bad:
            raise ValueError(f"unknown provider_defaults.{gname} key(s) "
                             f"{sorted(bad)}: {path}")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"providers must be a non-empty JSON list: {path}")
    out, seen = [], set()
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            raise ValueError(f"provider #{i}: must be an object")
        unknown = set(p) - set(_PROVIDER_FIELDS)
        if unknown:
            raise ValueError(f"provider #{i}: unknown key(s) {sorted(unknown)}")
        ref = p.get("defaults")
        if ref is not None and (not isinstance(ref, str) or ref not in defaults):
            raise ValueError(f"provider #{i}: unknown defaults group {ref!r} "
                             f"(have: {sorted(defaults)})")
        m = dict(defaults.get(ref) or {})
        m.update({k: v for k, v in p.items() if v is not None})
        name = m.get("name") or str(m.get("model") or "").rpartition("/")[2]
        if not name:
            raise ValueError(f"provider #{i}: missing required 'name' "
                             "(or 'model' to derive it from)")
        if name in seen:
            raise ValueError(f"duplicate provider name: {name!r}")
        seen.add(name)
        if m.get("api_key") is not None and m.get("api_key_env"):
            raise ValueError(f"provider {name!r}: 'api_key' and 'api_key_env' "
                             "are mutually exclusive")
        kw = {}
        for key in ("model", "base_url", "api_key", "track_cost", "max_tokens"):
            if m.get(key) is not None:
                kw[key] = m[key]
        if "api_key" not in kw and m.get("api_key_env"):
            env_val = os.environ.get(m["api_key_env"])
            if env_val:
                kw["api_key"] = env_val
        out.append({"name": name, "kwargs": kw})
    return out


def load_providers(path):
    """Parse a bare provider-list JSON file (back-compat entry point)."""
    return _providers_from_list(_read_json_config(path), path)


# ------------------------------------------------------------- scenarios

# Anonymous scenario used when the config is a bare provider list: the
# cohort then comes from CLI flags (legacy behavior, legacy iids).
DEFAULT_SCENARIO = {"name": None, "instances": 6, "sizes": ["small"],
                    "genre": "auto", "burial_min": 1, "burial_max": 2,
                    "seed_base": 1000, "renderer": "template"}

COHORT_FLAGS = ("--instances", "--sizes", "--genre", "--burial-min",
                "--burial-max", "--seed-base")

# A ladder level counts as solved for a provider when every instance at the
# level produced a response and mean fact recovery clears this bar.
LEVEL_SOLVED_FACTS = 0.6

# pipeline.build knobs a level may tune; mode/genre/burial are harness-owned.
_LEVEL_OVERRIDES = set(pipeline.SIZES["large"])


def _validate_level(lvl, where):
    if not isinstance(lvl, dict):
        raise ValueError(f"{where}: each level must be an object")
    s = {"size": lvl.get("size", "small"),
         "overrides": lvl.get("overrides") or {},
         "burial": lvl.get("burial")}
    if s["size"] not in pipeline.SIZES:
        raise ValueError(f"{where}: size must be one of {sorted(pipeline.SIZES)}")
    if not isinstance(s["overrides"], dict):
        raise ValueError(f"{where}: overrides must be an object")
    bad = set(s["overrides"]) - _LEVEL_OVERRIDES
    if bad:
        raise ValueError(f"{where}: unknown override(s) {sorted(bad)}; "
                         f"tunable knobs: {sorted(_LEVEL_OVERRIDES)}")
    if s["burial"] is not None and (isinstance(s["burial"], bool)
                                    or not isinstance(s["burial"], int)
                                    or not 0 <= s["burial"] <= 12):
        raise ValueError(f"{where}: burial must be an integer in 0..12")
    return s


def _validate_scenario(sc, i, default_renderer="template"):
    """Fill defaults and validate one scenario entry; ValueError on any
    malformation (fail-loud: a bad config must not reach the LLM calls)."""
    if not isinstance(sc, dict) or not sc.get("name"):
        raise ValueError(f"scenario #{i}: missing required 'name'")
    s = dict(DEFAULT_SCENARIO, name=str(sc["name"]))
    where = f"scenario {s['name']!r}"
    try:
        s["levels"] = None
        if sc.get("levels") is not None:
            if not isinstance(sc["levels"], list) or not sc["levels"]:
                raise ValueError("levels must be a non-empty list of level objects")
            s["levels"] = [_validate_level(l, f"{where} level #{j}")
                           for j, l in enumerate(sc["levels"])]
            if "sizes" in sc:
                raise ValueError("levels and sizes are mutually exclusive: "
                                 "each level carries its own size")
            default_instances = 1  # per level: 6 levels x 6 = 36 stories
        else:
            default_instances = s["instances"]
        s["instances"] = sc.get("instances", default_instances)
        if isinstance(s["instances"], bool) or not isinstance(s["instances"], int) \
                or s["instances"] < 1:
            raise ValueError("instances must be an integer >= 1")
        if s["levels"] is None:
            s["sizes"] = sc.get("sizes", s["sizes"])
            if (not isinstance(s["sizes"], list) or not s["sizes"]
                    or any(x not in pipeline.SIZES for x in s["sizes"])):
                raise ValueError(f"sizes must be a non-empty list from "
                                 f"{sorted(pipeline.SIZES)}")
        else:
            s["sizes"] = None
        s["genre"] = sc.get("genre", s["genre"])
        if s["genre"] not in ("auto", "llm") and s["genre"] not in PACKS:
            raise ValueError(f"genre must be 'auto', 'llm' or one of "
                             f"{sorted(PACKS)}")
        s["polish"] = bool(sc.get("polish", False))
        s["renderer"] = sc.get("renderer", default_renderer)
        if s["renderer"] not in ("template", "llm"):
            raise ValueError("renderer must be 'template' or 'llm'")
        for k in ("burial_min", "burial_max"):
            s[k] = sc.get(k, s[k])
            if isinstance(s[k], bool) or not isinstance(s[k], int) \
                    or not 0 <= s[k] <= 12:
                raise ValueError(f"{k} must be an integer in 0..12")
        if s["burial_min"] > s["burial_max"]:
            raise ValueError("burial_min must be <= burial_max")
        s["seed_base"] = sc.get("seed_base", s["seed_base"])
        if isinstance(s["seed_base"], bool) or not isinstance(s["seed_base"], int):
            raise ValueError("seed_base must be an integer")
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from e
    return s


def load_config(path):
    """Parse the benchmark config: a bare provider list yields one anonymous
    scenario (CLI-driven); an object yields {"providers", "scenarios"} with
    the cohort fully config-defined and scenario-named instance dirs."""
    raw = _read_json_config(path)
    if isinstance(raw, list):
        return {"providers": _providers_from_list(raw, path),
                "scenarios": [dict(DEFAULT_SCENARIO)], "corpus": None,
                "renderer_endpoint": None}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON list or object: {path}")
    corpus = raw.get("corpus")
    if corpus is not None and (not isinstance(corpus, str)
                               or not corpus.strip()):
        raise ValueError("'corpus' must be a non-empty directory path")
    corpus = corpus.strip() if corpus else None
    default_renderer = raw.get("renderer", "template")
    if default_renderer not in ("template", "llm"):
        raise ValueError("'renderer' must be 'template' or 'llm'")
    providers = _providers_from_list(raw.get("providers"), path,
                                     raw.get("provider_defaults"))
    raw_scenarios = raw.get("scenarios")
    if raw_scenarios is None:  # object with providers only: CLI drives cohort
        scenarios = [dict(DEFAULT_SCENARIO)]
    else:
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ValueError("'scenarios' must be a non-empty list of "
                             "scenario objects")
        scenarios, seen = [], set()
        for i, sc in enumerate(raw_scenarios):
            s = _validate_scenario(sc, i, default_renderer)
            slug = _slug(s["name"])
            if slug in seen:
                raise ValueError(f"duplicate scenario name: {s['name']!r}")
            seen.add(slug)
            scenarios.append(s)
    rend = {k: raw.get(f"renderer_{k}")
            for k in ("model", "base_url", "api_key_env")}
    if any(v is not None and not isinstance(v, str) for v in rend.values()):
        raise ValueError("renderer_model/renderer_base_url/"
                         "renderer_api_key_env must be strings")
    renderer_endpoint = {k: v for k, v in rend.items() if v} or None
    return {"providers": providers, "scenarios": scenarios, "corpus": corpus,
            "renderer_endpoint": renderer_endpoint}


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "provider"


# ------------------------------------------------------------- generation

def _tagged_seed(seed, tag):
    """Stable per-(seed, tag) derived seed (hash(), unlike sha256, is salted)."""
    return int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)


def _scenario_specs(sc):
    """Expand one validated scenario into instance specs. Instance i gets
    seed = seed_base + (running index)*17, an explicit genre or
    pick_genre(seed), and a burial depth: explicit per ladder level, else
    drawn from a (seed, 'burial')-derived rng in [burial_min, burial_max].
    Scenario names (and ladder levels) are baked into iids, so an unchanged
    config re-finds its instance dirs on every rerun."""
    slug = _slug(sc["name"]) if sc["name"] else None
    out = []
    if sc.get("levels") is not None:
        for lvl, lv in enumerate(sc["levels"]):
            for i in range(sc["instances"]):
                # stride 3 = the instances count of the original corpus:
                # pinning (level, index) -> seed keeps every cached story
                # reusable when 'instances' grows (only new indexes render).
                # Windows overlap across adjacent levels, but each level's
                # overrides make the overlapping worlds distinct.
                seed = sc["seed_base"] + (lvl * 3 + i) * 17
                genre = sc["genre"] if sc["genre"] != "auto" else pick_genre(seed)
                b = lv["burial"]
                if b is None:
                    b = Rng(_tagged_seed(seed, "burial")).range(
                        sc["burial_min"], sc["burial_max"])
                iid = (f"inst-{slug}-L{lvl:02d}-{i:03d}-{genre}" if slug
                       else f"inst-L{lvl:02d}-{i:03d}-{genre}")
                out.append({"iid": iid, "scenario": sc["name"], "level": lvl,
                            "level_overrides": lv["overrides"], "index": i,
                            "genre": genre, "seed": seed, "size": lv["size"],
                            "burial": b, "renderer": sc.get("renderer", "template"),
                            "polish": bool(sc.get("polish"))})
        return out
    sizes = sc["sizes"] or ["small"]
    for i in range(sc["instances"]):
        seed = sc["seed_base"] + i * 17
        genre = sc["genre"] if sc["genre"] != "auto" else pick_genre(seed)
        b = Rng(_tagged_seed(seed, "burial")).range(sc["burial_min"],
                                                    sc["burial_max"])
        iid = (f"inst-{slug}-{i:03d}-{genre}" if slug
               else f"inst-{i:03d}-{genre}")
        out.append({"iid": iid, "scenario": sc["name"], "index": i,
                    "genre": genre, "seed": seed, "size": sizes[i % len(sizes)],
                    "burial": b, "renderer": sc.get("renderer", "template"),
                    "polish": bool(sc.get("polish"))})
    return out


def build_specs(n_instances, sizes, genre, burial_min, burial_max, seed_base):
    """Legacy single-scenario entry point (CLI flags, anonymous scenario)."""
    return _scenario_specs({"name": None, "instances": n_instances,
                            "sizes": list(sizes) or ["small"],
                            "genre": genre or "auto",
                            "burial_min": burial_min, "burial_max": burial_max,
                            "seed_base": seed_base})


def _instance_dir(out_dir, iid):
    return os.path.join(out_dir, "instances", iid)


def generate_cohort(out_dir, specs=None, workers=1, renderer_endpoint=None):
    """Build and package every spec into <out>/instances/<iid>/, writing the
    grading payload instance.json. Existing instance dirs are skipped —
    resumable cohorts. Missing stories build in a bounded thread pool
    (workers = max concurrently rendering stories; each story's scenes
    render in parallel inside the renderer too); failed stories retry up
    to 3 attempts, run back-to-back (OpenRouter needs no cooldown). specs=None
    scans the dir instead — --skip-generate.
    Raises RuntimeError only if stories still fail after all attempts;
    successful stories stay cached, so rerunning retries only those.
    """
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
    instances, missing = [], []
    for spec in specs:
        meta = os.path.join(_instance_dir(out_dir, spec["iid"]),
                            "instance.json")
        if os.path.isfile(meta):
            with open(meta) as f:
                instances.append(json.load(f))
        else:
            missing.append(spec)
    if not missing:
        return instances

    def _llm_kwargs():
        ep = renderer_endpoint or {}
        kw = {}
        if ep.get("model"):
            kw["model"] = ep["model"]
        if ep.get("base_url"):
            kw["base_url"] = ep["base_url"]
        if ep.get("api_key_env"):
            val = os.environ.get(ep["api_key_env"])
            if val:
                kw["api_key"] = val
        return kw

    llm_renderer = None
    if any(sp.get("renderer") == "llm" for sp in missing):
        _note("LLM scene renderer enabled (endpoint via env/"
              "agent-config autodiscovery; planning + per-scene calls)")
        llm_renderer = llm_scene_renderer(**_llm_kwargs())  # shared across stories
    genre_gen = None
    if any(sp.get("genre") == "llm" for sp in missing):
        _note("LLM genre invention enabled (a fresh setting pack per story)")
        genre_gen = lambda seed: generate_genre_pack(seed=seed,
                                                     max_attempts=5,
                                                     **_llm_kwargs())
    polisher = None
    if any(sp.get("polish") for sp in missing):
        _note("Polish pass enabled (final LLM rewrite, clauses verbatim, "
              "re-gated)")
        polisher = lambda world, r: polish_realization(world, r,
                                                       **_llm_kwargs())

    def _build_one(spec):
        d = _instance_dir(out_dir, spec["iid"])
        spec_renderer = llm_renderer if spec.get("renderer") == "llm" else None
        spec_genre_gen = genre_gen if spec.get("genre") == "llm" else None
        spec_polisher = polisher if spec.get("polish") else None
        world = pipeline.build(spec["size"], spec["seed"], config_overrides={
            "mode": "story", "genre": spec["genre"], "burial": spec["burial"],
            **(spec.get("level_overrides") or {})},
            renderer=spec_renderer, polisher=spec_polisher,
            genre_gen=spec_genre_gen)
        pipeline.package(world, d)
        with open(os.path.join(d, "story.md")) as f:
            story_text = f.read()
        true_obj = next(o for o in world.objectives if o.true_objective)
        inst = {"iid": spec["iid"], "scenario": spec.get("scenario"),
                "level": spec.get("level"),
                "level_overrides": spec.get("level_overrides") or None,
                "genre": spec["genre"], "seed": spec["seed"],
                "size": spec["size"], "burial": spec["burial"],
                "renderer": spec.get("renderer", "template"),
                "polish": bool(spec.get("polish")),
                "story_path": os.path.join(d, "story.md"),
                "story_text": story_text,
                "ground_truth": world.meta["ground_truth"],
                "surfaces": {v.vid: v.surface_names[0] for v in world.variables},
                "final_action": true_obj.answer["final_action"],
                "n_constraints": len(world.constraints)}
        with open(os.path.join(d, "instance.json"), "w") as f:
            json.dump(inst, f, indent=2)
        _note(f"{spec['iid']}: story ready ({len(story_text)} chars)")
        return inst

    failures = []
    pending = list(missing)
    attempt = 0
    while pending and attempt < 3:  # transient endpoint failures recover
        if attempt:                 # retries run back-to-back
            _note(f"retrying {len(pending)} story generation(s) "
                  f"(attempt {attempt + 1}/3)")
        failures = []
        n = max(1, min(workers, len(pending)))
        if n == 1:
            for spec in pending:
                try:
                    instances.append(_build_one(spec))
                except Exception as e:
                    failures.append((spec["iid"], str(e)[:300]))
        else:
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = {pool.submit(_build_one, spec): spec
                           for spec in pending}
                for fut in as_completed(futures):
                    spec = futures[fut]
                    try:
                        instances.append(fut.result())
                    except Exception as e:
                        failures.append((spec["iid"], str(e)[:300]))
                        _note(f"{spec['iid']}: GENERATION FAILED "
                              f"{str(e)[:300]}")
        # keep only what failed this wave; success never re-queues
        failed_iids = {iid for iid, _ in failures}
        pending = [sp for sp in pending if sp["iid"] in failed_iids]
        attempt += 1
    if failures:
        for iid, err in failures:
            _note(f"generation failed: {iid}: {err}")
        raise RuntimeError(f"{len(failures)}/{len(missing)} story "
                           "generation(s) failed after 3 attempts; "
                           "successful stories are cached — rerun to retry "
                           "only the failures")
    return instances


# ------------------------------------------------------------- solving

def _response_path(out_dir, provider_name, iid):
    return os.path.join(out_dir, "responses", f"{_slug(provider_name)}--{iid}.txt")


def run_solvers(instances, providers, out_dir, timeout, call=True, workers=1):
    """One chat call per (provider, instance); response + wall time + token
    usage + cost + error are persisted under responses/. Successful
    responses are final; ERROR records retry when the run can call (so
    transient endpoint failures self-heal across reruns) — except
    deterministic policy refusals (content_filter), which are kept.
    Calls run in a bounded thread pool (workers = max concurrent calls).
    With call=False (--grade-only) only existing files are loaded."""
    os.makedirs(os.path.join(out_dir, "responses"), exist_ok=True)
    records, pairs = [], []
    for prov in providers:
        for inst in instances:
            path = _response_path(out_dir, prov["name"], inst["iid"])
            if os.path.isfile(path):
                with open(path) as f:
                    rec = json.load(f)
                if call and rec.get("error") \
                        and "content_filter" not in (rec.get("error") or ""):
                    os.remove(path)  # stale error: try again this run
                else:
                    records.append({"provider": prov["name"],
                                    "iid": inst["iid"], **rec})
                continue
            if call:
                pairs.append((prov, inst, path))

    def _solve_one(prov, inst, path):
        t0 = time.time()
        try:
            text, usage = chat_completion(
                [{"role": "system", "content": SOLVER_PROMPT},
                 {"role": "user", "content": inst["story_text"]}],
                timeout=timeout, with_usage=True, **prov["kwargs"])
            rec = {"text": text, "seconds": round(time.time() - t0, 2),
                   "error": None,
                   "tokens": {k: usage.get(k) for k in
                              ("prompt_tokens", "completion_tokens",
                               "total_tokens")} if usage else None,
                   "cost": usage.get("cost")}
            _note(f"{prov['name']} x {inst['iid']}: ok "
                  f"({rec['seconds']}s, {len(text)} chars)")
        except Exception as e:  # endpoint failure: record, keep going
            rec = {"text": None, "seconds": round(time.time() - t0, 2),
                   "error": str(e)[:500], "tokens": None, "cost": None}
            _note(f"{prov['name']} x {inst['iid']}: ERROR {rec['error']}")
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        return {"provider": prov["name"], "iid": inst["iid"], **rec}

    if pairs:
        n = max(1, min(workers, len(pairs)))
        if n == 1:
            for prov, inst, path in pairs:
                records.append(_solve_one(prov, inst, path))
        else:
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(_solve_one, prov, inst, path)
                           for prov, inst, path in pairs]
                for fut in as_completed(futures):
                    records.append(fut.result())
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
    comps = {"compliance": None, "comprehension": None, "decisions": None}
    graded = {"provider": record["provider"], "iid": record["iid"],
              "components": comps, "score": None,
              "seconds": record.get("seconds"), "error": record.get("error"),
              "cost": record.get("cost"), "tokens": record.get("tokens")}
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
    comps["comprehension"] = recovered / len(gt) if gt else 0.0

    resp_action = parsed.get("final_action") if isinstance(parsed, dict) else None
    graded["decision"] = resp_action if isinstance(resp_action, str) else None
    if not isinstance(resp_action, str):
        resp_action = raw
    sim = _action_overlap(_normalize_tokens(resp_action),
                          _normalize_tokens(instance.get("final_action")))
    graded["similarity"] = round(sim, 4)
    comps["decisions"] = 1.0 if sim >= 0.6 else (0.5 if sim >= 0.3 else 0.0)

    graded["score"] = round(sum(WEIGHTS[k] * comps[k] for k in WEIGHTS), 6)
    return graded


_STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
              "with"}


def _normalize_tokens(s):
    """Normalized content words: punctuation/number-free lowercase tokens
    minus function words, so 'the' alone never earns action credit."""
    return [t for t in _normalize(s).split() if t not in _STOPWORDS]


def _action_overlap(a, b):
    """Overlap coefficient between two token lists: how much of the smaller
    bag (the canonical action) the response covers, capped at 1.0. Unlike
    cosine, a longer substantive answer is not diluted for its length."""
    va, vb = Counter(a), Counter(b)
    if not va or not vb:
        return 0.0
    dot = sum(va[t] * vb[t] for t in va.keys() & vb.keys())
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return min(1.0, dot / min(na, nb)) if na and nb else 0.0


# ------------------------------------------------------------- aggregation

def _ladder_summary(graded, instances, providers):
    """Per-provider breakdown by ladder level: mean comprehension/composite, a
    solved flag, levels cleared (gap-blind count), and the highest level
    reached before the first failure — levels are walked easiest ->
    hardest in config order; the walk stops at the first uncleared rung."""
    lvl = [i for i in instances if i.get("level") is not None]
    if not lvl:
        return None
    keys = sorted({(i["scenario"], i["level"]) for i in lvl})
    meta, iids_at = {}, {}
    for key in keys:
        iids_at[key] = {i["iid"] for i in lvl
                        if (i["scenario"], i["level"]) == key}
        one = next(i for i in lvl if (i["scenario"], i["level"]) == key)
        ov = one.get("level_overrides") or {}
        preset = pipeline.SIZES[one["size"]]
        n = ov.get("n_variables", preset["n_variables"])
        d = ov.get("dependency_depth", preset["dependency_depth"])
        meta[key] = {"scenario": key[0], "level": key[1],
                     "label": f"n={n} d={d} b={one['burial']}",
                     "n_instances": len(iids_at[key])}
    by_provider = {}
    for prov in providers:
        rows = []
        for key in keys:
            rs = [g for g in graded
                  if g["provider"] == prov["name"] and g["iid"] in iids_at[key]]
            scored = [g for g in rs if g["score"] is not None]
            facts = (round(sum(g["components"]["comprehension"] for g in scored)
                           / len(scored), 4) if scored else None)
            comp = (round(sum(g["score"] for g in scored) / len(scored), 4)
                    if scored else None)
            solved = (len(scored) == meta[key]["n_instances"]
                      and facts is not None and facts >= LEVEL_SOLVED_FACTS)
            rows.append({"scenario": key[0], "level": key[1],
                         "label": meta[key]["label"], "comprehension": facts,
                         "composite": comp, "n_scored": len(scored),
                         "n_errors": sum(1 for g in rs if g["error"]),
                         "n_instances": meta[key]["n_instances"],
                         "solved": solved})
        max_row = None
        for r in rows:  # easiest -> hardest; stop at the first uncleared rung
            if r["solved"]:
                max_row = r
            else:
                break
        by_provider[prov["name"]] = {
            "levels": rows,
            "levels_cleared": sum(1 for r in rows if r["solved"]),
            "max_solved_level": max_row["level"] if max_row else None,
            "max_solved_scenario": max_row["scenario"] if max_row else None}
    return {"levels": [meta[k] for k in keys], "by_provider": by_provider,
            "solved_facts_threshold": LEVEL_SOLVED_FACTS}

def aggregate(graded, instances, providers):
    """Per-provider results, ranked by ladder achievement first (levels
    cleared, then highest solved level) so refusing or partial models do
    not outrank complete ones on a difficulty-blind mean; composite breaks
    ties. Providers from flat (non-ladder) runs rank by composite alone."""
    ladder = _ladder_summary(graded, instances, providers)
    cleared = {p: s["levels_cleared"]
               for p, s in (ladder or {}).get("by_provider", {}).items()}
    maxlvl = {p: s["max_solved_level"]
              for p, s in (ladder or {}).get("by_provider", {}).items()}
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
        costs = [g["cost"] for g in rs if g.get("cost") is not None]
        toks = [g["tokens"]["total_tokens"] for g in rs
                if g.get("tokens") and g["tokens"].get("total_tokens")]
        rows.append({"provider": pname, "composite": composite,
                     "comprehension": mean("comprehension"),
                     "decisions": mean("decisions"),
                     "compliance": mean("compliance"),
                     "n_scored": len(scored),
                     "n_errors": sum(1 for g in rs if g["error"]),
                     "total_seconds": round(sum(g.get("seconds") or 0 for g in rs), 2),
                     "tokens": sum(toks) if toks else None,
                     "cost": round(sum(costs), 4) if costs else None,
                     "levels_cleared": cleared.get(pname),
                     "max_solved_level": maxlvl.get(pname)})
    rows.sort(key=lambda r: (
        -(r["levels_cleared"] or 0),
        -(r["max_solved_level"] if r["max_solved_level"] is not None else -1),
        r["composite"] is None,
        -(r["composite"] or 0)))
    return {"generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_instances": len(instances),
            "leaderboard": rows,
            "per_instance": {iid: inst_by_iid[iid]["final_action"]
                             for iid in sorted(inst_by_iid)},
            "ladder": ladder,
            "runs": [{k: i.get(k) for k in ("iid", "scenario", "genre",
                                            "size", "seed", "burial")}
                     for i in sorted(instances, key=lambda x: x["iid"])],
            "records": sorted(graded, key=lambda g: (g["provider"], g["iid"]))}


# ------------------------------------------------------------- HTML report

_CSS = """
body { background:#12151c; color:#e6e9ef; margin:0; padding:2rem;
       font:15px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif; }
main { max-width:1200px; margin-inline:auto; }
h1 { font-size:1.5rem; margin-bottom:.2rem; }
h2 { font-size:1.05rem; margin:0 0 1rem; color:#9fb3d1;
     text-transform:uppercase; letter-spacing:.08em; }
table { border-collapse:collapse; width:100%; font-size:.92rem; }
th, td { padding:.35rem .6rem; border-bottom:1px solid #232a38;
         text-align:left; white-space:nowrap; }
th { color:#9fb3d1; font-weight:600; }
.bar-row { display:flex; align-items:center; gap:.8rem; margin:.22rem 0; }
.bar-name { width:170px; text-align:right; color:#c7d1e0; font-size:.9rem;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar-track { flex:1; background:#1a2030; border-radius:4px; height:22px; }
.bar-fill { height:100%; border-radius:4px; display:flex; overflow:hidden;
            min-width:2px; }
.seg-ladder { background:#a78bfa; }
.seg-facts { background:#4f9cf9; } .seg-action { background:#f9a94f; }
.seg-compliance { background:#58c98b; }
.bar-score { width:130px; font-variant-numeric:tabular-nums;
             line-height:1.15; }
.bar-meta { color:#8b96a8; font-size:.72rem; line-height:1.15;
            display:block; }
.legend { display:flex; gap:1.2rem; margin:.6rem 0 1rem; font-size:.85rem;
          color:#9fb3d1; }
.dot { display:inline-block; width:10px; height:10px; border-radius:2px;
       margin-right:.35rem; }
.muted { color:#8b96a8; } .err-cell { color:#e26d6d; }
.tabbar { display:flex; gap:.3rem; margin:2rem 0 1.2rem;
          border-bottom:1px solid #232a38; flex-wrap:wrap; }
.tabbtn { background:none; border:none; color:#9fb3d1; padding:.5rem .9rem;
          cursor:pointer; font:inherit; font-size:.92rem;
          border-bottom:2px solid transparent; }
.tabbtn.active { color:#fff; border-bottom-color:#4f9cf9; }
.tabpane { display:none; } .tabpane.active { display:block; }
.dec-canonical { background:#1d2b22; border:1px solid #2f5741; color:#c9ecd8;
                 padding:.5rem .8rem; border-radius:6px; margin:.3rem 0 .5rem; }
.dec-row { border-bottom:1px solid #1d2430; padding:.15rem 0; cursor:pointer; }
.dec-row .dec-text { display:block; overflow:hidden; text-overflow:ellipsis;
                     white-space:nowrap; color:#c7d1e0; }
.dec-row.open .dec-text { white-space:normal; }
.dec-row:hover .dec-text { color:#fff; }
.dec-chip { display:inline-block; min-width:2.2em; text-align:center;
            border-radius:3px; margin-right:.5rem; font-size:.78rem;
            padding:0 .3rem; }
.chip-good { background:#1d3a2a; color:#7ee2a8; }
.chip-mid { background:#3a3013; color:#f9cf7e; }
.chip-bad { background:#3a1d1d; color:#ff9d9d; }
.dec-level { margin-top:1.2rem; }
.dec-level > h3 { margin:.2rem 0 .4rem; color:#9fb3d1; font-size:.95rem; }
.dec-iid { color:#8b96a8; font-size:.8rem; margin:.6rem 0 .2rem; }
.tabpane { display:none; } .tabpane.active { display:block; }
.lvl-pass { color:#7ee2a8; font-weight:600; }
.lvl-fail { color:#ff9d9d; }
.stories-layout { display:flex; gap:1rem; align-items:flex-start; }
.story-list { width:300px; flex:none; max-height:640px; overflow-y:auto;
              display:flex; flex-direction:column; gap:.2rem; }
.story-item { background:#1a2030; border:none; color:#c7d1e0; text-align:left;
              padding:.4rem .6rem; border-radius:4px; cursor:pointer;
              font:12px/1.4 ui-monospace,Menlo,monospace; }
.story-item.active { background:#2c3b57; color:#fff; }
.modebar { display:flex; gap:.4rem; margin-bottom:.8rem; }
.modebtn { background:#1a2030; border:1px solid #2c3b57; color:#c7d1e0;
           padding:.25rem .7rem; border-radius:4px; cursor:pointer;
           font:inherit; font-size:.85rem; }
.modebtn.active { background:#2c3b57; color:#fff; }
.story-pane pre { white-space:pre-wrap; word-break:break-word;
                  font:13px/1.6 ui-monospace,Menlo,monospace;
                  background:#161a24; padding:1rem; border-radius:6px;
                  max-height:70vh; overflow-y:auto; }
.story-pane table { max-width:480px; margin-bottom:1rem; }
"""

_SEGMENTS = (("comprehension", "seg-facts"), ("decisions", "seg-action"),
             ("compliance", "seg-compliance"))


def render_html(agg, out_path, instances=None):
    """Single self-contained report: inline CSS + a few lines of vanilla JS
    for the tabbed layout — zero external assets. Leaderboard bars carry
    tooltips with the component breakdown; `instances` (with story_text /
    ground_truth / surfaces / final_action) enables the story browser."""
    esc = html.escape

    def num(x, nd=2):
        return "&mdash;" if x is None else f"{x:.{nd}f}"

    rows = agg["leaderboard"]

    # display order: best -> worst by the combined ladder.composite score
    # (ladder level as the whole number, composite as the decimals)
    def _combined(r):
        cl, c = r.get("levels_cleared"), r.get("composite")
        return (cl or 0) + c if (cl is not None and c is not None) else -1.0

    rows = sorted(rows, key=lambda r: (_combined(r), r["composite"] is None),
                  reverse=True)
    top = max((_combined(r) for r in rows), default=1.0) or 1.0

    chart = []
    for r in rows:
        name = f'<div class="bar-name" title="{esc(r["provider"])}">{esc(r["provider"])}</div>'
        tip = esc(
            f"composite={num(r['composite'], 3)} | comprehension={num(r['comprehension'])} "
            f"decisions={num(r['decisions'])} compliance={num(r['compliance'])} | "
            f"{r['n_scored']} scored, {r['n_errors']} errors | "
            f"{r['total_seconds']:.0f}s"
            + (f", {r['tokens']:,} tokens" if r.get("tokens") else "")
            + (f", ${r['cost']:.4f}" if r.get("cost") is not None else ""))
        c = r["composite"]
        if c is None:
            chart.append(
                f'<div class="bar-row" title="{tip}">{name}'
                f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:100%;background:#2a3142;"></div></div>'
                f'<div class="bar-score muted">no scored instances '
                f'({r["n_errors"]} errors)</div></div>')
            continue
        cl = r.get("levels_cleared")
        combined = f"{cl + c:.3f}" if cl is not None else f"{c:.3f}"
        # bar = ladder chunk (levels cleared) + weighted component segments,
        # all scaled to the best combined score so length == position order
        ladder_w = (cl or 0)
        segs = (f'<div class="seg-ladder" '
                f'style="width:{100.0 * (cl or 0) / top:.1f}%" '
                f'title="levels cleared={cl}"></div>'
                if cl is not None else "")
        segs += "".join(
            f'<div class="{cls}" style="width:'
            f'{100.0 * WEIGHTS[key] * (r[key] or 0.0) / top:.1f}%" '
            f'title="{key}={num(r[key])}"></div>'
            for key, cls in _SEGMENTS)
        cost = (f'<span class="bar-meta">${r["cost"]:.4f}</span>'
                if r.get("cost") is not None else "")
        chart.append(
            f'<div class="bar-row" title="{tip}">{name}'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{100.0 * ((cl or 0) + c) / top:.1f}%">{segs}</div></div>'
            f'<div class="bar-score"><b>{combined}</b>'
            f'<span class="bar-meta">{r["total_seconds"]:.0f}s'
            f'{(" &middot; $" + format(r["cost"], ".4f")) if r.get("cost") is not None else ""}'
            f'</span></div></div>')

    detailed_rows = []
    iids = [r["iid"] for r in agg["runs"]]
    by_pair = {(g["provider"], g["iid"]): g for g in agg["records"]}
    detailed_head = "".join(f"<th>{esc(i)}</th>" for i in iids)
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
                tip = esc(f"comprehension={c['comprehension']:.2f} decisions={c['decisions']:.2f} "
                          f"similarity={g.get('similarity')} "
                          f"compliance={c['compliance']:.2f} "
                          f"seconds={g['seconds']}")
                light = 8 + 20 * g["score"]
                cells.append(f'<td style="background:hsl(145,60%,{light:.0f}%)" '
                             f'title="{tip}">{g["score"]:.2f}</td>')
        detailed_rows.append(
            f"<tr><td>{esc(r['provider'])}</td>{''.join(cells)}"
            f"<td><b>{num(r['composite'], 3)}</b></td>"
            f'<td class="muted">{r["total_seconds"]:.1f}s</td>'
            f'<td class="muted">{r["cost"] if r["cost"] is not None else "&mdash;"}</td></tr>')

    run_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
        "</tr>".format(esc(r["iid"]),
                       esc(r["scenario"]) if r["scenario"] else "&mdash;",
                       esc(r["genre"]), esc(r["size"]), r["seed"], r["burial"])
        for r in agg["runs"])

    ladder = agg.get("ladder")
    ladder_html = '<p class="muted">no ladder in this run</p>'
    if ladder:
        prov_names = [r["provider"] for r in rows]
        bp = ladder["by_provider"]
        ladd_head = "".join(f"<th>{esc(p)}</th>" for p in prov_names)
        ladd_body = []
        for lm in ladder["levels"]:
            cells = []
            for p in prov_names:
                row = next(r for r in bp[p]["levels"]
                           if (r["scenario"], r["level"])
                           == (lm["scenario"], lm["level"]))
                if row["comprehension"] is None:
                    cells.append('<td class="err-cell">err</td>')
                    continue
                tip = esc(f"composite={row['composite']:.2f} "
                          f"scored={row['n_scored']}/{row['n_instances']}")
                if row["solved"]:
                    light = 14 + 16 * row["comprehension"]
                    cells.append(
                        f'<td style="background:hsl(145,55%,{light:.0f}%)" '
                        f'title="{tip}"><span class="lvl-pass">&#10003; '
                        f'{row["comprehension"]:.2f}</span></td>')
                else:
                    light = 12 + 12 * row["comprehension"]
                    cells.append(
                        f'<td style="background:hsl(0,55%,{light:.0f}%)" '
                        f'title="{tip}"><span class="lvl-fail">&#10007; '
                        f'{row["comprehension"]:.2f}</span></td>')
            ladd_body.append(
                f'<tr><td>L{lm["level"]:02d} {esc(lm["scenario"])}</td>'
                f'<td class="muted">{esc(lm["label"])}</td>{"".join(cells)}</tr>')
        tops = " &middot; ".join(
            (f'<b>{esc(p)}</b>&nbsp;L{bp[p]["max_solved_level"]:02d}'
             f'&nbsp;<span class="muted">({bp[p]["levels_cleared"]}/'
             f'{len(bp[p]["levels"])} cleared)</span>')
            if bp[p]["max_solved_level"] is not None
            else f'<b>{esc(p)}</b>&nbsp;&mdash;'
            for p in prov_names)
        ladder_html = (
            f'<p class="muted">a level is solved when every instance at the '
            f'level has a response and mean comprehension &ge; '
            f'{ladder["solved_facts_threshold"]:.2f}; &#10003; solved, '
            f'&#10007; missed and the walk stops.</p>\n'
            f'<p>solved up to: {tops}</p>\n'
            f'<table>\n<thead><tr><th>level</th><th>knobs</th>{ladd_head}'
            f'</tr></thead>\n<tbody>{"".join(ladd_body)}</tbody>\n</table>')

    story_items, story_sets = [], []
    for i in sorted(instances or [], key=lambda x: x["iid"]):
        sid = esc(i["iid"])
        lvl = f'L{i["level"]:02d} &middot; ' if i.get("level") is not None else ""
        meta = f"{i['size']} &middot; burial {i['burial']} &middot; " \
               f"{i['n_constraints']} constraints"
        story_items.append(
            f'<button class="story-item" id="btn-{sid}" '
            f'onclick="pickStory(\'{sid}\', this)">'
            f'{lvl}{esc(i["iid"])}</button>')
        gt_rows = "".join(
            f"<tr><td>{esc(s)}</td><td><b>{esc(str(v))}</b></td></tr>"
            for s, v in sorted((i.get("surfaces") or {}).items()))
        story_sets.append(
            f'<div class="story-set" id="set-{sid}" style="display:none">'
            f'<p class="muted">{lvl}{esc(i["iid"])} &middot; {meta}</p>'
            f'<div class="story-pane" id="{sid}-story">'
            f'<pre>{esc(i["story_text"])}</pre></div>'
            f'<div class="story-pane" id="{sid}-facts" style="display:none">'
            f'<p class="muted">Distilled fact pattern — the hidden variables '
            f'this world encodes:</p>'
            f'<table><thead><tr><th>surface noun</th><th>value</th></tr>'
            f'</thead><tbody>{gt_rows}</tbody></table></div>'
            f'<div class="story-pane" id="{sid}-solution" style="display:none">'
            f'<p>Solution — the canonical final action this story calls for:</p>'
            f'<p><code>{esc(i.get("final_action", ""))}</code></p></div></div>')
    stories_html = (f'<div class="stories-layout">\n'
                    f'<div class="story-list">{"".join(story_items)}</div>\n'
                    f'<div style="flex:1;min-width:0">\n'
                    f'<div class="modebar">'
                    f'<button class="modebtn active" data-mode="story" '
                    f'onclick="pickMode(\'story\', this)">Story</button>'
                    f'<button class="modebtn" data-mode="facts" '
                    f'onclick="pickMode(\'facts\', this)">Fact pattern</button>'
                    f'<button class="modebtn" data-mode="solution" '
                    f'onclick="pickMode(\'solution\', this)">Solution</button>'
                    f'</div>\n<div id="story-view">{"".join(story_sets)}</div>\n'
                    f'</div>\n</div>') if instances else \
        '<p class="muted">story browser needs instance data</p>'

    by_level = {}
    for rr in agg["runs"]:
        by_level.setdefault(rr.get("level") or 0, []).append(rr["iid"])
    prov_order = [r["provider"] for r in rows]
    dec_levels = []
    for lvl in sorted(k for k in by_level):
        blocks = []
        for iid in sorted(by_level[lvl]):
            canon = agg["per_instance"].get(iid, "")
            rows_html = []
            for p in prov_order:
                g = by_pair.get((p, iid))
                if g is None:
                    continue
                sim = g.get("similarity")
                if g.get("error"):
                    chip, cls = "err", "chip-bad"
                elif sim is None:
                    chip, cls = "?", "chip-mid"
                elif sim >= 0.6:
                    chip, cls = "&#10003;", "chip-good"
                elif sim >= 0.3:
                    chip, cls = "&#189;", "chip-mid"
                else:
                    chip, cls = "&#10007;", "chip-bad"
                dec = g.get("decision")
                text = esc(dec) if dec else '<span class="muted">&mdash;</span>'
                tip = esc(f"similarity={sim if sim is not None else 'n/a'}")
                rows_html.append(
                    f'<div class="dec-row" onclick="this.classList.toggle(\'open\')" '
                    f'title="{tip}"><span class="dec-chip {cls}">{chip}</span>'
                    f'<b>{esc(p)}</b><span class="dec-text">{text}</span></div>')
            blocks.append(
                f'<div class="dec-iid">{esc(iid)}</div>'
                f'<div class="dec-canonical" title="canonical decision">'
                f'{esc(canon)}</div>{"".join(rows_html)}')
        dec_levels.append(f'<div class="dec-level"><h3>Level {lvl:02d}</h3>'
                          f'{"".join(blocks)}</div>')
    decisions_html = ('<div class="dec-level">' + "".join(dec_levels)
                      + '</div>') if dec_levels else \
        '<p class="muted">no decisions recorded</p>'

    tabs = [("pane-ladder", "Difficulty ladder", ladder_html),
            ("pane-decisions", "Decisions", decisions_html),
            ("pane-stories", "Stories", stories_html),
            ("pane-detailed", "Detailed results",
             f'<table>\n<thead><tr><th>provider</th>{detailed_head}'
             f'<th>composite</th><th>seconds</th><th>cost ($)</th></tr>'
             f'</thead>\n<tbody>{"".join(detailed_rows)}</tbody>\n</table>'),
            ("pane-runs", "Runs",
             f'<table>\n<thead><tr><th>instance</th><th>scenario</th>'
             f'<th>genre</th><th>size</th><th>seed</th><th>burial</th></tr>'
             f'</thead>\n<tbody>{run_rows}</tbody>\n</table>')]
    tabbar = "".join(
        f'<button class="tabbtn{" active" if k == "pane-ladder" else ""}" '
        f'onclick="showTab(\'{k}\', this)">{label}</button>'
        for k, label, _ in tabs)
    panes = "".join(
        f'<div class="tabpane{" active" if k == "pane-ladder" else ""}" '
        f'id="{k}">{body}</div>'
        for k, _, body in tabs)

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
(decisions 0.66 / comprehension 0.24 / compliance 0.10) &middot; hover bars and cells for detail</p>

<h2>Leaderboard</h2>
<div class="legend">
<span><span class="dot seg-ladder"></span>levels cleared</span>
<span><span class="dot seg-facts"></span>comprehension &times;0.24</span>
<span><span class="dot seg-action"></span>decisions &times;0.66</span>
<span><span class="dot seg-compliance"></span>compliance &times;0.10</span>
</div>
{''.join(chart)}

<div class="tabbar">{tabbar}</div>
{panes}
</main>
<script>
function showTab(id, el) {{
  document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.tabbtn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
}}
let curStory = null;
function pickStory(id, el) {{
  curStory = id;
  document.querySelectorAll('.story-item').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  renderStory();
}}
function pickMode(mode, el) {{
  document.querySelectorAll('.modebtn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
  renderStory(mode);
}}
function renderStory(mode) {{
  mode = mode || (document.querySelector('.modebtn.active')
                  ? document.querySelector('.modebtn.active').dataset.mode
                  : 'story');
  document.querySelectorAll('.story-set').forEach(d => d.style.display = 'none');
  if (!curStory) return;
  document.getElementById('set-' + curStory).style.display = 'block';
  document.querySelectorAll('#set-' + curStory + ' > .story-pane')
    .forEach(p => p.style.display = 'none');
  document.getElementById(curStory + '-' + mode).style.display = 'block';
}}
(function () {{
  const first = document.querySelector('.story-item');
  if (!first) return;
  first.classList.add('active');
  curStory = first.id.slice(4);
  document.getElementById('set-' + curStory).style.display = 'block';
}})();
</script>
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(doc)


# ------------------------------------------------------------- CLI

def _print_leaderboard(agg):
    print("\n== Leaderboard ==")
    hdr = (f"{'provider':<24} {'composite':>9} {'compr.':>7} {'decision':>8} "
           f"{'compl':>6} {'scored':>6} {'errors':>6} {'seconds':>8} {'cost$':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in agg["leaderboard"]:
        f3 = lambda x: "\u2014" if x is None else f"{x:.3f}"
        cost = lambda x: "\u2014" if x is None else f"{x:.4f}"
        print(f"{r['provider']:<24} {f3(r['composite']):>9} {f3(r['comprehension']):>7} "
              f"{f3(r['decisions']):>8} {f3(r['compliance']):>6} {r['n_scored']:>6} "
              f"{r['n_errors']:>6} {r['total_seconds']:>8.1f} {cost(r.get('cost')):>9}")
    print()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="enigmaforge.harness",
        description="Generate a verified puzzle-story cohort, run it against "
                    "a list of LLM providers, grade mechanically, and emit "
                    "results.json + report.html.")
    ap.add_argument("--providers", required=True,
                    help="benchmark config: a JSON provider list, or an "
                         "object {providers, scenarios}")
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--sizes", default=None,
                    help="comma list, cycled across instances "
                         "(e.g. small,medium); ignored with config scenarios")
    ap.add_argument("--genre", default=None,
                    choices=["auto"] + sorted(PACKS))
    ap.add_argument("--burial-min", type=int, default=None)
    ap.add_argument("--burial-max", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=None,
                    help="instance i uses seed seed_base + i*17")
    ap.add_argument("--out", default=None,
                    help="output dir for reports (default runs/harness-<ts>)")
    ap.add_argument("--corpus", default=None,
                    help="persistent dir for instances + responses, reused "
                         "across runs (default: keep everything in --out)")
    ap.add_argument("--timeout", type=int, default=500,
                    help="per-call LLM timeout in seconds")
    ap.add_argument("--workers", type=int, default=4,
                    help="max concurrent solver calls (OpenRouter scales; "
                         "use per provider limits)")
    ap.add_argument("--render-workers", type=int, default=1,
                    help="max stories rendering concurrently (the renderer "
                         "endpoint is usually the rate-limited one)")
    ap.add_argument("--skip-generate", action="store_true",
                    help="scan existing instances/ instead of building")
    ap.add_argument("--grade-only", action="store_true",
                    help="reuse existing responses; re-run grading + reports only")
    a = ap.parse_args(argv)
    t0 = time.time()

    def fail(msg):
        print(f"[enigmaforge.harness] {msg}", file=sys.stderr)
        return 1

    try:
        cfg = load_config(a.providers)
    except ValueError as e:
        return fail(str(e))
    providers = cfg["providers"]
    overrides = {"instances": a.instances,
                 "sizes": ([s.strip() for s in a.sizes.split(",") if s.strip()]
                           if a.sizes else None),
                 "genre": a.genre, "burial_min": a.burial_min,
                 "burial_max": a.burial_max, "seed_base": a.seed_base}
    if any(s["name"] is not None for s in cfg["scenarios"]):
        used = [f for f, v in zip(COHORT_FLAGS, overrides.values())
                if v is not None]
        if used:
            return fail(f"{', '.join(used)} cannot be combined with "
                        "'scenarios' defined in the config")
        scenarios = cfg["scenarios"]
    else:  # bare provider list: cohort from flags (legacy behavior)
        legacy = dict(DEFAULT_SCENARIO)
        legacy.update({k: v for k, v in overrides.items() if v is not None})
        try:  # validate as a scenario would be; the cohort stays anonymous
            _validate_scenario(dict(legacy, name="x"), 0)
        except ValueError as e:
            return fail(str(e))
        scenarios = [legacy]
    out = a.out or f"runs/harness-{datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(out, exist_ok=True)
    corpus = a.corpus or cfg.get("corpus")
    cohort_dir = corpus or out  # corpus mode: stories + responses persist there
    if corpus:
        os.makedirs(corpus, exist_ok=True)

    specs = None
    if not (a.skip_generate or a.grade_only):  # grade-only: grade what exists
        specs = [sp for sc in scenarios for sp in _scenario_specs(sc)]
    try:
        instances = generate_cohort(cohort_dir, specs,
                                    workers=a.render_workers,
                                    renderer_endpoint=cfg.get(
                                        "renderer_endpoint"))
    except RuntimeError as e:
        return fail(str(e))
    if not instances:
        return fail(f"no instances found under {cohort_dir}/instances/")
    records = run_solvers(instances, providers, cohort_dir, timeout=a.timeout,
                          call=not a.grade_only, workers=a.workers)
    by_iid = {i["iid"]: i for i in instances}
    graded = [grade_instance(by_iid[r["iid"]], r) for r in records]

    agg = aggregate(graded, instances, providers)
    agg["finished_at"] = datetime.now().isoformat(timespec="seconds")
    agg["elapsed_seconds"] = round(time.time() - t0, 1)
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
    render_html(agg, os.path.join(out, "report.html"), instances=instances)
    _print_leaderboard(agg)
    if agg.get("ladder"):
        print(f"Solved-up-to (every instance answered, mean facts >= "
              f"{agg['ladder']['solved_facts_threshold']:.2f}):")
        for pname, s in agg["ladder"]["by_provider"].items():
            m = s["max_solved_level"]
            print(f"  {pname:<24} L{m:02d}" if m is not None
                  else f"  {pname:<24} -")
            print(f"    ({s['levels_cleared']}/{len(s['levels'])} levels cleared)")
        print()
    if corpus:
        print(f"cohort:   {cohort_dir}/instances  {cohort_dir}/responses")
    print(f"results: {out}/results.json  {out}/results-detailed.json  "
          f"{out}/report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())

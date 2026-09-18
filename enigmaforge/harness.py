"""Versioned, fail-closed benchmark generation, execution and read-only regrading.

Use ``python -m enigmaforge.harness --help``. Judge calibration is a separate
``python -m enigmaforge.judge`` command; no judge model is inferred here.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from . import pipeline, grading, reporting, experiments
from .genres import PACKS, pick_genre
from .llm import chat_completion, llm_scene_renderer, generate_genre_pack, polish_realization
from .rng import Rng

BENCHMARK_VERSION = "3"
DEFAULT_SCENARIO = {"name": None, "instances": 20, "sizes": ["small"],
                    "genre": "auto", "burial_min": 1, "burial_max": 2,
                    "seed_base": 1000, "renderer": "template", "polish": False}
COHORT_FLAGS = ("--instances", "--sizes", "--genre", "--burial-min", "--burial-max", "--seed-base")
CONDITIONS = ("formal", "explicit", "implicit")
_LEVEL_OVERRIDES = set(pipeline.SIZES["large"])
_REQUEST_FIELDS = {"model", "base_url", "temperature", "max_tokens", "track_cost", "reasoning_effort", "reasoning_max_tokens", "seed"}
_PROVIDER_FIELDS = _REQUEST_FIELDS | {"name", "api_key", "api_key_env", "defaults"}
_PROTECTED = (Path(__file__).resolve().parent.parent / "runs/eval-final",
              Path(__file__).resolve().parent.parent / "benchmarks/openrouter-ladder")


def _note(message):
    print(f"[enigmaforge.harness] {message}", file=sys.stderr)


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "provider"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    """Hash implementation, not just a hand-maintained version label."""
    root = Path(__file__).resolve().parent
    return {"benchmark_version": BENCHMARK_VERSION, "grader_version": grading.GRADER_VERSION,
            "sources": {p.name: _file_hash(p) for p in sorted(root.glob("*.py"))}}


def _generator_provenance():
    p = provenance()
    # Grading, reporting, judging and the execution driver do not affect a
    # verified world; only generation-side sources bind the corpus.
    p["sources"] = {k: v for k, v in p["sources"].items()
                    if k not in {"grading.py", "reporting.py", "judge.py",
                                 "harness.py", "llm.py"}}
    p.pop("grader_version")
    return p


def _provenance_matches(stored):
    """Compare generator provenance, tolerating manifests written before an
    execution-side source was (re)classified as non-generation. Instance
    content hashes remain the tamper guard either way."""
    current = _generator_provenance()
    if not isinstance(stored, dict) or stored.get("benchmark_version") != current["benchmark_version"]:
        return False
    stored_sources = {k: v for k, v in (stored.get("sources") or {}).items()
                      if k in current["sources"]}
    return stored_sources == current["sources"]


def _read_json_config(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _guard_writable(path):
    resolved = Path(path).resolve()
    for protected in _PROTECTED:
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"immutable pilot path: {path}; choose a NEW v3 directory")
    return resolved


def _atomic_json(path, value):
    path = _guard_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lock(path):
    path = _guard_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"concurrent writer or interrupted lock: {path}; inspect before removing") from exc
    try:
        os.close(fd)
        yield
    finally:
        path.unlink()


def _seal(value):
    return {**value, "integrity_sha256": _digest(value)}


def _unseal(value, path):
    if not isinstance(value, dict):
        raise ValueError(f"invalid versioned artifact: {path}")
    payload = {k: v for k, v in value.items() if k != "integrity_sha256"}
    if value.get("integrity_sha256") != _digest(payload):
        raise ValueError(f"artifact integrity mismatch: {path}; do not reuse modified caches")
    return payload


def _public_request(kwargs):
    # Allow-list ensures neither literal keys nor credential-bearing headers leak.
    return {k: kwargs[k] for k in sorted(_REQUEST_FIELDS) if k in kwargs}


def _providers_from_list(raw, path, defaults=None):
    defaults = {} if defaults is None else defaults
    if not isinstance(defaults, dict):
        raise ValueError(f"provider_defaults must be an object: {path}")
    for name, group in defaults.items():
        if not isinstance(group, dict) or set(group) - (_PROVIDER_FIELDS - {"name", "defaults"}):
            raise ValueError(f"invalid provider_defaults.{name}: {path}")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"providers must be a non-empty list: {path}")
    providers, seen, slugs = [], set(), set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) - _PROVIDER_FIELDS:
            raise ValueError(f"provider #{index}: invalid fields")
        ref = item.get("defaults")
        if ref is not None and (not isinstance(ref, str) or ref not in defaults):
            raise ValueError(f"provider #{index}: unknown defaults group {ref!r}")
        resolved = {**defaults.get(ref, {}), **item}
        model, endpoint = resolved.get("model"), resolved.get("base_url")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"provider #{index}: explicit model is required")
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise ValueError(f"provider #{index}: explicit http(s) base_url is required; no autodiscovery in benchmarks")
        name = resolved.get("name") or model.rpartition("/")[2]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"provider #{index}: name must be nonempty")
        if name in seen or _slug(name) in slugs:
            raise ValueError(f"duplicate provider name or filename collision: {name!r}")
        seen.add(name)
        slugs.add(_slug(name))
        if resolved.get("api_key") is not None and resolved.get("api_key_env"):
            raise ValueError(f"provider {name}: api_key and api_key_env are mutually exclusive")
        for key in ("max_tokens", "reasoning_max_tokens"):
            if key in resolved and (type(resolved[key]) is not int or resolved[key] < 1):
                raise ValueError(f"provider {name}: {key} must be a positive integer")
        kwargs = {k: resolved[k] for k in _REQUEST_FIELDS if k in resolved}
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_tokens", 4096)
        kwargs.setdefault("track_cost", False)
        if resolved.get("api_key") is not None:
            kwargs["api_key"] = resolved["api_key"]
        elif resolved.get("api_key_env"):
            kwargs["api_key"] = os.environ.get(resolved["api_key_env"], "")
        else:
            kwargs["api_key"] = ""
        providers.append({"name": name, "kwargs": kwargs})
    return providers


def load_providers(path):
    return _providers_from_list(_read_json_config(path), path)


def _validate_level(level, where):
    if not isinstance(level, dict) or set(level) - {"size", "overrides", "burial"}:
        raise ValueError(f"{where}: invalid level object")
    result = {"size": level.get("size", "small"), "overrides": level.get("overrides", {}),
              "burial": level.get("burial")}
    if result["size"] not in pipeline.SIZES:
        raise ValueError(f"{where}: size must be one of {sorted(pipeline.SIZES)}")
    if not isinstance(result["overrides"], dict) or set(result["overrides"]) - _LEVEL_OVERRIDES:
        raise ValueError(f"{where}: unsupported level overrides")
    for key, value in result["overrides"].items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{where}: {key} must be a nonnegative integer")
    if result["burial"] is not None and (type(result["burial"]) is not int or not 0 <= result["burial"] <= 12):
        raise ValueError(f"{where}: burial must be an integer in 0..12")
    return result


def _validate_scenario(scenario, index, default_renderer="template"):
    if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str) or not scenario["name"].strip():
        raise ValueError(f"scenario #{index}: missing required name")
    allowed = set(DEFAULT_SCENARIO) | {"levels"}
    if set(scenario) - allowed:
        raise ValueError(f"scenario {scenario['name']}: unknown fields {sorted(set(scenario) - allowed)}")
    result = {**DEFAULT_SCENARIO, "renderer": default_renderer, **scenario}
    where = f"scenario {result['name']!r}"
    if type(result["instances"]) is not int or result["instances"] < 1:
        raise ValueError(f"{where}: instances must be an integer >=1")
    if scenario.get("levels") is not None:
        if "sizes" in scenario or not isinstance(scenario["levels"], list) or not scenario["levels"]:
            raise ValueError(f"{where}: nonempty levels and sizes are mutually exclusive")
        result["levels"] = [_validate_level(v, f"{where} level {j}") for j, v in enumerate(scenario["levels"])]
        result["sizes"] = None
    elif not isinstance(result["sizes"], list) or not result["sizes"] or any(s not in pipeline.SIZES for s in result["sizes"]):
        raise ValueError(f"{where}: sizes must be a nonempty list from {sorted(pipeline.SIZES)}")
    if result["genre"] not in ("auto", "llm") and result["genre"] not in PACKS:
        raise ValueError(f"{where}: unknown genre {result['genre']!r}")
    if result["renderer"] not in ("template", "llm"):
        raise ValueError(f"{where}: renderer must be template or llm")
    if type(result["polish"]) is not bool:
        raise ValueError(f"{where}: polish must be boolean")
    for key in ("burial_min", "burial_max"):
        if type(result[key]) is not int or not 0 <= result[key] <= 12:
            raise ValueError(f"{where}: {key} must be an integer in 0..12")
    if result["burial_min"] > result["burial_max"] or type(result["seed_base"]) is not int:
        raise ValueError(f"{where}: invalid burial range or seed_base")
    return result


def _conditions(value):
    if not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value) or any(c not in CONDITIONS for c in value):
        raise ValueError(f"conditions must be a nonempty unique list from {CONDITIONS}")
    return list(value)


def load_config(path):
    raw = _read_json_config(path)
    if isinstance(raw, list):
        return {"providers": _providers_from_list(raw, path), "scenarios": [dict(DEFAULT_SCENARIO)],
                "corpus": None, "out": None, "conditions": list(CONDITIONS),
                "realizations": [1, 2], "renderer_endpoint": None, "baselines": False}
    allowed = {"providers", "provider_defaults", "scenarios", "corpus", "out", "conditions",
               "realizations", "renderer", "renderer_model", "renderer_base_url", "renderer_api_key_env", "baselines"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError(f"unknown benchmark config fields: {path}")
    result = {k: raw.get(k) for k in ("corpus", "out")}
    for key, value in result.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty directory path")
        if value:
            _guard_writable(value)
    result["providers"] = _providers_from_list(raw.get("providers"), path, raw.get("provider_defaults"))
    default_renderer = raw.get("renderer", "template")
    if default_renderer not in ("template", "llm"):
        raise ValueError("renderer must be template or llm")
    raw_scenarios = raw.get("scenarios")
    if raw_scenarios is None:
        result["scenarios"] = [{**DEFAULT_SCENARIO, "renderer": default_renderer}]
    else:
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ValueError("scenarios must be a nonempty list")
        result["scenarios"] = [_validate_scenario(s, i, default_renderer) for i, s in enumerate(raw_scenarios)]
        slugs = [_slug(s["name"]) for s in result["scenarios"]]
        if len(set(slugs)) != len(slugs):
            raise ValueError("duplicate scenario names or filename collision")
    result["conditions"] = _conditions(raw.get("conditions", list(CONDITIONS)))
    result["realizations"] = raw.get("realizations", [1, 2])
    if result["realizations"] != [1, 2]:
        raise ValueError("v3 requires both realizations: realizations must be [1, 2]")
    result["baselines"] = raw.get("baselines", False)
    if type(result["baselines"]) is not bool:
        raise ValueError("baselines must be boolean")
    endpoint = {k: raw[f"renderer_{k}"] for k in ("model", "base_url", "api_key_env") if raw.get(f"renderer_{k}")}
    if any(not isinstance(v, str) for v in endpoint.values()):
        raise ValueError("renderer endpoint settings must be strings")
    paid_renderer = any(s["renderer"] == "llm" or s["genre"] == "llm" or s["polish"] for s in result["scenarios"])
    if paid_renderer and (not endpoint.get("model") or not endpoint.get("base_url", "").startswith(("http://", "https://"))):
        raise ValueError("LLM rendering/genre/polish requires explicit renderer_model and renderer_base_url")
    result["renderer_endpoint"] = endpoint or None
    return result


def _tagged_seed(seed, tag):
    return int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:16], 16)


def _scenario_specs(scenario):
    result = []
    levels = scenario.get("levels")
    strata = list(enumerate(levels)) if levels is not None else [(None, None)]
    for level, settings in strata:
        for index in range(scenario["instances"]):
            seed = _tagged_seed(scenario["seed_base"], json.dumps([scenario["name"], level, index]))
            genre = scenario["genre"] if scenario["genre"] != "auto" else pick_genre(seed)
            burial = settings["burial"] if settings else None
            if burial is None:
                burial = Rng(_tagged_seed(seed, "burial")).range(scenario["burial_min"], scenario["burial_max"])
            prefix = f"inst-{_slug(scenario['name'])}" if scenario["name"] else "inst"
            iid = f"{prefix}{f'-L{level:02d}' if level is not None else ''}-{index:03d}-{genre}"
            result.append({"iid": iid, "scenario": scenario["name"], "level": level, "index": index,
                           "seed": seed, "genre": genre, "burial": burial,
                           "size": settings["size"] if settings else scenario["sizes"][index % len(scenario["sizes"])],
                           "level_overrides": settings["overrides"] if settings else {},
                           "renderer": scenario.get("renderer", "template"), "polish": scenario.get("polish", False)})
    return result


def build_specs(n_instances, sizes, genre, burial_min, burial_max, seed_base):
    return _scenario_specs({**DEFAULT_SCENARIO, "instances": n_instances, "sizes": list(sizes),
                            "genre": genre, "burial_min": burial_min, "burial_max": burial_max, "seed_base": seed_base})


def _resolved_generation(spec, endpoint):
    return {"generator": {**pipeline.SIZES[spec["size"]], "mode": "story", "genre": spec["genre"],
                          "burial": spec["burial"], **spec.get("level_overrides", {})},
            "renderer": {"kind": spec["renderer"], "polish": spec["polish"],
                         "endpoint": {k: v for k, v in (endpoint or {}).items() if k != "api_key_env"}}}


def _manifest_contract(specs, endpoint):
    iids = [s["iid"] for s in specs]
    if len(iids) != len(set(iids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in iids):
        raise ValueError("duplicate or unsafe instance identifiers")
    return {"benchmark_version": BENCHMARK_VERSION, "expected_specs": specs,
            "settings": {s["iid"]: _resolved_generation(s, endpoint) for s in specs},
            "provenance": _generator_provenance()}


def _validate_manifest(root, manifest, contract=None, require_complete=True):
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("not a v3 corpus; use --legacy-regrade with a NEW output directory")
    if contract is not None and any(
            manifest.get(k) != v for k, v in contract.items() if k != "provenance"):
        raise ValueError("corpus manifest/config/source mismatch; use a NEW corpus or the original exact configuration")
    if not _provenance_matches(manifest.get("provenance")):
        raise ValueError("corpus generator provenance mismatch; generate a NEW corpus")
    expected = {s["iid"] for s in manifest["expected_specs"]}
    entries = manifest.get("instances", {})
    if set(entries) - expected:
        raise ValueError("manifest contains unexpected instances")
    actual_dirs = {p.name for p in (root / "instances").iterdir() if p.is_dir()} if (root / "instances").exists() else set()
    if actual_dirs != set(entries):
        raise ValueError("corpus instance directory set differs from manifest (missing or untracked files)")
    instances = []
    for iid, entry in entries.items():
        directory = root / "instances" / iid
        paths = {str(p.relative_to(directory)): p for p in directory.rglob("*") if p.is_file()}
        if set(paths) != set(entry["files"]) or any(_file_hash(paths[p]) != digest for p, digest in entry["files"].items()):
            raise ValueError(f"corpus content hash mismatch for {iid}; cached files were modified or removed")
        instance = _read_json_config(directory / "instance.json")
        if instance.get("iid") != iid:
            raise ValueError(f"instance identity mismatch for {iid}")
        instances.append(instance)
    complete = set(entries) == expected and not manifest.get("exclusions")
    if manifest.get("complete") != complete:
        raise ValueError("manifest completeness flag does not match expected instance set")
    if require_complete and not complete:
        missing = sorted(expected - set(entries))
        raise ValueError(f"incomplete corpus: {len(missing)} expected worlds missing ({', '.join(missing[:5])}); resume generation, never grade a partial cohort")
    return sorted(instances, key=lambda item: item["iid"])


def generate_cohort(out_dir, specs=None, workers=1, renderer_endpoint=None, *, read_only=False, attempts=3):
    root = Path(out_dir)
    manifest_path = root / "manifest.json"
    contract = _manifest_contract(specs, renderer_endpoint) if specs is not None else None
    if specs is None or read_only:
        if not manifest_path.is_file():
            raise ValueError(f"missing v3 manifest: {manifest_path}; use --legacy-regrade for pilot artifacts")
        manifest = _unseal(_read_json_config(manifest_path), manifest_path)
        return _validate_manifest(root, manifest, contract)
    _guard_writable(root)
    with _lock(root / ".generation.lock"):
        if manifest_path.exists():
            manifest = _unseal(_read_json_config(manifest_path), manifest_path)
            instances = _validate_manifest(root, manifest, contract, require_complete=False)
        else:
            if (root / "instances").exists() and any((root / "instances").iterdir()):
                raise ValueError("unversioned corpus already exists; choose a NEW v3 corpus directory")
            manifest = {**contract, "instances": {}, "attempts": [], "exclusions": [], "complete": False}
            instances = []
            _atomic_json(manifest_path, _seal(manifest))
        pending = [s for s in specs if s["iid"] not in manifest["instances"]]
        if not pending:
            return instances
        ep = renderer_endpoint or {}
        llm_kwargs = {k: ep[k] for k in ("model", "base_url") if k in ep}
        llm_kwargs["api_key"] = os.environ.get(ep.get("api_key_env", ""), "")
        needs_llm = any(s["renderer"] == "llm" or s["genre"] == "llm" or s["polish"] for s in pending)
        if needs_llm and (not ep.get("model") or not ep.get("base_url")):
            raise ValueError("generation requires an explicit renderer model and endpoint")
        renderer = llm_scene_renderer(**llm_kwargs) if any(s["renderer"] == "llm" for s in pending) else None
        genre_gen = (lambda seed: generate_genre_pack(seed=seed, max_attempts=5, **llm_kwargs)) if any(s["genre"] == "llm" for s in pending) else None
        polisher = (lambda world, realization: polish_realization(world, realization, **llm_kwargs)) if any(s["polish"] for s in pending) else None

        def build_one(spec):
            world = pipeline.build(spec["size"], spec["seed"], config_overrides=_resolved_generation(spec, ep)["generator"],
                                   renderer=renderer if spec["renderer"] == "llm" else None,
                                   genre_gen=genre_gen if spec["genre"] == "llm" else None,
                                   polisher=polisher if spec["polish"] else None)
            staging = Path(tempfile.mkdtemp(prefix=".instance-", dir=root))
            try:
                pipeline.package(world, str(staging))
                hidden = pipeline._hidden(world)
                formal = {"variables": hidden["variables"], "constraints": hidden["constraints"]}
                instance = {**spec, "family_id": spec["iid"], "formal_world": formal,
                            "story_text": (staging / "story.md").read_text(),
                            "story_r2_text": (staging / "story_r2.md").read_text(),
                            "ground_truth": hidden["ground_truth"],
                            "surfaces": {v.vid: v.surface_names[0] for v in world.variables},
                            "decision_policy": world.meta["decision_policy"], "policy_text": world.meta["policy_text"],
                            "difficulty": experiments.measure_difficulty(formal),
                            "renderer_provenance": _resolved_generation(spec, ep)["renderer"],
                            "n_constraints": len(world.constraints)}
                _atomic_json(staging / "instance.json", instance)
                files = {str(p.relative_to(staging)): _file_hash(p) for p in staging.rglob("*") if p.is_file()}
                destination = root / "instances" / spec["iid"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError(f"instance destination collision: {destination}")
                os.rename(staging, destination)
                return instance, files
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

        for _ in range(attempts):
            if not pending:
                break
            failed = []
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(build_one, spec): spec for spec in pending}
                for future in as_completed(futures):
                    spec = futures[future]
                    attempt = {"iid": spec["iid"], "at": datetime.now(timezone.utc).isoformat()}
                    try:
                        instance, files = future.result()
                        instances.append(instance)
                        manifest["instances"][spec["iid"]] = {"files": files}
                        attempt["status"] = "generated"
                    except Exception as exc:
                        attempt.update(status="generation_error", error=str(exc))
                        failed.append(spec)
                    manifest["attempts"].append(attempt)
                    manifest["complete"] = len(manifest["instances"]) == len(specs)
                    _atomic_json(manifest_path, _seal(manifest))
            pending = failed
        if pending:
            raise ValueError(f"{len(pending)}/{len(specs)} generation failures; attempts retained in {manifest_path}; rerun the same configuration")
        return sorted(instances, key=lambda item: item["iid"])


def _response_path(out_dir, provider_name, iid):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", iid):
        raise ValueError(f"unsafe instance identifier: {iid!r}")
    return Path(out_dir) / "responses" / f"{_slug(provider_name)}--{iid}.json"


def _response_identity(instance, provider, timeout):
    return {"benchmark_version": BENCHMARK_VERSION, "provider": provider["name"], "iid": instance["iid"],
            "input_sha256": _digest(instance["input_text"]), "prompt_sha256": _digest(grading.SOLVER_PROMPT),
            "request": {**_public_request(provider["kwargs"]), "timeout": timeout},
            "sources": {name: digest for name, digest in provenance()["sources"].items()
                        if name in {"harness.py", "llm.py", "grading.py", "experiments.py"}}}


def _missing(provider, instance):
    return {"provider": provider["name"], "iid": instance["iid"], "status": "missing", "text": None,
            "error": "no cached solver response", "seconds": 0.0, "cost": None, "tokens": None, "attempts": []}


def _record(identity, attempts):
    last = {k: v for k, v in attempts[-1].items()
            if k not in ("provider",)}  # endpoint host must not shadow the benchmark provider name
    usages = [a.get("usage") or {} for a in attempts]
    return {"provider": identity["provider"], "iid": identity["iid"], "identity": identity,
            **last, "attempts": attempts,"seconds": sum(a["seconds"] for a in attempts),
            "evaluation_seconds": sum(a["seconds"] for a in attempts),
            "cost": sum(u["cost"] for u in usages if isinstance(u.get("cost"), (int, float))) if any(isinstance(u.get("cost"), (int, float)) for u in usages) else None,
            "tokens": {k: sum(u.get(k) or 0 for u in usages) for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if any(usages) else None}


def _semantic_identity(identity):
    """The fields that determine what was asked. Implementation-source hashes
    are informational: they are recorded but do not invalidate a cache, since
    the recorded request metadata already captures the effective payload and
    grading/reporting run after production."""
    return {k: v for k, v in identity.items() if k != "sources"}


def run_solvers(instances, providers, out_dir, timeout, call=True, workers=1, *, retry_errors=True,
                retry_refusals=False, attempts=1):
    if attempts < 1 or workers < 1 or timeout <= 0:
        raise ValueError("attempts, workers and timeout must be positive")
    names = [_slug(p["name"]) for p in providers]
    iids = [i["iid"] for i in instances]
    if len(names) != len(set(names)) or len(iids) != len(set(iids)):
        raise ValueError("provider or instance cache filename collision")
    retriable = {"transport_error", "empty", "invalid_response"} if retry_errors else set()
    if retry_refusals:
        retriable |= {"content_filter", "refusal"}
    records, pending = [], []
    for provider in providers:
        for instance in instances:
            path = _response_path(out_dir, provider["name"], instance["iid"])
            identity = _response_identity(instance, provider, timeout)
            cached = None
            if path.exists():
                cached = _unseal(_read_json_config(path), path)
                if _semantic_identity(cached.get("identity")) != _semantic_identity(identity):
                    raise ValueError(f"solver cache identity mismatch: {path}; model, input, prompt or request changed; choose a NEW response corpus")
                if not cached.get("attempts"):
                    raise ValueError(f"solver cache has no attempt history: {path}")
            if cached is not None and (not call or cached["status"] not in retriable):
                records.append(cached)
            elif call:
                pending.append((provider, instance, path, identity, cached))
            else:
                records.append(_missing(provider, instance))

    def solve_one(provider, instance, path, identity, cached):
        with _lock(str(path) + ".lock"):
            if path.exists():
                current = _unseal(_read_json_config(path), path)
                if current != cached:
                    raise ValueError(f"concurrent solver cache update: {path}; rerun to load it")
            history = list(cached["attempts"]) if cached else []
            for _ in range(attempts):
                started = time.monotonic()
                try:
                    metadata = chat_completion([{"role": "system", "content": grading.SOLVER_PROMPT},
                                                {"role": "user", "content": instance["input_text"]}],
                                               timeout=timeout, with_metadata=True, **provider["kwargs"])
                    if not isinstance(metadata, dict):
                        raise ValueError("transport did not return a metadata envelope")
                except Exception as exc:
                    metadata = dict(getattr(exc, "metadata", None) or {})
                    metadata.setdefault("status", "transport_error")
                    metadata.setdefault("text", None)
                    metadata["error"] = str(exc)
                metadata["seconds"] = round(time.monotonic() - started, 6)
                metadata["attempt"] = len(history) + 1
                metadata["at"] = datetime.now(timezone.utc).isoformat()
                history.append(metadata)
                record = _record(identity, history)
                _atomic_json(path, _seal(record))
                if record["status"] not in retriable:
                    break
            return record

    if pending:
        _guard_writable(out_dir)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(solve_one, *item) for item in pending]
            records.extend(f.result() for f in as_completed(futures))
    return sorted(records, key=lambda record: (record["provider"], record["iid"]))


def _prepare_output(path, *, must_be_new=False):
    root = _guard_writable(path)
    if must_be_new and root.exists():
        raise ValueError(f"legacy regrade requires a NEW output directory: {root}")
    if any((root / name).exists() for name in ("results.json", "results-detailed.json", "report.html")):
        raise ValueError(f"completed report output is immutable: {root}; choose a NEW --out")
    return root


def _write_report(out, instances, providers, records, graded, completeness, manifest, started):
    aggregate = reporting.aggregate(graded, instances, providers)
    aggregate.update(benchmark_version=BENCHMARK_VERSION, provenance=provenance(),
                     generation_completeness=completeness, corpus_manifest=manifest,
                     finished_at=datetime.now(timezone.utc).isoformat(),
                     regrading_seconds=round(time.monotonic() - started, 6),
                     evaluation_seconds=sum(r.get("seconds") or 0 for r in records))
    _atomic_json(out / "results.json", aggregate)
    _atomic_json(out / "results-detailed.json", {**aggregate, "instances": instances,
                 "raw_responses": {f"{r['provider']}--{r['iid']}": r for r in records}})
    fd, temporary = tempfile.mkstemp(prefix=".report-", suffix=".html", dir=out)
    os.close(fd)
    try:
        reporting.render_html(aggregate, temporary, instances=instances)
        os.replace(temporary, out / "report.html")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    reporting.print_leaderboard(aggregate)
    return aggregate


def _legacy_regrade(args):
    if not args.out or not args.legacy_corpus:
        raise ValueError("--legacy-regrade requires --legacy-corpus and --out pointing to a NEW directory")
    if args.providers or args.corpus or args.baselines or args.conditions or args.skip_generate:
        raise ValueError("legacy regrade uses saved provider identities and corpus, not new benchmark configuration")
    out = _prepare_output(args.out, must_be_new=True)
    source = Path(args.legacy_regrade).resolve()
    corpus = Path(args.legacy_corpus).resolve()
    if out == corpus or corpus in out.parents or out == source.parent or source.parent in out.parents:
        raise ValueError("legacy output must be outside the read-only source corpus and report directory")
    saved = _read_json_config(source)
    raw = saved.get("raw_responses")
    if not isinstance(raw, dict):
        raise ValueError("legacy input must be saved results-detailed.json with raw_responses")
    paths = sorted((corpus / "instances").glob("*/instance.json"))
    if not paths:
        raise ValueError(f"no legacy instances under {corpus}/instances")
    instances = [_read_json_config(p) for p in paths]
    for instance in instances:
        instance.setdefault("family_id", instance["iid"])
        instance.setdefault("condition", "implicit")
        instance.setdefault("realization", 1)
    providers = [{"name": row["provider"]} for row in saved.get("leaderboard", [])]
    if not providers or len({p["name"] for p in providers}) != len(providers):
        raise ValueError("legacy leaderboard is missing unique provider identities")
    human = _read_json_config(args.extractions) if args.extractions else {}
    if not isinstance(human, dict):
        raise ValueError("human extractions must map provider--iid to extraction objects")
    if args.allow_judge_calls and (not args.judge_model or not args.judge_base_url or args.judge_max_calls < 1):
        raise ValueError("paid judging requires --judge-model, --judge-base-url and positive --judge-max-calls")
    if args.grade_only and args.allow_judge_calls:
        raise ValueError("--grade-only never calls any model; remove --allow-judge-calls")
    if bool(args.judge_model) != bool(args.judge_base_url):
        raise ValueError("provide both --judge-model and --judge-base-url for cached or paid extraction")
    from . import judge
    budget = judge.CallBudget(args.judge_max_calls) if args.allow_judge_calls else None
    records, graded = [], []
    prior = {(r["provider"], r["iid"]): r for r in saved.get("records", [])}
    expected_keys = {f"{p['name']}--{i['iid']}" for p in providers for i in instances}
    if set(human) - expected_keys:
        raise ValueError("human extraction file has unknown provider--iid keys")
    if set(raw) - expected_keys:
        raise ValueError("legacy raw responses refer to missing corpus instances or unknown providers")
    manifest = {"benchmark_version": "legacy", "classification": "pilot", "complete": False,
                "expected_cohort_known": False, "observed_worlds": len(instances),
                "original_reported_worlds": saved.get("n_instances"),
                "source_report_sha256": _file_hash(source),
                "instance_hashes": {str(p.relative_to(corpus)): _file_hash(p) for p in paths},
                "limitations": ["Original expected generation manifest unavailable; not a complete prospective cohort.",
                                "Legacy decision policies are not reconstructed; decision metrics unavailable."]}
    started = time.monotonic()
    with _lock(out / ".report.lock"):
        for provider in providers:
            for instance in instances:
                key = f"{provider['name']}--{instance['iid']}"
                record = {**prior.get((provider["name"], instance["iid"]), {}),
                          **raw.get(key, _missing(provider, instance)),
                          "provider": provider["name"], "iid": instance["iid"], "legacy": True}
                if "status" not in record:
                    record["status"] = ("content_filter" if "content_filter" in str(record.get("error")) else "transport_error") if record.get("error") else ("answered" if record.get("text") else "missing")
                extraction = human.get(key)
                if extraction is None and record.get("text") and args.judge_model:
                    extraction = judge.extract_claims(record["text"], list(instance["surfaces"].values()),
                        model=args.judge_model, base_url=args.judge_base_url,
                        api_key=os.environ.get(args.judge_api_key_env, "") if args.judge_api_key_env else "",
                        cache_dir=args.judge_cache or str(out / "judge-cache"), call=args.allow_judge_calls, budget=budget)
                records.append(record)
                graded.append(grading.grade_instance(instance, record, extraction=extraction))
        _write_report(out, instances, providers, records, graded,
                      {"classification": "pilot", "complete": False, "expected": None, "available": len(instances)}, manifest, started)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", help="v3 JSON config; explicit model and endpoint required")
    parser.add_argument("--instances", type=int)
    parser.add_argument("--sizes")
    parser.add_argument("--genre", choices=["auto", "llm"] + sorted(PACKS))
    parser.add_argument("--burial-min", type=int)
    parser.add_argument("--burial-max", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--out", help="new report directory; completed reports are immutable")
    parser.add_argument("--corpus", help="v3 manifest-backed corpus/cache directory")
    parser.add_argument("--timeout", type=float, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--solver-attempts", type=int, default=1)
    parser.add_argument("--retry-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-refusals", action="store_true", help="explicitly retry cached filters/refusals")
    parser.add_argument("--skip-generate", action="store_true", help="validate complete corpus against configured expected specs")
    parser.add_argument("--grade-only", action="store_true", help="strictly cached-only: never call a solver, renderer or judge")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--conditions", help="comma-separated formal,explicit,implicit; both surface realizations required")
    parser.add_argument("--baselines", action="store_true", help="include deterministic no-API baselines")
    parser.add_argument("--baselines-only", action="store_true", help="run deterministic baselines without solver calls")
    parser.add_argument("--legacy-regrade", metavar="RESULTS_DETAILED", help="read-only pilot regrade; no decision retrofit")
    parser.add_argument("--legacy-corpus")
    parser.add_argument("--extractions", help="human extraction JSON mapping provider--iid to claims/conflicts/unmapped_claims")
    parser.add_argument("--judge-cache", help="cached-only extraction unless --allow-judge-calls is explicit")
    parser.add_argument("--judge-model", help="explicit model; calibrate separately with python -m enigmaforge.judge")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key-env")
    parser.add_argument("--allow-judge-calls", action="store_true")
    parser.add_argument("--judge-max-calls", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.legacy_regrade:
            return _legacy_regrade(args)
        if any((args.legacy_corpus, args.extractions, args.judge_model, args.judge_base_url, args.judge_cache, args.allow_judge_calls)):
            raise ValueError("judge/extraction and legacy flags require --legacy-regrade; calibration uses python -m enigmaforge.judge")
        if not args.providers:
            raise ValueError("--providers is required for v3 generation/evaluation")
        if min(args.workers, args.render_workers, args.generation_attempts, args.solver_attempts) < 1 or args.timeout <= 0:
            raise ValueError("workers, attempts and timeout must be positive")
        if args.grade_only and args.generate_only:
            raise ValueError("--grade-only and --generate-only are mutually exclusive")
        config = load_config(args.providers)
        overrides = {"instances": args.instances, "sizes": args.sizes.split(",") if args.sizes else None,
                     "genre": args.genre, "burial_min": args.burial_min, "burial_max": args.burial_max, "seed_base": args.seed_base}
        scenarios = config["scenarios"]
        if any(s["name"] is not None for s in scenarios):
            if any(v is not None for v in overrides.values()):
                raise ValueError("cohort CLI flags cannot be combined with config-defined scenarios")
        else:
            scenario = {**scenarios[0], **{k: v for k, v in overrides.items() if v is not None}}
            validated = _validate_scenario({**scenario, "name": "default"}, 0)
            validated["name"] = None
            scenarios = [validated]
        conditions = _conditions(args.conditions.split(",")) if args.conditions else config["conditions"]
        specs = [spec for scenario in scenarios for spec in _scenario_specs(scenario)]
        out = _prepare_output(args.out or config["out"] or f"runs/v3-{datetime.now():%Y%m%d-%H%M%S-%f}")
        corpus = args.corpus or config["corpus"] or str(out)
        _guard_writable(corpus)
        with _lock(out / ".report.lock"):
            base_instances = generate_cohort(corpus, specs, workers=args.render_workers,
                renderer_endpoint=config["renderer_endpoint"], read_only=args.grade_only or args.skip_generate,
                attempts=args.generation_attempts)
            if args.generate_only:
                _note(f"complete v3 corpus: {corpus}/manifest.json ({len(base_instances)} independent worlds)")
                return 0
            instances = experiments.expand_conditions(base_instances, conditions=conditions, realizations=(1, 2))
            providers = [] if args.baselines_only else config["providers"]
            records = run_solvers(instances, providers, corpus, args.timeout, call=not args.grade_only,
                                  workers=args.workers, retry_errors=args.retry_errors,
                                  retry_refusals=args.retry_refusals, attempts=args.solver_attempts)
            if args.baselines or args.baselines_only or config["baselines"]:
                baseline_records = experiments.baseline_records(instances)
                baseline_names = sorted({r["provider"] for r in baseline_records})
                if set(baseline_names) & {p["name"] for p in providers}:
                    raise ValueError("configured provider name collides with a baseline")
                providers += [{"name": name} for name in baseline_names]
                records += baseline_records
            started = time.monotonic()
            by_iid = {i["iid"]: i for i in instances}
            graded = [grading.grade_instance(by_iid[r["iid"]], r) for r in records]
            manifest = _read_json_config(Path(corpus) / "manifest.json")
            _write_report(out, instances, providers, records, graded,
                {"classification": "v3", "complete": True, "expected": len(specs), "available": len(base_instances),
                 "conditions": conditions, "realizations": [1, 2], "expected_variants": len(instances)}, manifest, started)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _note(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())

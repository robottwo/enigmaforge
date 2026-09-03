"""LLM scene renderer: plug a chat model into the story contract.

Zero-dependency (stdlib urllib) OpenAI-compatible client — works with the
OpenAI API and with local servers that speak the same shape (ollama at
http://localhost:11434/v1, vLLM, llama.cpp).

Config resolution order (first wins, per field):
  1. explicit arguments (llm_scene_renderer(model=..., base_url=...))
  2. environment: OPENAI_BASE_URL, OPENAI_API_KEY, ENIGMAFORGE_MODEL / OPENAI_MODEL
  3. autodiscovery from local agent configs (see discover_llm_config)
  4. built-in defaults

Autodiscovery reads only well-known config paths under $HOME and never
logs keys (debug output is redacted). Probed agents:
  - opencode  ~/.local/share/opencode/auth.json + ~/.config/opencode/opencode.jsonc
  - codex     ~/.codex/config.toml + ~/.codex/auth.json
  - goose     ~/.config/goose/config.yaml
  - continue  ~/.continue/config.json
Not probed (no OpenAI-compatible endpoint in their local configs): pi
(skills/extensions only), claude/gemini/cursor (hooks + model display names;
their native APIs do not speak /chat/completions).

The renderer is a drop-in for compile_story(renderer=...): the model writes
the scene prose but MUST embed every claim clause verbatim. The pipeline
does not trust it — span search and the extraction round-trip gate every
scene, and compile_story_verified rejection-samples failures.

Debug:  python3 -m enigmaforge.llm   # show candidates + resolved config
"""
import json
import os
import re
import sys
import time
import urllib.request

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# OpenAI-compatible endpoints, keyed by provider id (agent-config naming).
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "zai-coding-plan": "https://api.z.ai/api/coding/paas/v4",
    "zai": "https://api.z.ai/api/paas/v4",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "together": "https://api.together.xyz/v1",
    "xai": "https://api.x.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "cerebras": "https://api.cerebras.ai/v1",
}

# Providers whose native API is NOT OpenAI-compatible — never probe them
# with /chat/completions even when a key is on disk.
NON_COMPATIBLE = {"anthropic", "azure", "bedrock", "google", "gemini", "vertex"}

def _scene_prompt(world):
    from .genres import get_pack
    vibe = get_pack(world).vibe if world is not None else "small-harbor"
    return SYSTEM_PROMPT_TEMPLATE.format(vibe=vibe)


SYSTEM_PROMPT_TEMPLATE = """You are ghostwriting one scene of a quiet {vibe}
mystery. You will receive a PREMISE, a CHARACTER sheet, a SETTING, and CLAIM
clauses — residue of things that happened. Write 3 to 8 sentences of scene
prose that carries every claim into the story. Write the scene as if it is
happening now: concrete actions, spoken lines, plain crisp sentences. Never
reminisce, never summarize, never stack three clauses into one sentence.

Hard rules:
1. Every CLAIM must appear in your scene EXACTLY as given, character for
   character. Never reword, shorten, translate, or capitalize a claim, and
   never let a claim open a sentence — lead into it with your own words.
2. Stage each claim as something that HAPPENED: an errand, an argument, a
   habit, something said in passing ("...she said, not looking up...").
   Ground every claim in a person doing something concrete. NEVER stage a
   claim as someone consulting, quoting, or reading out a record, ledger,
   or paper — characters handling exhibits is itself the puzzle-frame we
   are hiding.
3. Obey the SETTING and timeline you are given. Do not reuse weather,
   rooms, or mannerisms from other scenes; each scene advances the story.
4. Keep every character consistent with the sheet (role, pronouns, habits).
5. Write around the claims: what people do, say, and want — small human
   business, not atmospheric padding. Do NOT add any further statement
   about a document, ledger, or record reading, agreeing, matching, or
   being stamped, signed, or marked.
6. No lists, no numbering, no headings. Never ask a question, and never
   suggest there is a puzzle, task, or anything to be figured out.
7. Return only the scene prose — no preamble, no explanation."""

def _plan_prompt(world):
    from .genres import get_pack
    pack = get_pack(world) if world is not None else None
    vibe = pack.vibe if pack else "small-harbor"
    return PLAN_PROMPT_TEMPLATE.format(vibe=vibe)


PLAN_PROMPT_TEMPLATE = """You are planning a short quiet {vibe} mystery told
in scenes. You will get a cast, what the story turns on, and the scene list
with timeline labels. Produce a plan that makes the scenes feel like ONE
continuous story: a premise, a character sheet, and a distinct setting per
scene (different locations and weather — never the same room twice in a
row). Return ONLY the JSON object requested; no prose around it."""

def _polish_prompt(world):
    from .genres import get_pack
    pack = get_pack(world) if world is not None else None
    vibe = pack.vibe if pack else "small-harbor"
    return POLISH_PROMPT_TEMPLATE.format(vibe=vibe)


POLISH_PROMPT_TEMPLATE = """You are revising the draft of a short {vibe}
mystery story for publication. Rewrite it as natural, flowing narrative:
break up run-on sentences, keep the language plain and colloquial, vary
sentence rhythm, merge and break sentences as the prose wants, smooth the
seams between scenes, cut repetition, and let the characters and the
season come alive. Scenes play out in the present, not as reminiscence.
You have full creative liberty over everything EXCEPT the claims.

Hard rules:
1. Every CLAIM clause (listed after the draft) must appear in your revision
   EXACTLY as given, character for character — never reworded, shortened,
   capitalized, or dropped. Rewrite freely around them.
2. Do not add any new statement about a document, ledger, or record
   reading, agreeing, matching, or being stamped, signed, or marked.
3. Keep the cast, the settings, the timeline, every scene, and the title
   as the first line.
4. No lists, no numbering, no headings, no questions; never suggest there
   is a puzzle or a task.
5. Return only the revised story."""


def polish_realization(world, realization, model=None, base_url=None,
                       api_key=None, temperature=0.6, timeout=500,
                       max_attempts=3):
    """Final creative pass: a chat model rewrites the contract-locked draft
    for flow. Acceptance = every claim clause present verbatim (spans
    relocated) AND the full surface gates (structure + extraction
    round-trip) passing on the rewritten text. Violations are fed back for
    the next attempt; after max_attempts the run FAILS — an unpolished or
    hint-breaking surface is never published."""
    from .narrative import Realization
    from .verify import verify_realization, verify_roundtrip
    claims = [realization.clauses[ref] for ref in realization.rendered]
    user = (realization.text + "\n\nCLAIMS:\n"
            + "\n".join(f"- {c}" for c in claims))
    failures = []
    for attempt in range(max_attempts):
        try:
            out = chat_completion(
                [{"role": "system", "content": _polish_prompt(world)},
                 {"role": "user", "content": user}],
                model=model, base_url=base_url, api_key=api_key,
                temperature=temperature, timeout=timeout)
        except RuntimeError as e:
            failures.append(f"attempt {attempt}: endpoint failure: {e}")
            continue
        spans, broken = {}, []
        for ref in realization.rendered:
            clause = realization.clauses[ref]
            pos = out.find(clause)
            if pos < 0:
                broken.append(clause)
                continue
            spans[ref] = (pos, pos + len(clause))
        if broken:
            failures.append(f"attempt {attempt}: dropped or rewrote "
                            f"{len(broken)} claim(s)")
            user = (realization.text + "\n\nCLAIMS:\n"
                    + "\n".join(f"- {c}" for c in claims)
                    + "\n\nYour previous revision DROPPED OR ALTERED these "
                    + "specific claims:\n"
                    + "\n".join(f"- {c}" for c in broken)
                    + "\n\nRevise again, keeping every single claim exact.")
            continue
        cand = Realization(mode=realization.mode, text=out, spans=spans,
                           clauses=dict(realization.clauses),
                           rendered=list(realization.rendered))
        g = verify_realization(world, cand)
        g["roundtrip"] = verify_roundtrip(world, cand)
        if g["pass"] and g["roundtrip"]["pass"]:
            g["polished"] = True
            cand.gates = g
            return cand
        failures.append(f"attempt {attempt}: gates failed: "
                        f"{g['issues'] or g['roundtrip']}")
    raise RuntimeError(
        f"polish could not preserve the hints in {max_attempts} attempts — "
        f"refusing to publish:\n  " + "\n  ".join(failures))


GENRE_PROMPT = """You are inventing the SETTING for a quiet mystery story —
not harbor, manor, hotel, theater, or observatory (those exist); something
fresh and specific. Fill the JSON madlib below. Rules:
- "nouns": 14+ names of documents/records/registers unique to this setting,
  all DISTINCT, none containing the words: agreed, matched, stamped, signed
  out, pointed to, disputed, coincided, whenever.
- "distractor_bodies"/"hypotheses"/"filler"/"titles"/"places": setting
  flavor; never mention any noun from "nouns" and never use the words
  listed above.
- "lore": 8 pairs [one-sentence piece of setting history, short name for it];
  short names must not contain any noun.
Return ONLY the JSON object:
{"name": "...", "vibe": "3-6 word setting description",
 "setting": "the <place>", "locale": "<where people talk>",
 "demonym": "<adjective for locals>", "chrono": "<what fixes a schedule>",
 "nouns": [...], "places": [...], "frames": [...], "filler": [...],
 "titles": [...], "distractor_bodies": [...], "hypotheses": [...],
 "lore": [[sentence, short], ...], "things": [...]}"""


def generate_genre_pack(model=None, base_url=None, api_key=None,
                        seed=None, max_attempts=3, timeout=400):
    """LLM-driven genre: a chat model invents the whole setting pack,
    madlib-style. The output must pass every construction-time GenrePack
    check (unique nouns, no formal leakage, frame slots) — violations are
    fed back and retried; after max_attempts the run fails rather than
    publish a broken pack. The returned pack is NOT seed-reproducible:
    callers should persist it alongside the instance."""
    from .genres import GenrePack
    user = "Invent the setting now. Aim for vivid, coherent, specific."
    if seed is not None:
        user += f" (variation seed {seed})"
    failures = []
    for attempt in range(max_attempts):
        try:
            raw = chat_completion(
                [{"role": "system", "content": GENRE_PROMPT},
                 {"role": "user", "content": user}],
                model=model, base_url=base_url, api_key=api_key,
                temperature=1.0, timeout=timeout)
        except RuntimeError as e:
            failures.append(f"attempt {attempt}: endpoint failure: {e}")
            continue
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            d = json.loads(m.group(0)) if m else {}
            nouns = [str(n) for n in d["nouns"]]
            claim_words = ("agreed", "matched", "stamped", "signed out",
                           "pointed to", "disputed", "coincided", "whenever",
                           "no two")

            def _clean(label, items, min_keep=1):
                # models routinely mention setting nouns in scenery, which
                # the pack forbids (scenery must carry no formal content).
                # Drop the offending lines instead of rejecting the pack —
                # deterministic repair beats reject-and-retry roulette.
                items = [str(x) for x in items]
                kept = [x for x in items
                        if not any(n in x for n in nouns)
                        and not any(w in x.lower() for w in claim_words)]
                if len(kept) < min_keep:
                    raise ValueError(f"{label}: every line referenced a "
                                     f"noun or claim word")
                if len(kept) < len(items):
                    _note(f"genre pack: dropped {len(items) - len(kept)} "
                          f"{label} line(s) referencing nouns/claim words")
                return kept

            pack = GenrePack(
                name=str(d["name"])[:40], vibe=str(d["vibe"])[:80],
                setting=str(d["setting"])[:60], locale=str(d["locale"])[:40],
                demonym=str(d["demonym"])[:40], chrono=str(d["chrono"])[:60],
                nouns=nouns,
                places=_clean("places", d["places"]),
                frames=[str(f) for f in d["frames"]],
                filler=_clean("filler", d["filler"], min_keep=6),
                titles=_clean("titles", d["titles"]),
                distractor_bodies=_clean("distractor", d["distractor_bodies"]),
                hypotheses=_clean("hypothesis", d["hypotheses"]),
                lore=[(str(a), str(b)) for a, b in d["lore"]],
                things=_clean("thing", d.get("things", []), min_keep=0))
            if len(pack.nouns) < 12:
                raise ValueError(f"only {len(pack.nouns)} nouns (need >=12; "
                                 f"series prefixes extend larger tiers)")
            if len(pack.frames) < 3 or len(pack.filler) < 6:
                raise ValueError("too few frames/filler")
            return pack
        except (AssertionError, KeyError, ValueError, TypeError,
                json.JSONDecodeError) as e:
            failures.append(f"attempt {attempt}: invalid pack: {e}")
            user = (f"Your previous pack was rejected: {e}. Fix it and "
                    f"return the complete JSON again, honoring every rule.")
    raise RuntimeError(
        f"could not generate a valid genre pack in {max_attempts} attempts "
        f"— refusing to publish:\n  " + "\n  ".join(failures))

# ------------------------------------------------------------- autodiscovery

def _redact(key):
    return (key[:6] + "***") if key and len(key) > 8 else ("***" if key else None)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_jsonc(path):
    """opencode configs are JSONC (// comments, trailing commas tolerated)."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(text)
    except ValueError:
        return None


def _localhost(base):
    return bool(base) and ("://localhost" in base or "://127.0.0.1" in base)


def _probe_opencode(home):
    """auth.json: {provider_id: {type: 'api'|'oauth', key}}; opencode.jsonc
    may add a provider block (npm:<id> -> options.baseURL) and a default
    model ('provider/model')."""
    auth = _read_json(os.path.join(home, ".local/share/opencode/auth.json"))
    if not isinstance(auth, dict):
        return []
    cfg = _read_jsonc(os.path.join(home, ".config/opencode/opencode.jsonc")) or {}
    blocks = cfg.get("provider", {}) if isinstance(cfg, dict) else {}
    default_model = cfg.get("model") if isinstance(cfg, dict) else None
    out = []
    for pid, entry in auth.items():
        if not isinstance(entry, dict) or entry.get("type") != "api":
            continue
        key = entry.get("key")
        if not key:
            continue
        block = blocks.get(pid) or blocks.get(f"npm:{pid}") or {}
        base = (block.get("options") or {}).get("baseURL") if isinstance(block, dict) else None
        base = base or PROVIDER_BASE_URLS.get(pid)
        if not base:
            continue
        model = None
        if isinstance(default_model, str) and default_model.startswith(pid + "/"):
            model = default_model.split("/", 1)[1]
        out.append(("opencode", base, key, model))
    return out


def _probe_codex(home):
    """config.toml [model_providers.<id>] base_url + auth.json OPENAI_API_KEY.
    ChatGPT-OAuth auth (auth_mode != 'apikey') yields no usable key."""
    cfg_path = os.path.join(home, ".codex/config.toml")
    if not os.path.exists(cfg_path):
        return []
    try:
        import tomllib
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
    except Exception:
        return []
    auth = _read_json(os.path.join(home, ".codex/auth.json")) or {}
    key = auth.get("OPENAI_API_KEY")
    if not key or not isinstance(key, str):
        return []
    providers = cfg.get("model_providers", {})
    pid = cfg.get("model_provider", "openai")
    entry = providers.get(pid, {})
    base = entry.get("base_url") or PROVIDER_BASE_URLS.get(pid)
    if not base:
        return []
    return [("codex", base, key, cfg.get("model"))]


def _probe_goose(home):
    """Top-level YAML scalars only (GOOSE_PROVIDER/GOOSE_MODEL, OPENAI_HOST,
    OPENAI_BASE_PATH, and an optional explicit key). No stdlib YAML: a flat
    line scan is enough for goose's config shape."""
    path = os.path.join(home, ".config/goose/config.yaml")
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return []
    scal = {}
    for ln in lines:
        m = re.match(r"^([A-Za-z_][\w]*)\s*:\s*(\S.*)$", ln)
        if m:
            scal[m.group(1)] = m.group(2).strip().strip("'\"")
    pid = scal.get("GOOSE_PROVIDER", "openai")
    base = scal.get("OPENAI_BASE_URL")
    if not base and scal.get("OPENAI_HOST"):
        base = scal["OPENAI_HOST"].rstrip("/") + "/v1"
    base = base or PROVIDER_BASE_URLS.get(pid)
    key = scal.get("OPENAI_API_KEY") or scal.get("LLM_API_KEY")
    if not base or pid in NON_COMPATIBLE:
        return []
    if not key and not _localhost(base):
        return []  # goose keeps keys in its own keychain; nothing to use here
    return [("goose", base, key, scal.get("GOOSE_MODEL"))]


def _probe_continue(home):
    """config.json models[]: {model, provider, apiKey, apiBase?}."""
    cfg = _read_json(os.path.join(home, ".continue/config.json"))
    if not isinstance(cfg, dict):
        return []
    out = []
    for m in cfg.get("models", []):
        if not isinstance(m, dict):
            continue
        pid = str(m.get("provider", "")).lower()
        key = m.get("apiKey")
        if not key or pid in NON_COMPATIBLE:
            continue
        base = m.get("apiBase") or PROVIDER_BASE_URLS.get(pid)
        if base:
            out.append(("continue", base, key, m.get("model")))
    return out


_PROBES = (
    ("opencode", _probe_opencode),
    ("codex", _probe_codex),
    ("goose", _probe_goose),
    ("continue", _probe_continue),
)


def discover_llm_config(home=None):
    """Scan local agent configs for an OpenAI-compatible endpoint. Returns
    (candidates, resolved): candidates is every usable (source, base_url,
    api_key, model) in probe order; resolved is the first candidate whose
    endpoint is usable as-is (localhost without key, else key required)."""
    home = home or os.path.expanduser("~")
    candidates = []
    for name, probe in _PROBES:
        try:
            candidates.extend(probe(home))
        except Exception:
            continue  # a malformed foreign config must never break the run
    resolved = next(((s, b, k, m) for s, b, k, m in candidates
                     if k or _localhost(b)), None)
    return candidates, resolved


def resolve_llm_config(model=None, base_url=None, api_key=None):
    """Full precedence: explicit args > environment > autodiscovery >
    defaults. Returns a dict with base_url/api_key/model plus provenance."""
    env_base = os.environ.get("OPENAI_BASE_URL")
    env_key = os.environ.get("OPENAI_API_KEY")
    env_model = (os.environ.get("ENIGMAFORGE_MODEL")
                 or os.environ.get("OPENAI_MODEL"))
    if env_key or env_base:
        cand = ("environment", env_base or DEFAULT_BASE_URL, env_key, env_model)
    else:
        _cands, cand = discover_llm_config()
        cand = cand or ("default", DEFAULT_BASE_URL, None, None)
    source, cbase, ckey, cmodel = cand
    return {"source": source,
            "base_url": base_url or cbase,
            "api_key": api_key or ckey,
            "model": model or cmodel or env_model or DEFAULT_MODEL}


# ------------------------------------------------------------- chat client

def _note(msg):
    print(f"[enigmaforge.llm] {msg}", file=sys.stderr)


def _list_models(base, key, timeout):
    req = urllib.request.Request(
        base.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return [m.get("id") for m in json.load(resp).get("data", []) if m.get("id")]


# process-level caches: LLM story runs make one call per scene; without
# these, every call re-probes a dead endpoint and re-fetches /models
_DEAD_ENDPOINTS = set()      # (base, key) that hard-failed (auth/bad request)
_MODEL_PICK_CACHE = {}       # (base, key) -> model id chosen from /models

_SPEED_TIERS = ("flash", "turbo", "air", "mini", "fast")
_NON_PRIMARY = re.compile(r"-(air|flash|turbo|mini|fast|pro|max|preview|thinking)")


def _pick_model(ids):
    """From an endpoint's /models list: newest version wins; within that
    version prefer the fast tier — this workload is many short calls where
    speed variants are near-parity and cheaper (e.g. glm-5.3-flash over
    glm-5.3 over glm-4.5). Explicit --model always overrides."""
    if not ids:
        return None

    def version(mid):
        nums = re.findall(r"\d+(?:\.\d+)?", mid)
        return tuple(float(x) for x in nums) or (0.0,)

    newest = max(version(m) for m in ids)
    top = [m for m in ids if version(m) == newest]
    for tier in _SPEED_TIERS:
        for m in top:
            if tier in m.lower():
                return m
    for m in top:
        if not _NON_PRIMARY.search(m.lower()):
            return m
    return top[0]


def _post_chat(base, key, model, payload, timeout):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, **payload}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    msg = out["choices"][0]["message"]
    content = msg.get("content")
    if isinstance(content, list):  # some providers return text blocks
        content = "".join(b.get("text", "") for b in content
                          if isinstance(b, dict))
    if content is None:  # thinking models may park the answer in reasoning
        content = msg.get("reasoning") or ""
    if not str(content).strip():
        finish = out["choices"][0].get("finish_reason")
        raise ValueError(f"model returned no content "
                         f"(finish_reason={finish}); raise max_tokens")
    return content, out.get("usage") or {}


def chat_completion(messages, model=None, base_url=None, api_key=None,
                    temperature=0.7, timeout=120, with_usage=False,
                    track_cost=False, max_tokens=None):
    """One call against an OpenAI-compatible /chat/completions endpoint.
    Unset fields fall back through env/autodiscovery/defaults. If the
    resolved endpoint fails hard (auth, bad request, unreachable, timeout),
    the remaining autodiscovered candidates are tried in order — env keys
    on dev machines are often dead placeholders. An explicit base_url
    disables fallback: you asked for that endpoint, you get its error."""
    import urllib.error
    cfg = resolve_llm_config(model=model, base_url=base_url, api_key=api_key)
    chain = [cfg]
    if base_url is None:
        seen = {(cfg["base_url"], cfg["api_key"])}
        for src, b, k, m in discover_llm_config()[0]:
            if ((k or _localhost(b)) and (b, k) not in seen
                    and (b, k) not in _DEAD_ENDPOINTS):
                chain.append({"source": src, "base_url": b, "api_key": k,
                              "model": model or m})
                seen.add((b, k))
        # the resolved primary is subject to the dead-cache too: without
        # this, every call re-probes a dead env key first
        chain = [c for c in chain
                 if (c["base_url"], c["api_key"]) not in _DEAD_ENDPOINTS] \
            or chain[-1:]
    failures = []
    for cand in chain:
        base = cand["base_url"]
        key = cand["api_key"]
        if not key and "api.openai.com" not in base:
            key = "none"  # local servers commonly accept any bearer
        use_model = cand["model"] or model or DEFAULT_MODEL
        payload = {"messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if track_cost:  # OpenRouter: opt in to usage.cost in the response
            payload["usage"] = {"include": True}
        if cand["model"] is None and model is None:
            ckey = (base, key)
            if ckey in _MODEL_PICK_CACHE:
                use_model = _MODEL_PICK_CACHE[ckey]
            else:
                try:  # endpoint knows its models better than any catalog
                    picked = _pick_model(_list_models(base, key, timeout))
                    if picked:
                        use_model = _MODEL_PICK_CACHE[ckey] = picked
                        _note(f"{cand['source']}: no model configured, "
                              f"using {picked} (newest from /models)")
                except (urllib.error.URLError, OSError, ValueError, KeyError):
                    pass
        for attempt in (1, 2):  # one retry on transient errors, then next candidate
            try:
                content, usage = _post_chat(base, key, use_model,
                                            payload, timeout)
                return (content, usage) if with_usage else content
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                failures.append(f"{cand['source']} {base} -> HTTP {e.code}: {body}")
                if e.code in (400, 401, 403, 404):
                    _DEAD_ENDPOINTS.add((base, key))  # auth/bad request: not transient
                if attempt == 1 and e.code in (429, 500, 502, 503, 504):
                    time.sleep(2)
                    continue
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                # read timeouts surface as bare TimeoutError, not URLError.
                # No in-call retry on timeouts: a stalled generation just
                # doubles the wait — record and fall through to the next
                # candidate / the caller's retry loop.
                failures.append(f"{cand['source']} {base} -> "
                                f"{getattr(e, 'reason', e)}")
            break
        if cand is not chain[-1]:
            _note(f"{cand['source']} endpoint failed, trying next candidate…")
    raise RuntimeError("all LLM endpoints failed:\n  " + "\n  ".join(failures))

def llm_scene_renderer(model=None, base_url=None, api_key=None,
                       temperature=0.7, timeout=240):
    """Build a two-phase scene renderer backed by a chat model.

    prepare() (called once per realization by compile_story) drafts a shared
    story plan in one call — premise, character sheet with pronouns, and a
    distinct setting for every scene — so scenes rendered independently
    (and in parallel) still tell one continuous story instead of converging
    on the same rainy room. Scene calls carry the plan plus the skeleton's
    timeline label. render() itself matches the template renderer's
    signature and stays bound by the verbatim-clause contract."""
    state = {"plan": None}

    def prepare(world, skeleton):
        names = [e.name for e in (world.entities if world else [])][:6] or ["the clerk"]
        stake = _stake_core(world)
        scenes = "; ".join(f"{s.sid} ({s.when})" for s in skeleton.scenes)
        user = (f"Cast: {', '.join(names)}. The story turns on: {stake}.\n"
                f"Scenes: {scenes}.\n\nReturn ONLY a JSON object with keys "
                f'"premise" (2 sentences), "characters" (object: name -> one '
                f'line with role AND pronouns), "settings" (object: scene id '
                f'-> one line: a DIFFERENT location and weather for that '
                f"scene — vary rooms, buildings, quays, times of day).")
        try:
            raw = chat_completion(
                [{"role": "system", "content": _plan_prompt(world)},
                 {"role": "user", "content": user}],
                model=model, base_url=base_url, api_key=api_key,
                temperature=temperature, timeout=timeout)
            state["plan"] = _parse_plan(raw, skeleton, names)
        except (RuntimeError, ValueError) as e:
            _note(f"plan call failed, using fallback plan: {e}")
            state["plan"] = None

    def render(scene, beat_clauses, rng, world=None):
        names = [e.name for e in (world.entities if world else [])][:6] or ["the clerk"]
        focus = rng.pick(names)
        claims = "\n".join(f"- {clause}" for _ref, clause in beat_clauses)
        plan = state["plan"] or {}
        ctx = [f"Scene {scene.sid} of the story, taking place {scene.when}."]
        if plan.get("premise"):
            ctx.append(f"PREMISE: {plan['premise']}")
        if plan.get("characters"):
            ctx.append("CHARACTERS: " + "; ".join(
                f"{k} — {v}" for k, v in list(plan["characters"].items())[:6]))
        if plan.get("settings", {}).get(scene.sid):
            ctx.append(f"SETTING: {plan['settings'][scene.sid]}")
        ctx.append(f"The scene settles on {focus}. Continue the same story; "
                   f"keep characters consistent with the sheet.")
        depth_open = getattr(scene, "depth_open", 0)
        depth_close = getattr(scene, "depth_close", 0)
        if depth_open:
            ctx.append(f"Open with {depth_open} pure-story paragraph(s) — "
                       f"life, weather, small events — BEFORE any claim.")
        if depth_close:
            ctx.append(f"Close with {depth_close} pure-story paragraph(s) "
                       f"after the last claim; let the clues sink.")
        if not beat_clauses:
            ctx.append("This scene carries NO claims: write pure story that "
                       "deepens the premise. Do not mention any record, "
                       "figure, or match.")
            user = "\n".join(ctx)
        else:
            user = "\n".join(ctx) + f"\n\nCLAIMS:\n{claims}"
        return chat_completion(
            [{"role": "system", "content": _scene_prompt(world)},
             {"role": "user", "content": user}],
            model=model, base_url=base_url, api_key=api_key,
            temperature=temperature, timeout=timeout)

    render.prepare = prepare
    render.__name__ = "llm_scene_renderer"
    return render


def _stake_core(world):
    """Diegetic stake for the planning prompt: in-world lore if present,
    else generic unease. Never the formal objective statement — that text
    must not leak toward the surface."""
    if world.bridges:
        return f"a season quietly gone wrong ({world.bridges[0].fact})"
    return "a season quietly gone wrong at the harbor"


def _parse_plan(raw, skeleton, names):
    m = re.search(r"\{.*\}", raw, re.S)
    plan = json.loads(m.group(0)) if m else {}
    if not isinstance(plan, dict):
        raise ValueError("plan is not a JSON object")
    characters = plan.get("characters") if isinstance(plan.get("characters"), dict) else {}
    settings = plan.get("settings") if isinstance(plan.get("settings"), dict) else {}
    return {"premise": str(plan.get("premise", ""))[:400],
            "characters": {str(k): str(v)[:120]
                           for k, v in list(characters.items())[:6]},
            "settings": {s.sid: str(settings.get(s.sid, ""))[:160]
                         for s in skeleton.scenes}}


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        prog="enigmaforge.llm",
        description="Show LLM config autodiscovery (keys redacted).")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args()
    candidates, resolved = discover_llm_config()
    cfg = resolve_llm_config()
    if a.json:
        print(json.dumps({
            "candidates": [{"source": s, "base_url": b,
                            "api_key": _redact(k), "model": m}
                           for s, b, k, m in candidates],
            "resolved": {"source": cfg["source"], "base_url": cfg["base_url"],
                         "api_key": _redact(cfg["api_key"]),
                         "model": cfg["model"]}}, indent=2))
        return
    if not candidates:
        print("no local agent configs found — falling back to env/defaults")
    for s, b, k, m in candidates:
        print(f"[{s:9s}] {b}  key={_redact(k)}  model={m or '-'}")
    print(f"\nresolved : {cfg['source']} -> {cfg['base_url']}  "
          f"key={_redact(cfg['api_key'])}  model={cfg['model']}")


if __name__ == "__main__":
    _main()

"""Test battery: LLM scene renderer against a local OpenAI-compatible mock.
The mock parses CLAIMS out of the prompt and embeds them verbatim — exactly
what a well-behaved model must do — so these tests exercise the real HTTP
client, the verbatim-contract span search, and the rejection loop end to end
with zero external dependencies."""
import sys, os, json, threading, tempfile, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from http.server import BaseHTTPRequestHandler, HTTPServer

from enigmaforge.llm import llm_scene_renderer
from enigmaforge.story import build_skeleton, compile_story, RenderContractError

CFG = dict(n_variables=8, n_constraints=10, dependency_depth=3, domain_size=4,
           n_people=4, n_bridges=2, n_distractors=2, n_objective_stages=2,
           narrative_tokens=600)

def _world(seed):
    from enigmaforge.generator import generate_world
    from enigmaforge.populate import (populate_evidence, populate_bridges,
                                      populate_objectives)
    w = generate_world(dict(CFG), seed)
    populate_evidence(w, seed)
    populate_bridges(w, seed)
    populate_objectives(w, seed)
    return w


def _fake_server(drop_first_claim=False, genre_fail_first=False):
    """OpenAI-shaped server on an ephemeral port. Embeds prompt CLAIMS into
    prose verbatim (or drops the first one to simulate a bad model);
    genre-mode invents a valid (or once-invalid) pack; polish-mode rewrites
    around intact claims."""
    state = {"calls": 0, "genre_calls": 0}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"] += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = body["messages"][0]["content"]
            user = body["messages"][-1]["content"]
            state.setdefault("prompts", []).append(user)
            if "inventing the SETTING" in system:
                state["genre_calls"] += 1
                content = _fake_genre_json(
                    invalid=genre_fail_first and state["genre_calls"] == 1)
            elif "planning a short quiet" in system and "mystery told" in system:
                import re as _re
                sids = _re.findall(r"(S\d+) \(", user)
                plan = {"premise": "the winter the manifests stopped balancing",
                        "characters": {"Vela": "the harbormaster (she)"},
                        "settings": {s: "the chandlery back room, snowing" for s in sids}}
                content = json.dumps(plan)
            elif "revising the draft" in system:
                draft = user.split("\n\nCLAIMS:\n")[0]
                claims = [ln[2:] for ln in user.splitlines()
                          if ln.startswith("- ")]
                if drop_first_claim and len(claims) > 1:
                    # bad polisher: drops the first clause from the text
                    draft = draft.replace(claims[0], "", 1)
                content = "A POLISHED OPENING.\n\n" + draft
            else:
                claims = [ln[2:] for ln in user.splitlines() if ln.startswith("- ")]
                if drop_first_claim and len(claims) > 1:
                    claims = claims[1:]
                content = ("It was still early, and the yard was quiet. " +
                           " ".join(f"The paper lay where it had fallen: {c}."
                                    for c in claims) +
                           " The kettle went unpoured.")
            out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1", state


def _fake_genre_json(invalid=False):
    """A valid madlib pack (32 nouns, all rules honored), or an invalid one
    (duplicate nouns) to exercise retry-then-fail behavior."""
    words = ["kelp", "salt", "spar", "tackle", "beacon", "sluice", "net",
             "oar", "skiff", "tidepool", "brine", "gull", "knot", "wharf",
             "heron", "shoal", "oyster", "lamp", "rope", "hook", "dory",
             "pile", "hull", "sail", "mast", "keel", "winch", "buoy",
             "ferry", "barge", "eel", "cod"]
    nouns = [f"{w}-warden sheet" for w in words]
    if invalid:
        nouns = nouns[:3] + nouns[:3]  # duplicates -> construction assert
    return json.dumps({
        "name": "tidewatch", "vibe": "estuary warden station",
        "setting": "the tidewatch", "locale": "sluice gate",
        "demonym": "estuary", "chrono": "warden roster",
        "nouns": nouns,
        "places": ["the sluice shed", "the sparrow loft", "the eel smokehouse"],
        "frames": [
            "In the back room of the {place}, {name} sat out the bad hour with the stove drawn close.",
            "The {place} smelled of brine and lamp oil. {name} had an hour before the gate turned.",
            "It was quiet in the {place}. The waders had stopped for the day."],
        "filler": ["The reeds ticked in the wind.",
                   "A heron lifted off the mud and went.",
                   "The stove ticked and settled.",
                   "Somewhere out, a buoy bell worked the chop.",
                   "The kettle murmured and went unanswered.",
                   "Frost ferned the watch-glass from the corner."],
        "titles": ["The Long Ebb", "What the Sluice Kept", "Brine and Paper",
                   "The Warden's Winter", "High and Dry", "The Quiet Reach"],
        "distractor_bodies": [
            "the matter of the unpaid warden's fee surfaced again",
            "a discrepancy in the eel count was noted",
            "a stranger at the sluice gate had been discussed",
            "the smokehouse fire that winter was never explained",
            "the warden's dog barked at nobody that night"],
        "hypotheses": ["the oldest letter is the forgery",
                       "the warden's deputy altered the tally",
                       "the second partner acted alone",
                       "the smokehouse book was redrawn after the fire"],
        "lore": [["the winter the sluice froze for six weeks", "the frozen sluice"],
                 ["the year the eel boats went elsewhere", "the missing eels"],
                 ["the summer the beacon oil was cut", "the thin beacon"],
                 ["the storm that took the middle spar", "the lost spar"]],
        "things": ["the stove", "the kettle", "the lamp", "the shutters",
                   "the mud boots", "the coat hooks", "the biscuit tin",
                   "the boot rack", "the wood box", "the lamp oil can",
                   "the watch kettle", "the bunk curtain"],
    })


def test_llm_renderer_passes_gates():
    w = _world(5)
    sk = build_skeleton(w, 5)
    srv, url, state = _fake_server()
    try:
        from enigmaforge.story import compile_story_verified
        r = compile_story_verified(w, sk, 900,
                                   renderer=llm_scene_renderer(base_url=url),
                                   max_attempts=2)
        assert r.gates["pass"] and r.gates["roundtrip"]["pass"]
        assert state["calls"] > 0
        # the prose is the model's, but every clause sits verbatim at its span
        for u in w.evidence:
            if u.is_distractor:
                continue
            s, e = r.spans[u.euid]
            assert r.text[s:e] == r.clauses[u.euid]
    finally:
        srv.shutdown()


def test_llm_renderer_dropping_claim_rejected():
    w = _world(5)
    sk = build_skeleton(w, 5)
    srv, url, _ = _fake_server(drop_first_claim=True)
    try:
        try:
            compile_story(w, sk, 900,
                          renderer=llm_scene_renderer(base_url=url))
            assert False, "dropped claim must fail the span contract"
        except RenderContractError:
            pass
    finally:
        srv.shutdown()


def test_cli_llm_renderer_end_to_end():
    from enigmaforge.pipeline import main
    srv, url, state = _fake_server()
    try:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "pkg")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = main(["--size", "small", "--seed", "9", "--mode", "story",
                          "--renderer", "llm", "--base-url", url, "--out", out])
            assert r["summary"]["mode"] == "story"
            assert os.path.exists(os.path.join(out, "story.md"))
            v = json.load(open(os.path.join(out, "verification.json")))
            for k, g in v["realization"].items():
                assert g["pass"] and g["roundtrip"]["pass"], (k, g)
            assert state["calls"] > 0
            # renderer provenance recorded in the hidden config
            hidden = json.load(open(os.path.join(out, "hidden_formal.json")))
            assert hidden["config"]["renderer"] == "llm"
    finally:
        srv.shutdown()


# ------------------------------------------------ config autodiscovery

import subprocess

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

def _synthetic_home(tmp, with_opencode=True, with_continue=True):
    """A $HOME holding the agent configs found on real machines."""
    if with_opencode:
        _write(os.path.join(tmp, ".local/share/opencode/auth.json"),
               json.dumps({"zai-coding-plan": {"type": "api",
                                               "key": "e19ca4e566synthetic"}}))
        _write(os.path.join(tmp, ".config/opencode/opencode.jsonc"),
               '{\n  // providers\n  "model": "zai-coding-plan/glm-4.7",\n}\n')
    if with_continue:
        _write(os.path.join(tmp, ".continue/config.json"),
               json.dumps({"models": [
                   {"model": "claude-3-5", "provider": "anthropic",
                    "apiKey": "sk-ant-synthetic"},
                   {"model": "codestral-latest", "provider": "mistral",
                    "apiKey": "qhq0synthetic"}]}))
    return tmp

def test_discovery_parses_agent_configs(tmp_path):
    from enigmaforge.llm import discover_llm_config
    cands, resolved = discover_llm_config(home=str(tmp_path))
    assert cands == [] and resolved is None, "empty HOME must find nothing"
    _synthetic_home(str(tmp_path))
    cands, resolved = discover_llm_config(home=str(tmp_path))
    sources = [c[0] for c in cands]
    assert "opencode" in sources and "continue" in sources
    base = next(c[1] for c in cands if c[0] == "opencode")
    assert base == "https://api.z.ai/api/coding/paas/v4"
    assert resolved[0] == "opencode" and resolved[3] == "glm-4.7"

def test_discovery_env_wins_over_agent_configs(tmp_path, monkeypatch):
    from enigmaforge.llm import resolve_llm_config
    _synthetic_home(str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-envsynthetic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    monkeypatch.delenv("ENIGMAFORGE_MODEL", raising=False)
    cfg = resolve_llm_config()
    assert cfg["source"] == "environment"
    assert cfg["base_url"] == "https://env.example/v1"
    assert cfg["api_key"] == "sk-envsynthetic"

def test_discovery_skips_malformed_configs(tmp_path):
    from enigmaforge.llm import discover_llm_config
    _write(os.path.join(tmp_path, ".local/share/opencode/auth.json"), "{not json")
    _write(os.path.join(tmp_path, ".config/opencode/opencode.jsonc"), "]]totally broken[[")
    cands, resolved = discover_llm_config(home=str(tmp_path))
    assert resolved is None and cands == []

def test_chat_routes_through_discovered_base_url(tmp_path, monkeypatch):
    """No explicit base_url anywhere: the renderer must find the endpoint in
    the synthetic opencode config (its provider block points at the mock)."""
    srv, url, state = _fake_server()
    try:
        home = str(tmp_path)
        _write(os.path.join(home, ".local/share/opencode/auth.json"),
               json.dumps({"mock": {"type": "api", "key": "synthetic"}}))
        _write(os.path.join(home, ".config/opencode/opencode.jsonc"), json.dumps(
            {"provider": {"mock": {"options": {"baseURL": url}}}}))
        monkeypatch.setenv("HOME", home)
        for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        from enigmaforge.llm import chat_completion
        out = chat_completion([{"role": "user", "content":
                                "CLAIMS:\n- the lamp was trimmed"}])
        assert "the lamp was trimmed" in out
        assert state["calls"] == 1
    finally:
        srv.shutdown()

def test_debug_print_redacts_keys(tmp_path):
    from enigmaforge.llm import _redact
    assert _redact("e19ca4e566synthetic") == "e19ca4***"
    assert _redact("") is None
    home = _synthetic_home(str(tmp_path))
    env = {**os.environ, "HOME": home}
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        env.pop(var, None)
    out = subprocess.run([sys.executable, "-m", "enigmaforge.llm"],
                         capture_output=True, text=True, env=env, cwd=os.path.dirname(
                             os.path.dirname(os.path.abspath(__file__))))
    assert "e19ca4e566synthetic" not in out.stdout
    assert "e19ca4***" in out.stdout

def _server_401():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(401)
            out = b'{"error": {"message": "Incorrect API key provided"}}'
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"

def test_fallback_across_discovered_candidates(tmp_path, monkeypatch):
    """First discovered endpoint is dead (401): the call must fall through
    to the next discovered candidate and succeed."""
    dead, dead_url = _server_401()
    live, live_url, _state = _fake_server()
    try:
        home = str(tmp_path)
        _write(os.path.join(home, ".local/share/opencode/auth.json"), json.dumps({
            "dead": {"type": "api", "key": "synthetic-dead"},
            "live": {"type": "api", "key": "synthetic-live"}}))
        _write(os.path.join(home, ".config/opencode/opencode.jsonc"), json.dumps({
            "provider": {"dead": {"options": {"baseURL": dead_url}},
                         "live": {"options": {"baseURL": live_url}}}}))
        monkeypatch.setenv("HOME", home)
        for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        from enigmaforge.llm import chat_completion
        out = chat_completion([{"role": "user", "content":
                                "CLAIMS:\n- the lamp was trimmed"}])
        assert "the lamp was trimmed" in out
    finally:
        dead.shutdown()
        live.shutdown()

def test_explicit_base_url_reports_error_body(monkeypatch):
    """Explicit endpoint: no fallback, and the failure must carry the body."""
    dead, dead_url = _server_401()
    try:
        from enigmaforge.llm import chat_completion
        try:
            chat_completion([{"role": "user", "content": "x"}],
                            base_url=dead_url, api_key="synthetic")
            assert False, "must raise"
        except RuntimeError as e:
            assert "HTTP 401" in str(e)
            assert "Incorrect API key" in str(e)
    finally:
        dead.shutdown()


def test_plan_phase_threads_premise_into_scene_prompts():
    """prepare() runs once per compile; every scene prompt carries the plan."""
    w = _world(5)
    sk = build_skeleton(w, 5)
    srv, url, state = _fake_server()
    try:
        from enigmaforge.story import compile_story
        r = compile_story(w, sk, 900, renderer=llm_scene_renderer(base_url=url))
        plan_prompts = [p for p in state["prompts"] if "Cast:" in p]
        scene_prompts = [p for p in state["prompts"] if "CLAIMS:" in p]
        assert len(plan_prompts) == 1, "one planning call per compile"
        assert len(scene_prompts) == len(sk.scenes)
        for p in scene_prompts:
            assert "PREMISE: the winter the manifests stopped balancing" in p
            assert "SETTING: the chandlery back room, snowing" in p
            assert "taking place " in p  # skeleton timeline label present
        # gates still hold with the two-phase renderer
        from enigmaforge.verify import verify_realization, verify_roundtrip
        assert verify_realization(w, r)["pass"]
        assert verify_roundtrip(w, r)["pass"]
    finally:
        srv.shutdown()


def test_polish_pass_rewrites_and_ships():
    """Good polisher: text rewritten, clauses verbatim at relocated spans,
    gates green, marked polished."""
    from enigmaforge.pipeline import build
    from enigmaforge.llm import polish_realization
    srv, url, _state = _fake_server()
    try:
        w = build("small", seed=5,
                  config_overrides={"mode": "story", "burial": 0},
                  polisher=lambda world, r: polish_realization(
                      world, r, base_url=url))
        r = w.meta["realizations"][0]
        assert r.text.startswith("A POLISHED OPENING."), "polish not applied"
        entry = w.verification["realization"]["r1"]
        assert entry["polished"] is True
        assert entry["pass"] and entry["roundtrip"]["pass"]
        for u in w.evidence:
            if u.is_distractor:
                continue
            s, e = r.spans[u.euid]
            assert r.text[s:e] == r.clauses[u.euid]
    finally:
        srv.shutdown()


def test_polish_that_breaks_hints_fails_the_run():
    """No fallback: a polisher that drops a claim fails after 3 attempts —
    an unpolished or hint-breaking surface is never published."""
    from enigmaforge.pipeline import build
    from enigmaforge.llm import polish_realization
    srv, url, state = _fake_server(drop_first_claim=True)
    try:
        try:
            build("small", seed=5,
                  config_overrides={"mode": "story", "burial": 0},
                  polisher=lambda world, r: polish_realization(
                      world, r, base_url=url))
            assert False, "must raise"
        except RuntimeError as e:
            assert "refusing to publish" in str(e)
    finally:
        srv.shutdown()


def test_dynamic_genre_end_to_end():
    """--genre llm: the model's madlib pack drives the whole pipeline, all
    gates pass on the invented setting, and the pack is persisted."""
    from enigmaforge.pipeline import build, package
    from enigmaforge.llm import generate_genre_pack
    srv, url, _state = _fake_server()
    try:
        w = build("small", seed=3,
                  config_overrides={"mode": "story", "genre": "llm",
                                    "burial": 1},
                  genre_gen=lambda seed: generate_genre_pack(
                      seed=seed, base_url=url))
        for k, g in w.verification["realization"].items():
            assert g["pass"] and g["roundtrip"]["pass"], (k, g)
        # invented nouns are the variable surfaces in the story
        assert "kelp-warden sheet" in w.meta["realizations"][0].text
        with tempfile.TemporaryDirectory() as td:
            package(w, td)
            gp = json.load(open(os.path.join(td, "genre_pack.json")))
            assert gp["name"] == "tidewatch" and len(gp["nouns"]) >= 30
    finally:
        srv.shutdown()


def test_dynamic_genre_retries_then_fails():
    """Invalid pack -> one corrective retry succeeds; always-invalid -> the
    run refuses to publish."""
    from enigmaforge.llm import generate_genre_pack
    srv, url, state = _fake_server(genre_fail_first=True)
    try:
        pack = generate_genre_pack(seed=1, base_url=url)
        assert pack.name == "tidewatch"
        assert state["genre_calls"] == 2, "expected one corrective retry"
    finally:
        srv.shutdown()
    srv2, url2, state2 = _fake_server(genre_fail_first=True)
    try:
        # always-invalid: server returns duplicates only on first call;
        # force permanent failure by max_attempts=1 on an invalid first try
        try:
            generate_genre_pack(seed=1, base_url=url2, max_attempts=1)
        except RuntimeError as e:
            assert "could not generate a valid genre pack" in str(e)
        else:
            assert False, "must raise"
    finally:
        srv2.shutdown()

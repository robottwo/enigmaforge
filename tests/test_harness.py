"""Test battery: benchmark harness — grader units, provider loading, and the
full generate -> solve -> grade -> report pipeline against a local
OpenAI-shaped mock (same pattern as tests/test_llm.py)."""
import sys, os, re, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from enigmaforge.harness import (SOLVER_PROMPT, load_providers, load_config,
                                 DEFAULT_SCENARIO, build_specs,
                                 _scenario_specs, _ladder_summary,
                                 generate_cohort, grade_instance, main)

# Canned grading fixture: 3 ground-truth facts (int + name + int values).
INST = {"iid": "inst-000-maritime",
        "ground_truth": {"V0": 3, "V1": "Vela", "V2": 2},
        "surfaces": {"V0": "logbook-warden sheet",
                     "V1": "crew manifest",
                     "V2": "signal lamp"},
        "final_action": "act on the corrected record"}


def _rec(text, error=None, seconds=1.0):
    return {"provider": "p", "iid": INST["iid"], "text": text,
            "seconds": seconds, "error": error}


def _perfect_text():
    facts = [f"{INST['surfaces'][vid]} = {val}"
             for vid, val in INST["ground_truth"].items()]
    return json.dumps({"observations": ["one", "two", "three"],
                       "fixed_facts": facts,
                       "final_action": INST["final_action"]})


# ------------------------------------------------------------- grader units

def test_perfect_structured_response():
    g = grade_instance(INST, _rec(_perfect_text()))
    assert g["components"] == {"compliance": 1.0, "comprehension": 1.0, "decisions": 1.0}
    assert g["score"] == 1.0
    assert g["error"] is None


def test_garbage_text_scores_zero():
    g = grade_instance(INST, _rec("I have no idea what this document says."))
    assert g["components"]["comprehension"] == 0.0
    assert g["components"]["compliance"] == 0.0
    assert g["score"] is not None  # low, but mechanical


def test_partially_correct_facts_exact_fraction():
    text = json.dumps({
        "observations": ["the sheet says three"],
        "fixed_facts": [f"{INST['surfaces']['V0']} = 3"],  # only V0 recovered
        "final_action": "file the records away for now"})
    g = grade_instance(INST, _rec(text))
    assert g["components"]["comprehension"] == pytest.approx(1 / 3)
    assert g["components"]["compliance"] == 1.0
    # facts is low enough that the similarity tier must be 0.0
    assert g["components"]["decisions"] == 0.0
    assert g["score"] == pytest.approx(round(0.24 * (1 / 3) + 0.10, 6))


def test_fenced_json_parses():
    text = "```json\n" + _perfect_text() + "\n```"
    g = grade_instance(INST, _rec(text))
    assert g["components"]["compliance"] == 1.0
    assert g["components"]["comprehension"] == 1.0


def test_unparseable_json_falls_back_to_raw_scan():
    # no JSON at all, but one fact co-occurs in the raw prose
    raw = (f"The {INST['surfaces']['V0']} was marked 3, said everyone. "
           "Nothing else survives.")
    g = grade_instance(INST, _rec(raw))
    assert g["components"]["comprehension"] == pytest.approx(1 / 3)
    assert g["components"]["compliance"] == 0.0


def test_int_word_boundary_precisely():
    text = json.dumps({
        "observations": ["o"],
        "fixed_facts": [f"{INST['surfaces']['V0']} = 13",
                        f"{INST['surfaces']['V2']} = 2"],
        "final_action": "x"})
    g = grade_instance(INST, _rec(text))
    # V0=3 must NOT match "13"; V2=2 matches
    assert g["components"]["comprehension"] == pytest.approx(1 / 3)


def test_error_record_all_none():
    g = grade_instance(INST, _rec(None, error="HTTP 401: nope"))
    assert g["score"] is None
    assert g["components"] == {"compliance": None, "comprehension": None, "decisions": None}
    assert g["error"] == "HTTP 401: nope"


# ------------------------------------------------------------- providers

def test_load_providers_kwargs(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY_VAR", "env-secret")
    p = tmp_path / "providers.json"
    p.write_text(json.dumps([
        {"name": "a", "model": "m1", "base_url": "http://x/v1"},
        {"name": "b", "api_key_env": "MY_KEY_VAR"},
        {"name": "c", "api_key": "lit"}]))
    provs = load_providers(str(p))
    assert provs[0]["kwargs"] == {"model": "m1", "base_url": "http://x/v1"}
    assert provs[1]["kwargs"] == {"api_key": "env-secret"}
    assert provs[2]["kwargs"] == {"api_key": "lit"}


def test_load_providers_missing_name(tmp_path):
    p = tmp_path / "providers.json"
    # neither 'name' nor a 'model' to derive it from
    p.write_text(json.dumps([{"base_url": "http://x/v1"}]))
    with pytest.raises(ValueError, match="missing required 'name'"):
        load_providers(str(p))


def test_load_providers_duplicate_names(tmp_path):
    p = tmp_path / "providers.json"
    p.write_text(json.dumps([{"name": "x"}, {"name": "x"}]))
    with pytest.raises(ValueError, match="duplicate provider name"):
        load_providers(str(p))


def test_load_providers_api_key_conflict(tmp_path):
    p = tmp_path / "providers.json"
    p.write_text(json.dumps([{"name": "x", "api_key": "a", "api_key_env": "B"}]))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_providers(str(p))


# ------------------------------------------------------------- fake servers

def _solver_server(content):
    """OpenAI-shaped server returning a canned completion for every call.
    'content' may be a str (every call), a list (popped per call, last
    repeats), or a dict {user-message substring: content} — order-safe
    under parallel solving."""
    state = {"calls": 0}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"] += 1
            body = json.loads(
                self.rfile.read(int(self.headers["Content-Length"])))
            state.setdefault("payloads", []).append(body)
            user = body["messages"][-1]["content"]
            if isinstance(content, dict):
                c = next((v for k, v in content.items() if k in user), "")
            elif isinstance(content, list):
                c = content.pop(0) if content else content[-1]
            else:
                c = content
            out = json.dumps(
                {"choices": [{"message": {"content": c}}],
                 "usage": {"prompt_tokens": 120, "completion_tokens": 60,
                           "total_tokens": 180, "cost": 0.0002}}).encode()
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


def _failing_server(status):
    """Always replies with the given HTTP status — simulates a dead provider."""
    state = {"calls": 0}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"] += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()
        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1", state


def _providers_file(tmp_path, entries):
    p = tmp_path / "providers.json"
    p.write_text(json.dumps(entries))
    return str(p)


# ------------------------------------------------------------- pipeline

def test_full_pipeline(tmp_path):
    out = str(tmp_path / "run")
    instances = generate_cohort(out, build_specs(1, ["small"], "maritime",
                                                 1, 1, 4000))
    assert len(instances) == 1
    inst = instances[0]
    assert os.path.isfile(inst["story_path"])
    assert inst["ground_truth"] and inst["surfaces"] and inst["final_action"]
    # the solver prompt must never hint at a puzzle
    assert "puzzle" not in SOLVER_PROMPT.lower()

    facts = [f"{inst['surfaces'][vid]} = {val}"
             for vid, val in inst["ground_truth"].items()]
    content = json.dumps({"observations": ["a", "b", "c"],
                          "fixed_facts": facts,
                          "final_action": inst["final_action"]})
    srv, url, state = _solver_server(content)
    bad_srv, bad_url, bad_state = _failing_server(401)
    prov = _providers_file(tmp_path, [
        {"name": "fake-good", "model": "fake-model", "base_url": url},
        {"name": "fake-bad", "model": "fake-model", "base_url": bad_url}])

    rc = main(["--providers", prov, "--out", out, "--timeout", "30",
               "--instances", "1", "--seed-base", "4000", "--genre", "maritime"])
    assert rc == 0

    iid = inst["iid"]
    resp = os.path.join(out, "responses", f"fake-good--{iid}.txt")
    assert os.path.isfile(resp)
    rec = json.load(open(resp))
    assert rec["error"] is None and rec["seconds"] >= 0
    assert rec["tokens"]["total_tokens"] == 180 and rec["cost"] == 0.0002

    results = json.load(open(os.path.join(out, "results.json")))
    lb = results["leaderboard"]
    assert lb[0]["provider"] == "fake-good"
    assert lb[0]["composite"] > 0.8
    assert lb[0]["n_scored"] == 1
    bad_row = next(r for r in lb if r["provider"] == "fake-bad")
    assert bad_row["composite"] is None
    assert bad_row["n_errors"] == 1 and bad_row["n_scored"] == 0
    assert bad_row in lb and lb[-1]["provider"] == "fake-bad"

    detailed = json.load(open(os.path.join(out, "results-detailed.json")))
    assert detailed["raw_responses"][f"fake-good--{iid}"]["text"] == content

    report = open(os.path.join(out, "report.html")).read()
    assert "fake-good" in report and "Leaderboard" in report
    # standalone: no EXTERNAL assets (an inline <script> is fine; error-cell
    # titles may quote a URL)
    assert not re.search(
        r"""(<script[^>]*\bsrc|src=["']?http|href=["']?http|url\(["']?http)""",
        report)
    assert state["calls"] == 1 and bad_state["calls"] == 1


def test_resume_no_new_calls(tmp_path):
    out = str(tmp_path / "run")
    generate_cohort(out, build_specs(1, ["small"], "maritime", 1, 1, 4100))
    inst = json.load(open(os.path.join(out, "instances",
                                       os.listdir(os.path.join(out, "instances"))[0],
                                       "instance.json")))
    content = json.dumps({"observations": ["a"], "fixed_facts": [],
                          "final_action": "unknown"})
    srv, url, state = _solver_server(content)
    prov = _providers_file(tmp_path, [
        {"name": "fake", "model": "m", "base_url": url}])
    args = ["--providers", prov, "--out", out, "--timeout", "30",
            "--instances", "1", "--seed-base", "4100", "--genre", "maritime"]

    rc1 = main(args)
    results1 = open(os.path.join(out, "results.json"), "rb").read()
    resp_file = os.path.join(out, "responses",
                             f"fake--{inst['iid']}.txt")
    mtime1 = os.path.getmtime(resp_file)
    calls1 = state["calls"]
    assert rc1 == 0 and calls1 == 1

    rc2 = main(args)  # same out dir: instance + response reused
    results2 = open(os.path.join(out, "results.json"), "rb").read()
    assert rc2 == 0
    assert state["calls"] == calls1  # no new solver calls
    assert os.path.getmtime(resp_file) == mtime1
    # results identical apart from the generation timestamp
    r1 = json.loads(results1); r2 = json.loads(results2)
    for k in ("generated_at", "finished_at", "elapsed_seconds"):
        r1.pop(k); r2.pop(k)
    assert r1 == r2

    rc3 = main(args + ["--grade-only"])
    assert rc3 == 0
    assert state["calls"] == calls1  # grade-only never calls the endpoint
    assert os.path.getmtime(resp_file) == mtime1


def test_skip_generate_scans_existing(tmp_path):
    out = str(tmp_path / "run")
    specs = build_specs(2, ["small"], "maritime", 1, 1, 4200)
    instances = generate_cohort(out, specs)
    scanned = generate_cohort(out)  # specs=None: scan mode
    assert [i["iid"] for i in scanned] == [i["iid"] for i in instances]
    assert scanned == instances


# ------------------------------------------------------------- scenarios

def _write_config(tmp_path, providers, scenarios, corpus=None):
    d = {"providers": providers, "scenarios": scenarios}
    if corpus:
        d["corpus"] = corpus
    p = tmp_path / "benchmark.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_load_config_scenarios(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "providers": [{"name": "a"}],
        "scenarios": [
            {"name": "easy", "instances": 2, "sizes": ["small"],
             "genre": "maritime", "burial_min": 1, "burial_max": 1,
             "seed_base": 4000},
            {"name": "buried", "instances": 1, "sizes": ["medium"],
             "burial_min": 2, "burial_max": 3, "seed_base": 5000}]}))
    out = load_config(str(cfg))
    assert [p["name"] for p in out["providers"]] == ["a"]
    easy, buried = out["scenarios"]
    assert easy["name"] == "easy" and easy["burial_max"] == 1
    assert buried["genre"] == "auto" and buried["seed_base"] == 5000
    specs = [sp for sc in out["scenarios"] for sp in _scenario_specs(sc)]
    assert [sp["iid"] for sp in specs[:2]] == ["inst-easy-000-maritime",
                                               "inst-easy-001-maritime"]
    assert specs[2]["iid"].startswith("inst-buried-000-")
    assert specs[2]["size"] == "medium"
    assert all(sp["burial"] == 1 for sp in specs[:2])
    assert specs[2]["burial"] in (2, 3)


def test_load_config_bare_list_is_legacy(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"name": "a"}]))
    out = load_config(str(p))
    assert out["scenarios"] == [DEFAULT_SCENARIO]


@pytest.mark.parametrize("bad", [
    {"providers": [{"name": "a"}], "scenarios": [{"instances": 2}]},
    {"providers": [{"name": "a"}], "scenarios": [{"name": "x"}, {"name": "x"}]},
    {"providers": [{"name": "a"}], "scenarios": [{"name": "x", "sizes": ["huge"]}]},
    {"providers": [{"name": "a"}],
     "scenarios": [{"name": "x", "burial_min": 3, "burial_max": 1}]},
    {"providers": [{"name": "a"}], "scenarios": [{"name": "x", "instances": 0}]},
    {"scenarios": [{"name": "x"}]},
])
def test_load_config_rejects_bad(tmp_path, bad):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_config(str(p))


def test_scenarios_end_to_end(tmp_path):
    out = str(tmp_path / "run")
    scenario_cfg = [
        {"name": "easy", "instances": 1, "sizes": ["small"],
         "genre": "maritime", "burial_min": 1, "burial_max": 1,
         "seed_base": 4000},
        {"name": "buried", "instances": 1, "sizes": ["small"],
         "genre": "maritime", "burial_min": 2, "burial_max": 2,
         "seed_base": 4100}]
    content = json.dumps({"observations": ["a", "b"], "fixed_facts": [],
                          "final_action": "act on the corrected record"})
    srv, url, state = _solver_server(content)
    args = ["--providers", _write_config(tmp_path, [
        {"name": "fake", "model": "m", "base_url": url}], scenario_cfg),
        "--out", out, "--timeout", "30"]
    assert main(args) == 0
    assert state["calls"] == 2  # 1 provider x 2 scenarios
    iids = sorted(os.listdir(os.path.join(out, "instances")))
    assert iids[0].startswith("inst-buried-000-maritime")
    assert iids[1].startswith("inst-easy-000-maritime")
    buried = json.load(open(os.path.join(out, "instances", iids[0],
                                         "instance.json")))
    easy = json.load(open(os.path.join(out, "instances", iids[1],
                                       "instance.json")))
    assert buried["scenario"] == "buried" and buried["burial"] == 2
    assert easy["scenario"] == "easy" and easy["burial"] == 1

    # idempotent: same config rerun -> no new solver calls
    calls1 = state["calls"]
    assert main(args) == 0
    assert state["calls"] == calls1

    # config-driven scale-out: grow one scenario -> only the new story
    # is generated and solved; the rest is reused
    scenario_cfg[0]["instances"] = 2
    args = ["--providers", _write_config(tmp_path, [
        {"name": "fake", "model": "m", "base_url": url}], scenario_cfg),
        "--out", out, "--timeout", "30"]
    assert main(args) == 0
    assert state["calls"] == calls1 + 1

    # cohort flags conflict with config scenarios: fail loud, rc 1
    assert main(args + ["--instances", "3"]) == 1


def test_config_add_provider_only_calls_new(tmp_path):
    out = str(tmp_path / "run")
    scenario = [{"name": "core", "instances": 1, "sizes": ["small"],
                 "genre": "maritime", "seed_base": 4200}]
    content = json.dumps({"observations": ["a"], "fixed_facts": [],
                          "final_action": "x"})
    srv, url, state = _solver_server(content)
    base_prov = [{"name": "a", "model": "m", "base_url": url}]
    args = ["--providers", _write_config(tmp_path, base_prov, scenario),
            "--out", out, "--timeout", "30"]
    assert main(args) == 0
    calls1 = state["calls"]
    assert main(["--providers", _write_config(
        tmp_path, base_prov + [{"name": "b", "model": "m", "base_url": url}],
        scenario), "--out", out, "--timeout", "30"]) == 0
    assert state["calls"] == calls1 + 1  # only provider b was called
    results = json.load(open(os.path.join(out, "results.json")))
    assert {r["provider"] for r in results["leaderboard"]} == {"a", "b"}


def test_provider_defaults_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "provider_defaults": {
            "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                           "api_key_env": "OPENROUTER_API_KEY"}},
        "providers": [
            {"model": "moonshotai/kimi-k3", "defaults": "openrouter"},
            {"name": "custom", "model": "qwen/qwen3.6-27b",
             "base_url": "http://override/v1", "defaults": "openrouter"},
            {"name": "openai", "model": "gpt-4o-mini"}]}))
    provs = load_config(str(p))["providers"]
    assert [x["name"] for x in provs] == ["kimi-k3", "custom", "openai"]
    assert provs[0]["kwargs"] == {"model": "moonshotai/kimi-k3",
                                  "base_url": "https://openrouter.ai/api/v1",
                                  "api_key": "or-key"}
    assert provs[1]["kwargs"] == {"model": "qwen/qwen3.6-27b",
                                  "base_url": "http://override/v1",
                                  "api_key": "or-key"}
    assert provs[2]["kwargs"] == {"model": "gpt-4o-mini"}


def test_provider_defaults_rejections(tmp_path):
    def cfg(providers, defaults=None):
        d = {"providers": providers}
        if defaults is not None:
            d["provider_defaults"] = defaults
        p = tmp_path / f"c{abs(hash(json.dumps(d)))}.json"
        p.write_text(json.dumps(d))
        return str(p)

    with pytest.raises(ValueError, match="unknown provider_defaults"):
        load_config(cfg([{"name": "a"}],
                        {"g": {"base_ul": "typo"}}))
    with pytest.raises(ValueError, match="unknown defaults group"):
        load_config(cfg([{"model": "m", "defaults": "nope"}]))
    with pytest.raises(ValueError, match="missing required 'name'"):
        load_config(cfg([{"base_url": "http://x/v1"}]))
    with pytest.raises(ValueError, match="unknown key"):
        load_config(cfg([{"name": "a", "bases_url": "typo"}]))
    with pytest.raises(ValueError, match="duplicate provider name"):
        load_config(cfg([{"model": "x/kimi-k3"}, {"name": "kimi-k3"}]))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(cfg([{"name": "a", "api_key": "lit", "defaults": "g"}],
                        {"g": {"api_key_env": "X"}}))


# ------------------------------------------------------- difficulty ladder

def test_ladder_scenario_specs(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "providers": [{"name": "a"}],
        "scenarios": [{"name": "lad", "instances": 2, "genre": "maritime",
                       "seed_base": 4000,
                       "levels": [
                           {"size": "small", "burial": 0,
                            "overrides": {"n_variables": 4}},
                           {"size": "small"}]}]}))
    cfg = load_config(str(p))
    specs = [sp for sc in cfg["scenarios"] for sp in _scenario_specs(sc)]
    assert [sp["iid"] for sp in specs] == [
        "inst-lad-L00-000-maritime", "inst-lad-L00-001-maritime",
        "inst-lad-L01-000-maritime", "inst-lad-L01-001-maritime"]
    assert [sp["level"] for sp in specs] == [0, 0, 1, 1]
    assert len({sp["seed"] for sp in specs}) == 4  # unique seeds across levels
    assert specs[0]["burial"] == 0                 # explicit per level
    assert specs[2]["burial"] in (1, 2)            # drawn from burial min/max
    assert specs[0]["level_overrides"] == {"n_variables": 4}
    assert specs[2]["level_overrides"] == {}


def test_ladder_scenario_bad_configs(tmp_path):
    for scen in [
        {"name": "x", "levels": []},
        {"name": "x", "levels": [{"size": "huge"}]},
        {"name": "x", "levels": [{"overrides": {"n_variable": 4}}]},  # typo
        {"name": "x", "levels": [{"burial": 13}]},
        {"name": "x", "sizes": ["small"], "levels": [{"size": "small"}]},
    ]:
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"providers": [{"name": "a"}],
                                 "scenarios": [scen]}))
        with pytest.raises(ValueError):
            load_config(str(p))


def test_ladder_summary_tolerance_and_threshold():
    insts = [{"iid": f"i{k}", "scenario": "lad", "level": k, "size": "small",
              "burial": 1, "final_action": "x", "ground_truth": {},
              "surfaces": {}} for k in range(5)]

    def g(iid, facts):
        return {"provider": "p", "iid": iid,
                "components": {"comprehension": facts, "decisions": 0.0,
                               "compliance": 1.0},
                "score": round(0.5 * facts + 0.2, 6), "seconds": 1.0,
                "error": None}

    # L1 (comprehension 0.5 < 0.6) is the first miss; a strict walk stops
    # there even though later levels happen to be solved
    graded = [g("i0", 1.0), g("i1", 0.5), g("i2", 1.0),
              g("i3", 0.3), g("i4", 1.0)]
    s = _ladder_summary(graded, insts, [{"name": "p"}])
    bp = s["by_provider"]["p"]
    assert [r["solved"] for r in bp["levels"]] == [True, False, True, False, True]
    assert bp["max_solved_level"] == 0   # walk stops at the L1 miss
    assert bp["levels_cleared"] == 3     # gap-blind count
    assert s["solved_facts_threshold"] == 0.6


def test_ladder_end_to_end(tmp_path):
    out = str(tmp_path / "run")
    scen = {"name": "lad", "instances": 1, "genre": "maritime",
            "seed_base": 4000,
            "levels": [{"size": "small", "burial": 1},
                       {"size": "small", "burial": 1}]}
    cfg = load_config(_write_config(tmp_path, [
        {"name": "fake", "model": "m", "base_url": "http://x/v1"}], [scen]))
    specs = [sp for sc in cfg["scenarios"] for sp in _scenario_specs(sc)]
    instances = generate_cohort(out, specs)
    assert len(instances) == 2
    inst0 = instances[0]
    perfect = json.dumps({
        "observations": ["a", "b"],
        "fixed_facts": [f"{inst0['surfaces'][vid]} = {val}"
                        for vid, val in inst0["ground_truth"].items()],
        "final_action": inst0["final_action"]})
    garbage = "I cannot tell what this document asks of anyone."
    srv, url, state = _solver_server({
        instances[0]["story_text"][:80]: perfect,   # L0: solves
        instances[1]["story_text"][:80]: garbage})  # L1: fails
    args = ["--providers", _write_config(tmp_path, [
        {"name": "fake", "model": "m", "base_url": url}], [scen]),
        "--out", out, "--timeout", "30"]
    assert main(args) == 0
    assert state["calls"] == 2
    lad = json.load(open(os.path.join(out, "results.json")))["ladder"]
    bp = lad["by_provider"]["fake"]
    assert [r["solved"] for r in bp["levels"]] == [True, False]
    assert bp["max_solved_level"] == 0
    assert lad["levels"][0]["label"].startswith("n=8")
    report = open(os.path.join(out, "report.html")).read()
    assert "Difficulty ladder" in report and "solved up to" in report


# ------------------------------------------------------------- llm renderer

def test_renderer_config(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "renderer": "llm",
        "providers": [{"name": "a"}],
        "scenarios": [{"name": "x"},
                      {"name": "y", "renderer": "template"}]}))
    cfg = load_config(str(p))
    assert cfg["scenarios"][0]["renderer"] == "llm"      # inherited
    assert cfg["scenarios"][1]["renderer"] == "template"  # per-scenario win
    for body in (
        {"renderer": "gpt", "providers": [{"name": "a"}],
         "scenarios": [{"name": "x"}]},
        {"providers": [{"name": "a"}],
         "scenarios": [{"name": "x", "renderer": "magic"}]},
    ):
        q = tmp_path / "bad.json"
        q.write_text(json.dumps(body))
        with pytest.raises(ValueError, match="renderer"):
            load_config(str(q))


def test_llm_renderer_wiring(tmp_path, monkeypatch):
    import enigmaforge.pipeline as pl
    made = []

    def fake_factory(**kw):
        made.append(kw)
        return ("sentinel-renderer",)

    monkeypatch.setattr("enigmaforge.harness.llm_scene_renderer",
                        fake_factory)
    real_build = pl.build
    seen = {}

    def spy_build(size, seed, config_overrides=None, renderer=None, **kw):
        seen[seed] = renderer
        return real_build(size, seed, config_overrides=config_overrides)

    monkeypatch.setattr(pl, "build", spy_build)
    scen = [{"name": "a", "instances": 1, "genre": "maritime",
             "seed_base": 4000, "renderer": "llm"},
            {"name": "b", "instances": 1, "genre": "maritime",
             "seed_base": 4100}]
    cfg = load_config(_write_config(tmp_path, [{"name": "p"}], scen))
    specs = [sp for sc in cfg["scenarios"] for sp in _scenario_specs(sc)]
    instances = generate_cohort(str(tmp_path / "run"), specs)
    assert len(instances) == 2
    assert len(made) == 1  # one renderer for the whole cohort
    assert seen[4000] == ("sentinel-renderer",)
    assert seen[4100] is None  # template scenario renders without a model


def test_corpus_split(tmp_path):
    out = str(tmp_path / "out")
    corpus = str(tmp_path / "corpus")
    scen = [{"name": "core", "instances": 1, "sizes": ["small"],
             "genre": "maritime", "seed_base": 4300}]
    content = json.dumps({"observations": ["a"], "fixed_facts": [],
                          "final_action": "x"})
    srv, url, state = _solver_server(content)
    args = ["--providers", _write_config(tmp_path, [
        {"name": "fake", "model": "m", "base_url": url}], scen),
        "--corpus", corpus, "--out", out, "--timeout", "30"]
    assert main(args) == 0
    inst_dir = os.path.join(corpus, "instances")
    iid = os.listdir(inst_dir)[0]
    assert os.path.isfile(os.path.join(inst_dir, iid, "instance.json"))
    assert os.listdir(os.path.join(corpus, "responses"))
    assert os.path.isfile(os.path.join(out, "results.json"))
    assert not os.path.exists(os.path.join(out, "instances"))
    calls1 = state["calls"]
    assert main(args) == 0            # rerun: corpus fully reused
    assert state["calls"] == calls1

    # corpus from the config key, no flag: same reuse semantics
    corpus2 = str(tmp_path / "corpus2")
    assert main(["--providers", _write_config(tmp_path, [
        {"name": "fake2", "model": "m", "base_url": url}], scen, corpus2),
        "--out", out, "--timeout", "30"]) == 0
    assert os.path.isdir(os.path.join(corpus2, "instances"))
    assert state["calls"] == calls1 + 1  # only the new provider was called


def test_track_cost_opt_in_payload(tmp_path):
    scen = [{"name": "core", "instances": 1, "sizes": ["small"],
             "genre": "maritime", "seed_base": 4400}]
    content = json.dumps({"observations": ["a"], "fixed_facts": [],
                          "final_action": "x"})
    srv, url, state = _solver_server(content)
    prov = _write_config(tmp_path, [
        {"name": "fake", "model": "m", "base_url": url, "track_cost": True}],
        scen)
    assert main(["--providers", prov, "--out", str(tmp_path / "run"),
                 "--timeout", "30"]) == 0
    assert state["payloads"] and all(
        p.get("usage") == {"include": True} for p in state["payloads"])
    results = json.load(open(os.path.join(str(tmp_path / "run"),
                                          "results.json")))
    assert results["leaderboard"][0]["cost"] == 0.0002
    assert results["leaderboard"][0]["tokens"] == 180


def test_action_similarity_tiers():
    def grade(final_action):
        return grade_instance(INST, _rec(json.dumps(
            {"observations": ["o"], "fixed_facts": [],
             "final_action": final_action})))

    # canonical content words embedded in a longer sentence -> full credit
    assert grade("the crew should act on the corrected record immediately"
                 )["components"]["decisions"] == 1.0
    # right verb, wrong object -> partial credit
    assert grade("discard the ledger; act on annex"
                 )["components"]["decisions"] == 0.5
    # unrelated -> zero
    assert grade("review the tide tables")["components"]["decisions"] == 0.0
    # the similarity value itself is recorded
    assert grade("act on the corrected record")["similarity"] == 1.0


# ------------------------------------------------- generation quality

def test_populate_objectives_vary_with_world(tmp_path):
    from enigmaforge.pipeline import build
    actions = []
    for seed in (4000, 4017, 4034, 4051):
        w = build("small", seed, config_overrides={"mode": "story"})
        obj = next(o for o in w.objectives if o.true_objective)
        actions.append(obj.answer["final_action"])
        # the action references the origin variable's resolved value
        assert str(w.meta["ground_truth"]["V0"]) in obj.answer["final_action"]
    assert len(set(actions)) >= 2  # verbs vary across seeds
    assert "act on the corrected record" not in actions


def test_genre_llm_and_polish_validation(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "providers": [{"name": "a"}],
        "scenarios": [{"name": "x", "genre": "llm", "polish": True}]}))
    scen = load_config(str(p))["scenarios"][0]
    assert scen["genre"] == "llm" and scen["polish"] is True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "providers": [{"name": "a"}],
        "scenarios": [{"name": "x", "genre": "dreamscape"}]}))
    with pytest.raises(ValueError, match="genre"):
        load_config(str(bad))


def test_genre_llm_and_polish_wiring(tmp_path, monkeypatch):
    import enigmaforge.pipeline as pl
    import enigmaforge.harness as hh
    seen = {}
    monkeypatch.setattr(hh, "generate_genre_pack",
                        lambda seed, **kw: seen.setdefault("pack_seed", seed))
    monkeypatch.setattr(hh, "polish_realization",
                        lambda world, r, **kw: r)
    real_build = pl.build

    def spy(size, seed, config_overrides=None, renderer=None,
            polisher=None, genre_gen=None, **kw):
        seen["genre_gen"] = genre_gen
        seen["polisher"] = polisher
        cfg = dict(config_overrides or {})
        cfg["genre"] = "maritime"  # template path: keep the test fast
        return real_build(size, seed, config_overrides=cfg)

    monkeypatch.setattr(pl, "build", spy)
    scen = [{"name": "x", "instances": 1, "sizes": ["small"],
             "genre": "llm", "polish": True, "seed_base": 4500}]
    prov = _write_config(tmp_path, [{"name": "p"}], scen)
    assert main(["--providers", prov, "--out", str(tmp_path / "run"),
                 "--timeout", "30"]) == 0
    assert callable(seen["genre_gen"]) and callable(seen["polisher"])

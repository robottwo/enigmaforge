"""Test battery: benchmark harness — grader units, provider loading, and the
full generate -> solve -> grade -> report pipeline against a local
OpenAI-shaped mock (same pattern as tests/test_llm.py)."""
import sys, os, re, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from enigmaforge.harness import (SOLVER_PROMPT, load_providers, build_specs,
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
    assert g["components"] == {"compliance": 1.0, "facts": 1.0, "action": 1.0}
    assert g["score"] == 1.0
    assert g["error"] is None


def test_garbage_text_scores_zero():
    g = grade_instance(INST, _rec("I have no idea what this document says."))
    assert g["components"]["facts"] == 0.0
    assert g["components"]["compliance"] == 0.0
    assert g["score"] is not None  # low, but mechanical


def test_partially_correct_facts_exact_fraction():
    text = json.dumps({
        "observations": ["the sheet says three"],
        "fixed_facts": [f"{INST['surfaces']['V0']} = 3"],  # only V0 recovered
        "final_action": "file the records away for now"})
    g = grade_instance(INST, _rec(text))
    assert g["components"]["facts"] == pytest.approx(1 / 3)
    assert g["components"]["compliance"] == 1.0
    # facts is low enough that the similarity tier must be 0.0
    assert g["components"]["action"] == 0.0
    assert g["score"] == pytest.approx(round(0.5 * (1 / 3) + 0.2, 6))


def test_fenced_json_parses():
    text = "```json\n" + _perfect_text() + "\n```"
    g = grade_instance(INST, _rec(text))
    assert g["components"]["compliance"] == 1.0
    assert g["components"]["facts"] == 1.0


def test_unparseable_json_falls_back_to_raw_scan():
    # no JSON at all, but one fact co-occurs in the raw prose
    raw = (f"The {INST['surfaces']['V0']} was marked 3, said everyone. "
           "Nothing else survives.")
    g = grade_instance(INST, _rec(raw))
    assert g["components"]["facts"] == pytest.approx(1 / 3)
    assert g["components"]["compliance"] == 0.0


def test_int_word_boundary_precisely():
    text = json.dumps({
        "observations": ["o"],
        "fixed_facts": [f"{INST['surfaces']['V0']} = 13",
                        f"{INST['surfaces']['V2']} = 2"],
        "final_action": "x"})
    g = grade_instance(INST, _rec(text))
    # V0=3 must NOT match "13"; V2=2 matches
    assert g["components"]["facts"] == pytest.approx(1 / 3)


def test_error_record_all_none():
    g = grade_instance(INST, _rec(None, error="HTTP 401: nope"))
    assert g["score"] is None
    assert g["components"] == {"compliance": None, "facts": None, "action": None}
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
    p.write_text(json.dumps([{"model": "m"}]))
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
    """OpenAI-shaped server returning a canned completion for every call."""
    state = {"calls": 0}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"] += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps(
                {"choices": [{"message": {"content": content}}]}).encode()
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
    # standalone: no external assets (error-cell titles may quote a URL)
    assert not re.search(r"""(src=|href=|<script|<link|url\()""", report)
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
    r1.pop("generated_at"); r2.pop("generated_at")
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

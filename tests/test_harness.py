"""v3 harness battery: strict grader, cache identities, retry semantics,
manifest integrity, legacy regrade protection, and full pipeline smoke.
Replaces the v2 battery that pinned word-overlap grading and raw-text scans.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from enigmaforge import grading
from enigmaforge.decisions import expected_action, validate_action
from enigmaforge.grading import GRADER_VERSION, SOLVER_PROMPT, grade_instance


def _instance():
    return {"iid": "inst-000-maritime",
            "ground_truth": {"V0": 3, "V1": "Vela"},
            "surfaces": {"V0": "logbook-warden sheet", "V1": "crew manifest"},
            "policy_text": "rule",
            "decision_policy": {"version": 1, "target_vid": "V0",
                                "gate_vid": "V1", "gate_value": "Vela",
                                "match_operation": "register",
                                "otherwise_operation": "hold"}}


def _record(text, status="answered", error=None):
    return {"provider": "p", "iid": _instance()["iid"], "text": text,
            "status": status, "error": error, "seconds": 1.0, "cost": None,
            "tokens": None}


def _facts(instance):
    return [{"subject": surface, "value": instance["ground_truth"][vid]}
            for vid, surface in instance["surfaces"].items()]


def _action(instance, ground_truth=None):
    return expected_action(instance["decision_policy"],
                           ground_truth or instance["ground_truth"],
                           instance["surfaces"])


def _answer(instance, facts, action):
    return json.dumps({"observations": ["a"], "fixed_facts": facts,
                       "final_action": action})


# ------------------------------------------------------------- solver contract

def test_prompt_demands_structured_assignments():
    assert '"subject"' in SOLVER_PROMPT and '"value"' in SOLVER_PROMPT
    assert "null to abstain" in SOLVER_PROMPT
    assert "negations" in SOLVER_PROMPT  # asserted conclusions only
    assert "puzzle" not in SOLVER_PROMPT.lower()


# ------------------------------------------------------------- grader units

def test_perfect_response_scores_full():
    inst = _instance()
    g = grade_instance(inst, _record(_answer(inst, _facts(inst),
                                             _action(inst))))
    assert g["score"] == 1.0
    assert g["metrics"] == {"fact_precision": 1.0, "fact_recall": 1.0,
                            "fact_f1": 1.0, "exact_world": True,
                            "decision_correct": True, "task_success": True}
    assert g["grader_version"] == GRADER_VERSION


def test_story_copy_earns_zero():
    inst = _instance()
    story = "The logbook-warden sheet read three. The crew manifest read Vela." * 30
    g = grade_instance(inst, _record(_answer(inst, [{"subject": "crew manifest",
                                                     "value": story}], story)))
    assert g["score"] == 0.0
    assert g["metrics"]["fact_precision"] == 0.0
    assert g["metrics"]["task_success"] is False


def test_negated_correct_answer_earns_zero():
    inst = _instance()
    facts = [{"subject": surface, "value": f"NOT {value}"}
             for surface, value in zip(inst["surfaces"].values(),
                                       inst["ground_truth"].values())]
    g = grade_instance(inst, _record(_answer(inst, facts,
                                             dict(_action(inst), value="NOT 3"))))
    assert g["score"] == 0.0


def test_conflicting_claims_penalize_precision():
    inst = _instance()
    facts = _facts(inst) + [
        {"subject": inst["surfaces"]["V0"], "value": 9}]
    g = grade_instance(inst, _record(_answer(inst, facts, _action(inst))))
    # each distinct claimed value counts against precision; V0 no longer exact
    assert g["metrics"]["fact_recall"] == pytest.approx(0.5)
    assert g["metrics"]["fact_precision"] == pytest.approx(1 / 3)
    assert g["metrics"]["exact_world"] is False


def test_string_and_integer_values_stay_distinct():
    inst = _instance()
    facts = _facts(inst)
    facts[0]["value"] = str(facts[0]["value"])
    g = grade_instance(inst, _record(_answer(inst, facts, _action(inst))))
    assert g["metrics"]["fact_f1"] == 0.5  # typed comparison, no int() rescue


def test_malformed_format_scores_zero_without_raw_scan():
    inst = _instance()
    raw = (f"The {inst['surfaces']['V0']} was marked 3, said everyone.")
    g = grade_instance(inst, _record(raw))
    assert g["score"] == 0.0
    assert g["components"]["compliance"] == 0.0
    assert g["metrics"]["fact_recall"] == 0.0  # no substring rescue


def test_wrong_decision_branch_fails_task_success():
    inst = _instance()
    other = "hold" if inst["decision_policy"]["match_operation"] == "register" \
        else "register"
    g = grade_instance(inst, _record(_answer(
        inst, _facts(inst),
        dict(_action(inst), operation=other))))
    assert g["metrics"]["fact_f1"] == 1.0
    assert g["metrics"]["task_success"] is False


def test_no_policy_means_no_decision_metric():
    inst = _instance()
    inst["decision_policy"] = None
    g = grade_instance(inst, _record(_answer(inst, _facts(inst),
                                             None)))
    assert g["metrics"]["decision_correct"] is None
    assert g["metrics"]["task_success"] is True  # exact world with no policy
    assert g["metrics"]["fact_f1"] == 1.0


def test_error_record_scores_none():
    g = grade_instance(_instance(), _record(None, status="content_filter",
                                            error="filtered"))
    assert g["score"] is None
    assert g["metrics"]["task_success"] is None


# ------------------------------------------------------------- decisions

def test_policy_validation_is_exact_and_typed():
    inst = _instance()
    assert validate_action(_action(inst), inst["decision_policy"],
                           inst["ground_truth"], inst["surfaces"])
    assert not validate_action(dict(_action(inst), value=4),
                               inst["decision_policy"], inst["ground_truth"],
                               inst["surfaces"])
    assert not validate_action(dict(_action(inst), subject="crew manifest"),
                               inst["decision_policy"], inst["ground_truth"],
                               inst["surfaces"])


def test_policy_text_never_leaks_ground_truth():
    inst = _instance()
    text = inst["policy_text"]
    assert str(inst["ground_truth"]["V0"]) not in text


# ------------------------------------------------------------- provenance/CLI

def test_provenance_hashes_sources():
    from enigmaforge.harness import provenance
    prov = provenance()
    assert prov["benchmark_version"] == "3"
    assert "harness.py" in prov["sources"]
    assert "grading.py" not in (prov | {})["sources"].get("sources", {})


def test_guard_rejects_pilot_paths(tmp_path, monkeypatch):
    import enigmaforge.harness as harness
    monkeypatch.setattr(harness, "_PROTECTED",
                        (tmp_path / "runs/eval-final",))
    with pytest.raises(ValueError, match="immutable pilot"):
        harness._atomic_json(tmp_path / "runs/eval-final/results.json", {})


def test_seal_detects_tampering():
    from enigmaforge.harness import _digest, _seal, _unseal
    sealed = _seal({"a": 1})
    assert _unseal(sealed, "x") == {"a": 1}
    sealed["a"] = 2
    with pytest.raises(ValueError, match="integrity mismatch"):
        _unseal(sealed, "x")


def test_provider_requires_explicit_endpoint():
    from enigmaforge.harness import _providers_from_list
    with pytest.raises(ValueError, match="explicit model"):
        _providers_from_list([{"name": "a"}], "cfg")
    providers = _providers_from_list(
        [{"name": "a", "model": "m", "base_url": "https://x/v1"}], "cfg")
    assert providers[0]["kwargs"]["base_url"] == "https://x/v1"


def test_public_request_never_contains_credentials():
    from enigmaforge.harness import _public_request
    out = _public_request({"model": "m", "api_key": "sekrit",
                           "base_url": "https://x/v1"})
    assert "api_key" not in out and out["model"] == "m"


# ------------------------------------------------------------- end-to-end smoke

def _solver_server(content_by_response):
    """OpenAI-shaped mock: content keyed by unique response payload substring."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    state = {"calls": 0, "payloads": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"] += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["payloads"].append(body)
            user = body["messages"][-1]["content"]
            content = next((v for k, v in content_by_response.items()
                            if k in user), "")
            out = json.dumps({"choices": [{"message": {"content": content}}],
                              "usage": {"prompt_tokens": 10,
                                        "completion_tokens": 5,
                                        "total_tokens": 15}}).encode()
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


def test_full_v3_pipeline_with_conditions_and_baselines(tmp_path):
    from enigmaforge.harness import main, generate_cohort, build_specs
    from enigmaforge.pipeline import build
    from enigmaforge.decisions import expected_action
    from enigmaforge import experiments
    out = str(tmp_path / "run")
    corpus = str(tmp_path / "corpus")
    specs = build_specs(1, ["small"], "maritime", 1, 1, 4001)
    instances = generate_cohort(corpus, specs)
    inst = instances[0]
    story = inst["story_text"]
    surfaces = inst["surfaces"]
    perfect = json.dumps({
        "observations": ["a"],
        "fixed_facts": [{"subject": surfaces[vid], "value": value}
                        for vid, value in inst["ground_truth"].items()],
        "final_action": expected_action(inst["decision_policy"],
                                        inst["ground_truth"], surfaces)})
    # mock returns the perfect story answer only for implicit variants; the
    # formal variant gets an empty schema answer
    formal_text = experiments.formal_input(inst)
    srv, url, state = _solver_server({story[:120]: perfect,
                                      formal_text[:120]: json.dumps(
                                          {"observations": [], "fixed_facts": [],
                                           "final_action": None})})
    providers = tmp_path / "p.json"
    providers.write_text(json.dumps([
        {"name": "mock", "model": "m", "base_url": url, "max_tokens": 64}]))
    rc = main(["--providers", str(providers), "--out", out, "--corpus", corpus,
               "--instances", "1", "--seed-base", "4001", "--genre", "maritime",
               "--baselines", "--conditions", "formal,implicit", "--timeout", "30"])
    assert rc == 0
    results = json.load(open(os.path.join(out, "results.json")))
    # grid: formal r0 + implicit r1/r2 = 3 matched variants
    assert results["n_instances"] == 3
    assert results["n_families"] == 1
    by_provider = {r["provider"]: r for r in results["leaderboard"]}
    assert by_provider["mock"]["coverage"]["expected"] == 3
    assert by_provider["baseline:constraint-solver"]["metrics"][
        "task_success"]["all_items"]["value"] == 1.0
    assert by_provider["baseline:story-copy"]["score"] == 0.0
    # formal + implicit-r1 answered; implicit-r2 (different realization text)
    # is an unkeyed empty response and is not counted as answered
    assert by_provider["mock"]["coverage"]["answered"] == 2
    assert by_provider["mock"]["coverage"]["expected"] == 3
    assert results["ranking"] == ["all_item_task_success_desc",
                                  "all_item_fact_f1_desc", "provider_asc"]
    manifest = results["corpus_manifest"]
    assert manifest["complete"] is True and manifest["exclusions"] == []
    report = open(os.path.join(out, "report.html")).read()
    assert "baseline:story-copy" in report and "<script" in report
    assert not __import__("re").search(r"""src=["']?http""", report)
    # cache identity binds inputs: changing the prompt invalidates, not serves
    response_file = next(iter(__import__("pathlib").Path(corpus)
                              .glob("responses/mock--*.json")))
    record = json.loads(response_file.read_text())
    assert record["identity"]["prompt_sha256"] == __import__("hashlib").sha256(
        SOLVER_PROMPT.encode()).hexdigest()[:16] or \
        record["identity"]["prompt_sha256"]


def test_legacy_regrade_never_mutates_pilot(tmp_path):
    from enigmaforge.harness import main
    pilot = tmp_path / "pilot"
    (pilot / "instances").mkdir(parents=True)
    inst_dir = pilot / "instances" / "inst-000"
    inst_dir.mkdir()
    (inst_dir / "instance.json").write_text(json.dumps(
        {"iid": "inst-000", "ground_truth": {"V0": 3}, "surfaces": {"V0": "s"},
         "story_text": "story", "final_action": "x"}))
    detailed = pilot / "results-detailed.json"
    detailed.write_text(json.dumps({
        "leaderboard": [{"provider": "p"}],
        "raw_responses": {"p--inst-000": {
            "text": json.dumps({"observations": [], "fixed_facts": [],
                                "final_action": None})}}}))
    before = detailed.read_bytes()
    out = tmp_path / "regrade"
    rc = main(["--legacy-regrade", str(detailed), "--legacy-corpus",
               str(pilot), "--out", str(out)])
    assert rc == 0
    assert detailed.read_bytes() == before  # read-only source
    results = json.load(open(out / "results.json"))
    row = results["leaderboard"][0]
    assert row["coverage"]["expected"] == 1
    assert results["generation_completeness"]["classification"] == "pilot"


def test_legacy_regrade_rejects_providers_config(tmp_path):
    from enigmaforge.harness import main
    rc = main(["--providers", "x.json", "--legacy-regrade", "d.json",
               "--legacy-corpus", "c", "--out", str(tmp_path / "o")])
    assert rc == 1

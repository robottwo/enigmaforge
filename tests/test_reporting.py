"""Reporting contracts: denominators, pairing, rankings, and untrusted HTML."""
from contextlib import redirect_stdout
from html.parser import HTMLParser
import io
import json

import pytest

from enigmaforge import reporting
from enigmaforge.reporting import aggregate, print_leaderboard, render_html


def instance(iid, *, family=None, condition="implicit", realization=1, policy=True,
             level=7):
    return {"iid": iid, "family_id": family or iid, "condition": condition,
            "realization": realization, "level": level, "scenario": "test",
            "genre": "mystery", "renderer": "template", "story_text": "A story.",
            "ground_truth": {"V0": 42}, "surfaces": {"V0": "signal lamp"},
            "decision_policy": {"version": 1} if policy else None,
            "difficulty": {"solutions": 1, "propagation_depth": 3}}


def grade(provider, iid, f1=1, task=1, status="answered", cost=None):
    available = status in ("answered", "invalid_response")
    return {"provider": provider, "iid": iid, "status": status,
            "error": None, "score": f1 if available else None,
            "components": {"compliance": int(status == "answered")},
            "metrics": {"fact_precision": f1 if available else None,
                        "fact_recall": f1 if available else None,
                        "fact_f1": f1 if available else None,
                        "exact_world": int(f1 == 1) if available else None,
                        "decision_correct": task if available else None,
                        "task_success": task if available else None},
            "decision": {"operation": "register", "subject": "signal lamp", "value": 42},
            "seconds": 1, "cost": cost, "tokens": {"total_tokens": 10}}


def metric(row, name="fact_f1", population="all_items"):
    return row["metrics"][name][population]


def test_cartesian_denominators_and_policy_only_decisions():
    instances = [instance("a"), instance("b"), instance("legacy", policy=False)]
    result = aggregate([grade("p", "a", cost=.1), grade("p", "legacy", task=None)],
                       instances, ["p", "absent"], bootstrap_resamples=0)
    row, absent = result["leaderboard"]
    assert row["provider"] == "p"
    assert row["coverage"]["expected"] == 3
    assert row["coverage"]["answered"] == row["coverage"]["valid"] == 2
    assert row["coverage"]["missing"] == 1
    assert metric(row)["value"] == pytest.approx(2 / 3)
    assert metric(row, population="conditional")["value"] == 1
    assert metric(row, population="conditional")["denominator"] == 2
    assert metric(row, "task_success")["denominator"] == 2
    assert metric(row, "task_success")["value"] == .5
    assert absent["coverage"]["missing"] == 3
    assert metric(absent)["value"] == 0
    assert metric(absent, population="conditional")["value"] is None
    assert len(result["records"]) == 6
    assert row["recorded_attempt_cost"] == .1
    assert row["cost_known_records"] == 1
    assert result["difficulty"]["families_measured"] == 3


def test_rank_order_task_then_fact_then_name_shared_by_html_and_cli(tmp_path):
    instances = [instance("i")]
    graded = [grade("z-task", "i", f1=.2), grade("a-facts", "i", task=0),
              grade("d-tie", "i", f1=.8), grade("b-tie", "i", f1=.8)]
    agg = aggregate(graded, instances, [g["provider"] for g in graded], bootstrap_resamples=0)
    names = [r["provider"] for r in agg["leaderboard"]]
    assert names == ["b-tie", "d-tie", "z-task", "a-facts"]
    stream = io.StringIO()
    with redirect_stdout(stream):
        print_leaderboard(agg)
    out = tmp_path / "report.html"
    render_html(agg, out, instances)
    for text in (stream.getvalue(), out.read_text()):
        assert [text.index(name) for name in names] == sorted(text.index(name) for name in names)


def test_errors_are_distinguished_from_unadjudicated_capability():
    statuses = ["content_filter", "transport_error", "invalid_response", "judge_error",
                "adjudication_required", "missing", "answered"]
    instances = [instance(str(n)) for n in range(len(statuses))]
    records = [grade("p", str(n), f1=0 if s == "invalid_response" else 1, status=s)
               for n, s in enumerate(statuses)]
    row = aggregate(records, instances, ["p"], bootstrap_resamples=0)["leaderboard"][0]
    assert row["status_counts"] == dict.fromkeys(statuses, 1)
    assert row["coverage"]["answered"] == 4
    assert row["coverage"]["valid"] == 1
    assert metric(row)["value"] is None
    assert metric(row)["denominator"] == 7
    assert metric(row)["unavailable"] == 2
    assert metric(row)["lower_bound"] == pytest.approx(1 / 7)
    assert metric(row, population="conditional")["value"] == .5
    assert row["warnings"]



def test_length_truncation_rate_warns():
    def empty(iid):
        # grading maps transport "empty" to invalid_response but keeps the
        # finish reason for diagnosis — replicate that shape here.
        row = grade("p", iid, f1=0, status="invalid_response")
        row["finish_reason"] = "length"
        return row
    # 30% length-capped empties -> loud config warning.
    instances = [instance(str(n)) for n in range(10)]
    records = [grade("p", str(n)) for n in range(7)] + [empty(str(n)) for n in range(7, 10)]
    row = aggregate(records, instances, ["p"], bootstrap_resamples=0)["leaderboard"][0]
    assert any("finish_reason=length" in w and "3/10" in w for w in row["warnings"])
    # Healthy runs get no such warning.
    records = [grade("p", str(n)) for n in range(10)]
    row = aggregate(records, instances, ["p"], bootstrap_resamples=0)["leaderboard"][0]
    assert not any("finish_reason=length" in w for w in row["warnings"])


def test_condition_pairs_average_realizations_then_resample_families():
    instances = []
    records = []
    # Worlds have opposite condition effects, with unequal realization counts.
    for family, realizations, effect in (("a", 1, 1), ("b", 3, 0)):
        for condition in ("explicit", "implicit"):
            for realization in range(realizations):
                iid = f"{family}-{condition}-{realization}"
                instances.append(instance(iid, family=family, condition=condition,
                                          realization=realization))
                score = effect if condition == "explicit" else 1 - effect
                records.append(grade("p", iid, f1=score, task=score))
    kwargs = {"bootstrap_resamples": 400, "bootstrap_seed": 23}
    agg = aggregate(records, instances, ["p"], **kwargs)
    repeated = aggregate(list(reversed(records)), list(reversed(instances)), ["p"], **kwargs)
    paired = agg["paired_differences"]["conditions"][0]["metrics"]["fact_f1"]
    assert paired["difference"] == 0  # Not the item-weighted -0.5.
    assert paired["complete_pairs"] == paired["families"] == 2
    assert paired["ci"]["low"] == -1
    assert paired["ci"]["high"] == 1
    assert agg["paired_differences"] == repeated["paired_differences"]
    assert agg["leaderboard"] == repeated["leaderboard"]
    assert agg["difficulty"]["families_measured"] == 2


def test_missing_pair_and_cluster_counts_are_not_realization_counts():
    instances = [instance(f"{f}-{c}", family=f, condition=c)
                 for f in ("a", "b") for c in ("explicit", "implicit")]
    records = [grade("p", i["iid"]) for i in instances if i["iid"] != "b-implicit"]
    records += [grade("q", i["iid"], f1=0, task=0) for i in instances]
    result = aggregate(records, instances, ["p", "q"], bootstrap_resamples=100)
    conditions = result["paired_differences"]["conditions"][0]["metrics"]["fact_f1"]
    assert conditions["expected_pairs"] == 2
    assert conditions["complete_pairs"] == conditions["missing_pairs"] == 1
    assert conditions["ci"]["low"] is None  # One family is not independent replication.
    models = result["paired_differences"]["models"][0]["metrics"]["fact_f1"]
    assert models["expected_pairs"] == 4
    assert models["complete_pairs"] == 3
    assert models["missing_pairs"] == 1
    assert models["families"] == 2
    assert models["ci"]["low"] == models["ci"]["high"] == 1


def test_duplicate_and_out_of_grid_records_rejected():
    i = instance("i")
    g = grade("p", "i")
    with pytest.raises(ValueError):
        aggregate([g, g], [i], ["p"], bootstrap_resamples=0)
    with pytest.raises(ValueError):
        aggregate([g], [i], ["other"], bootstrap_resamples=0)


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attrs = []
        self.text = []
        self.cells = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend(attrs)
        if tag in ("td", "th"):
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.cells.append("".join(self.cell))
            self.cell = None

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)


def test_html_escapes_all_dynamic_text_and_maps_hidden_facts(tmp_path):
    attack = '</script><img src=x onerror="alert(1)"><script>'
    i = instance('world" onclick="alert(2)')
    i["story_text"] = attack
    i["surfaces"]["V0"] = "lamp <west>"
    g = grade(attack, i["iid"])
    g["decision"]["subject"] = attack
    agg = aggregate([g], [i], [attack], bootstrap_resamples=0)
    agg["generation_completeness"] = {"failure": attack}
    out = tmp_path / "report.html"
    render_html(agg, out, [i])
    parser = ReportParser()
    parser.feed(out.read_text())
    assert parser.tags.count("script") == 1
    assert "img" not in parser.tags
    assert not any(name.startswith("on") for name, _ in parser.attrs)
    assert attack in "".join(parser.text)
    pos = parser.cells.index("V0")
    assert parser.cells[pos:pos + 3] == ["V0", "lamp <west>", "42"]
    assert agg["runs"][0]["level"] == 7
    assert any(item["value"] == 7 for item in agg["strata"][attack]["level"])
    assert json.loads(json.dumps(agg))["records"][0]["decision"]["subject"] == attack


def test_top_line_and_summary_chart_render():
    instances = [instance(str(n)) for n in range(4)]
    records = [grade("model-a", str(n)) for n in range(4)]
    records += [grade("baseline:story-copy", str(n), f1=0, task=0) for n in range(4)]
    agg = aggregate(records, instances, ["model-a", "baseline:story-copy"],
                    bootstrap_resamples=0)
    # Baselines never win the top line even when ranked nearby.
    top = reporting._top_line(agg["leaderboard"])
    assert top and top.startswith("model-a leads")
    assert "task success" in top and "warning" not in top.split("·")[1] if "·" in top else True
    chart = reporting._summary_chart(agg["leaderboard"])
    assert chart.startswith("<svg") and "</svg>" in chart
    # Only the model provider is plotted; baselines stay out of the chart.
    assert chart.count("<rect") == 2 + 2  # two bars + 2 legend swatches
    assert "baseline:story-copy" not in chart
    assert "model-a" in chart
    # A leaderboard with no valid answers yields the explicit no-outcome line.
    empty = [grade("x", str(n), f1=0, task=0, status="transport_error") for n in range(4)]
    agg2 = aggregate(empty, instances, ["x"], bootstrap_resamples=0)
    assert reporting._top_line(agg2["leaderboard"]) is None


def test_derived_diagnostics_measure_failure_modes():
    def cond_record(iid, condition, f1, decision=None, exact=False, length=False):
        inst = instance(iid, condition=condition)
        row = grade("p", iid, f1=f1, task=int(decision) if decision is not None else 1)
        row["metrics"]["exact_world"] = int(exact)
        row["metrics"]["decision_correct"] = decision
        if length:
            row["status"] = "invalid_response"
            row["finish_reason"] = "length"
        return inst, row
    pairs = []
    # 4 explicit items F1=1.0, 4 implicit items F1=0.5 (+4 decision items at
    # F1=1.0, also implicit) → implicit mean 0.75, discovery cost 0.25
    pairs += [cond_record(f"e{n}", "explicit", 1.0) for n in range(4)]
    pairs += [cond_record(f"i{n}", "implicit", 0.5) for n in range(4)]
    # 4 decision-graded items: 3 correct-with-imperfect-world (guesses), 1 exact
    pairs += [cond_record(f"d{n}", "implicit", 1.0, decision=True, exact=False) for n in range(3)]
    pairs += [cond_record("d3", "implicit", 1.0, decision=True, exact=True)]
    # 2 length-truncated non-answers out of 14 items
    pairs += [cond_record(f"x{n}", "implicit", None, length=True) for n in range(2)]
    row = aggregate([r for _, r in pairs], [i for i, _ in pairs], ["p"],
                    bootstrap_resamples=0)["leaderboard"][0]
    d = row["derived"]
    assert d["discovery_cost"]["difference"] == pytest.approx(0.25)
    assert d["discovery_cost"]["explicit_f1"] == pytest.approx(1.0)
    assert d["discovery_cost"]["implicit_f1"] == pytest.approx(0.75)
    assert d["discovery_cost"]["explicit_n"] == 4 and d["discovery_cost"]["implicit_n"] == 8
    assert d["reasoning_overflow"] == {"rate": 0.1429, "count": 2, "denominator": 14}
    assert d["educated_guess"] == {"rate": 0.75, "count": 3, "denominator": 4}
    # Length-truncated items are excluded from answered coverage.
    assert row["coverage"]["answered"] == 12


def test_scores_100_scale_and_direction():
    instances = [instance(str(n)) for n in range(4)]
    records = [grade("p", str(n)) for n in range(4)]  # perfect run
    records[0]["metrics"]["exact_world"] = 0  # one imperfect world
    records[0]["metrics"]["task_success"] = 0  # exact world feeds task success
    row = aggregate(records, instances, ["p"], bootstrap_resamples=0)["leaderboard"][0]
    s = row["scores_100"]
    assert s["task_success"] == 75.0          # 3/4 items fully solved
    assert s["fact_f1"] == 100.0
    assert s["exact_world"] == 75.0
    assert s["earned_decisions"] == 75.0      # 1 of 4 decisions was a guess
    assert s["reasoning_discipline"] == 100.0
    assert s["discovery_retention"] is None   # single condition: no gap


def test_report_leads_with_chart_and_demotes_completeness(tmp_path):
    class ChartParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.first_heading = None
            self.svg_count = 0
            self.in_h2 = False

        def handle_starttag(self, tag, attrs):
            if tag == "svg":
                self.svg_count += 1
            if tag == "h2":
                self.in_h2 = True
                if self.first_heading is None:
                    self.first_heading = "__h2__"

        def handle_data(self, data):
            if self.in_h2 and self.first_heading == "__h2__":
                self.first_heading = data.strip()

        def handle_endtag(self, tag):
            if tag == "h2":
                self.in_h2 = False

    instances = [instance(str(n), level=n % 2) for n in range(6)]
    for inst in instances:
        inst["difficulty"] = {"forward_chain_rounds": 2, "direct_assignment_fraction": .5}
        inst["input_words"] = 900
    records = []
    for n, inst in enumerate(instances):
        row = grade("p", inst["iid"])
        if inst["condition"] == "implicit":
            row["metrics"]["fact_f1"] = .5
        records.append(row)
    agg = aggregate(records, instances, ["p"], bootstrap_resamples=0)
    agg["generation_completeness"] = {"complete": True}
    out = tmp_path / "r.html"
    render_html(agg, out, instances)
    html_text = out.read_text()
    parser = ChartParser()
    parser.feed(html_text)
    # Hero leads the page; the Benchmark tab opens with the top line + chart.
    assert html_text.index('class="hero"') < html_text.index("<svg")
    assert html_text.index("leads: task success") < html_text.index("<svg")
    assert parser.first_heading == "The leaderboard at a glance"
    # Every tab carries at least one chart; completeness is appendix-only.
    assert parser.svg_count >= 6          # overview + strata + paired + difficulty + stories + decisions
    assert html_text.index("Generation completeness") > html_text.index('id="tab-appendix"')
    assert 'aside class="warning"' not in html_text.split("<nav")[0]

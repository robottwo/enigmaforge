"""Coverage-aware benchmark summaries and an offline, escaped HTML report.

Means are item-weighted. Uncertainty resamples whole world families, never
individual realizations. Paired conditions first average realizations within
a family; model comparisons retain item weights but resample whole families.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import html
import json
import math
from pathlib import Path
import random


METRICS = ("fact_precision", "fact_recall", "fact_f1", "exact_world",
           "decision_correct", "task_success")
STRATA = ("level", "scenario", "condition", "realization", "genre", "renderer")
STATUSES = ("answered", "missing", "content_filter", "transport_error",
            "invalid_response", "judge_error", "adjudication_required")
UNAVAILABLE = {"judge_error", "adjudication_required"}
FAILURES = {"missing", "content_filter", "transport_error", "invalid_response"}
POLICY_METRICS = {"decision_correct", "task_success"}


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _family(instance):
    return instance.get("family_id") or instance["iid"]


def _eligible(instance, metric):
    return metric not in POLICY_METRICS or bool(instance.get("decision_policy"))


def _value(record, metric, all_items=False):
    status = record["status"]
    if status in UNAVAILABLE:
        return None
    if all_items and status in FAILURES:
        return 0.0
    value = record.get("metrics", {}).get(metric)
    return float(value) if _number(value) else None


def _interval(clusters, resamples, seed):
    """Percentile interval for a ratio of sums, sampling clusters uniformly."""
    groups = list(clusters.values())
    if not groups or not resamples:
        return None
    # A single world cannot estimate between-world uncertainty.
    if len(groups) < 2:
        return {"low": None, "high": None, "clusters": len(groups),
                "reason": "fewer_than_two_families"}
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        total = count = 0
        for _ in groups:
            subtotal, n = groups[rng.randrange(len(groups))]
            total += subtotal
            count += n
        draws.append(total / count)
    draws.sort()

    def quantile(q):
        pos = (len(draws) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(draws) - 1)
        return draws[lo] + (draws[hi] - draws[lo]) * (pos - lo)

    return {"low": quantile(.025), "high": quantile(.975),
            "clusters": len(groups)}


def _estimate(values, expected, unavailable, resamples, seed):
    clusters = defaultdict(lambda: [0.0, 0])
    for family, value in values:
        clusters[family][0] += value
        clusters[family][1] += 1
    total = sum(value for _, value in values)
    value = total / expected if expected and not unavailable else None
    return {"value": value, "numerator": total, "denominator": expected,
            "unavailable": unavailable,
            "lower_bound": total / expected if expected else None,
            "ci": _interval(clusters, resamples, seed) if value is not None else None}


def _summary(pairs, resamples, seed):
    counts = Counter(record["status"] for _, record in pairs)
    coverage = {status: counts[status] for status in STATUSES}
    length_empty = sum(1 for _, r in pairs
                       if r.get("finish_reason") == "length"
                       and r.get("status") == "invalid_response")
    coverage.update(expected=len(pairs),
                    recorded=sum(not record.get("synthetic_missing", False)
                                 for _, record in pairs),
                    answered=sum(counts[s] for s in
                                 ("answered", "invalid_response", "judge_error",
                                  "adjudication_required"))
                    - length_empty,
                    valid=sum(record["status"] == "answered" and
                              record.get("components", {}).get("compliance") == 1
                              for _, record in pairs))
    # The answered field is response coverage — responses that actually said
    # something gradable. Length-truncated non-answers are failures, not
    # coverage; status counts stay separate.
    metrics = {}
    for metric in METRICS:
        eligible = [(instance, record) for instance, record in pairs
                    if _eligible(instance, metric)]
        conditional = [(_family(instance), value)
                       for instance, record in eligible
                       if (value := _value(record, metric)) is not None]
        all_values = [(_family(instance), value)
                      for instance, record in eligible
                      if (value := _value(record, metric, True)) is not None]
        metrics[metric] = {
            "conditional": _estimate(conditional, len(conditional), 0, resamples, seed),
            "all_items": _estimate(all_values, len(eligible),
                                   len(eligible) - len(all_values), resamples, seed)}
    costs = [r["cost"] for _, r in pairs if _number(r.get("cost"))]
    tokens = [r["tokens"]["total_tokens"] for _, r in pairs
              if isinstance(r.get("tokens"), dict) and
              _number(r["tokens"].get("total_tokens"))]

    # Derived diagnostics. All are conditional (scored items only) with
    # explicit denominators; each isolates one failure mode.
    def _cond_f1(condition):
        vals = [v for inst, r in pairs
                if inst.get("condition") == condition
                and (v := _value(r, "fact_f1")) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    formal_f1, formal_n = _cond_f1("formal")
    explicit_f1, explicit_n = _cond_f1("explicit")
    implicit_f1, implicit_n = _cond_f1("implicit")
    # Discovery cost: same story, task stated (explicit) vs not (implicit).
    # The formal comparison is reported alongside as the compression bound.
    discovery_cost = (round(explicit_f1 - implicit_f1, 4)
                      if explicit_f1 is not None and implicit_f1 is not None else None)
    derived = {
        "discovery_cost": {"difference": discovery_cost,
                           "explicit_f1": explicit_f1, "implicit_f1": implicit_f1,
                           "explicit_n": explicit_n, "implicit_n": implicit_n},
        "compression_bound": {"formal_f1": formal_f1, "explicit_f1": explicit_f1,
                              "formal_n": formal_n, "explicit_n": explicit_n},
        "reasoning_overflow": {
            "rate": round(length_empty / len(pairs), 4) if pairs else None,
            "count": length_empty, "denominator": len(pairs)},
    }
    # Educated guesses: policy-graded decisions that were correct while the
    # submitted world was NOT exactly right — right action, imperfect facts.
    decision_items = [(inst, r) for inst, r in pairs
                      if _eligible(inst, "decision_correct")
                      and _value(r, "decision_correct") is not None
                      and r.get("metrics", {}).get("exact_world") is not None]
    guesses = sum(1 for inst, r in decision_items
                  if r["metrics"]["decision_correct"]
                  and not r["metrics"]["exact_world"])
    derived["educated_guess"] = {
        "rate": round(guesses / len(decision_items), 4) if decision_items else None,
        "count": guesses, "denominator": len(decision_items)}

    warnings = (["Scoring unavailable: judge failures or legacy responses "
                 "require adjudication; lower bounds are not capability estimates."]
                if any(counts[s] for s in UNAVAILABLE) else [])
    # A reasoning-mode model that exhausts max_tokens before emitting any
    # visible text is graded invalid_response with finish_reason=length.
    # Warn loudly: this is a config/model interaction, not zero capability.
    # Answered-with-length items carried partial visible text and are graded
    # normally.
    if length_empty and length_empty >= max(1, round(0.1 * len(pairs))):
        warnings.append(
            f"{length_empty}/{len(pairs)} responses exhausted max_tokens on "
            "reasoning without emitting visible text (empty, "
            "finish_reason=length). This is a budget configuration problem; "
            "raise max_tokens and/or cap the reasoning budget "
            "(reasoning_max_tokens) before comparing these scores.")
    return {"coverage": coverage, "status_counts": {s: counts[s] for s in STATUSES},
            "metrics": metrics, "score": metrics["fact_f1"]["all_items"]["value"],
            "derived": derived,
            "warnings": warnings,
            "recorded_attempt_cost": sum(costs) if costs else None,
            "cost_known_records": len(costs),
            "total_seconds": sum(r.get("seconds") or 0 for _, r in pairs),
            "tokens": sum(tokens) if tokens else None}


def _paired_summary(pairs, expected, resamples, seed):
    clusters = defaultdict(lambda: [0.0, 0])
    for family, difference in pairs:
        clusters[family][0] += difference
        clusters[family][1] += 1
    return {"difference": sum(d for _, d in pairs) / len(pairs) if pairs else None,
            "expected_pairs": expected, "complete_pairs": len(pairs),
            "missing_pairs": expected - len(pairs), "families": len(clusters),
            "ci": _interval(clusters, resamples, seed)}


def _paired(instances, names, records, resamples, seed):
    models = []
    for left, right in combinations(names, 2):
        results = {}
        for metric in METRICS:
            pairs = []
            eligible = [i for i in instances if _eligible(i, metric)]
            for instance in eligible:
                a = _value(records[left, instance["iid"]], metric)
                b = _value(records[right, instance["iid"]], metric)
                if a is not None and b is not None:
                    pairs.append((_family(instance), a - b))
            results[metric] = _paired_summary(pairs, len(eligible), resamples, seed)
        models.append({"left": left, "right": right, "metrics": results})

    grouped = defaultdict(lambda: defaultdict(list))
    conditions = set()
    for instance in instances:
        condition = instance.get("condition")
        if condition is not None:
            conditions.add(condition)
            grouped[_family(instance)][condition].append(instance)
    comparisons = []
    for provider in names:
        for left, right in combinations(sorted(conditions), 2):
            results = {}
            for metric in METRICS:
                pairs = []
                expected = 0
                for family, variants in sorted(grouped.items()):
                    relevant = variants.get(left, []) + variants.get(right, [])
                    if not any(_eligible(i, metric) for i in relevant):
                        continue
                    expected += 1
                    sides = []
                    for condition in (left, right):
                        items = [i for i in variants.get(condition, [])
                                 if _eligible(i, metric)]
                        values = [_value(records[provider, i["iid"]], metric) for i in items]
                        sides.append(sum(values) / len(values) if values and
                                     all(v is not None for v in values) else None)
                    if all(v is not None for v in sides):
                        pairs.append((family, sides[0] - sides[1]))
                results[metric] = _paired_summary(pairs, expected, resamples, seed)
            comparisons.append({"provider": provider, "left": left, "right": right,
                                "metrics": results})
    return {"direction": "left minus right", "models": models,
            "conditions": comparisons,
            "population": "Complete scored pairs only; condition pairs require all "
                          "configured realizations on both sides. Missing pairs are excluded."}


def _difficulty(instances):
    families = {}
    for instance in instances:
        if instance.get("difficulty") is not None:
            families.setdefault(_family(instance), instance["difficulty"])
    numeric = defaultdict(list)
    for difficulty in families.values():
        if isinstance(difficulty, dict):
            for key, value in difficulty.items():
                if _number(value) and not isinstance(value, bool):
                    numeric[key].append(value)
    return {"families_measured": len(families),
            "numeric_summary": {key: {"n": len(values), "mean": sum(values) / len(values),
                                      "min": min(values), "max": max(values)}
                                for key, values in sorted(numeric.items())},
            "by_family": dict(sorted(families.items()))}


def _score100(value):
    """Convert a 0-1 rate to a 0-100 score; higher is always better."""
    return None if value is None else round(value * 100, 1)


def _scores_100(summary):
    """Unified 0-100 scale, lower = worse. Failure-mode rates are inverted
    into goodness scores so every column reads the same direction:
    discovery retention (100 − explicit−implicit F1 gap), reasoning
    discipline (100 − overflow), earned decisions (100 − guess rate)."""
    m = summary["metrics"]
    d = summary["derived"]
    disc = d["discovery_cost"]["difference"]
    return {
        "task_success": _score100(m["task_success"]["all_items"]["value"]),
        "fact_f1": _score100(m["fact_f1"]["all_items"]["value"]),
        "exact_world": _score100(m["exact_world"]["all_items"]["value"]),
        "discovery_retention": None if disc is None else round(100 - disc * 100, 1),
        "reasoning_discipline": _score100(1 - (d["reasoning_overflow"]["rate"] or 0))
        if d["reasoning_overflow"]["denominator"] else None,
        "earned_decisions": _score100(1 - d["educated_guess"]["rate"])
        if d["educated_guess"]["denominator"] else None,
    }


def aggregate(graded, instances, providers, *, bootstrap_resamples=2000, bootstrap_seed=0):
    """Build the expected provider × instance grid before computing any mean.

    Unknown adjudications make all-item estimates unavailable rather than zero.
    Missing/filtered/transport/invalid attempts count as zero achieved success in
    all-item estimates. Conditional denominators contain only scored items.
    Decision and task metrics require a declared decision policy.
    """
    if not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be a nonnegative integer")
    instances = sorted(instances, key=lambda i: i["iid"])
    names = sorted(p if isinstance(p, str) else p["name"] for p in providers)
    if len(set(names)) != len(names) or len({i["iid"] for i in instances}) != len(instances):
        raise ValueError("provider names and instance IDs must be unique")
    expected = {(name, instance["iid"]) for name in names for instance in instances}
    records = {}
    for record in graded:
        key = record["provider"], record["iid"]
        if key not in expected:
            raise ValueError(f"record outside configured provider/instance grid: {key!r}")
        if key in records:
            raise ValueError(f"duplicate graded record: {key!r}")
        if record.get("status") not in STATUSES:
            raise ValueError(f"unknown grading status: {record.get('status')!r}")
        records[key] = dict(record)
    for name, iid in sorted(expected - records.keys()):
        records[name, iid] = {"provider": name, "iid": iid, "status": "missing",
                              "error": None, "score": None, "components": {},
                              "metrics": {}, "decision": None, "seconds": 0,
                              "cost": None, "tokens": None, "synthetic_missing": True}
    rows = []
    strata = {}
    for name in names:
        pairs = [(i, records[name, i["iid"]]) for i in instances]
        summary = _summary(pairs, bootstrap_resamples, bootstrap_seed)
        summary["scores_100"] = _scores_100(summary)
        rows.append({"provider": name, **summary})
        strata[name] = {}
        for dimension in STRATA:
            groups = defaultdict(list)
            for instance, record in pairs:
                groups[instance.get(dimension)].append((instance, record))
            strata[name][dimension] = [
                {"value": value, **_summary(group, 0, bootstrap_seed)}
                for value, group in sorted(groups.items(), key=lambda pair: str(pair[0]))]

    def ranking(row):
        task = row["metrics"]["task_success"]["all_items"]["value"]
        fact = row["score"]
        return task is None, -(task or 0), fact is None, -(fact or 0), row["provider"]

    rows.sort(key=ranking)
    return {"version": 3, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_instances": len(instances), "n_families": len({_family(i) for i in instances}),
            "ranking": ["all_item_task_success_desc", "all_item_fact_f1_desc", "provider_asc"],
            "leaderboard": rows, "strata": strata,
            "uncertainty": {"method": "family-cluster percentile bootstrap", "confidence": .95,
                            "resamples": bootstrap_resamples, "seed": bootstrap_seed,
                            "strata_intervals": False,
                            "caution": "Descriptive intervals, not multiplicity-adjusted; "
                                       "few families give unstable uncertainty estimates."},
            "paired_differences": _paired(instances, names, records,
                                           bootstrap_resamples, bootstrap_seed),
            "difficulty": _difficulty(instances),
            "runs": [{k: i.get(k) for k in ("iid", "family_id", *STRATA, "difficulty")}
                     for i in instances],
            "records": [records[key] for key in sorted(records)],
            "cost_scope": "Recorded solver attempt costs only; not full experimental spend."}


def _escape(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return _escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def _top_line(rows):
    """One-sentence outcome: the leading real (non-baseline) provider under
    the report's single ranking, with honest coverage caveats. Returns
    None when no provider produced a scored answer."""
    candidates = [r for r in rows
                  if not r["provider"].startswith("baseline:")
                  and r["coverage"]["valid"] > 0]
    if not candidates:
        return None
    best = candidates[0]  # rows are already in the report's ranking order
    s = best["scores_100"]
    cov = best["coverage"]
    return (f"{best['provider']} leads: task success {s['task_success']}/100 "
            f"({best['metrics']['task_success']['all_items']['numerator']:.0f}/"
            f"{best['metrics']['task_success']['all_items']['denominator']:.0f} items), "
            f"fact F1 {s['fact_f1']}/100, "
            f"{cov['answered']}/{cov['expected']} items answered"
            + (f" · {len(best['warnings'])} warning(s)" if best["warnings"] else "")
            + ". Solid baselines: constraint-solver and story-copy bound the scale.")


def _summary_chart(rows):
    """Static inline SVG: paired horizontal bars (task success, fact F1) per
    model provider in ranking order. Baselines stay in the tables but are not
    plotted. All text is escaped; no JS data."""
    rows = [r for r in rows if not r["provider"].startswith("baseline:")]
    bar_h, gap, label_w, plot_w = 14, 6, 190, 420
    chart_h = max(len(rows), 1) * (bar_h * 2 + gap) + 30
    parts = [f'<svg role="img" aria-label="All-item task success and fact F1 per provider" '
             f'viewBox="0 0 {label_w + plot_w + 60} {chart_h}" width="100%" '
             f'style="max-width:700px" xmlns="http://www.w3.org/2000/svg">']
    y = 4
    for row in rows:
        s = row["scores_100"]
        task = (s["task_success"] or 0) / 100
        f1 = (s["fact_f1"] or 0) / 100
        label = _escape(row["provider"][:28])
        parts.append(f'<text x="{label_w - 8}" y="{y + bar_h}" text-anchor="end" '
                     f'dominant-baseline="middle" font-size="11" fill="#c1d7f5">{label}</text>')
        w_task = round(plot_w * task)
        w_f1 = round(plot_w * f1)
        parts.append(f'<rect x="{label_w}" y="{y}" width="{w_task}" height="{bar_h - 2}" '
                     f'fill="#4f9cf0"><title>{_escape(row["provider"])} task success: {s["task_success"]}/100</title></rect>')
        parts.append(f'<rect x="{label_w}" y="{y + bar_h}" width="{w_f1}" height="{bar_h - 2}" '
                     f'fill="#2f7d4f"><title>{_escape(row["provider"])} fact F1: {s["fact_f1"]}/100</title></rect>')
        parts.append(f'<text x="{label_w + plot_w + 6}" y="{y + bar_h}" dominant-baseline="middle" '
                     f'font-size="10" fill="#c1c8d4">{s["task_success"]}</text>')
        y += bar_h * 2 + gap
    parts.append(f'<rect x="{label_w}" y="{chart_h - 14}" width="12" height="10" fill="#4f9cf0"/>'
                 f'<text x="{label_w + 16}" y="{chart_h - 6}" font-size="10" fill="#c1c8d4">task success</text>'
                 f'<rect x="{label_w + 110}" y="{chart_h - 14}" width="12" height="10" fill="#2f7d4f"/>'
                 f'<text x="{label_w + 126}" y="{chart_h - 6}" font-size="10" fill="#c1c8d4">fact F1</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _metric_cell(estimate):
    result = f"{_percent(estimate['value'])} <small>(n={estimate['denominator']})</small>"
    ci = estimate.get("ci")
    if ci and ci["low"] is not None:
        result += f"<br><small>95% CI {_percent(ci['low'])}–{_percent(ci['high'])}</small>"
    if estimate.get("unavailable"):
        result += (f"<br><strong>{estimate['unavailable']} unscored</strong>"
                   f"<br><small>observed lower bound {_percent(estimate['lower_bound'])}</small>")
    return result


def _table(headers, rows, caption):
    return ('<div class="table-scroll"><table><caption>' + _escape(caption) +
            '</caption><thead><tr>' + ''.join('<th scope="col">' + _escape(h) + '</th>'
                                             for h in headers) + '</tr></thead><tbody>' +
            ''.join('<tr>' + ''.join((f'<th scope="row">{cell}</th>' if n == 0
                                      else f'<td>{cell}</td>')
                                     for n, cell in enumerate(row)) + '</tr>'
                    for row in rows) + '</tbody></table></div>')


_CSS = """
:root { color-scheme: light dark; font: 16px/1.5 system-ui,sans-serif; }
body { margin:0; background:#12151c; color:#e6e9ef; }
main { max-width:1400px; padding:1.5rem; margin:auto; }
a { color:#a9d0ff; } button { font:inherit; cursor:pointer; padding:.5rem .8rem;
background:#26334a; color:#fff; border:1px solid #7187a8; border-radius:.3rem; }
button[aria-selected=true] { background:#375b8d; } :focus-visible { outline:3px solid #ffd35e; }
nav { display:flex; gap:.5rem; flex-wrap:wrap; margin:1rem 0; }
.table-scroll { overflow-x:auto; } table { width:100%; border-collapse:collapse; margin:1rem 0; }
caption { text-align:left; font-weight:600; padding:.5rem 0; }
th,td { text-align:left; padding:.5rem; border-bottom:1px solid #46516a; vertical-align:top; }
th { color:#c1d7f5; } small { color:#c1c8d4; } pre { white-space:pre-wrap; overflow-wrap:anywhere;
background:#1d2533; padding:1rem; } details { margin:.7rem 0; border:1px solid #46516a; padding:.5rem; }
summary { cursor:pointer; font-weight:600; overflow-wrap:anywhere; }
.warning { padding:1rem; border:2px solid #dfb158; background:#332b1e; }
[hidden] { display:none !important; }
"""

_SCRIPT = """
const tabs = [...document.querySelectorAll('[role=tab]')];
function activate(tab, focus=false) {
  tabs.forEach(t => {
    const selected = t === tab;
    t.setAttribute('aria-selected', String(selected));
    t.tabIndex = selected ? 0 : -1;
    document.getElementById(t.getAttribute('aria-controls')).hidden = !selected;
  });
  if (focus) tab.focus();
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activate(tab));
  tab.addEventListener('keydown', event => {
    let target;
    if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') target = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') target = 0;
    if (event.key === 'End') target = tabs.length - 1;
    if (target !== undefined) { event.preventDefault(); activate(tabs[target], true); }
  });
});
if (tabs.length) activate(tabs[0]);
"""


def render_html(agg, out_path, instances=None):
    """Write an offline report. Dynamic text never enters JavaScript or IDs."""
    rows = agg["leaderboard"]  # The sole ranking order is aggregate's order.
    names = [r["provider"] for r in rows]
    records = {(r["provider"], r["iid"]): r for r in agg["records"]}
    blocks = []
    performance_rows = []
    coverage_rows = []
    for row in rows:
        s = row["scores_100"]
        metric_cells = [_escape(row["provider"])]
        for key in ("task_success", "fact_f1", "exact_world"):
            for population in ("all_items", "conditional"):
                metric_cells.append(_metric_cell(row["metrics"][key][population]))
        metric_cells += [f"{s['task_success']}" if s["task_success"] is not None else "—",
                         f"{s['fact_f1']}" if s["fact_f1"] is not None else "—",
                         f"{s['discovery_retention']}" if s["discovery_retention"] is not None else "—",
                         f"{s['reasoning_discipline']}" if s["reasoning_discipline"] is not None else "—",
                         f"{s['earned_decisions']}" if s["earned_decisions"] is not None else "—"]
        performance_rows.append(metric_cells)
        c = row["coverage"]
        coverage_rows.append([_escape(row["provider"])] +
                             [str(c[k]) for k in ("expected", "recorded", "answered", "valid",
                                                 "missing", "content_filter", "transport_error",
                                                 "invalid_response", "judge_error",
                                                 "adjudication_required")] +
                             ["—" if row["recorded_attempt_cost"] is None else
                              f"${row['recorded_attempt_cost']:.4f}",
                              str(row["cost_known_records"])])
        for warning in row["warnings"]:
            blocks.append(f'<p class="warning"><strong>{_escape(row["provider"])}</strong>: '
                          f'{_escape(warning)}</p>')
    performance = _table(
        ["Provider"] + [f"{m} · {p}" for m in ("Task success", "Fact F1", "Exact world")
                        for p in ("all items /100", "scored only")] +
        ["Task success /100", "Fact F1 /100", "Discovery retention /100",
         "Reasoning discipline /100", "Earned decisions /100"],
        performance_rows,
        "Performance — all scores 0-100, higher is better (lower is worse). "
        "Discovery retention = 100 − (explicit − implicit F1 gap); reasoning "
        "discipline = 100 − reasoning overflow; earned decisions = 100 − share "
        "of correct decisions reached without an exactly-right world. "
        "Discovery retention above 100 means the unstated task outperformed "
        "the stated one for that provider.")
    coverage = _table(["Provider", "Expected", "Recorded", "Answered", "Valid", "Missing",
                       "Filtered", "Transport error", "Invalid response", "Judge error",
                       "Adjudication required", "Recorded attempt cost", "Known-cost records"],
                      coverage_rows, "Coverage and recorded costs")
    strata_parts = []
    for dimension in STRATA:
        entries = []
        for name in names:
            for group in agg["strata"][name][dimension]:
                entries.append([_escape(name), _escape(group["value"]),
                                str(group["coverage"]["expected"]),
                                str(group["coverage"]["answered"])] +
                               [_metric_cell(group["metrics"][metric][population])
                                for metric in ("fact_f1", "exact_world", "task_success")
                                for population in ("all_items", "conditional")])
        strata_parts.append(_table(["Provider", dimension, "Expected", "Answered"] +
                                   [f"{m} · {p}" for m in ("Fact F1", "Exact world", "Task success")
                                    for p in ("all items", "scored only")], entries, dimension))
    paired_parts = []
    for kind in ("models", "conditions"):
        entries = []
        for comparison in agg["paired_differences"][kind]:
            for metric in ("fact_f1", "exact_world", "task_success", "decision_correct"):
                result = comparison["metrics"][metric]
                ci = result["ci"]
                interval = (f"{_percent(ci['low'])} to {_percent(ci['high'])}"
                            if ci and ci["low"] is not None else "—")
                entries.append([_escape(comparison.get("provider", "All providers")),
                                _escape(comparison["left"]), _escape(comparison["right"]),
                                _escape(metric), _percent(result["difference"]), interval,
                                str(result["expected_pairs"]), str(result["complete_pairs"]),
                                str(result["missing_pairs"]), str(result["families"])])
        paired_parts.append(_table(["Provider", "Left", "Right", "Metric", "Difference (pp)",
                                    "95% CI (pp)", "Expected pairs", "Complete pairs", "Missing pairs",
                                    "Families"], entries, kind))
    stories = []
    for instance in sorted(instances or [], key=lambda i: i["iid"]):
        fact_rows = [[_escape(vid), _escape((instance.get("surfaces") or {}).get(vid, vid)),
                      _json(value)] for vid, value in sorted((instance.get("ground_truth") or {}).items())]
        stories.append('<details><summary>' + _escape(instance["iid"]) + '</summary>' +
                       '<h3>Solver input</h3><pre>' +
                       _escape(instance.get("input_text", instance.get("story_text", ""))) +
                       '</pre><details><summary>Hidden fact pattern</summary>' +
                       _table(["Variable", "Surface subject", "Ground truth"], fact_rows,
                              "Ground truth mapped by variable ID") + '</details>' +
                       '<details><summary>Decision policy and measured difficulty</summary><pre>' +
                       _json({"decision_policy": instance.get("decision_policy"),
                              "difficulty": instance.get("difficulty")}) + '</pre></details></details>')
    decisions = []
    by_level = defaultdict(list)
    for run in agg["runs"]:
        by_level[run.get("level")].append(run)
    for level, runs in sorted(by_level.items(), key=lambda pair: str(pair[0])):
        entries = []
        for run in runs:
            for name in names:
                record = records[name, run["iid"]]
                entries.append([_escape(run["iid"]), _escape(name), _escape(record["status"]),
                                _percent(record.get("metrics", {}).get("decision_correct")),
                                '<pre>' + _json(record.get("decision")) + '</pre>'])
        decisions.append(_table(["Instance", "Provider", "Status", "Correct", "Submitted action"],
                                entries, f"Configured level: {level}"))
    detailed = []
    for name in names:
        for run in agg["runs"]:
            record = records[name, run["iid"]]
            detailed.append('<details><summary>' + _escape(name) + ' / ' + _escape(run["iid"]) +
                            ' · ' + _escape(record["status"]) + '</summary><pre>' +
                            _json(record) + '</pre></details>')
    top_line = _top_line(rows)
    headline = (f'<h2>Top line</h2><p><strong>{_escape(top_line)}</strong></p>'
                if top_line else
                '<h2>Top line</h2><p>No provider produced a scored answer.</p>')
    summary = ('<details open><summary>Summary</summary>' + _summary_chart(rows) +
               '<p>All-item values include failed and missing attempts as zero; '
               'pale bars are deterministic baselines (constraint-solver: perfect '
               'information bound; story-copy: must score zero).</p></details>')
    tabs = [("overview", "Overview", headline + summary + performance + coverage),
            ("strata", "Strata", '<p>Levels are configured strata, not calibrated capability ceilings. '
             'Stratum estimates are descriptive; intervals below use family clusters.</p>' + ''.join(strata_parts)),
            ("uncertainty", "Paired comparisons", '<p>Differences are left minus right, in percentage points. '
             + _escape(agg["paired_differences"]["population"]) + '</p><pre>' +
             _json(agg["uncertainty"]) + '</pre>' + ''.join(paired_parts)),
            ("difficulty", "Measured difficulty", '<pre>' + _json(agg["difficulty"]) + '</pre>'),
            ("stories", "Stories", ''.join(stories) or '<p>No story data supplied.</p>'),
            ("decisions", "Decisions", ''.join(decisions)),
            ("details", "Record details", ''.join(detailed))]
    nav = ''.join(f'<button type="button" role="tab" id="tab-{key}" aria-controls="pane-{key}" '
                  f'aria-selected="false">{label}</button>' for key, label, _ in tabs)
    panes = ''.join(f'<section role="tabpanel" tabindex="0" id="pane-{key}" '
                    f'aria-labelledby="tab-{key}">{body}</section>' for key, _, body in tabs)
    completeness = agg.get("generation_completeness")
    banner = ('<aside class="warning"><strong>Generation completeness</strong><pre>' +
              _json(completeness) + '</pre></aside>') if completeness is not None else ''
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           '<title>EnigmaForge benchmark report</title><style>' + _CSS + '</style></head><body><main>'
           '<h1>EnigmaForge benchmark</h1><p>' + _escape(agg["generated_at"]) +
           f' · {agg["n_instances"]} instances · {agg["n_families"]} world families</p>' + banner +
           '<p>All scores are 0-100; lower is worse. Headline measure: fact F1. Ranking: all-item task success, then all-item fact F1, '
           'then provider name. No weighted composite. Conditional means use scored-item denominators; '
           'all-item means include missing, filtered, invalid, and failed attempts as zero achieved success. '
           'Unadjudicated outcomes remain unavailable. Decisions and task success use policy-bearing items only. '
           'Derived diagnostics: discovery cost is the conditional-F1 drop from a stated task (explicit) to an '
           'unstated one (implicit) — the price of finding the problem; reasoning overflow is the share of items '
           'where the model exhausted max_tokens without visible output; educated guesses are decisions that were '
           'correct while the submitted world was not exactly right. '
           'Answered includes invalid content; valid means a schema-compliant scored answer. '
           'Confidence intervals resample world families, not realizations.</p><p>' +
           _escape(agg["cost_scope"]) + '</p>' + ''.join(blocks) +
           '<nav role="tablist" aria-label="Report sections">' + nav + '</nav>' + panes +
           '</main><script>' + _SCRIPT + '</script></body></html>')
    Path(out_path).write_text(doc, encoding="utf-8")


def print_leaderboard(agg):
    """Print precisely the same provider order as the HTML report."""
    top_line = _top_line(agg["leaderboard"])
    print("\n== Top line ==")
    print(top_line or "No provider produced a scored answer.")
    print("\n== Leaderboard: scores out of 100, lower is worse ==")
    print(f"{'provider':<28} {'task':>6} {'F1':>6} {'exact':>6} {'answer':>8} "
          f"{'disco':>6} {'ovfl':>6} {'guess':>6}")
    for row in agg["leaderboard"]:
        sc = row["scores_100"]
        fmt = lambda v: "—" if v is None else f"{v:g}"
        coverage = row["coverage"]
        count = f"{coverage['answered']}/{coverage['expected']}"
        print(f"{row['provider']:<28} {fmt(sc['task_success']):>6} {fmt(sc['fact_f1']):>6} "
              f"{fmt(sc['exact_world']):>6} {count:>8}"
              f" {fmt(sc['discovery_retention']):>6} {fmt(sc['reasoning_discipline']):>6}"
              f" {fmt(sc['earned_decisions']):>6}")
        for warning in row["warnings"]:
            print(f"  WARNING: {warning}")
    print("All-item denominator includes missing attempts; — means unavailable, not zero capability.")
    print(agg["cost_scope"])

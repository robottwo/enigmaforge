# 🧩 EnigmaForge

**Procedurally generated benchmarks for the hardest thing an LLM can do: discover the problem before solving it.**

Most benchmarks hand a model the question. EnigmaForge hands it a *record* — letters, receipts, logbooks, marginalia — and asks nothing else. The real task is hidden inside: the solver must infer latent entities, decide which evidence matters, supply world knowledge the narrative never states, abandon objectives that turn out to be intermediate, and justify a final answer against a **mechanically verified ground truth**.

> *You have been given the complete record of an unusual sequence of events. Determine what the record ultimately requires you to figure out. Then figure it out.*

## Why it is different

| Conventional benchmarks | EnigmaForge |
|---|---|
| Question is stated | Problem must be **discovered** |
| Puzzle handcrafted | Instance **procedurally generated** at any complexity |
| Solution trusted by convention | Uniqueness **proved by SAT** |
| Distractors are noise | Distractors **support plausible false hypotheses** |
| Score = final answer | **Answer-shaped scoring**: exact assignments, policy-validated actions, coverage-first reporting |
| One canonical wording | Multiple **surface realizations** of one hidden instance |

## The pipeline

```
config ──► World Generator ──► Constraint/Objective Generator ──► Formal Verifier
                                                                        │
   ◄── package ── Narrative Compiler ◄── Knowledge Bridges ◄── Distractors ◄──┘
                    │
                    ├─► Interactive Environment (budgeted investigation)
                    └─► Trajectory Recorder ──► Evaluator ──► Difficulty Calibrator
```

Every published challenge ships with:

- the **solver-visible narrative** (and a second realization with identical solution);
- the **public decision policy** — a conditional register/hold rule stated in the text (with its superseded provisional rule), whose correct action follows from the verified world;
- the **hidden formal world** — variables, constraints, dependency graph;
- a **machine-verified solution** — uniqueness proven by ban-clause UNSAT;
- **ablation certificates** — removing *any* clue provably admits a second model;
- the **evidence→constraint map**, in-world lore references, distractor annotations;
- a **realization map** — span-level provenance from every clue's prose to its
  evidence unit, plus the story skeleton and pacing policy in story mode;

## Quick start

```bash
git clone https://github.com/robottwo/enigmaforge.git
cd enigmaforge
python3 -m enigmaforge.pipeline --size small --seed 2026 --out runs/demo
# or, equivalently:  ./run.py --size small --seed 2026 --out runs/demo
```

Story mode embeds the same verified instance in plain prose — no exhibit
list, no stated task; whether there is anything to figure out is itself
part of the challenge:

```bash
python3 -m enigmaforge.pipeline --size small --seed 2026 --mode story --out runs/demo-story
```

The setting is a seeded axis too — `--genre` picks the pack (maritime,
manor, hotel, theater, observatory); `auto` (default) selects by seed, so
instances vary in setting while staying reproducible per seed. `--genre llm`
goes further: a model invents the entire setting pack, madlib-style — nouns,
places, frames, lore — and the generated pack must pass every construction
time check (unique extractable nouns, no formal leakage) with corrective
retries, else the run fails. Invented packs are persisted per instance as
`genre_pack.json` (they are not seed-reproducible):

```bash
./run.py --size small --seed 2026 --mode story --genre theater --out runs/theater
./run.py --size small --seed 7 --mode story --genre llm --out runs/invented
```

Clue burial is a measured dial: `--burial 0-12` (default 1) adds seeded
pure-story paragraphs before and after clue-bearing paragraphs — at 2+,
whole clue-free story scenes appear between them. Depth is fixed by the
world seed (both realizations bury identically), recorded in
`skeleton.json`, and gated: buried or not, the extraction round-trip must
still recover the model from the prose alone. A final creative `--polish`
pass lets a model rewrite the finished draft for natural prose with full
liberty around the claim clauses — clauses stay verbatim, the polished text
is re-gated, and if the hints cannot be preserved in 3 attempts the run
fails rather than publish an unpolished or broken surface.

### Running with an LLM renderer

Story mode can hand scene prose to a chat model instead of the template
renderer. Rendering is two-phase: one planning call drafts a shared premise,
a character sheet (roles, pronouns), and a distinct setting per scene; the
scene calls then render in parallel (default 3 workers, per-scene retry)
inside that one story — so scenes differ in place and weather and the cast
stays consistent instead of converging on the same rainy room. The model
writes everything *around* the clues but must embed every claim clause
**verbatim**; the pipeline trusts nothing — span search and the extraction
round-trip gate every scene, and failures are rejection-sampled.

```bash
# OpenAI (reads OPENAI_API_KEY; model via --model or ENIGMAFORGE_MODEL)
export OPENAI_API_KEY=sk-...
./run.py --size small --seed 2026 --mode story --renderer llm --out runs/llm-story

# any OpenAI-compatible local server (ollama, vLLM, llama.cpp)
./run.py --size small --seed 2026 --mode story --renderer llm \
         --base-url http://localhost:11434/v1 --model llama3.1
```

With no flags and no env vars, the endpoint is **autodiscovered from local
agent configs** — opencode (`auth.json` + `opencode.jsonc`), codex
(`config.toml` + `auth.json`), goose, and continue are probed for an
OpenAI-compatible base URL, key, and model (precedence: flags > env
`OPENAI_*` > agent configs > defaults; native non-compatible providers like
Anthropic are skipped). Inspect what would be used — keys always redacted:

```bash
python3 -m enigmaforge.llm          # human-readable
python3 -m enigmaforge.llm --json   # machine-readable
```

Trade-off: the formal world, skeleton, pacing, and every verified guarantee
stay seed-deterministic, but the LLM prose itself is not reproducible and
each run re-renders (small ≈ 10-20 calls, large ≈ 100+).

### Benchmarking models

`enigmaforge.harness` generates a verified story cohort, runs every instance
through a list of OpenAI-compatible providers, grades each (provider,
instance) pair mechanically from the hidden formal world, and writes
`results.json`, `results-detailed.json` (raw responses, attempt histories,
and response identities inline), and a self-contained dark-theme
`report.html` with a top-line outcome sentence and a summary bar chart of
all providers on the 0–100 scale:

```bash
python3 -m enigmaforge.harness --providers benchmark.json --out runs/harness
```

The config file is either a bare JSON list of providers (the cohort then
comes from CLI flags — `--instances` (default 6), `--sizes` comma list
cycled across instances, `--genre` (`auto` derives it from the seed),
`--burial-min/max`, `--seed-base`; instance *i* uses seed
`seed_base + i*17`) — or an object that also defines the scenarios:

```json
{
  "provider_defaults": {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "api_key_env": "OPENROUTER_API_KEY"}
  },
  "providers": [
    {"model": "moonshotai/kimi-k3", "defaults": "openrouter"},
    {"model": "qwen/qwen3.6-27b", "defaults": "openrouter"},
    {"name": "openai", "model": "gpt-4o-mini"}
  ],
  "scenarios": [
    {"name": "core", "instances": 6, "sizes": ["small"],
     "genre": "auto", "burial_min": 1, "burial_max": 2, "seed_base": 1000},
    {"name": "buried-medium", "instances": 3, "sizes": ["medium"],
     "burial_min": 2, "burial_max": 3, "seed_base": 5000}
  ]
}
```

- `providers[]` — one entry per model. `name` labels the response files
  and leaderboard rows; omit it and it defaults to the last path segment
  of `model` (`"moonshotai/kimi-k3"` → `kimi-k3`). Request knobs:
  `temperature` (default 0.2), `max_tokens` (default 4096 — set this
  generously; reasoning-mode models can exhaust small budgets on internal
  reasoning without emitting any visible text), `track_cost` (OpenRouter
  `usage: {"include": true}`), `reasoning_effort`, and
  `reasoning_max_tokens` — the latter caps the model's internal reasoning
  budget via OpenRouter's unified `reasoning` object (it replaces
  `reasoning_effort` on the wire, since endpoints reject both together;
  the harness records which was sent in the response identity). An entry
  that exhausts `max_tokens` on reasoning is graded as a failed item and
  flagged in the report when it exceeds 10% of a provider's records.
  `api_key` (literal) and `api_key_env` (env var name; mutually exclusive)
  resolve entry > defaults group > environment.
- `provider_defaults` — optional map of named default groups; an entry
  inherits one with `"defaults": "<group>"`, and per-entry fields win.
  Unknown group references and unknown keys are rejected.
- `scenarios[].name` — required, unique; it is baked into the instance
  directory names (`inst-<scenario>-000-<genre>/`), so each scenario's
  stories are stable across runs. Optional per-scenario knobs with
  defaults: `instances` 6, `sizes` ["small"], `genre` "auto",
  `burial_min` 1, `burial_max` 2, `seed_base` 1000. The cohort CLI flags
  are rejected when scenarios are defined in the config.
- `renderer` — `"template"` (default) or `"llm"`, set per scenario or
  once at the config top level. `"llm"` hands scene prose to a chat model
  (endpoint via the env/agent-config chain; `ENIGMAFORGE_MODEL` picks the
  model) under the same rejection-sampled contract as `--renderer llm`:
  clue clauses stay verbatim, every surface re-passes the gates, and the
  extraction round-trip re-proves uniqueness *as read*. Costs ~10-20
  extra calls per story at generation time and the prose is no longer
  seed-reproducible — existing instance dirs are never re-rendered, so
  flip the flag only before first generation. The rendering model itself
  is set by `renderer_model` (with optional `renderer_base_url` /
  `renderer_api_key_env`); unset, it falls back to `ENIGMAFORGE_MODEL` /
  `OPENAI_*` env vars and agent-config autodiscovery.
- `corpus` — a persistent directory for the generated cohort (instances +
  per-provider responses), kept across runs so every benchmarked model is
  evaluated on the *same* rendered stories — essential once `renderer` is
  `"llm"`, where prose is not seed-reproducible. Config key or `--corpus`
  flag; reports still land in `--out`. Keep it in a gitignored directory
  (`benchmarks/` is ignored by default). A signed **manifest** records the
  expected specs, content hashes, generation attempts and exclusions;
  cached corpora are validated against it and any mismatch fails the run.
  Benchmark responses additionally require an explicit `model` +
  `base_url` per provider — no endpoint autodiscovery inside benchmarks —
  and every cache/report record carries identity hashes (inputs, prompt,
  request parameters, implementation sources) so stale artifacts can never
  be silently served. Costs are recorded per solver attempt; they are not
  a full experimental spend.
- **Cost & time tracking** — every solver response records wall-clock
  seconds, token usage, and (when the endpoint reports it) dollar cost.
  Set `"track_cost": true` on a provider (or in `provider_defaults`) to
  send OpenRouter's `usage: {"include": true}` flag, which returns
  `usage.cost` per call. Totals land in the leaderboard (`tokens`,
  `cost`, `seconds` columns) and in `results.json`; the report's detailed
  table shows cost per provider.
- `scenarios[].levels` — ordered **difficulty strata** (not calibrated
  ceilings): each level may set `size`, `overrides` (any generator knob),
  and `burial` (0–12). `instances` counts stories *per level*; seeds are
  hashed from (scenario, level, index), so every world is independent.
  Reports break results down per stratum alongside measured structural
  difficulty (direct-assignment fraction, forward-chaining depth) — the
  configured `dependency_depth` is a generation knob, not a measured
  property.

Runs are **idempotent** at both stages: successful instance dirs and
response files are never rebuilt or re-called; **error records retry
automatically on the next run** (`--no-retry-errors` to disable; use
`--retry-refusals` to also retry cached content filters/refusals).
Rerun the same command after an interruption and only the missing pieces
execute; adding a provider or growing a scenario's `instances` creates and
calls only the new combinations. `--grade-only` skips generation and
solving entirely and just re-grades and re-renders the reports.

### Grading (benchmark v3)

Solvers reply with a strict JSON schema — `fixed_facts` as
`{"subject", "value"}` assignments (exact typed values, `null` to
abstain) and `final_action` as `{"operation", "subject", "value"}`.
Grading is deterministic and **answer-shaped, never lexical**:

- **facts** — exact subject/value assignments vs the hidden world
  (precision/recall/F1 + exact-world); copied prose, negations,
  alternative values, and conflicting claims earn nothing;
- **decisions** — the story ships a public conditional *policy*
  (`register`/`hold` on a gate variable, with an explicitly superseded
  provisional rule); an action is correct only if all three fields match
  the policy applied to the true world;
- **compliance** — strict schema conformance; malformed output scores 0,
  with no raw-text fallback;
- **headline score** is all-item fact F1; `task_success` (exact world +
  correct action) is reported separately and drives ranking. There is no
  weighted composite and no threshold "ladder ceiling".

Every provider row also carries **`scores_100`** — a unified 0–100 scale,
higher is better (lower is worse) — plus three derived diagnostics that
each isolate one failure mode:

- **discovery retention** = 100 − (explicit − implicit conditional-F1
  gap): how much capability survives when the task isn't stated. Values
  above 100 mean the unstated task outperformed the stated one.
- **reasoning discipline** = 100 − share of items where the model
  exhausted its completion budget on internal reasoning without emitting
  visible text;
- **earned decisions** = 100 − share of policy-graded decisions that were
  correct while the submitted world was *not* exactly right (right
  action, imperfect facts).

Raw 0–1 metrics and failure-mode rates are retained in `results.json`
alongside the scores, so every number is auditable to its counts.

Every story **family** is evaluated under matched **conditions**:
`formal` (published constraints, explicit question), `explicit` (story +
stated task), and `implicit` (story alone, two surface realizations) —
`--conditions formal,explicit,implicit` selects them; paired
condition/model differences come with family-cluster bootstrap CIs.
Deterministic no-API baselines (`--baselines`: direct-facts,
forward-chain, constraint-solver, story-copy) calibrate each run; the
story-copy control that once scored 0.99 under the old grader now scores
0. `--skip-generate` / `--grade-only` validate the corpus against its
signed manifest and fail closed on any modification. Pilot results
(`runs/eval-final`) are immutable; regrade them read-only with
`--legacy-regrade` (optionally with blind judge extraction via
`python -m enigmaforge.judge`, which also handles calibration
export/audit/evaluate).

Three difficulty tiers, all gates verified:

| Size | Latent vars | What it tests |
|---|---|---|
| `small` | 8 | Human-inspectable: latent structure, basic discovery |
| `medium` | 30 | Distractors, knowledge bridges, hidden intermediate objective |
| `large` | 60+ | Budgeted interactive investigation, competing hypotheses, nested objectives |

## Verified guarantees

1. **Solvability** — the world is generated ground-truth-first; a solution exists by construction.
2. **Uniqueness** — banning the intended solution makes the instance UNSAT (DPLL proof).
3. **No dead clues** — dropping any published clue provably admits a second model.
4. **Engine ≡ oracle** — the DPLL engine agrees with exhaustive enumeration on the committed test battery (for large worlds the oracle cross-check is skipped, recorded as skipped, and never reported as proved).
5. **Determinism** — same seed → identical challenge; realization seed changes only the surface, never the solution.
6. **Inert distractors & lore bridges** — narrative red herrings and knowledge "bridges" carry no formal content; they can mislead, never invalidate (bridges are lore, not reasoning requirements).

## Configurable difficulty

Latent entity count, constraint topology, dependency depth, distractor ratio,
competing-hypothesis count, world-knowledge bridges, narrative length,
information budget, planning horizon — every axis is a config knob, from a
6-variable puzzle to 100+ variables spanning tens of thousands of tokens.

## Status

**v0.3** (benchmark contract v3) — deterministic, answer-shaped grading over
strict structured solver responses; public conditional decision policies
with an explicitly superseded provisional rule; matched formal/explicit/
implicit conditions over two surface realizations; deterministic
baselines (including a story-copy control that must score 0); signed
corpus manifests with fail-closed cache validation; transport that keeps
reasoning separate from visible answers, never falls back to it, and
supports explicit reasoning-budget caps (`reasoning_effort`,
`reasoning_max_tokens`); a blind, cached-only judge
(`python -m enigmaforge.judge`) for legacy regrading with an auditable
human-calibration workflow; and coverage-first reporting — unified 0–100
scores, derived diagnostics (discovery retention, reasoning discipline,
earned decisions), family-cluster bootstrap CIs — where missing attempts
count as failures and adjudication-pending never masquerades as zero
capability. First full seven-model comparison on one 600-variant cohort
in `runs/v3-models-7`. The pilot leaderboard (`runs/eval-final`) predates
v3 and is retained read-only as a negative result: its word-overlap
grader was beaten by copying the story (0.993).

## Limits (v3)

- The extraction round-trip is the deterministic inverse of the template
  grammar; it proves the supported claim set, not the full prose
  semantics of every sentence. LLM-rendered stories are gated the same
  way, but no gate currently proves that arbitrary prose *entails* the
  model.
- Measured difficulty (forward-chaining rounds, direct-assignment
  fraction) characterizes one propagator; it is not a proof of minimum
  reasoning complexity, and instances remain solvable by shallow
  propagation.
- The judge is shipped uncalibrated: it must pass a human-adjudicated
  audit (false-accept/false-reject rates by stratum) before its
  extractions are trusted. Seed fixtures are explicitly not human labels.
- Story compilers remain template-based; the LLM scene renderer is
  wired behind the rejection-sampled contract. World-knowledge bridges
  requiring real external inference are still roadmap.
- Decisions are two-operation policies; richer action spaces
  (multi-step, competing objectives) are roadmap.

## License

MIT — © 2026 Robottwo


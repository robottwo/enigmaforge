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
| Score = final answer | **Trajectory scored**: insight latency, evidence efficiency, objective revision |
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
`results.json`, `results-detailed.json` (raw responses inline), and a
self-contained dark-theme `report.html` with a ranked bar chart:

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
  of `model` (`"moonshotai/kimi-k3"` → `kimi-k3`). `model` is optional
  too: omit it and the endpoint's own `/models` list is probed for the
  newest version (fast tier preferred). `base_url`, `api_key` (literal)
  and `api_key_env` (env var name; mutually exclusive with `api_key`)
  resolve entry > defaults group > env `OPENAI_*` > agent-config
  autodiscovery > defaults.
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
  (`benchmarks/` is ignored by default).
- **Cost & time tracking** — every solver response records wall-clock
  seconds, token usage, and (when the endpoint reports it) dollar cost.
  Set `"track_cost": true` on a provider (or in `provider_defaults`) to
  send OpenRouter's `usage: {"include": true}` flag, which returns
  `usage.cost` per call. Totals land in the leaderboard (`tokens`,
  `cost`, `seconds` columns) and in `results.json`; the report's detailed
  table shows cost per provider.
- `scenarios[].levels` — an ordered **difficulty ladder** that replaces
  the coarse small/medium/large dial: each level may set `size` (a preset
  base), `overrides` (any generator knob — `n_variables`,
  `n_constraints`, `dependency_depth`, `domain_size`, `n_people`,
  `n_bridges`, `n_distractors`, ...), and `burial` (0–12). `instances`
  counts stories *per level* (default 1; each (level, index) pair pins its
  seed, so growing `instances` never re-renders existing stories), and
  levels are walked easiest → hardest. After grading, the report shows a
  level × provider matrix with a ✓ where a level is solved (every story
  answered, mean facts ≥ 0.6) and each provider's highest level reached
  before its first miss — the walk stops there — plus a gap-blind
  `levels_cleared` count.
  `results.json` carries the same breakdown under `ladder`.

Runs are **idempotent** at both stages: existing instance dirs are reused,
never rebuilt, and existing response files are never re-called. Rerun the
same command after an interruption and only the missing pieces execute;
adding a provider or growing a scenario's `instances` creates and calls
only the new combinations. `--grade-only` skips generation and solving
entirely and just re-grades and re-renders the reports. Grading is a fixed
mechanical rubric — ground-truth fact recovery (0.5), similarity of the
final action to the canonical answer (0.3), and response-format compliance
(0.2) — never an LLM judge.

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
4. **Engine ≡ oracle** — the DPLL engine agrees with exhaustive enumeration across a 75-instance differential corpus (committed test battery).
5. **Determinism** — same seed → identical challenge; realization seed changes only the surface, never the solution.
6. **Inert distractors** — narrative red herrings carry no formal content; they can mislead, never invalidate.

## Configurable difficulty

Latent entity count, constraint topology, dependency depth, distractor ratio,
competing-hypothesis count, world-knowledge bridges, narrative length,
information budget, planning horizon — every axis is a config knob, from a
6-variable puzzle to 100+ variables spanning tens of thousands of tokens.

## Status

v0.2 — working end-to-end prototype in two surfaces: the exhibit **record**
and **story mode**, where the instance is embedded in prose with no stated
task. Both run under the same contract: every clue's clause is embedded
verbatim and span-mapped (`realization_map.json`), and an extraction
round-trip re-derives the formal model from the prose alone, re-proving
uniqueness *as read*. Story macro-structure (scene allocation, clue
sequencing, pacing policy) is fixed per instance; realizations vary only
texture. Custom (e.g. LLM) scene renderers plug in behind the same contract
and are rejection-sampled against the gates. See the honest
[Limits](#limits-v2) section. Roadmap: LLM renderer wired to a live
adversarial extractor, underdetermined-scenario mode, subtlety calibration
(pass-rate vs. burial depth). Technical detail:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Limits (v2)

Honest state: both compilers (record and story) are template-based; the LLM
scene renderer is an API hook behind the rejection-sampled contract, not
shipped wired to a model. The extractor is the deterministic inverse of the
template grammar — it covers the constraint kinds the generator emits
(eq/neq/implies/alldiff) and treats out-of-domain matches as scenery; a live
adversarial LLM extractor, testimony gating, underdetermined-scenario
generation, and subtlety calibration are designed but not yet wired.
Technical detail: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License

MIT — © 2026 Robottwo


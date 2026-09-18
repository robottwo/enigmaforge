# EnigmaForge — Technical Architecture

The technical companion to the [README](README.md).

## Modules

```
config ─→ generator ─→ populate ─→ verify ─→ narrative ─→ package
            │                          │          │
   HiddenWorld (HFW)            SAT gates   multiple surface
   ground truth first           + oracle    realizations
```

- `enigmaforge/rng.py` — seeded mulberry32; every artifact reproducible
- `enigmaforge/world.py` — Hidden Formal World: variables, constraints
  (eq/neq/implies/alldiff/exactly-one/arith), evidence units, knowledge
  bridges, staged objectives (apparent → intermediate → true)
- `enigmaforge/generator.py` — ground-truth-first sampling; strengthen to
  uniqueness; minimize (PINs dropped first so relational clues survive);
  chain-depth instrumentation
- `enigmaforge/sat.py` — DPLL engine with unit propagation (branch-local
  assignment copies — propagations must not leak across siblings)
- `enigmaforge/compile.py` — CSP → CNF; negation of (v,val) expands to all
  other domain literals; ARITH bans each violating combination
- `enigmaforge/oracle.py` — brute-force enumeration; the correctness oracle
- `enigmaforge/verify.py` — ban-clause uniqueness (SAT, early exit),
  ablation certificates, engine-vs-oracle differential
- `enigmaforge/populate.py` — evidence units across 9 channels, in-world lore
  references (diegetic only; real-world knowledge grounding: designed, not
  yet wired), staged objectives whose final action follows a public
  conditional decision policy
- `enigmaforge/decisions.py` — conditional register/hold policies
  (target/gate variables, superseded provisional rule), policy text
  rendering, exact typed action validation
- `enigmaforge/grading.py` — benchmark v3 grader: strict structured
  response schema, exact assignment scoring (precision/recall/F1,
  exact-world), policy-validated decisions, no lexical overlap, no raw
  text fallback
- `enigmaforge/judge.py` — blind, cached-only legacy extraction with
  quote validation, review queues, and a human-adjudication calibration
  workflow (`python -m enigmaforge.judge`)
- `enigmaforge/experiments.py` — matched formal/explicit/implicit
  condition expansion over surface realizations, measured structural
  difficulty, deterministic no-API baselines
- `enigmaforge/reporting.py` — coverage-first aggregation (all-item vs
  conditional metrics, family-cluster bootstrap CIs, paired
  condition/model differences), derived diagnostics rendered as unified
  0–100 scores (discovery retention, reasoning discipline, earned
  decisions), inline SVG summary chart, top-line outcome, and the
  offline HTML report
- `enigmaforge/narrative.py` — replaceable compiler returning a `Realization`
  (text + verbatim span map per evidence unit and the public decision
  policy); clauses are natural wrappers
  around extractable cores; surface-noun lexicon assigned at build time,
  unique per variable so extraction is unambiguous
- `enigmaforge/story.py` — story realization: deterministic skeleton (scenes,
  clue sequencing, pacing policy, stakes) from the world seed; scene renderer
  contract with verbatim clause embedding; rejection-sampled compile loop
- `enigmaforge/genres.py` — genre packs (maritime, manor, hotel, theater,
  observatory): surface nouns, places, frames, filler, distractor bodies,
  hypotheses, lore; seeded pick per instance, pinnable via config/CLI;
  construction-time checks keep pack flavor free of variable surfaces
- `enigmaforge/llm.py` — stdlib-only OpenAI-compatible client + scene
  renderer behind the same verbatim-clause contract; `with_metadata`
  envelopes keep visible text, model reasoning, refusals and raw
  responses separate (reasoning is never a content fallback), and carry
  finish reason, usage, and response IDs; `reasoning_max_tokens` caps
  the model's internal reasoning budget via OpenRouter's unified
  `reasoning` object; explicit model/endpoint configurations never
  reroute, fall back, or autodiscover credentials; gates verify every
  model output, rejection-sample failures
- `enigmaforge/interactive.py` — budgeted investigation; irreversible actions
  destroy evidence classes
- `enigmaforge/evaluate.py` — trajectory scoring from the hidden world
- `enigmaforge/pipeline.py` — driver + adaptive verification gates
- `enigmaforge/harness.py` — benchmark v3 driver: hashed independent
  seeds, condition expansion, solver execution with structured transport
  envelopes and per-attempt history, signed corpus manifests
  (fail-closed validation: instance content hashes are the tamper guard;
  response identity binds semantic fields — input, prompt, request
  params — while implementation hashes are recorded for forensics but do
  not block regrading across grading/reporting/driver changes), baseline
  runs, and read-only legacy regrading

## Verification battery (per published instance)

| Gate | Method | Cost |
|---|---|---|
| engine agreement | oracle vs SAT model-set equality | small shapes only |
| uniqueness | ban-clause SAT early exit (UNSAT proof) | any size |
| ablation | drop each clue → must admit second model | any size |
| distractor safety | structural: distractors carry no constraints | free |
| realizations | ≥2 surfaces, identical solution | free |
| realization contract | coverage, verbatim spans, distractor inertness, leak/frame checks | free |
| extraction round-trip | template extractor: prose → constraints → same unique model; scope: supported template claims, not full prose semantics | any size (SAT) |
| decision-policy gate | public rule rendered verbatim; canonical action validates against the policy | free |

## Design invariants

1. **Oracle defines correctness.** The DPLL engine is validated against
   exhaustive enumeration on the committed test battery — never trusted on
   agreement of counts alone (capped
   enumeration fakes disagreement; dict order fakes inequality).
2. **Ground truth first.** Solvability is by construction; uniqueness is
   verified, never assumed.
3. **Every published clue is load-bearing.** Minimization proves no clue can
   be dropped; ablation re-proves it at publish time.
4. **Surface never leaks structure.** Variables render through a fixed
   surface-noun lexicon; formal descriptions never reach the narrative.
5. **Surface faithfulness is verified, not trusted.** Every realization ships
   a span map; each clue's clause must appear verbatim at its span, and the
   extractor must round-trip the prose alone back to exactly the formal model
   with the same unique solution. Story macro-pacing is fixed per instance;
   realizations vary texture only, so surfaces stay difficulty-matched.
6. **Grading is answer-shaped, never lexical.** Scores come from exact
   structured assignments and policy-validated actions; there is no word
   overlap, no raw-text fallback, and no judge in the default path.

## Known bugs caught by the battery (kept for the record)

- unit-propagation state leaking across DPLL branch siblings → 0 models
- EQ-pair compiled as `(A,v)∨(B,v)` instead of the biconditional
- capped model enumeration compared as counts → false disagreements
- cross-domain NEQ anchors compiled to vacuous-truth/unit-clause mismatch
- lazy per-render surface nouns broke realization determinism
- `omission_note` template read "Nothing in the record states {clause}" —
  literally negating the constraint it carried (surfaced while building the
  extraction round-trip)

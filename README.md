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
- the **evidence→constraint map**, external-knowledge facts, distractor annotations;
- a **scoring rubric** computed from the hidden representation, not an LLM judge.

## Quick start

```bash
git clone https://github.com/robottwo/enigmaforge.git
cd enigmaforge
python3 -m enigmaforge.pipeline --size small --seed 2026 --out runs/demo
```

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

v0.1 — working end-to-end prototype: generation, verification, narrative
compilation, interactive mode, trajectory scoring. See the honest
[Limits](#limits-v1) section in the docs. Roadmap: LLM narrative compiler
under the same constraint-evidence contract, underdetermined-scenario mode,
adversarial LLM validation harness.

## Limits (v1)

Honest state of the prototype: the narrative compiler is template-based (an
LLM compiler under the same constraint-evidence contract is the v2 path);
testimony gating, underdetermined-scenario generation, and adversarial LLM
validation are designed but not yet wired. Technical detail:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## License

MIT — © 2026 Robottwo


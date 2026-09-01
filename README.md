# EnigmaForge

Procedural generator of hidden-formal-world strategic-reasoning benchmarks.

A fictional narrative surface (case file, correspondence, records) is *compiled
from* a machine-verified formal instance. The solver must discover the problem,
the latent variables, and the relevant evidence before solving. Uniqueness of
the intended solution is mechanically proven; ablation proves every clue is
load-bearing; distractors are formally inert.

## Architecture

```
config ─→ generator ─→ populate ─→ verify ─→ narrative ─→ package
            │                          │          │
   HiddenWorld (HFW)            SAT gates   multiple surface
   ground truth first           + oracle    realizations
```

- `enigmaforge/rng.py` — seeded mulberry32, all artifacts reproducible
- `world.py` — Hidden Formal World: vars, constraints, evidence units,
  knowledge bridges, staged objectives (apparent → intermediate → true)
- `generator.py` — ground-truth-first sampling; strengthen; minimize
  (PINs dropped first); chain-depth instrumentation
- `verify.py` — SAT gates: uniqueness via ban-clause SAT, ablation
  (each clue removal must admit a second model), engine-vs-oracle
  differential on small shapes
- `oracle.py` — brute-force truth for differential validation
- `narrative.py` — replaceable compiler: channels (letter/receipt/logbook/
  dialogue/marginalia/photo/chronology/omission), surface-noun lexicon,
  deterministic per-realization paraphrase
- `interactive.py` — budgeted investigation; irreversible actions
- `evaluate.py` — trajectory scoring from the hidden representation

## Guarantees (enforced by tests)

1. SAT engine ≡ brute-force oracle on the small-shape corpus (model-set
   equality, order-insensitive).
2. Uniqueness: banning the ground-truth assignment makes the instance UNSAT.
3. Essentiality: removing ANY published clue admits a second model.
4. Determinism: same seed → identical challenge; different realization seed →
   different surface, same hidden structure and solution.
5. Distractors are formally inert by construction (carry no constraints).

## Use

    python3 -m enigmaforge.pipeline --size small|medium|large --seed N --out runs/X

Sizes: small (8 vars), medium (30 vars), large (60 vars, interactive mode).

## Verification battery per published instance

| Gate | Method |
|---|---|
| engine agreement | oracle vs SAT, small shapes |
| uniqueness | ban-clause SAT (early exit) |
| ablation | drop-each-clue → must loosen |
| distractor safety | structural (no formal content) |
| realizations | ≥2 surfaces, same solution |

## Limits (v1, honest)

- Narrative compiler is template-based, not LLM-compiled; prose quality is
  serviceable, not literary. The module is replaceable by design — an LLM
  compiler subject to the same constraint-evidence map is the v2 path.
- Testimony gating (speaker reliability self-reference) is designed but not
  yet wired into generation.
- Underdetermined-scenario mode (recognizing ambiguity is the answer) is
  designed but not yet generated; the verifier supports it (`want_unique=False`).
- Adversarial LLM validation (running solvers against the challenge) is not
  in this prototype.

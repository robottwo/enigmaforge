from .rng import Rng
from .world import *
from .sat import Sat
from .compile import compile_to_sat
from .oracle import oracle_models, check_constraint
from .generator import generate_world
from .populate import populate_evidence, populate_bridges, populate_objectives
from .verify import sat_vs_oracle, verify_uniqueness, verify_ablation, verify_distractor_safety
from .narrative import compile_narrative, Realization
from .verify import verify_realization, verify_roundtrip, extract_claims
from .evaluate import score_trajectory
# `pipeline` doubles as the CLI entry point (`python -m enigmaforge.pipeline`);
# importing it here would make runpy execute the module twice (RuntimeWarning).
# PEP 562 lazy re-export keeps `from enigmaforge import build, ...` working.
_LAZY_EXPORTS = {
    "build": "pipeline", "package": "pipeline", "SIZES": "pipeline",
    "compile_story": "story", "compile_story_verified": "story",
    "build_skeleton": "story",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib
        mod = importlib.import_module("." + _LAZY_EXPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

"""Hidden Formal World (HFW) — the machine-readable model beneath the narrative.

Everything the solver sees is a *rendering* of this structure. The HFW is the
ground truth artifact: variables, constraints, evidence gates, knowledge
bridges, objective stages, and the verified solution.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class VarType(Enum):
    ENUM = "enum"      # finite domain, listed
    BOOL = "bool"
    INT = "int"        # finite range, listed

@dataclass
class Variable:
    vid: str
    vtype: VarType
    domain: list          # for ENUM/INT: list of allowed values; BOOL: [True, False]
    desc: str = ""        # human-facing description (kept hidden from solver)
    surface_names: list = field(default_factory=list)  # synonym class for narrative

@dataclass
class Entity:
    eid: str
    kind: str             # person / place / object / organization / event ...
    name: str             # narrative name (public)
    attrs: dict = field(default_factory=dict)

class ConstraintKind(Enum):
    EQ = "eq"             # var == value, or varA == varB
    NEQ = "neq"
    IMPLIES = "implies"   # (a==va) -> (b==vb)
    ALLDIFF = "alldiff"   # all listed vars pairwise distinct
    EXACTLY_ONE = "exactly_one"  # exactly one of listed (var,value) literals true
    ARITH = "arith"       # linear over bool indicators: sum(coeffs*lits) op rhs

@dataclass
class Constraint:
    cid: str
    kind: ConstraintKind
    vars: list = field(default_factory=list)
    values: list = field(default_factory=list)   # aligned with vars for EQ/NEQ/IMPLIES
    lits: list = field(default_factory=list)     # for EXACTLY_ONE/ARITH: [(vid, val), ...]
    coeffs: list = field(default_factory=list)   # for ARITH
    op: str = "="                           # =, !=, <=, >=  (for ARITH)
    rhs: int = 0
    note: str = ""

@dataclass
class EvidenceUnit:
    """One piece of latent evidence. Rendered to prose by the narrative compiler.

    An evidence unit ENCODES one or more constraints (or fragments). The solver
    must infer that e.g. 'the manifest was signed' encodes  at_location(doc)=harbor.
    Testimony is GATED: its truth is conditional on latent speaker reliability.
    """
    euid: str
    channel: str                      # letter, receipt, logbook, dialogue, marginalia, photo_caption, chronology, omission_note, rule_text
    encodes: list = field(default_factory=list)   # cids this unit carries
    speaker: Optional[str] = None     # entity id if testimony
    subject_entities: list = field(default_factory=list)
    surface: dict = field(default_factory=dict)   # compiler hints (verbs, tone)
    is_distractor: bool = False
    distractor_hypothesis: str = ""   # which false lead this supports

@dataclass
class KnowledgeBridge:
    """External fact required (or seductive). Stored explicitly for ground truth."""
    kbid: str
    fact: str                          # stable, broadly-known fact
    entity_ref: str                    # narrative entity or name mentioned
    role: str = "essential"            # essential | confirmatory | seductive | multi_hop
    encodes: list = field(default_factory=list)   # constraints it grounds
    chain: list = field(default_factory=list)     # for multi_hop: ordered sub-facts

@dataclass
class ObjectiveStage:
    sid: str
    level: int                         # 0 = apparent, 1 = intermediate, ...
    statement: str                     # what it asks, formally
    answer: dict                       # ground truth: {vid: value} fragment
    unlocks: Optional[str] = None      # next stage revealed by solving this
    reveal_text: str = ""              # narrative beat on solution
    true_objective: bool = False

@dataclass
class HiddenWorld:
    wid: str
    seed: int
    config: dict
    entities: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    bridges: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    distractors: list = field(default_factory=list)   # annotations
    meta: dict = field(default_factory=dict)
    # verification results attached post-verify
    verification: dict = field(default_factory=dict)

    def var(self, vid):
        for v in self.variables:
            if v.vid == vid:
                return v
        raise KeyError(vid)

    def constraint(self, cid):
        for c in self.constraints:
            if c.cid == cid:
                return c
        raise KeyError(cid)

    def solution_fragments(self):
        return {o.sid: o.answer for o in self.objectives}

    def public_summary(self):
        return {
            "wid": self.wid, "seed": self.seed,
            "n_entities": len(self.entities),
            "n_variables": len(self.variables),
            "n_constraints": len(self.constraints),
            "n_evidence": len(self.evidence),
            "n_bridges": len(self.bridges),
            "objective_stages": [
                {"sid": o.sid, "level": o.level, "true": o.true_objective}
                for o in sorted(self.objectives, key=lambda o: o.level)
            ],
        }

"""CSP -> CNF compiler. Negation of (v,val) expands to the clause literals
[(v,x) for x in domain if x != val] — works for any domain size."""
from itertools import product
from .world import ConstraintKind
from .sat import Sat

def _neg_literals(lit, sat):
    return [(lit[0], x) for x in sat.var_domain[lit[0]] if x != lit[1]]

def compile_to_sat(world, skip_cids=(), extra_clauses=()):
    sat = Sat()
    for v in world.variables:
        sat.encode_domain(v.vid, v.domain)
    for c in world.constraints:
        if c.cid not in skip_cids:
            _compile_constraint(c, sat)
    for cl in extra_clauses:
        sat.add(cl)
    return sat

def _compile_constraint(c, sat):
    k = c.kind
    if k == ConstraintKind.EQ and len(c.vars) == 1:
        sat.add([(c.vars[0], c.values[0])])
    elif k == ConstraintKind.EQ:  # varA == varB  (biconditional)
        for v in sat.var_domain[c.vars[0]]:
            sat.add(_neg_literals((c.vars[0], v), sat) + [(c.vars[1], v)])
            sat.add(_neg_literals((c.vars[1], v), sat) + [(c.vars[0], v)])
    elif k == ConstraintKind.NEQ:  # never both equal to v
        for v in sat.var_domain[c.vars[0]]:
            sat.add(_neg_literals((c.vars[0], v), sat)
                    + _neg_literals((c.vars[1], v), sat))
    elif k == ConstraintKind.IMPLIES:
        ante = (c.vars[0], c.values[0])
        cons = (c.vars[1], c.values[1])
        sat.add(_neg_literals(ante, sat) + [cons])
    elif k == ConstraintKind.ALLDIFF:
        for i in range(len(c.vars)):
            for j in range(i + 1, len(c.vars)):
                for v in sat.var_domain[c.vars[i]]:
                    sat.add([(c.vars[i], v), (c.vars[j], v)])
    elif k == ConstraintKind.EXACTLY_ONE:
        sat.add(c.lits)
        for i in range(len(c.lits)):
            for j in range(i + 1, len(c.lits)):
                sat.add(_neg_literals(c.lits[i], sat) + _neg_literals(c.lits[j], sat))
    elif k == ConstraintKind.ARITH:
        _compile_arith(c, sat)
    else:
        raise ValueError(k)

def _compile_arith(c, sat):
    lit_vars = sorted({v for (v, _) in c.lits})
    doms = [sat.var_domain[v] for v in lit_vars]
    for combo in product(*doms):
        a = dict(zip(lit_vars, combo))
        total = sum(co for ((v, val), co) in zip(c.lits, c.coeffs) if a[v] == val)
        bad = ((c.op == "=" and total != c.rhs) or (c.op == "!=" and total == c.rhs)
               or (c.op == "<=" and total > c.rhs) or (c.op == ">=" and total < c.rhs))
        if bad:
            # ban combo a: clause false exactly when every var equals its a-value
            sat.add([(v, x) for v in lit_vars
                     for x in sat.var_domain[v] if x != a[v]])

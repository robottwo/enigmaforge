"""Brute-force oracle: exhaustive enumeration, obviously correct.
The DPLL engine must agree with this on every instance."""
from itertools import product
from .world import ConstraintKind

def check_constraint(c, a) -> bool:
    k = c.kind
    if k == ConstraintKind.EQ:
        if len(c.vars) == 1:
            return a[c.vars[0]] == c.values[0]
        return a[c.vars[0]] == a[c.vars[1]]
    if k == ConstraintKind.NEQ:
        return a[c.vars[0]] != a[c.vars[1]]
    if k == ConstraintKind.IMPLIES:
        return a[c.vars[0]] != c.values[0] or a[c.vars[1]] == c.values[1]
    if k == ConstraintKind.ALLDIFF:
        vals = [a[v] for v in c.vars]
        return len(set(vals)) == len(vals)
    if k == ConstraintKind.EXACTLY_ONE:
        return sum(1 for (v, val) in c.lits if a[v] == val) == 1
    if k == ConstraintKind.ARITH:
        total = sum(co for ((v, val), co) in zip(c.lits, c.coeffs) if a[v] == val)
        return {"=": total == c.rhs, "!=": total != c.rhs,
                "<=": total <= c.rhs, ">=": total >= c.rhs}[c.op]
    raise ValueError(k)

def oracle_models(world, constraints=None):
    vids = [v.vid for v in world.variables]
    doms = [v.domain for v in world.variables]
    cs = world.constraints if constraints is None else constraints
    out = []
    for combo in product(*doms):
        a = dict(zip(vids, combo))
        if all(check_constraint(c, a) for c in cs):
            out.append(a)
    return out

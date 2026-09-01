"""CNF SAT engine over (vid, value) literals, with exhaustive model enumeration."""
from itertools import product

class Sat:
    def __init__(self):
        self.clauses = []
        self.var_domain = {}

    def add(self, clause):
        if not clause:
            raise ValueError("empty clause: instance UNSAT by construction")
        self.clauses.append(list(clause))

    def encode_domain(self, vid, domain):
        self.var_domain[vid] = list(domain)

    def _sat(self, assign):
        for cl in self.clauses:
            if not any(assign.get(v) == val for (v, val) in cl):
                return False
        return True

    def _unit_propagate(self, assign, vids, idx):
        """Cheap unit propagation: repeatedly satisfy unit clauses consistent
        with already-assigned vars; abort branch on conflict."""
        changed = True
        while changed:
            changed = False
            for cl in self.clauses:
                if any(assign.get(v) == val for (v, val) in cl):
                    continue
                unassigned = [(v, val) for (v, val) in cl if v not in assign]
                if not unassigned:
                    return False
                if len(unassigned) == 1:
                    v, val = unassigned[0]
                    assign[v] = val
                    changed = True
        return True

    def enumerate_models(self, max_models=2000):
        vids = sorted(self.var_domain)
        models = []
        def dpll(assign, idx):
            if len(models) >= max_models:
                return
            assign = dict(assign)  # branch-local copy: propagations must not leak
            if not self._unit_propagate(assign, vids, idx):
                return
            while idx < len(vids) and vids[idx] in assign:
                idx += 1
            if idx == len(vids):
                if self._sat(assign):
                    models.append(dict(assign))
                return
            vid = vids[idx]
            for val in self.var_domain[vid]:
                child = dict(assign)
                child[vid] = val
                dpll(child, idx)
        dpll({}, 0)
        return models

    def solve(self):
        m = self.enumerate_models(max_models=2)
        return m[0] if m else None

    def count_models(self, cap=2000):
        return len(self.enumerate_models(max_models=1 + cap))

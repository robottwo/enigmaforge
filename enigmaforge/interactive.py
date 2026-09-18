"""Interactive investigation environment: large inspectable world, budget,
irreversible actions. Trajectory recorded for the evaluator."""
from .rng import Rng
from .narrative import render_unit

class InteractiveSession:
    """Solver-visible API. The world holds MORE units than the budget allows:
    choosing what to inspect is itself strategic."""
    def __init__(self, world, budget=20, seed=0):
        self.world = world
        self.budget = budget
        self.rng = Rng(seed + 13)
        self.trajectory = []
        self.seen = set()
        self.closed = set()          # options eliminated by irreversible actions
        self.final_answer = None
        self.policy_text = world.meta["policy_text"]

    def actions(self):
        opts = [("inspect", u.euid) for u in self.world.evidence
                if u.euid not in self.seen and u.euid not in self.closed]
        opts.append(("submit", "final"))
        if self.budget >= 5:
            opts.append(("irreversible:burn_archive", "burn"))
        return opts

    def step(self, action, target):
        """Inspect evidence or submit an {operation, subject, value} action."""
        if self.budget <= 0 and action == "inspect":
            return "BUDGET EXHAUSTED."
        if action == "submit":
            stage = next(o for o in self.world.objectives if o.true_objective)
            rec = {"t": len(self.trajectory), "type": "answer",
                   "payload": {"stage": stage.sid, "answer": {"final_action": target}}}
        else:
            rec = {"t": len(self.trajectory), "type": action, "payload": {"euid": target}}
        self.trajectory.append(rec)
        if action == "inspect":
            for u in self.world.evidence:
                if u.euid == target:
                    self.seen.add(target)
                    self.budget -= 1
                    return render_unit(u, self.world, self.rng.below(10**6))
            return "no such artifact."
        if action.startswith("irreversible"):
            self.budget -= 5
            self.closed.update(e.euid for e in self.world.evidence
                               if e.channel in ("logbook", "receipt"))
            return "The archive burns. Ledger and logbook evidence is now unreachable."
        if action == "submit":
            self.final_answer = target
            return "submitted."
        return "unknown action."

    def score(self):
        from .evaluate import score_trajectory
        return score_trajectory(self.world, self.trajectory)

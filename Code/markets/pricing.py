"""Select a specific price vector among multiple dual-optimal solutions, when the
day-ahead LP is dual-degenerate (ROADMAP.md, "Price formation & dispatch degeneracy").

Builds the explicit LP dual of an already-solved Gurobi model generically, from the
model's own data (`model.getVars()`, `model.getConstrs()`, `model.getCol()`) -- this
works for any Gurobi model, not just this package's day-ahead LP -- then re-solves it
to minimize or maximize a chosen set of dual variables, subject to matching the
primal's own optimal objective value exactly. Any dual-feasible point achieving that
value is, by LP strong duality, itself an optimal dual solution (this is the standard
"second-stage pricing" technique real ISOs use to pick among several prices that are
all consistent with the same optimal dispatch -- see e.g. Gribik et al. on
convex-hull/minimum-uplift pricing; Paul Gribik is among Baldick's acknowledgments).
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB


def _build_dual(model: gp.Model, env: gp.Env) -> tuple[gp.Model, dict[str, gp.Var]]:
    """Build the explicit dual of `model` (already solved to GRB.OPTIMAL), with a
    constraint fixing its objective to `model`'s own optimal value. Returns the dual
    model and a dict mapping each primal constraint's name to its dual variable.
    """
    model.update()
    dual = gp.Model("dual", env=env)
    y: dict[str, gp.Var] = {}
    u: dict[str, gp.Var] = {}
    v: dict[str, gp.Var] = {}

    for c in model.getConstrs():
        if c.Sense == GRB.EQUAL:
            y[c.ConstrName] = dual.addVar(lb=-GRB.INFINITY, name=f"y[{c.ConstrName}]")
        elif c.Sense == GRB.LESS_EQUAL:
            y[c.ConstrName] = dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"y[{c.ConstrName}]")
        else:
            y[c.ConstrName] = dual.addVar(lb=0.0, name=f"y[{c.ConstrName}]")

    for var in model.getVars():
        if var.LB > -GRB.INFINITY:
            u[var.VarName] = dual.addVar(lb=0.0, name=f"u[{var.VarName}]")
        if var.UB < GRB.INFINITY:
            v[var.VarName] = dual.addVar(lb=-GRB.INFINITY, ub=0.0, name=f"v[{var.VarName}]")
    dual.update()

    for var in model.getVars():
        col = model.getCol(var)
        expr = gp.LinExpr()
        for k in range(col.size()):
            expr += col.getCoeff(k) * y[col.getConstr(k).ConstrName]
        if var.VarName in u:
            expr += u[var.VarName]
        if var.VarName in v:
            expr += v[var.VarName]
        dual.addConstr(expr == var.Obj, name=f"stat[{var.VarName}]")

    dual_obj = gp.LinExpr()
    for c in model.getConstrs():
        dual_obj += c.RHS * y[c.ConstrName]
    for var in model.getVars():
        if var.VarName in u:
            dual_obj += var.LB * u[var.VarName]
        if var.VarName in v:
            dual_obj += var.UB * v[var.VarName]
    dual.addConstr(dual_obj == model.ObjVal, name="fix_optimality")

    return dual, y


def tie_break_prices(model: gp.Model, balance: dict, T: int, env: gp.Env, rule: str) -> list[float]:
    """Among all dual-optimal solutions of the solved `model`, pick the one minimizing
    (`rule="min"`) or maximizing (`rule="max"`) the sum of the balance constraints'
    shadow prices. Returns the raw (not yet /dt) price for each interval 0..T-1.
    """
    if rule not in ("min", "max"):
        raise ValueError(f"rule must be 'min' or 'max', got {rule!r}")

    dual, y = _build_dual(model, env)
    sense = GRB.MINIMIZE if rule == "min" else GRB.MAXIMIZE
    dual.setObjective(gp.quicksum(y[balance[t].ConstrName] for t in range(T)), sense)
    dual.optimize()
    if dual.Status != GRB.OPTIMAL:
        raise RuntimeError(f"price tie-break LP failed to solve: Gurobi status {dual.Status}")

    return [y[balance[t].ConstrName].X for t in range(T)]


def price_range_at(model: gp.Model, balance: dict, interval: int, env: gp.Env) -> tuple[float, float]:
    """The full range [min, max] of dual-optimal raw prices at `interval` (0-indexed),
    consistent with `model`'s own optimal objective value. A research/diagnostic tool
    for checking exactly which intervals are genuinely price-degenerate -- see
    ROADMAP.md, "Price formation & dispatch degeneracy" for what this found when run
    against Baldick's own three examples.
    """
    dual, y = _build_dual(model, env)
    yt = y[balance[interval].ConstrName]
    dual.setObjective(yt, GRB.MINIMIZE)
    dual.optimize()
    lo = yt.X
    dual.setObjective(yt, GRB.MAXIMIZE)
    dual.optimize()
    hi = yt.X
    return lo, hi

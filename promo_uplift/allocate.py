"""Who gets which e-mail: greedy rules and an integer program, side by side.

Two greedy rules: `greedy` ranks customers by how much they respond and
cuts at the budget; `regret_greedy` also asks what a customer loses
without their best e-mail, the rule a careful person would use by hand.

Each customer gets at most one e-mail. The predicted uplift of every
customer and e-mail comes out of the uplift model as a fixed number, so
the objective (total predicted uplift) and every constraint (one e-mail
per customer, the budget, a capacity per e-mail, floors per group) are
linear in the yes/no decisions. That makes it an integer program; its LP
relaxation is solved too, to show how far from integer it is.

Two formulations give the same optimum when customers the model cannot
tell apart are grouped:

- `solve_customers`: one binary per customer and e-mail.
- `solve_groups`: one integer per group of identical customers and
  e-mail (how many of the group get it). Exact, not an approximation,
  because customers in a group have the same uplift and the same group
  memberships.
"""

from __future__ import annotations

import warnings

import numpy as np
import pulp


def _solve(prob: pulp.LpProblem) -> str:
    # CBC as bundled with PuLP 3, as in atm-cash; PuLP warns that the
    # bundled solver moves to an extra in PuLP 4.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[prob.status]


def solve_groups(sizes, uplift, budget, cost=None, capacity=None, floors=(), relax=False):
    """Integer program over groups of identical customers.

    sizes:    {group: number of customers}
    uplift:   {(group, arm): predicted uplift per customer}
    budget:   total cost allowed
    cost:     {arm: cost per e-mail}, default 1 for every arm
    capacity: {arm: most e-mails of that arm}, optional
    floors:   iterable of (kind, groups, level) with kind "reach" (at least
              `level` share of those groups' customers get an e-mail) or
              "budget" (at least `level` share of the budget is spent on
              them)
    relax:    solve the LP relaxation (counts may be fractional)

    Returns (counts {(group, arm): n}, objective, status).
    """
    arms = sorted({a for _, a in uplift})
    cost = cost or {a: 1.0 for a in arms}
    cat = "Continuous" if relax else "Integer"
    prob = pulp.LpProblem("allocation", pulp.LpMaximize)
    n = {(g, a): prob.add_variable(f"n_{i}_{j}", 0, sizes[g], cat=cat)
         for i, g in enumerate(sizes) for j, a in enumerate(arms)}
    prob += pulp.lpSum(uplift[k] * v for k, v in n.items())
    for g in sizes:
        prob += pulp.lpSum(n[g, a] for a in arms) <= sizes[g]
    prob += pulp.lpSum(cost[a] * n[g, a] for g in sizes for a in arms) <= budget
    for a, cap in (capacity or {}).items():
        prob += pulp.lpSum(n[g, a] for g in sizes) <= cap
    for kind, groups, level in floors:
        treated = pulp.lpSum(n[g, a] for g in groups for a in arms)
        if kind == "reach":
            prob += treated >= level * sum(sizes[g] for g in groups)
        elif kind == "budget":
            prob += pulp.lpSum(cost[a] * n[g, a] for g in groups for a in arms) >= level * budget
        else:
            raise ValueError(kind)
    status = _solve(prob)
    counts = {k: v.value() or 0.0 for k, v in n.items()}
    return counts, pulp.value(prob.objective), status


def solve_customers(uplift, budget, cost=None, capacity=None, relax=False):
    """The same program with one binary per customer and e-mail.

    uplift: array (customers, arms). Returns (assignment, objective,
    status, fractional) where assignment[i] is the arm index or -1, and
    fractional counts customers whose relaxed solution is not 0/1.
    """
    u = np.asarray(uplift, float)
    n, k = u.shape
    cost = np.ones(k) if cost is None else np.asarray(cost, float)
    cat = "Continuous" if relax else "Binary"
    prob = pulp.LpProblem("allocation", pulp.LpMaximize)
    x = [[prob.add_variable(f"x_{i}_{j}", 0, 1, cat=cat) for j in range(k)] for i in range(n)]
    prob += pulp.lpSum(u[i, j] * x[i][j] for i in range(n) for j in range(k))
    for i in range(n):
        prob += pulp.lpSum(x[i]) <= 1
    prob += pulp.lpSum(cost[j] * x[i][j] for i in range(n) for j in range(k)) <= budget
    for j, cap in (capacity or {}).items():
        prob += pulp.lpSum(x[i][j] for i in range(n)) <= cap
    status = _solve(prob)
    val = np.array([[x[i][j].value() or 0.0 for j in range(k)] for i in range(n)])
    fractional = int(((val > 1e-6) & (val < 1 - 1e-6)).any(axis=1).sum())
    assign = np.where(val.max(axis=1) > 0.5, val.argmax(axis=1), -1)
    return assign, pulp.value(prob.objective), status, fractional


def greedy(uplift, budget, cost=None, capacity=None):
    """Rank and cut: best customers first, each given their best e-mail.

    Customers are sorted by their best uplift per unit of cost. Each gets
    the e-mail with the highest uplift per cost that still has capacity
    and fits the budget, if its uplift is positive. Optimal when every
    e-mail costs the same and nothing but the budget binds; not in
    general.
    """
    u = np.asarray(uplift, float)
    n, k = u.shape
    cost = np.ones(k) if cost is None else np.asarray(cost, float)
    left = {j: np.inf for j in range(k)} | dict(capacity or {})
    ratio = u / cost
    assign = np.full(n, -1)
    spent = 0.0
    for i in np.argsort(-ratio.max(axis=1), kind="stable"):
        for j in np.argsort(-ratio[i], kind="stable"):
            if u[i, j] <= 0:
                break
            if left[j] >= 1 and spent + cost[j] <= budget + 1e-9:
                assign[i] = j
                left[j] -= 1
                spent += cost[j]
                break
    return assign


def _steps_up(u_i, cost):
    """One customer's steps from no e-mail to dearer, better e-mails.

    Returns [(from, to), ...] where -1 is no e-mail. An e-mail is kept on
    the path only if it adds uplift over the one before it, and a step is
    dropped when jumping past it gives more uplift per extra unit of cost,
    so the gain per unit falls along the path.
    """
    def val(j):
        return 0.0 if j < 0 else u_i[j]

    def price(j):
        return 0.0 if j < 0 else cost[j]

    def gain_per_unit(a, b):
        return (val(b) - val(a)) / (price(b) - price(a))

    path = [-1]
    for j in sorted(range(len(u_i)), key=lambda j: (cost[j], -u_i[j])):
        if u_i[j] <= val(path[-1]) or cost[j] == price(path[-1]):
            continue
        while len(path) >= 2 and gain_per_unit(path[-2], j) >= gain_per_unit(path[-2], path[-1]):
            path.pop()
        path.append(j)
    return list(zip(path[:-1], path[1:]))


def regret_greedy(uplift, budget, cost=None, capacity=None):
    """Greedy that asks what each customer loses without their best e-mail.

    Three steps a person could follow by hand:

    1. Spend the budget on steps ranked by uplift gained per unit of
       cost. A step is giving a customer their first e-mail, or moving
       them up to a dearer e-mail that is worth its extra cost. Capacities
       are ignored here.
    2. Where an e-mail went to more customers than its capacity, keep it
       for the ones who lose most if they get their next-best option (the
       other e-mail if it has room and costs no more, else nothing), and
       move the rest to that option.
    3. Spend any budget left with the plain `greedy` on customers who
       still have no e-mail.

    Optimal with equal costs when the budget covers every customer and one
    e-mail has a capacity: the scarce e-mail should go to whoever loses
    most without it, which step 2 does. With a budget alone and equal
    costs it gives the plain greedy's plan. With unequal costs step 1 is
    the textbook greedy for choosing one option per customer under a
    budget: it can only lose at the budget's edge, where the next step
    does not fit whole. Not optimal in general when the budget and a
    capacity both bind: step 1 chooses who is mailed without knowing that
    some of them will lose the scarce e-mail in step 2.
    """
    u = np.asarray(uplift, float)
    n, k = u.shape
    cost = np.ones(k) if cost is None else np.asarray(cost, float)
    capacity = dict(capacity or {})
    assign = np.full(n, -1)

    # step 1: every customer's steps up, ranked together by gain per unit
    steps = []
    for i in range(n):
        for a, b in _steps_up(u[i], cost):
            extra = cost[b] - (0.0 if a < 0 else cost[a])
            gain = u[i, b] - (0.0 if a < 0 else u[i, a])
            steps.append((gain / extra, i, a, b, extra))
    steps.sort(key=lambda s: -s[0])  # stable: ties keep customer order
    spent = 0.0
    for _, i, a, b, extra in steps:
        if assign[i] == a and spent + extra <= budget + 1e-9:
            assign[i] = b
            spent += extra

    # step 2: an over-used e-mail stays with whoever loses most without it
    for j, cap in capacity.items():
        holders = np.flatnonzero(assign == j)
        if len(holders) <= cap:
            continue
        others = [m for m in range(k) if m != j]
        used = {m: int((assign == m).sum()) for m in others}
        fallback, loss = {}, {}
        for i in holders:
            best = -1
            for m in sorted(others, key=lambda m: -u[i, m]):
                if u[i, m] > 0 and cost[m] <= cost[j]:
                    best = m
                    break
            fallback[i] = best
            loss[i] = u[i, j] - (0.0 if best < 0 else u[i, best])
        ranked = sorted(holders, key=lambda i: -loss[i])  # stable
        for i in ranked[cap:]:
            m = fallback[i]
            if m >= 0 and used[m] >= capacity.get(m, np.inf):
                m = -1
            assign[i] = m
            spent += (0.0 if m < 0 else cost[m]) - cost[j]
            if m >= 0:
                used[m] += 1

    # step 3: budget left over goes to customers without an e-mail
    left = {j: capacity.get(j, np.inf) - int((assign == j).sum()) for j in range(k)}
    rest = np.flatnonzero(assign == -1)
    if len(rest) and budget - spent >= cost.min() - 1e-9:
        extra = greedy(u[rest], budget - spent, cost=cost, capacity=left)
        assign[rest] = extra
    return assign


def greedy_reach_floor(uplift, group, budget, level):
    """Greedy with a reach floor per group, equal costs, no capacity.

    Each group first gets its own best customers up to `level` of its
    size; the rest of the budget goes to the best remaining customers
    overall. With equal costs this is optimal (exchange argument).
    """
    u = np.asarray(uplift, float)
    group = np.asarray(group)
    best = u.max(axis=1)
    assign = np.full(len(u), -1)
    used = 0
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        need = int(np.ceil(level * len(idx) - 1e-9))
        take = idx[np.argsort(-best[idx], kind="stable")[:need]]
        assign[take] = u[take].argmax(axis=1)
        used += len(take)
    rest = np.flatnonzero(assign == -1)
    rest = rest[np.argsort(-best[rest], kind="stable")]
    rest = rest[best[rest] > 0][: max(int(budget) - used, 0)]
    assign[rest] = u[rest].argmax(axis=1)
    return assign


def value(uplift, assign):
    """Total predicted uplift of an assignment."""
    u = np.asarray(uplift, float)
    a = np.asarray(assign)
    m = a >= 0
    return float(u[np.flatnonzero(m), a[m]].sum())

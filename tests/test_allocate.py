import numpy as np
import pytest

from promo_uplift.allocate import greedy, greedy_reach_floor, regret_greedy, solve_customers, solve_groups, value

# two e-mails; columns are (A, B)
A_FAN = [0.10, 0.02]   # likes A, B nearly useless
B_OK = [0.08, 0.07]    # likes both about equally


def test_budget_only_greedy_matches_solver():
    rng = np.random.default_rng(3)
    u = rng.normal(0.05, 0.03, (60, 2))
    for budget in (10, 30, 60):
        g = value(u, greedy(u, budget))
        _, obj, status, _ = solve_customers(u, budget)
        assert status == "Optimal"
        assert g == pytest.approx(obj, abs=1e-9)


def test_capacity_breaks_greedy_and_solver_fixes_it():
    u = np.array([[0.09, 0.08], [0.085, 0.01]])  # customer 0 slightly prefers A; customer 1 needs A
    g = greedy(u, budget=2, capacity={0: 1})
    assert g.tolist() == [0, 1]                 # greedy gives the scarce A to customer 0
    assign, obj, _, _ = solve_customers(u, budget=2, capacity={0: 1})
    assert assign.tolist() == [1, 0]            # solver swaps them
    assert obj == pytest.approx(0.08 + 0.085)
    assert obj > value(u, g)


def test_groups_equal_customers_when_customers_are_identical():
    sizes = {"fan": 30, "ok": 50}
    per = {"fan": A_FAN, "ok": B_OK}
    uplift = {(g, a): per[g][a] for g in sizes for a in (0, 1)}
    u = np.array([A_FAN] * 30 + [B_OK] * 50)
    for budget, cap in [(40, None), (80, {0: 35}), (70, {1: 10})]:
        _, obj_g, s1 = solve_groups(sizes, uplift, budget, capacity=cap)
        _, obj_c, s2, _ = solve_customers(u, budget, capacity=cap)
        assert s1 == s2 == "Optimal"
        assert obj_g == pytest.approx(obj_c)


def test_reach_floor_is_met_and_greedy_matches_solver():
    rng = np.random.default_rng(5)
    u = np.abs(rng.normal(0.05, 0.03, (90, 2)))
    group = np.repeat(["x", "y", "z"], 30)
    u[group == "z"] *= 0.2                      # group z responds least
    a = greedy_reach_floor(u, group, budget=30, level=0.3)
    assert (a[group == "z"] >= 0).sum() >= 9
    # solver on the same problem, one group per customer
    sizes = {i: 1 for i in range(90)}
    uplift = {(i, j): u[i, j] for i in range(90) for j in (0, 1)}
    floors = [("reach", [i for i in range(90) if group[i] == g], 0.3) for g in "xyz"]
    _, obj, status = solve_groups(sizes, uplift, 30, floors=floors)
    assert status == "Optimal"
    assert value(u, a) == pytest.approx(obj)


def test_costs_respected():
    u = np.array([[0.10, 0.06]] * 10)
    a = greedy(u, budget=10, cost=[2.0, 1.0])
    spent = sum({0: 2.0, 1: 1.0}[j] for j in a if j >= 0)
    assert spent <= 10
    assign, obj, _, _ = solve_customers(u, budget=10, cost=[2.0, 1.0])
    assert obj >= value(u, a) - 1e-9


def test_negative_uplift_never_assigned():
    u = np.array([[-0.01, -0.02], [0.03, -0.01]])
    assert greedy(u, budget=5).tolist() == [-1, 0]


def test_regret_greedy_gives_the_scarce_email_to_who_loses_most():
    u = np.array([[0.09, 0.08], [0.085, 0.01]])  # same case plain greedy gets wrong above
    a = regret_greedy(u, budget=2, capacity={0: 1})
    assert a.tolist() == [1, 0]
    assert value(u, a) == pytest.approx(0.08 + 0.085)


def test_regret_greedy_matches_solver_when_budget_covers_everyone():
    rng = np.random.default_rng(11)
    u = rng.normal(0.05, 0.04, (80, 2))
    for cap in (0, 10, 40, 80):
        a = regret_greedy(u, budget=80, capacity={0: cap})
        _, obj, status, _ = solve_customers(u, budget=80, capacity={0: cap})
        assert status == "Optimal"
        assert value(u, a) == pytest.approx(obj)


def test_regret_greedy_is_plain_greedy_on_a_budget_alone():
    rng = np.random.default_rng(3)
    u = rng.normal(0.05, 0.03, (60, 2))
    for budget in (10, 30, 60):
        assert regret_greedy(u, budget).tolist() == greedy(u, budget).tolist()


def test_regret_greedy_upgrades_when_it_is_worth_the_extra_cost():
    # e-mail A costs 2, B costs 1. Plain greedy ranks by uplift per unit and
    # sends B to the first customer; moving them up to A adds 0.06 for one
    # unit, more than the 0.03 per unit the second customer is worth.
    u = np.array([[0.16, 0.10], [0.06, 0.01]])
    plain = greedy(u, budget=2, cost=[2.0, 1.0])
    smart = regret_greedy(u, budget=2, cost=[2.0, 1.0])
    _, obj, _, _ = solve_customers(u, budget=2, cost=[2.0, 1.0])
    assert value(u, plain) == pytest.approx(0.10 + 0.01)
    assert smart.tolist() == [0, -1]
    assert value(u, smart) == pytest.approx(obj)


def test_regret_greedy_can_lose_when_budget_and_capacity_both_bind():
    # step 1 mails the two customers who like A most; the capacity then
    # moves one of them to B, where it is worth less than mailing Z.
    u = np.array([[0.10, 0.01], [0.09, 0.02], [0.05, 0.04]])
    a = regret_greedy(u, budget=2, capacity={0: 1})
    _, obj, _, _ = solve_customers(u, budget=2, capacity={0: 1})
    assert value(u, a) == pytest.approx(0.10 + 0.02)
    assert obj == pytest.approx(0.10 + 0.04)


def test_regret_greedy_respects_budget_and_capacity():
    rng = np.random.default_rng(7)
    cost = np.array([2.0, 1.0])
    for _ in range(50):
        u = rng.normal(0.05, 0.04, (40, 2))
        budget, cap = rng.integers(5, 80), rng.integers(0, 40)
        a = regret_greedy(u, budget, cost=cost, capacity={0: cap})
        assert sum(cost[j] for j in a if j >= 0) <= budget + 1e-9
        assert (a == 0).sum() <= cap
        assert (u[np.flatnonzero(a >= 0), a[a >= 0]] > 0).all()

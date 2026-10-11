"""Focused regression checks for the Part B demos; kept outside the published repo."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).parents[1]


def load(folder):
    path = ROOT / folder / "main.py"
    spec = importlib.util.spec_from_file_location(folder, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[folder] = module
    spec.loader.exec_module(module)
    return module


class PartBChecks(unittest.TestCase):
    def test_ao_star_replans_after_road_closure(self):
        m = load("01_ao_star_dynamic_routes")
        actions = {"S": [m.Action("fast", ("A",), 1), m.Action("slow", ("G",), 4)],
                   "A": [m.Action("finish", ("G",), 1)]}
        planner = m.AOPlanner(actions, {"G"}, {"G": 0})
        self.assertEqual(planner.plan("S")[0], 2)
        planner.block_action("S", "fast")
        self.assertEqual(planner.plan("S")[0], 4)

    def test_connect_four_finds_immediate_win(self):
        m = load("02_adversarial_connect_four")
        game = m.ConnectFour()
        for col, player in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (5, 2)]:
            game.play(col, player)
        move, _, _ = game.choose(1, max_depth=4, time_limit=1.0)
        self.assertEqual(move, 3)
        game.play(move, 1)
        self.assertTrue(game.winning(1))
        self.assertEqual(sum(game.heights), 7)

    def test_first_order_unification_and_forward_chaining(self):
        m = load("03_first_order_logic_agent")
        subst = m.unify(m.Atom("P", ("?x", "?y")), m.Atom("P", ("Ada", "?x")))
        self.assertIsNotNone(subst)
        facts, _ = m.infer([m.Atom("Parent", ("Ada", "Bo"))], [
            m.Rule((m.Atom("Parent", ("?x", "?y")),), m.Atom("Ancestor", ("?x", "?y")), "R1")
        ])
        self.assertIn(m.Atom("Ancestor", ("Ada", "Bo")), facts)

    def test_bayesian_decision_probabilities_and_information_value(self):
        m = load("04_bayesian_decision_advisor")
        belief = m.posterior({"a": .6, "b": .4}, [{"a": .8, "b": .2}])
        self.assertAlmostEqual(sum(belief.values()), 1.0)
        self.assertAlmostEqual(belief["a"], .8571428571)
        utility = {"act_a": {"a": 10, "b": -2}, "act_b": {"a": 0, "b": 4}}
        self.assertGreater(m.value_of_information(belief, {"a": {"yes": .9, "no": .1}, "b": {"yes": .2, "no": .8}}, utility, 0), -1)

    def test_bayesian_network_posterior_and_evidence_query(self):
        m = load("05_bayesian_network_engine")
        domains = {"Rain": ("yes", "no"), "Wet": ("yes", "no")}
        factors = [m.Factor(("Rain",), {("yes",): .2, ("no",): .8}),
                   m.Factor(("Wet", "Rain"), {("yes", "yes"): .9, ("no", "yes"): .1,
                                                   ("yes", "no"): .1, ("no", "no"): .9})]
        bn = m.BayesianNetwork(domains, factors)
        result = bn.query("Rain", {"Wet": "yes"})
        self.assertAlmostEqual(result.table[("yes",)], .6923076923)
        self.assertAlmostEqual(bn.query("Rain", {"Rain": "no"}).table[("no",)], 1.0)

    def test_expert_system_combines_support(self):
        m = load("06_explainable_expert_system")
        beliefs, trace = m.infer({"start": .8, "lights": .7}, [
            m.Rule("R1", frozenset({"start", "lights"}), "battery", .9, "inspect battery")
        ])
        self.assertAlmostEqual(beliefs["battery"], .63)
        self.assertIn("R1", trace["battery"][0])

    def test_cbs_paths_avoid_vertex_and_edge_conflicts(self):
        m = load("07_multi_robot_cbs")
        grid = [[0] * 5 for _ in range(3)]
        result = m.cbs(grid, [(1, 0), (1, 4)], [(1, 4), (1, 0)])
        self.assertIsNotNone(result)
        paths, _ = result
        self.assertIsNone(m.first_conflict(paths))

    def test_nsga_front_is_unique_and_nondominated(self):
        m = load("08_multiobjective_delivery_optimizer")
        routes = [(0,), (1,), (2,), (3,)]
        scores = [(1, 5), (3, 3), (5, 1), (4, 4)]
        ranks, _ = m.rank_population(routes, scores)
        self.assertEqual(ranks, [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)

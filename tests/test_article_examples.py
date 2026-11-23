import csv
import unittest
from itertools import combinations
from fractions import Fraction as Q
from random import Random

import networkx as nx

from article_core import GraphMetrics, ReplayPredictor, Weights, singleton, singleton_path
from run_article_examples import (
    REFERENCE, admissible, canonical_edges, make_grid, one_episode,
    run_set_valued, run_singletons, singleton_reward,
)


class ArticleExamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls.positions = make_grid(0)

    def test_graph_exactly_matches_manuscript(self):
        with (REFERENCE / "figure2_edges.csv").open(newline="") as f:
            expected = [(int(r["u"]), int(r["v"])) for r in csv.DictReader(f)]
        self.assertEqual(len(self.graph), 100)
        self.assertEqual(self.graph.number_of_edges(), 144)
        self.assertTrue(nx.is_connected(self.graph))
        self.assertEqual(canonical_edges(self.graph), expected)

    def test_historical_trajectory_objective(self):
        with (REFERENCE / "figure2_trajectory.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        path = [int(r["vertex"]) for r in rows]
        values = [int(r["f"]) for r in rows]
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-2:], [99, 99])
        self.assertEqual(values, [0, 2, 3, 5, 7, 9, 11, 13, 14, 15,
                                  16, 17, 19, 21, 22, 24, 25, 26, 27, 27])
        self.assertTrue(all(self.graph.has_edge(u, v)
                            for u, v in zip(path[:-2], path[1:-1])))
        self.assertEqual(singleton_reward(path[:-1], path[-1], self.positions),
                         Q(416, 21))

    def test_exact_trajectory_distance_and_tail(self):
        graph = nx.path_graph(3)
        phi = singleton_path([0, 1])
        psi = singleton_path([0, 1, 2])
        self.assertEqual(GraphMetrics(graph, Weights()).trajectory_distance(phi, psi), Q(1, 4))
        self.assertEqual(GraphMetrics(graph, Weights(5)).trajectory_distance(phi, psi), Q(3, 5))
        repeated = singleton_path([0, 1, 1])
        self.assertEqual(GraphMetrics(graph, Weights()).trajectory_distance(phi, repeated), 0)

    def test_set_valued_example(self):
        values = run_set_valued(self.graph)
        self.assertEqual(values["coordinate_hausdorff_distances"], [0, 1, 1])
        self.assertEqual(values["eta_H"]["exact"], "1/2")
        self.assertEqual(values["R_D_phi"]["exact"], "-9/4")
        self.assertEqual(values["R_D_psi"]["exact"], "-5/2")
        self.assertEqual(values["absolute_reward_difference"]["exact"], "1/4")

    def test_incremental_seminorm_batch_and_interpolation(self):
        metrics = GraphMetrics(nx.path_graph(4), Weights())
        replay = ReplayPredictor(metrics)
        # Two batches, including an exact duplicate.
        for v in (0, 1):
            replay.observe(singleton_path([v]), singleton(v), Q(v*v))
        replay.update()
        self.assertEqual(replay.pair_comparisons, 1)
        self.assertFalse(replay.observe(singleton_path([0]), singleton(0), Q(0)))
        for v in (2, 3):
            replay.observe(singleton_path([v]), singleton(v), Q(v*v))
        replay.update()
        self.assertEqual(replay.pair_comparisons, 6)
        self.assertEqual(replay.L, replay.full_seminorm())
        for row in replay.records:
            self.assertEqual(replay.predict(row.trajectory, row.action), row.reward)
        self.assertEqual(replay.pair_comparisons, 6)

    def test_zero_distance_consistency(self):
        metrics = GraphMetrics(nx.path_graph(2), Weights())
        phi, psi = singleton_path([0]), singleton_path([0, 0])
        a = singleton(1)
        replay = ReplayPredictor(metrics)
        replay.observe(phi, a, Q(2))
        replay.observe(psi, a, Q(2))
        self.assertEqual(replay.update(), 0)
        bad = ReplayPredictor(metrics)
        bad.observe(phi, a, Q(2))
        bad.observe(psi, a, Q(3))
        with self.assertRaisesRegex(ValueError, "zero trajectory"):
            bad.update()

    def test_one_record_and_ties(self):
        metrics = GraphMetrics(nx.path_graph(3), Weights())
        replay = ReplayPredictor(metrics)
        phi = singleton_path([0])
        replay.observe(phi, singleton(1), Q(5))
        self.assertEqual(replay.predict(phi, singleton(2)), 5)
        self.assertEqual(replay.choose_greedy(phi, [singleton(0), singleton(1)], Random(0)),
                         singleton(1))

    def test_admissible_actions_and_terminal_repeat(self):
        self.assertEqual(admissible(self.graph, [0]), [1, 10, 0])
        self.assertNotIn(0, admissible(self.graph, [0, 10]))
        # A fixed RNG that always chooses the last action chooses stop at
        # the first transition. The terminal repeat is not queried again.
        class StopRNG(Random):
            def choice(self, values):
                return values[-1]
        replay = ReplayPredictor(GraphMetrics(self.graph, Weights(100)))
        ep = one_episode(self.graph, self.positions, replay, StopRNG(0), .05, 100, True)
        self.assertEqual(ep["path"], [0, 0])
        self.assertEqual(ep["stop"], "terminal_repeat")
        self.assertEqual(len(replay.records), 1)

    def test_default_run_checkpoint_and_final_buffer(self):
        out = run_singletons(self.graph, self.positions, epochs=15)
        cp = out["checkpoint_execution"]
        self.assertEqual(cp["vertices"][-1]["f"], 8)
        self.assertEqual(cp["empirical_seminorm"]["exact"], "375/14")
        self.assertEqual(cp["buffer_records"], 17)
        self.assertEqual(cp["pair_comparisons"], 136)
        self.assertEqual(len(out["history"]), 15)
        self.assertTrue(all(len(row["rewards"]) == row["steps"] for row in out["history"]))
        self.assertTrue(all(row["reward"] for row in out["snapshots"]["final"]))


if __name__ == "__main__":
    unittest.main()

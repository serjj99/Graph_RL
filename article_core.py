"""Exact finite-graph metrics and McShane--Whitney reward learning.

The examples use rational rewards and weights so equality of action scores
is not dependent on floating-point round-off.  Only vertex distances are
cached (on demand); comparisons between replay records are not cached.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from random import Random
from typing import FrozenSet, Tuple

import networkx as nx


State = FrozenSet[int]
Trajectory = Tuple[State, ...]


def singleton(vertex: int) -> State:
    return frozenset({vertex})


def singleton_path(vertices) -> Trajectory:
    return tuple(singleton(vertex) for vertex in vertices)


@dataclass(frozen=True)
class Weights:
    """Uniform weights on a finite horizon, or 2**(-i) for i >= 1."""

    horizon: int | None = None

    def __post_init__(self):
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("The uniform-weight horizon must be positive.")

    def weight(self, i: int) -> Fraction:
        if i < 1:
            raise ValueError("Weights are indexed starting at 1.")
        if self.horizon is None:
            return Fraction(1, 2**i)
        return Fraction(1, self.horizon) if i <= self.horizon else Fraction(0)

    def tail(self, h: int) -> Fraction:
        """Sum of weights with index greater than h."""
        if h < 0:
            raise ValueError("The tail index cannot be negative.")
        if self.horizon is None:
            return Fraction(1, 2**h)
        return Fraction(max(self.horizon - h, 0), self.horizon)


class GraphMetrics:
    """Path metric, Hausdorff distance and terminal-padded trajectory distance."""

    def __init__(self, graph: nx.Graph, weights: Weights):
        if graph.is_directed() or not graph or not nx.is_connected(graph):
            raise ValueError("A nonempty connected undirected graph is required.")
        self.graph = nx.freeze(graph.copy())
        self.weights = weights
        self._vertex_distances: dict[tuple[int, int], int] = {}

    def path_distance(self, u: int, v: int) -> int:
        key = tuple(sorted((u, v)))
        if key not in self._vertex_distances:
            self._vertex_distances[key] = nx.shortest_path_length(self.graph, u, v)
        return self._vertex_distances[key]

    def hausdorff(self, A: State, B: State) -> int:
        if not A or not B:
            raise ValueError("Hausdorff states must be nonempty.")
        if not A <= self.graph.nodes or not B <= self.graph.nodes:
            raise ValueError("Every state vertex must belong to the graph.")
        if A == B:
            return 0
        # The singleton case is exactly the ordinary graph path metric.
        if len(A) == len(B) == 1:
            return self.path_distance(next(iter(A)), next(iter(B)))
        return max(
            max(min(self.path_distance(x, y) for y in B) for x in A),
            max(min(self.path_distance(x, y) for x in A) for y in B),
        )

    def _check_trajectory(self, phi: Trajectory):
        if not phi:
            raise ValueError("A trajectory must have at least one state.")
        if self.weights.horizon is not None and len(phi) > self.weights.horizon:
            raise ValueError("A trajectory exceeds the fixed weight support.")
        if any(not A or not A <= self.graph.nodes for A in phi):
            raise ValueError("Trajectory states must be nonempty graph subsets.")

    def trajectory_distance(self, phi: Trajectory, psi: Trajectory) -> Fraction:
        self._check_trajectory(phi)
        self._check_trajectory(psi)
        h = max(len(phi), len(psi))
        # Beyond h both trajectories repeat their terminal sets. Evaluate
        # the tail exactly instead of constructing an infinite sequence.
        distance = sum(
            (
                self.weights.weight(i)
                * self.hausdorff(
                    phi[min(i - 1, len(phi) - 1)],
                    psi[min(i - 1, len(psi) - 1)],
                )
                for i in range(1, h + 1)
            ),
            Fraction(0),
        )
        return distance + self.weights.tail(h) * self.hausdorff(phi[-1], psi[-1])

    def nu(self, phi: Trajectory, a: State, psi: Trajectory, b: State) -> Fraction:
        return self.trajectory_distance(phi, psi) + self.hausdorff(a, b)


@dataclass(frozen=True)
class Record:
    trajectory: Trajectory
    action: State
    reward: Fraction


class ReplayPredictor:
    """Append-only replay buffer with incremental empirical seminorm."""

    def __init__(self, metrics: GraphMetrics):
        self.metrics = metrics
        self.records: list[Record] = []
        self._observed: dict[tuple[Trajectory, State], Fraction] = {}
        self._processed = 0
        self.L = Fraction(0)
        self.pair_comparisons = 0

    def observe(self, phi: Trajectory, a: State, reward: Fraction) -> bool:
        self.metrics._check_trajectory(phi)
        if not a or not a <= self.metrics.graph.nodes:
            raise ValueError("Actions must be nonempty subsets of the graph.")
        reward = Fraction(reward)
        key = (phi, a)
        if key in self._observed:
            if self._observed[key] != reward:
                raise ValueError("Different rewards for the same trajectory-action pair.")
            return False
        self.records.append(Record(phi, a, reward))
        self._observed[key] = reward
        return True

    def _ratio(self, x: Record, y: Record) -> Fraction:
        d = self.metrics.nu(x.trajectory, x.action, y.trajectory, y.action)
        diff = abs(x.reward - y.reward)
        if d == 0:
            if diff:
                raise ValueError("Inconsistent rewards at zero trajectory-action distance.")
            return Fraction(0)
        return diff / d

    def update(self) -> Fraction:
        """Compare each new record with all earlier records, once per pair."""
        while self._processed < len(self.records):
            i = self._processed
            x = self.records[i]
            for j in range(i):
                self.L = max(self.L, self._ratio(x, self.records[j]))
                self.pair_comparisons += 1
            self._processed += 1
        return self.L

    def full_seminorm(self) -> Fraction:
        """Independent reference calculation; not used during training."""
        return max(
            (self._ratio(x, y) for x, y in combinations(self.records, 2)),
            default=Fraction(0),
        )

    def extensions(self, phi: Trajectory, a: State) -> tuple[Fraction, Fraction]:
        if not self.records:
            raise ValueError("At least one observed reward is required.")
        self.update()
        lower = upper = None
        for record in self.records:
            radius = self.L * self.metrics.nu(phi, a, record.trajectory, record.action)
            lo, hi = record.reward - radius, record.reward + radius
            lower = lo if lower is None else max(lower, lo)
            upper = hi if upper is None else min(upper, hi)
        return lower, upper

    def predict(self, phi: Trajectory, a: State) -> Fraction:
        lower, upper = self.extensions(phi, a)
        return (lower + upper) / 2

    def choose_greedy(self, phi: Trajectory, actions: list[State], rng: Random) -> State:
        if not actions:
            raise ValueError("No admissible actions.")
        self.update()
        scores = [self.predict(phi, a) for a in actions]
        best = max(scores)
        tied = [a for a, value in zip(actions, scores) if value == best]
        # Do not consume an RNG draw if there is no tie.
        return rng.choice(tied) if len(tied) > 1 else tied[0]

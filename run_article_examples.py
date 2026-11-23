"""Run and export the two examples for the revised JCO manuscript."""

import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from fractions import Fraction
from pathlib import Path
from random import Random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from article_core import GraphMetrics, ReplayPredictor, Weights, singleton, singleton_path


ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"
GRID_SIZE = 10
SOURCE = 0
REFERENCE = ROOT / "reference"


def make_grid(seed: int = 0, fraction_removed: Fraction = Fraction(1, 5)):
    """Preserve the grid/edge order of the original graph-generation routine."""
    if not 0 <= fraction_removed <= 1:
        raise ValueError("The removal fraction must be between 0 and 1.")
    base = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
    positions = {
        x * GRID_SIZE + y: (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
    }
    mapping = {xy: xy[0] * GRID_SIZE + xy[1] for xy in base}
    graph = nx.relabel_nodes(base, mapping)
    edges = list(graph.edges)
    target = int(len(edges) * fraction_removed)
    if target > len(edges) - len(graph) + 1:
        raise ValueError("Cannot remove that many edges and preserve connectivity.")
    rng = Random(seed)
    removed = 0
    while removed < target:
        edge = rng.choice(edges)
        graph.remove_edge(*edge)
        if nx.is_connected(graph):
            edges.remove(edge)
            removed += 1
        else:
            graph.add_edge(*edge)
    return graph, positions


def canonical_edges(graph):
    return sorted(tuple(sorted(edge)) for edge in graph.edges)


def objective(vertex: int, positions: dict[int, tuple[int, int]]) -> int:
    x, y = positions[vertex]
    return 2 * x + y


def singleton_reward(path, action, positions) -> Fraction:
    n = len(path)
    weighted = sum(i * objective(v, positions) for i, v in enumerate(path, 1))
    return Fraction(2 * weighted, (n + 1) * (n + 2)) + Fraction(
        2 * objective(action, positions), n + 2
    )


def admissible(graph, path):
    """Sorted unvisited neighbors followed by the current vertex (stop)."""
    current = path[-1]
    return [v for v in sorted(graph[current]) if v not in path] + [current]


def one_episode(graph, positions, predictor, rng, epsilon, max_steps, training):
    path = [SOURCE]
    rewards = []
    exploration_steps = 0
    reason = "step_limit"
    for _ in range(max_steps):
        actions = admissible(graph, path)
        if training and (not predictor.records or rng.random() < epsilon):
            action = rng.choice(actions)
            exploration_steps += 1
        else:
            chosen = predictor.choose_greedy(
                singleton_path(path), [singleton(v) for v in actions], rng
            )
            action = next(iter(chosen))
        reward = singleton_reward(path, action, positions)
        if training:
            predictor.observe(singleton_path(path), singleton(action), reward)
        rewards.append(reward)
        current = path[-1]
        path.append(action)
        if action == current:
            reason = "terminal_repeat"
            break
    return {
        "path": path,
        "reward": rewards[-1],
        "rewards": rewards,
        "steps": len(rewards),
        "stop": reason,
        "exploration_steps": exploration_steps,
    }


def fraction_value(value: Fraction) -> dict:
    return {"exact": str(value), "decimal": float(value)}


def export_path(episode, positions):
    return {
        "vertices": [
            {"step": i, "vertex": v, "x": positions[v][0], "y": positions[v][1],
             "f": objective(v, positions)}
            for i, v in enumerate(episode["path"])
        ],
        "stop": episode["stop"],
        "steps": episode["steps"],
        "terminal_reward": fraction_value(episode["reward"]),
        "rewards": [fraction_value(r) for r in episode["rewards"]],
        "exploration_steps": episode["exploration_steps"],
    }


def run_singletons(graph, positions, *, epochs=50, checkpoint=15, max_steps=100,
                   epsilon=0.05, train_seed=0, execution_seed=0):
    if not 1 <= checkpoint <= epochs:
        raise ValueError("Require 1 <= checkpoint <= epochs.")
    if not 1 <= max_steps <= len(graph):
        raise ValueError("The step limit must be in [1, |V|].")
    if not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be in [0, 1].")
    metrics = GraphMetrics(graph, Weights(horizon=len(graph)))
    predictor = ReplayPredictor(metrics)
    rng = Random(train_seed)
    history = []
    evaluation = None
    evaluation_metadata = {}
    snapshots = {}
    for epoch in range(1, epochs + 1):
        ep = one_episode(graph, positions, predictor, rng, epsilon, max_steps, True)
        history.append({"epoch": epoch, **export_path(ep, positions)})
        if epoch == checkpoint:
            # Synchronization with the entire buffer is needed before the
            # final predictor is used. Evaluation consumes a separate RNG.
            predictor.update()
            assert predictor.L == predictor.full_seminorm()
            assert predictor.pair_comparisons == len(predictor.records) * (len(predictor.records) - 1) // 2
            evaluation = one_episode(
                graph, positions, predictor, Random(execution_seed), 0.0,
                max_steps, False
            )
            evaluation_metadata = {
                "epochs_trained": epoch,
                "buffer_records": len(predictor.records),
                "empirical_seminorm": fraction_value(predictor.L),
                "pair_comparisons": predictor.pair_comparisons,
            }
            snapshots["checkpoint"] = [
                export_record(record) for record in predictor.records
            ]
    predictor.update()
    snapshots["final"] = [export_record(record) for record in predictor.records]
    final_evaluation = one_episode(
        graph, positions, predictor, Random(execution_seed), 0.0, max_steps, False
    )
    # The reference calculation verifies both value and counting of pairs.
    assert predictor.L == predictor.full_seminorm()
    m = len(predictor.records)
    assert predictor.pair_comparisons == m * (m - 1) // 2
    return {
        "history": history,
        "checkpoint_execution": {**evaluation_metadata, **export_path(evaluation, positions)},
        "final_execution": {
            "epochs_trained": epochs,
            "buffer_records": m,
            "empirical_seminorm": fraction_value(predictor.L),
            "pair_comparisons": predictor.pair_comparisons,
            **export_path(final_evaluation, positions),
        },
        "snapshots": snapshots,
    }


def export_record(record):
    return {
        "trajectory": [sorted(A) for A in record.trajectory],
        "action": sorted(record.action),
        "reward": str(record.reward),
    }


def run_set_valued(graph):
    """The analytic example of Remark 3, using exact infinite tails."""
    o, b, c, z = 0, 10, 1, 11
    for u, v in [(o, b), (o, c), (b, z), (c, z)]:
        if not graph.has_edge(u, v):
            raise ValueError("The graph does not contain the square of Remark 3.")
    target = frozenset({o, b, z})
    other = frozenset({o, c, z})
    phi = (singleton(o), frozenset({o, b}), target)
    psi = (singleton(o), frozenset({o, c}), other)
    action = frozenset({o, b, c, z})
    metrics = GraphMetrics(graph, Weights())
    weights = metrics.weights

    def u(A):
        return -metrics.hausdorff(A, target)

    def reward(trajectory, a):
        return (
            sum(
                (weights.weight(i) * u(A) for i, A in enumerate(trajectory, 1)),
                Fraction(0),
            )
            + weights.tail(len(trajectory)) * u(trajectory[-1]) + u(a)
        )

    distances = [metrics.hausdorff(A, B) for A, B in zip(phi, psi)]
    eta = metrics.trajectory_distance(phi, psi)
    nu = metrics.nu(phi, action, psi, action)
    r_phi, r_psi = reward(phi, action), reward(psi, action)
    predictor = ReplayPredictor(metrics)
    predictor.observe(phi, action, r_phi)
    predictor.observe(psi, action, r_psi)
    predictor.update()
    assert distances == [0, 1, 1]
    assert eta == nu == Fraction(1, 2)
    assert r_phi == Fraction(-9, 4) and r_psi == Fraction(-5, 2)
    assert abs(r_phi - r_psi) <= nu
    assert predictor.L == Fraction(1, 2)
    assert predictor.predict(phi, action) == r_phi
    assert predictor.predict(psi, action) == r_psi
    return {
        "vertex_coordinates": {str(v): [v // GRID_SIZE, v % GRID_SIZE] for v in [o, b, c, z]},
        "phi": [sorted(A) for A in phi],
        "psi": [sorted(A) for A in psi],
        "target": sorted(target),
        "action": sorted(action),
        "coordinate_hausdorff_distances": distances,
        "padded_tail_hausdorff_distance": distances[-1],
        "weights": "w_i = 2^(-i); the tail after h equals 2^(-h)",
        "eta_H": fraction_value(eta),
        "nu": fraction_value(nu),
        "phi_scores": [u(A) for A in phi],
        "psi_scores": [u(A) for A in psi],
        "action_score": u(action),
        "R_D_phi": fraction_value(r_phi),
        "R_D_psi": fraction_value(r_psi),
        "absolute_reward_difference": fraction_value(abs(r_phi - r_psi)),
        "empirical_seminorm": fraction_value(predictor.L),
        "lipschitz_upper_bound": 1,
        "verified": True,
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def reference_comparison(graph, evaluation):
    with (REFERENCE / "figure2_edges.csv").open(newline="", encoding="utf-8") as f:
        ref_edges = [(int(r["u"]), int(r["v"])) for r in csv.DictReader(f)]
    with (REFERENCE / "figure2_trajectory.csv").open(newline="", encoding="utf-8") as f:
        ref_path = [int(r["vertex"]) for r in csv.DictReader(f)]
    path = [p["vertex"] for p in evaluation["vertices"]]
    return {
        "graph_matches_manuscript": canonical_edges(graph) == ref_edges,
        "checkpoint_trajectory_matches_manuscript": path == ref_path,
        "historical_figure2_terminal_vertex": ref_path[-1],
        "historical_figure2_terminal_value": 27,
        "new_checkpoint_terminal_vertex": path[-1],
        "new_checkpoint_terminal_value": evaluation["vertices"][-1]["f"],
        "note": "Reference data were transcribed from the manuscript. They do not identify the original training RNG or reproduce the historical reward progression.",
    }


def plots(output, graph, positions, results, checkpoint):
    history = results["history"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([r["epoch"] for r in history],
            [r["terminal_reward"]["decimal"] for r in history], "o-",
            markersize=3, color="tab:blue")
    ax.axvline(checkpoint, linestyle="--", color="0.45", label=f"Epoch {checkpoint}")
    ax.set(xlabel="Training epoch", ylabel="Last observed reward",
           title="Reproducible singleton run: terminal reward per epoch")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reward_progression.png", dpi=180)
    plt.close(fig)

    path_data = results["checkpoint_execution"]["vertices"]
    path = [r["vertex"] for r in path_data]
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="0.65", width=0.8)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=10, node_color="black")
    nx.draw_networkx_edges(graph, positions, ax=ax,
                           edgelist=[(u, v) for u, v in zip(path, path[1:]) if u != v],
                           edge_color="tab:blue", width=2.5)
    nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=[path[0]],
                           node_size=65, node_color="red")
    nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=[path[-1]],
                           node_size=70, node_color="tab:green")
    nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=[99], node_size=100,
                           node_color="none", edgecolors="tab:orange", linewidths=1.5)
    ax.set_title(f"Greedy execution after {checkpoint} epochs (new run)")
    ax.legend(handles=[
        Line2D([0], [0], color="tab:blue", lw=2.5, label="Executed path"),
        Line2D([0], [0], marker="o", color="red", linestyle="none", label="Source"),
        Line2D([0], [0], marker="o", color="tab:green", linestyle="none", label="Terminal vertex"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="tab:orange",
               linestyle="none", label="Global maximizer"),
    ], loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2, frameon=False)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "checkpoint_trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([r["step"] for r in path_data], [r["f"] for r in path_data],
            "o-", color="tab:blue")
    ax.axhline(27, color="tab:orange", linestyle="--", label="Global maximum")
    ax.set(xlabel="Execution step", ylabel="f(x,y)",
           title=f"Vertex objectives after {checkpoint} epochs (new run)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "checkpoint_objective.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "article_run")
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--execution-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()
    graph, positions = make_grid(args.graph_seed)
    results = run_singletons(
        graph, positions, epochs=args.epochs, checkpoint=args.checkpoint,
        max_steps=args.max_steps, epsilon=args.epsilon, train_seed=args.train_seed,
        execution_seed=args.execution_seed,
    )
    # Remark 3 is defined on the manuscript graph, independent of the
    # optional graph seed used by a numerical sensitivity run.
    remark_graph, _ = make_grid(0)
    set_results = run_set_valued(remark_graph)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "graph_edges.csv", ["u", "v"], canonical_edges(graph))
    write_csv(output / "reward_progression.csv",
              ["epoch", "terminal_reward_exact", "terminal_reward", "steps",
               "exploration_steps", "stop"],
              [(r["epoch"], r["terminal_reward"]["exact"], r["terminal_reward"]["decimal"],
                r["steps"], r["exploration_steps"], r["stop"]) for r in results["history"]])
    write_csv(output / "checkpoint_trajectory.csv", ["step", "vertex", "x", "y", "f"],
              [(r["step"], r["vertex"], r["x"], r["y"], r["f"])
               for r in results["checkpoint_execution"]["vertices"]])
    write_json(output / "episodes.json", results["history"])
    write_json(output / "checkpoint_execution.json", results["checkpoint_execution"])
    write_json(output / "final_execution.json", results["final_execution"])
    write_json(output / "replay_buffers.json", results["snapshots"])
    write_json(output / "set_valued_example.json", set_results)
    comparisons = reference_comparison(graph, results["checkpoint_execution"])
    write_json(output / "reference_comparison.json", comparisons)
    write_json(output / "manifest.json", {
        "implementation_version": VERSION,
        "python": platform.python_version(),
        "networkx": nx.__version__,
        "matplotlib": matplotlib.__version__,
        "dependencies": {
            line.split("==", 1)[0]: version(line.split("==", 1)[0])
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if "==" in line and not line.startswith("#")
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("article_core.py", "run_article_examples.py")
        },
        "graph_seed": args.graph_seed, "training_seed": args.train_seed,
        "execution_seed": args.execution_seed, "epochs": args.epochs,
        "checkpoint": args.checkpoint, "max_steps": args.max_steps,
        "epsilon": args.epsilon, "vertices": len(graph), "edges": graph.number_of_edges(),
        "objective": "f(x,y) = 2*x + y",
        "reward_statistic": "Exact observed reward of the last executed transition of each epoch.",
        "weights": "Uniform, 1/|V| for coordinates 1..|V|; terminal padding.",
        "set_example_graph_seed": 0,
        "ties": "Exact maximizer equality, uniform random tie-breaking; no draw for singleton argmax.",
        "stop": "Repeat current vertex or reach the step limit.",
        "historical_match": comparisons,
    })
    plots(output, graph, positions, results, args.checkpoint)
    print("Graph:", len(graph), "vertices,", graph.number_of_edges(), "edges")
    print("Matches manuscript graph:", comparisons["graph_matches_manuscript"])
    print("New checkpoint terminal objective:", comparisons["new_checkpoint_terminal_value"])
    print("Matches historical trajectory:", comparisons["checkpoint_trajectory_matches_manuscript"])
    print("Remark 3:", set_results["R_D_phi"]["exact"], set_results["R_D_psi"]["exact"])
    print("Results:", output.resolve())


if __name__ == "__main__":
    main()

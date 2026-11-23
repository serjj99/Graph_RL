"""Optional small sensitivity check; all five fixed seeds are reported."""

import argparse
from pathlib import Path

from run_article_examples import ROOT, make_grid, run_singletons, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "seed_check.csv")
    args = parser.parse_args()
    graph, positions = make_grid(seed=0)
    rows = []
    for seed in range(5):
        result = run_singletons(graph, positions, train_seed=seed)
        cp = result["checkpoint_execution"]
        final = result["final_execution"]
        x, y = cp["vertices"][-1], final["vertices"][-1]
        rows.append([
            seed, x["x"], x["y"], x["f"], cp["buffer_records"],
            y["x"], y["y"], y["f"], final["buffer_records"],
        ])
        print(seed, x["f"], y["f"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, [
        "training_seed", "x_after_15", "y_after_15", "f_after_15",
        "records_after_15", "x_after_50", "y_after_50", "f_after_50",
        "records_after_50",
    ], rows)


if __name__ == "__main__":
    main()

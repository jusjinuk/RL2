"""
Calculate pass@1 metrics for both MultiPL-E and PDDL datasets.

For MultiPL-E: checks status == "OK" and exit_code == 0
For PDDL: checks specified metric (parseable/valid/equivalent)

Output columns: Dataset, Pass@1, NumProblems, MinCompletions, MaxCompletions
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def estimator(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def load_result(path: Path):
    """Load result from JSON or gzipped JSON file."""
    open_fn = gzip.open if path.suffix == '.gz' else open
    with open_fn(path, 'rt') as f:
        return json.load(f)


def for_file(path: Path, metric: str = "equivalent"):
    """Calculate pass@1 for a single problem file.

    Handles both MultiPL-E format (results key) and PDDL format (eval_results key).
    """
    data = load_result(path)
    if data is None:
        return None

    # Check for MultiPL-E format
    if "results" in data:
        n = len(data["results"])
        c = sum(1 for r in data["results"]
                if r.get("status") == "OK" and r.get("exit_code") == 0)
    # Check for PDDL format
    elif "eval_results" in data and data["eval_results"]:
        n = len(data["eval_results"])
        c = sum(1 for r in data["eval_results"] if r.get(metric, False))
    else:
        return None

    return {
        "pass@1": estimator(n, c, 1),
        "n": n,
        "c": c
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate pass@1 for MultiPL-E and PDDL problems')
    parser.add_argument("dirs", type=str, nargs="+", help="Directories with results")
    parser.add_argument("--suppress-header", action="store_true", help="Suppress CSV header")
    parser.add_argument("--metric", type=str, default="equivalent",
                       choices=["parseable", "valid", "equivalent"],
                       help="PDDL metric to evaluate (default: equivalent)")

    args = parser.parse_args()

    if not args.suppress_header:
        print("Dataset,Pass@1,NumProblems,MinCompletions,MaxCompletions")

    for d in args.dirs:
        # Look for both MultiPL-E and PDDL file patterns
        results = [for_file(p, args.metric) for p in Path(d).glob("*.json*")]
        results = [r for r in results if r is not None]

        if not results:
            continue

        name = Path(d).name
        num_problems = len(results)
        min_completions = min(r["n"] for r in results)
        max_completions = max(r["n"] for r in results)

        pass_1 = np.mean([r["pass@1"] for r in results])
        print(f"{name},{pass_1:.4f},{num_problems},{min_completions},{max_completions}")


if __name__ == "__main__":
    main()

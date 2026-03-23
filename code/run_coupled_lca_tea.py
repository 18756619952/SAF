"""
Coupled workflow runner for the manuscript-aligned LCA and TEA models.

This script:
1. Runs the LCA model and captures the explicit output directory returned by it.
2. Runs the TEA model into the same output directory.
3. Reports the shared results location.

"""

from __future__ import annotations

import argparse
from pathlib import Path

from lca_model import run_lca
from tea_model import run_tea_model


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def run_coupled_workflow(out_dir: str | Path | None = None, verbose: bool = False) -> Path:
    """
    Run the LCA model followed by the TEA model and return the shared output directory.
    """
    target_dir = Path(out_dir).resolve() if out_dir is not None else None

    run_dir_str = run_lca(
        out_dir=str(target_dir) if target_dir is not None else None,
        verbose=verbose,
    )
    run_dir = Path(run_dir_str).resolve()

    run_tea_model(out_dir=run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the coupled SAF LCA-TEA workflow.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Optional output directory. If omitted, the LCA model creates a timestamped "
            "results/run_* directory that is then reused by the TEA model."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostic information from the LCA stage.",
    )
    args = parser.parse_args()

    run_dir = run_coupled_workflow(out_dir=args.out_dir, verbose=args.verbose)
    print(f"Coupled LCA-TEA workflow completed. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()

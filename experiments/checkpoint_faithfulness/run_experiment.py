"""
Main experiment orchestrator for checkpoint faithfulness experiment.

This script coordinates:
1. Training all 40 meta-networks (4 datasets x 2 conditions x 5 seeds)
2. Running faithfulness evaluation for each meta-network
3. Aggregating results into a summary

Usage:
    # Run full experiment
    python run_experiment.py --all

    # Run only training
    python run_experiment.py --train-only

    # Run only evaluation (assumes training is done)
    python run_experiment.py --eval-only

    # Run for specific dataset
    python run_experiment.py --dataset mnist
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from train_metanetworks import train_all_metanetworks, DATASETS, CONDITIONS, DEFAULT_SEEDS
from run_evaluation import run_evaluation


def run_full_experiment(
    datasets: Optional[list[str]] = None,
    conditions: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    n_models: int = 100,
    max_steps: int = 10000,
    lr: float = 0.1,
    output_base_dir: str = ".",
    skip_existing: bool = True,
    device: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run the full checkpoint faithfulness experiment.

    Args:
        datasets: List of datasets (default: all 4)
        conditions: List of conditions (default: both)
        seeds: List of seeds (default: 5 seeds)
        n_models: Number of models per evaluation
        max_steps: Max unlearning steps
        lr: Learning rate
        output_base_dir: Base directory for outputs
        skip_existing: Skip existing models/evaluations
        device: Training device
        verbose: Verbose output

    Returns:
        Aggregated results DataFrame
    """
    datasets = datasets or DATASETS
    conditions = conditions or CONDITIONS
    seeds = seeds or DEFAULT_SEEDS

    metanetworks_dir = Path(output_base_dir) / "metanetworks"
    results_dir = Path(output_base_dir) / "results"

    # Phase 1: Train meta-networks
    print("\n" + "=" * 60)
    print("PHASE 1: Training Meta-Networks")
    print("=" * 60)

    train_all_metanetworks(
        datasets=datasets,
        conditions=conditions,
        seeds=seeds,
        output_dir=str(metanetworks_dir),
        device=device,
        verbose=verbose,
        skip_existing=skip_existing,
    )

    # Phase 2: Run evaluations
    print("\n" + "=" * 60)
    print("PHASE 2: Running Faithfulness Evaluations")
    print("=" * 60)

    all_results = []

    total_evals = len(datasets) * len(conditions) * len(seeds)
    with tqdm(total=total_evals, desc="Evaluations") as pbar:
        for dataset in datasets:
            for condition in conditions:
                for seed in seeds:
                    condition_slug = "final_only" if condition == "final-only" else "multi_stage"
                    model_path = metanetworks_dir / condition_slug / f"{dataset}_seed{seed}.pt"
                    result_path = results_dir / f"{dataset}_{condition_slug}_seed{seed}.csv"

                    # Skip if results exist
                    if skip_existing and result_path.exists():
                        print(f"Skipping existing: {result_path}")
                        df = pd.read_csv(result_path)
                        all_results.append(df)
                        pbar.update(1)
                        continue

                    if not model_path.exists():
                        print(f"Warning: Model not found: {model_path}")
                        pbar.update(1)
                        continue

                    print(f"\nEvaluating: {dataset} | {condition} | seed={seed}")

                    df = run_evaluation(
                        meta_network_path=str(model_path),
                        dataset=dataset,
                        condition=condition,
                        seed=seed,
                        n_models=n_models,
                        output_dir=str(results_dir),
                        max_steps=max_steps,
                        lr=lr,
                    )
                    all_results.append(df)
                    pbar.update(1)

    # Phase 3: Aggregate results
    print("\n" + "=" * 60)
    print("PHASE 3: Aggregating Results")
    print("=" * 60)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = results_dir / "all_results.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined results saved to: {combined_path}")

        # Create summary
        summary = create_summary(combined_df)
        summary_path = results_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")

        return combined_df

    return pd.DataFrame()


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics from results."""
    # Group by dataset and condition
    summary = df.groupby(["dataset", "condition"]).agg(
        n_samples=("model_idx", "count"),
        n_unique_models=("model_idx", "nunique"),
        mean_initial_mae=("initial_mae", "mean"),
        std_initial_mae=("initial_mae", "std"),
        mean_final_mae=("final_mae", "mean"),
        std_final_mae=("final_mae", "std"),
        mean_total_steps=("total_steps", "mean"),
        mean_distance_travelled=("distance_travelled", "mean"),
    ).reset_index()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run checkpoint faithfulness experiment."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full experiment (train + evaluate).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train meta-networks.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluations (assumes training done).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        help="Run for specific dataset only.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=CONDITIONS,
        help="Run for specific condition only.",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=100,
        help="Number of models to evaluate per condition.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum unlearning steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for unlearning.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    args = parser.parse_args()

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Determine what to run
    datasets = [args.dataset] if args.dataset else None
    conditions = [args.condition] if args.condition else None

    if args.train_only:
        train_all_metanetworks(
            datasets=datasets,
            conditions=conditions,
            output_dir="metanetworks",
            device=args.device,
            verbose=args.verbose,
            skip_existing=not args.force,
        )
    elif args.eval_only:
        # Run evaluations only
        datasets = datasets or DATASETS
        conditions = conditions or CONDITIONS
        seeds = DEFAULT_SEEDS

        for dataset in datasets:
            for condition in conditions:
                for seed in seeds:
                    condition_slug = "final_only" if condition == "final-only" else "multi_stage"
                    model_path = f"metanetworks/{condition_slug}/{dataset}_seed{seed}.pt"
                    result_path = f"results/{dataset}_{condition_slug}_seed{seed}.csv"

                    if not args.force and Path(result_path).exists():
                        print(f"Skipping existing: {result_path}")
                        continue

                    if not Path(model_path).exists():
                        print(f"Model not found: {model_path}")
                        continue

                    run_evaluation(
                        meta_network_path=model_path,
                        dataset=dataset,
                        condition=condition,
                        seed=seed,
                        n_models=args.n_models,
                        output_dir="results",
                        max_steps=args.max_steps,
                        lr=args.lr,
                    )
    else:
        # Run full experiment
        run_full_experiment(
            datasets=datasets,
            conditions=conditions,
            n_models=args.n_models,
            max_steps=args.max_steps,
            lr=args.lr,
            skip_existing=not args.force,
            device=args.device,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_runs(csv_pattern, label):
    files = sorted(Path(csv_pattern).parent.glob(Path(csv_pattern).name))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["class"] = int(f.stem.split("class")[-1])
        df["source"] = label
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files matched pattern: {csv_pattern}")
    df = pd.concat(frames, ignore_index=True)
    before = df["base_target_acc"]
    after = df["best_target_acc"]
    non_target_drop = df["nontarget_drop"]
    df["clipped_negative_mean_difference"] = np.maximum(0, before - after) - np.maximum(0, non_target_drop)
    return df


def summarize(df):
    return (
        df.groupby("class")
        .agg(
            mean_target_drop=("target_drop", "mean"),
            std_target_drop=("target_drop", "std"),
            mean_clip=("clipped_negative_mean_difference", "mean"),
            std_clip=("clipped_negative_mean_difference", "std"),
            success_rate=("target_drop", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )


def plot_comparison(summary_ens, summary_single, out_path):
    classes = summary_ens["class"]
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    # # target_drop bars
    # ax1.bar(classes - 1.5 * width, summary_ens["mean_target_drop"], width=width, yerr=summary_ens["std_target_drop"], capsize=3, label="Ensemble target drop", color="steelblue")
    # ax1.bar(classes - 0.5 * width, summary_single["mean_target_drop"], width=width, yerr=summary_single["std_target_drop"], capsize=3, label="Single target drop", color="skyblue")

    # clipped metric bars
    ax1.bar(
        classes + 0.5 * width,
        summary_ens["mean_clip"],
        width=width,
        yerr=summary_ens["std_clip"],
        capsize=3,
        label="Ensemble clipped",
        color="seagreen",
    )
    ax1.bar(
        classes + 1.5 * width,
        summary_single["mean_clip"],
        width=width,
        yerr=summary_single["std_clip"],
        capsize=3,
        label="Single clipped",
        color="mediumseagreen",
    )

    ax1.set_xlabel("Class")
    ax1.set_ylabel("Score")
    ax1.axhline(0, color="k", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(classes, summary_ens["success_rate"], color="orange", marker="o", label="Ensemble success")
    ax2.plot(classes, summary_single["success_rate"], color="red", marker="s", label="Single success")
    ax2.set_ylabel("Success rate")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ensemble vs single metanet unlearning results.")
    parser.add_argument(
        "--ensemble_pattern",
        default="experiments/ensemble_unlearning/data/ensemble_holdout/fashion_mnist_class*.csv",
    )
    parser.add_argument(
        "--single_pattern",
        default="experiments/ensemble_unlearning/data/single_self/fashion_mnist_single_self_class*.csv",
    )
    parser.add_argument(
        "--single_holdout_pattern",
        default=None,
        help="Optional: single-with-holdout CSVs",
    )
    parser.add_argument(
        "--out",
        default="experiments/ensemble_unlearning/plots/fashion_mnist_ensemble_vs_single_min0.6_std.png",
    )
    parser.add_argument(
        "--out_single_compare",
        default="experiments/ensemble_unlearning/plots/fashion_mnist_single_holdout_vs_single_self_min0.6_std.png",
    )
    args = parser.parse_args()

    df_ens = load_runs(args.ensemble_pattern, "ensemble")
    df_single = load_runs(args.single_pattern, "single_self")
    summary_ens = summarize(df_ens)
    summary_single = summarize(df_single)
    plot_comparison(summary_ens, summary_single, args.out)

    # Optional: compare single-with-holdout vs single-self-scored
    if args.single_holdout_pattern:
        df_single_holdout = load_runs(args.single_holdout_pattern, "single_holdout")
        summary_holdout = summarize(df_single_holdout)
        plot_comparison(summary_holdout, summary_single, args.out_single_compare)


if __name__ == "__main__":
    main()

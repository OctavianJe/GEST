from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histograms for all metrics from scores.csv."
    )
    parser.add_argument(
        "--scores", default="results/Narrative Similarity Task/metrics/scores.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="results/Narrative Similarity Task/metrics/histograms",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores)
    out_dir = Path(args.output_dir)
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores.csv at {scores_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scores_path)
    metrics = sorted({c[:-2] for c in df.columns if c.endswith("_a")})

    for metric in metrics:
        a_col = f"{metric}_a"
        b_col = f"{metric}_b"
        if a_col not in df.columns or b_col not in df.columns:
            continue

        # A vs B histogram
        plt.figure(figsize=(6, 4))
        plt.hist(df[a_col], bins=50, alpha=0.6, label="A")
        plt.hist(df[b_col], bins=50, alpha=0.6, label="B")
        plt.title(f"Score distribution: {metric}")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{metric}.png")
        plt.close()

        # Diff histogram
        diff = df[a_col] - df[b_col]
        plt.figure(figsize=(6, 4))
        plt.hist(diff, bins=50, alpha=0.8)
        plt.title(f"Score diff (A-B): {metric}")
        plt.xlabel("score_a - score_b")
        plt.ylabel("count")
        plt.axvline(0, color="black", linewidth=1)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_diff_{metric}.png")
        plt.close()

    print(f"Saved {len(metrics) * 2} plots in {out_dir}")


if __name__ == "__main__":
    main()

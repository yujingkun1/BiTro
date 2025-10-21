#!/usr/bin/env python3
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend for servers


LOGS_DIR = Path(__file__).parent


def parse_metrics_file(file_path: Path) -> Optional[Dict[str, float]]:
    """Parse a metrics text file and return required metrics.

    Expected lines include:
      - Overall MSE: <float>
      - Overall Correlation: <float>
      - Mean Gene Correlation: <float>
    """
    overall_mse: Optional[float] = None
    overall_correlation: Optional[float] = None
    mean_gene_correlation: Optional[float] = None

    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Overall MSE:"):
            try:
                overall_mse = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Overall Correlation:"):
            try:
                overall_correlation = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Mean Gene Correlation:"):
            try:
                mean_gene_correlation = float(line.split(":", 1)[1].strip())
            except Exception:
                pass

    if (
        overall_mse is None
        or overall_correlation is None
        or mean_gene_correlation is None
    ):
        return None

    return {
        "overall_mse": overall_mse,
        "overall_correlation": overall_correlation,
        "mean_gene_correlation": mean_gene_correlation,
    }


def load_all_metrics(logs_dir: Path) -> List[Dict[str, float]]:
    metrics_list: List[Dict[str, float]] = []
    for file_path in sorted(logs_dir.glob("fold_*_metrics.txt")):
        parsed = parse_metrics_file(file_path)
        if parsed is None:
            continue
        # attach fold number if present
        match = re.search(r"fold_(\d+)_metrics\\.txt$", file_path.name)
        if match:
            parsed["fold"] = int(match.group(1))
        else:
            parsed["fold"] = None
        metrics_list.append(parsed)
    return metrics_list


def compute_means(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {"overall_mse": float("nan"), "overall_correlation": float("nan"), "mean_gene_correlation": float("nan")}

    def safe_mean(values: List[float]) -> float:
        if not values:
            return float("nan")
        return sum(values) / len(values)

    return {
        "overall_mse": safe_mean([m["overall_mse"] for m in metrics_list]),
        "overall_correlation": safe_mean([m["overall_correlation"] for m in metrics_list]),
        "mean_gene_correlation": safe_mean([m["mean_gene_correlation"] for m in metrics_list]),
    }


def save_csv(metrics_list: List[Dict[str, float]], out_path: Path) -> None:
    if not metrics_list:
        return
    # sort by fold if available
    sorted_list = sorted(
        metrics_list,
        key=lambda m: (m["fold"] is None, -1 if m["fold"]
                       is None else m["fold"]),
    )
    header = ["fold", "overall_mse",
              "overall_correlation", "mean_gene_correlation"]
    lines = [",".join(header)]
    for m in sorted_list:
        fold_str = "" if m["fold"] is None else str(m["fold"])
        line = [
            fold_str,
            f"{m['overall_mse']:.6f}",
            f"{m['overall_correlation']:.6f}",
            f"{m['mean_gene_correlation']:.6f}",
        ]
        lines.append(",".join(line))
    out_path.write_text("\n".join(lines) + "\n")


def save_summary_text(means: Dict[str, float], count: int, out_path: Path) -> None:
    content = (
        "=== Aggregated Metrics Summary ===\n"
        f"Num folds: {count}\n"
        f"Mean Overall MSE: {means['overall_mse']:.6f}\n"
        f"Mean Overall Correlation: {means['overall_correlation']:.6f}\n"
        f"Mean Gene Correlation: {means['mean_gene_correlation']:.6f}\n"
    )
    out_path.write_text(content)


def plot_boxplots(metrics_list: List[Dict[str, float]], out_path: Path) -> None:
    if not metrics_list:
        return
    mse_values = [m["overall_mse"] for m in metrics_list]
    corr_values = [m["overall_correlation"] for m in metrics_list]
    gene_corr_values = [m["mean_gene_correlation"] for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].boxplot(mse_values, showmeans=True)
    axes[0].set_title("Overall MSE")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].boxplot(corr_values, showmeans=True)
    axes[1].set_title("Overall Correlation")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    axes[2].boxplot(gene_corr_values, showmeans=True)
    axes[2].set_title("Mean Gene Correlation")
    axes[2].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Metrics Distribution Across Folds")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    metrics_list = load_all_metrics(LOGS_DIR)
    if not metrics_list:
        print("No valid metrics found in logs directory.")
        return

    means = compute_means(metrics_list)

    # Print summary to stdout
    print(f"Found {len(metrics_list)} folds with valid metrics.")
    print(f"Mean Overall MSE: {means['overall_mse']:.6f}")
    print(f"Mean Overall Correlation: {means['overall_correlation']:.6f}")
    print(f"Mean Gene Correlation: {means['mean_gene_correlation']:.6f}")

    # Persist artifacts
    save_csv(metrics_list, LOGS_DIR / "metrics_per_fold.csv")
    save_summary_text(means, len(metrics_list),
                      LOGS_DIR / "metrics_summary.txt")
    plot_boxplots(metrics_list, LOGS_DIR / "metrics_boxplots.png")


if __name__ == "__main__":
    main()



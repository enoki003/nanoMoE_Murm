import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt

ITER_PATTERN = re.compile(r"iter (\d+): loss ([0-9.]+)")
ROUTING_PATTERN = re.compile(r"routing balance: min ([0-9.]+), max ([0-9.]+)")
STEP_PATTERN = re.compile(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")


def _load_lines(log_path: Path) -> Iterable[str]:
    raw = log_path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Unable to decode log file with UTF-8/UTF-16 variants", b"", 0, 1, "decode error")

    return text.splitlines()


def parse_log(log_path: Path) -> Tuple[List[dict], List[dict]]:
    iter_rows: List[dict] = []
    eval_rows: List[dict] = []

    lines = list(_load_lines(log_path))

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        step_match = STEP_PATTERN.match(line)
        if step_match:
            eval_rows.append(
                {
                    "step": int(step_match.group(1)),
                    "train_loss": float(step_match.group(2)),
                    "val_loss": float(step_match.group(3)),
                }
            )
            i += 1
            continue

        iter_match = ITER_PATTERN.match(line)
        if iter_match:
            iter_row = {
                "iter": int(iter_match.group(1)),
                "loss": float(iter_match.group(2)),
                "routing_min": None,
                "routing_max": None,
            }

            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                routing_match = ROUTING_PATTERN.match(next_line)
                if routing_match:
                    iter_row["routing_min"] = float(routing_match.group(1))
                    iter_row["routing_max"] = float(routing_match.group(2))
                    i += 1  # skip routing line handled here

            iter_rows.append(iter_row)

        i += 1

    return iter_rows, eval_rows


def write_csv(rows: List[dict], headers: List[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def plot_loss(iter_rows: List[dict], eval_rows: List[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot([row["iter"] for row in iter_rows], [row["loss"] for row in iter_rows], label="iter loss", linewidth=1.2)

    if eval_rows:
        ax.scatter(
            [row["step"] for row in eval_rows],
            [row["val_loss"] for row in eval_rows],
            color="tab:orange",
            label="val loss",
            zorder=5,
        )
        ax.scatter(
            [row["step"] for row in eval_rows],
            [row["train_loss"] for row in eval_rows],
            color="tab:green",
            label="train loss (avg)",
            zorder=4,
            marker="x",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("MoE training loss trend")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_routing(iter_rows: List[dict], output_path: Path) -> None:
    routing_min = [row["routing_min"] for row in iter_rows if row["routing_min"] is not None]
    routing_max = [row["routing_max"] for row in iter_rows if row["routing_max"] is not None]

    if not routing_min or not routing_max:
        return

    iters = [row["iter"] for row in iter_rows if row["routing_min"] is not None]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, routing_min, label="min load", linewidth=1.1)
    ax.plot(iters, routing_max, label="max load", linewidth=1.1)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="ideal 0.5")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expert load fraction")
    ax.set_title("Routing load balance (per-iter)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MoE training log for loss and routing balance trends.")
    parser.add_argument("--log", type=Path, required=True, help="Path to the captured training log file.")
    parser.add_argument(
        "--outdir", type=Path, default=None, help="Directory to place CSV summaries and plots (defaults to log directory)."
    )
    args = parser.parse_args()

    log_path: Path = args.log
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    outdir = args.outdir or log_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    iter_rows, eval_rows = parse_log(log_path)

    if not iter_rows:
        raise RuntimeError("No iteration metrics were parsed from the log. Check log formatting.")

    iter_csv = outdir / "moe_iter_metrics.csv"
    eval_csv = outdir / "moe_eval_metrics.csv"
    loss_plot = outdir / "moe_loss.png"
    routing_plot = outdir / "moe_routing.png"

    write_csv(iter_rows, ["iter", "loss", "routing_min", "routing_max"], iter_csv)
    if eval_rows:
        write_csv(eval_rows, ["step", "train_loss", "val_loss"], eval_csv)
    else:
        eval_csv = None

    plot_loss(iter_rows, eval_rows, loss_plot)
    plot_routing(iter_rows, routing_plot)

    print(f"Wrote iteration metrics to {iter_csv}")
    if eval_csv is not None:
        print(f"Wrote eval metrics to {eval_csv}")
    print(f"Saved loss trend figure to {loss_plot}")
    print(f"Saved routing balance figure to {routing_plot}")


if __name__ == "__main__":
    main()

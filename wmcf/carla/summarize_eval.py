"""Summarise CarDreamer evaluation logs for the report.

``eval_dm3.sh`` writes ``metrics.jsonl`` into the evaluation logdir.  This
script turns the tail of that file into a small table, CSV/JSON and an optional
figure.  It works on any flat JSONL produced by the CarDreamer / DreamerV3
logger and does not require CARLA or JAX.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_PATTERN = (
    r"(collision|destination_reached|out_of_lane|time_exceeded|"
    r"ttc|speed|travel_distance|wpt_dis|score|length)"
)
RATE_PATTERN = r"(collision|out_of_lane|time_exceeded|destination_reached)"
CONTINUOUS_PATTERN = r"(ttc|speed|travel_distance|wpt_dis|score|length)"


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    if not records:
        raise ValueError(f"no JSON objects found in {path}")
    return records


def summarise(records: list[dict], pattern: str, tail: float) -> dict[str, dict]:
    regex = re.compile(pattern)
    series: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if key == "step" or not regex.search(key):
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                series.setdefault(key, []).append(float(value))

    summary: dict[str, dict] = {}
    for key, values in sorted(series.items()):
        keep = max(1, int(round(len(values) * tail)))
        values = values[-keep:]
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "n": int(array.size),
        }
    return summary


def print_table(summary: dict[str, dict]) -> None:
    if not summary:
        print("[warn] no matching scalar keys found")
        return
    width = max(len(key) for key in summary)
    print(f"{'key'.ljust(width)}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}  {'n':>5}")
    for key, stat in summary.items():
        print(
            f"{key.ljust(width)}  {stat['mean']:10.4f}  {stat['std']:10.4f}  "
            f"{stat['min']:10.4f}  {stat['max']:10.4f}  {stat['n']:5d}"
        )


def save_csv(summary: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "mean", "std", "min", "max", "n"])
        for key, stat in summary.items():
            writer.writerow([key, stat["mean"], stat["std"], stat["min"], stat["max"], stat["n"]])


def save_figure(summary: dict[str, dict], path: Path) -> bool:
    rates = {k: v for k, v in summary.items() if re.search(RATE_PATTERN, k)}
    continuous = {k: v for k, v in summary.items() if re.search(CONTINUOUS_PATTERN, k)}
    if not rates and not continuous:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = int(bool(rates)) + int(bool(continuous))
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 3.6))
    if panels == 1:
        axes = [axes]

    index = 0
    if rates:
        ax = axes[index]
        index += 1
        names = list(rates)
        values = [rates[name]["mean"] for name in names]
        ax.barh(names, values, color="#c0504d")
        ax.set_xlabel("rate")
        ax.set_title("Safety events")
        ax.grid(axis="x", alpha=0.3)
    if continuous:
        ax = axes[index]
        names = list(continuous)
        values = [continuous[name]["mean"] for name in names]
        ax.barh(names, values, color="#4f81bd")
        ax.set_xlabel("mean")
        ax.set_title("Efficiency / dynamics")
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True, help="CarDreamer eval logdir")
    parser.add_argument("--file", default="metrics.jsonl", help="log filename inside logdir")
    parser.add_argument("--out", default="summary", help="output directory")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="key regex")
    parser.add_argument("--tail", type=float, default=0.2, help="fraction of last values to keep")
    args = parser.parse_args(argv)

    logdir = Path(args.logdir)
    logfile = logdir / args.file
    if not logfile.exists():
        raise FileNotFoundError(logfile)

    records = load_jsonl(logfile)
    summary = summarise(records, args.pattern, args.tail)
    print(f"[log] {logfile}: {len(records)} records")
    print_table(summary)

    out_dir = logdir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_csv(summary, out_dir / "summary.csv")
    if save_figure(summary, out_dir / "summary.png"):
        print(f"[out] {out_dir / 'summary.png'}")
    print(f"[out] {out_dir / 'summary.csv'}")
    print(f"[out] {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

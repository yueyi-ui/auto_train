"""Evaluator: extracts the best epoch metrics from an Ultralytics results.csv."""

import csv
import math
from pathlib import Path


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate(run_dir, metric):
    results = Path(run_dir) / "results.csv"
    if not results.exists():
        return None
    with results.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = None
    for row in rows:
        val = parse_float(row.get(metric))
        if val is None or not math.isfinite(val):
            continue
        if best is None or val > best[0]:
            best = (val, row)
    if best is None:
        return None
    return {"fitness": best[0], "metrics": best[1], "best_epoch": best[1]["epoch"]}

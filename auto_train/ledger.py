"""Experiment ledger: persistent JSON history of tuning trials."""

import json
import time
from pathlib import Path


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.trials = []
        if self.path.exists():
            self.trials = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.trials, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, trial):
        trial["id"] = len(self.trials) + 1
        trial["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.trials.append(trial)
        self.save()
        return trial

    def history(self):
        return [
            {
                "id": t["id"],
                "config": t.get("config", {}),
                "fitness": t.get("fitness"),
                "metrics": t.get("metrics", {}),
                "status": t.get("status", "unknown"),
            }
            for t in self.trials
        ]

    def best(self):
        scored = [t for t in self.trials if isinstance(t.get("fitness"), float)]
        return max(scored, key=lambda t: t["fitness"]) if scored else None

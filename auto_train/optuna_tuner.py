"""Optuna proposer: Bayesian optimization without an LLM."""

import optuna
from pathlib import Path


class OptunaTuner:
    def __init__(self, storage, study_name="auto_train"):
        storage = str(storage).replace("\\", "/")
        if not storage.startswith("sqlite:///"):
            storage = "sqlite:///" + storage
        db_path = storage[len("sqlite:///"):]
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
        )
        self._pending = None

    def propose(self, history, search_space, defaults):
        trial = self.study.ask()
        raw = {}
        for key, spec in search_space.items():
            if isinstance(spec, dict):
                if key == "lr0":
                    raw[key] = trial.suggest_float(key, spec["min"], spec["max"], log=True)
                elif key in ("epochs", "warmup", "patience"):
                    raw[key] = trial.suggest_int(key, int(spec["min"]), int(spec["max"]))
                else:
                    raw[key] = trial.suggest_float(key, spec["min"], spec["max"])
            else:
                raw[key] = trial.suggest_categorical(key, spec)
        self._pending = trial
        return raw

    def report(self, fitness):
        if self._pending is None:
            return
        value = fitness if isinstance(fitness, (int, float)) else float("-inf")
        self.study.tell(self._pending, value)
        self._pending = None

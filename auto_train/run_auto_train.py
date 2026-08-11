"""Minimal LLM-in-the-loop auto training loop.

Usage:
    python run_auto_train.py --iterations 3 --mock
    python run_auto_train.py --iterations 1 --epochs 1 --fraction 0.005 --imgsz 320
"""

import argparse
import csv
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

from evaluator import evaluate
from ledger import Ledger
from trainer import Trainer
from tuner import Tuner
from validator import Validator

BASE = Path(__file__).resolve().parent
LOG_PATH = None


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    local = BASE / "config.local.json"
    if local.exists():
        cfg.update(json.loads(local.read_text(encoding="utf-8")))
    return cfg


def random_config(validator, space, defaults, history, rng):
    for _ in range(10):
        raw = {}
        for key, spec in space.items():
            if isinstance(spec, dict):
                if key == "lr0":
                    raw[key] = 10 ** rng.uniform(math.log10(spec["min"]), math.log10(spec["max"]))
                elif key in ("epochs", "warmup", "patience"):
                    raw[key] = rng.randint(int(spec["min"]), int(spec["max"]))
                else:
                    raw[key] = rng.uniform(spec["min"], spec["max"])
            else:
                raw[key] = rng.choice(spec)
        candidate, ok = validator.validate(raw, history)
        if ok:
            return candidate
    return validator.defaults_config()


def monitor_progress(run_dir, label, stop):
    last_epoch = 0
    while not stop.wait(60):
        try:
            results = Path(run_dir) / "results.csv"
            if not results.exists():
                continue
            with results.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            row = rows[-1]
            epoch = int(float(row["epoch"]))
            if epoch != last_epoch:
                last_epoch = epoch
                m = float(row["metrics/mAP50(M)"])
                log(f"[auto-train] {label}: epoch {epoch} mask_mAP50={m:.4f}")
        except Exception:
            pass


def find_latest_trial(project):
    """Return (trial_no, run_dir) for the highest auto_trial_NNN directory."""
    latest_no = 0
    latest_dir = None
    for d in project.glob("auto_trial_*"):
        try:
            no = int(d.name.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if no > latest_no:
            latest_no, latest_dir = no, d
    return latest_no, latest_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(BASE / "config.json"))
    ap.add_argument("--ledger", default=str(BASE / "experiments.json"))
    ap.add_argument("--optuna-db", default=str(BASE / "optuna.db"))
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--proposer", choices=["llm", "optuna", "random", "defaults"], default=None,
                    help="llm: LLM proposes; optuna: Bayesian optimization; "
                         "random: no-LLM random search; defaults: fixed config")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="cap the search-space epochs upper bound (budget control)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip completed trials, continue an interrupted trial "
                         "from its last.pt, then keep tuning")
    args = ap.parse_args()

    cfg = load_config(args.config)
    global LOG_PATH
    LOG_PATH = Path(args.ledger).with_name("auto_train.log")
    training = dict(cfg["training"])
    if args.max_epochs:
        space = cfg["search_space"]
        cur = space["epochs"]
        lo = min(cur["min"], args.max_epochs)
        space["epochs"] = {"min": lo, "max": args.max_epochs}
    for key in ("data", "weights", "project", "device", "epochs", "fraction",
                "imgsz", "batch", "workers"):
        val = getattr(args, key)
        if val is not None:
            training[key] = val

    target = args.target if args.target is not None else cfg["target_value"]
    iterations = args.iterations if args.iterations is not None else cfg["max_trials"]
    max_seconds = cfg["max_wall_minutes"] * 60
    ledger = Ledger(args.ledger)
    validator = Validator(cfg["search_space"], training)
    trainer = Trainer(training, sys.executable, BASE.parent / "train_cubit.py")
    proposer = args.proposer
    if proposer is None:
        proposer = "defaults" if args.skip_llm else "llm"
    tuner = None
    if proposer == "llm":
        tuner = Tuner(**cfg["tuner"])
    optuna_tuner = None
    if proposer == "optuna":
        from optuna_tuner import OptunaTuner
        optuna_tuner = OptunaTuner(storage=args.optuna_db)
    rng = random.Random(0)

    log(f"[auto-train] target={target} iterations={iterations} mock={args.mock} "
        f"proposer={proposer}")
    start = time.time()
    consecutive_failures = 0
    best_before = ledger.best()
    best_fitness = best_before["fitness"] if best_before else 0.0

    trial_count = 0
    batch_cap = int(training.get("batch", 16))
    if args.resume:
        latest_no, latest_dir = find_latest_trial(Path(training["project"]))
        if latest_dir is None:
            log("[auto-train] resume: no existing trial dirs, starting fresh")
            trial_count = 0
        else:
            done = (latest_dir / "args.json").exists()
            last_pt = latest_dir / "weights" / "last.pt"
            if not done and last_pt.exists():
                log(f"[auto-train] resume: continuing interrupted trial "
                    f"{latest_no} from {last_pt}")
                stop_monitor = threading.Event()
                monitor = threading.Thread(
                    target=monitor_progress,
                    args=(latest_dir, f"trial {latest_no} (resume)", stop_monitor),
                    daemon=True,
                )
                monitor.start()
                try:
                    run_dir = trainer.run({}, latest_no, mock=args.mock, resume=True)
                    result = evaluate(run_dir, cfg["target_metric"])
                    trial = {
                        "config": {"resume": str(last_pt)},
                        "fitness": result["fitness"] if result else None,
                        "metrics": result["metrics"] if result else {},
                        "status": "ok" if result else "nan",
                        "run_dir": str(run_dir),
                    }
                    ledger.add(trial)
                    if optuna_tuner:
                        optuna_tuner.report(trial["fitness"])
                    trial_count = latest_no
                    log(f"[auto-train] resume trial {latest_no}: "
                        f"fitness={trial['fitness']}")
                except Exception as exc:
                    if optuna_tuner:
                        optuna_tuner.report(None)
                    log(f"[auto-train] resume trial {latest_no} failed: {exc}")
                    trial_count = latest_no - 1
                finally:
                    stop_monitor.set()
            elif not done:
                log(f"[auto-train] resume: trial {latest_no} has no last.pt; "
                    f"it will be restarted as a fresh trial")
                trial_count = latest_no - 1
            else:
                log(f"[auto-train] resume: trial {latest_no} already complete; "
                    f"continuing with new trials")
                trial_count = latest_no
    while trial_count < iterations:
        if time.time() - start > max_seconds:
            log("[auto-train] wall-time budget exhausted")
            break
        trial_no = trial_count + 1
        history = ledger.history()
        config = validator.defaults_config()
        ok = True
        if proposer == "llm":
            try:
                for attempt in range(3):
                    raw = tuner.propose(
                        dataset_summary={"data": training["data"], "weights": training["weights"]},
                        history=history,
                        search_space=cfg["search_space"],
                        defaults=validator.defaults_config(),
                        target_metric=cfg["target_metric"],
                        target_value=target,
                    )
                    if getattr(tuner, "last_raw", None):
                        log(f"[auto-train] trial {trial_no}: LLM raw proposal: "
                            f"{str(tuner.last_raw)[:1500]}")
                    candidate, ok = validator.validate(raw, history)
                    if ok:
                        config = candidate
                        break
            except Exception as exc:
                raw_txt = getattr(tuner, "last_raw", None)
                msg = f"[auto-train] trial {trial_no}: LLM call failed ({exc})"
                if raw_txt:
                    msg += f" last_raw={str(raw_txt)[:400]}"
                log(msg + ", fallback to random proposal")
                config = random_config(validator, cfg["search_space"],
                                       validator.defaults_config(), history, rng)
                ok = True
        elif proposer == "random":
            config = random_config(validator, cfg["search_space"],
                                   validator.defaults_config(), history, rng)
        elif proposer == "optuna":
            config = optuna_tuner.propose(history, cfg["search_space"],
                                          validator.defaults_config())
            ok = True
        if not ok:
            config = validator.defaults_config()
            log(f"[auto-train] trial {trial_no}: proposal invalid/duplicate, using defaults")
        config["batch"] = min(int(config.get("batch", batch_cap)), batch_cap)
        log(f"[auto-train] trial {trial_no}: starting training epochs={config['epochs']} "
            f"batch={config['batch']} imgsz={config['imgsz']} lr0={config['lr0']} "
            f"amp={config['amp']}")
        run_dir = Path(training["project"]) / f"auto_trial_{trial_no:03d}"
        stop_monitor = threading.Event()
        monitor = threading.Thread(
            target=monitor_progress,
            args=(run_dir, f"trial {trial_no}", stop_monitor),
            daemon=True,
        )
        monitor.start()
        try:
            run_dir = trainer.run(config, trial_no, mock=args.mock)
            result = evaluate(run_dir, cfg["target_metric"])
            trial = {
                "config": config,
                "fitness": result["fitness"] if result else None,
                "metrics": result["metrics"] if result else {},
                "status": "ok" if result else "nan",
                "run_dir": str(run_dir),
            }
            ledger.add(trial)
            if optuna_tuner:
                optuna_tuner.report(trial["fitness"])
            consecutive_failures = 0
            trial_count += 1
            log(f"[auto-train] trial {trial_no}: fitness={trial['fitness']} "
                f"config={json.dumps(config)}")
        except Exception as exc:
            if optuna_tuner:
                optuna_tuner.report(None)
            if "OOM_RETRIES_EXHAUSTED" in str(exc):
                batch_cap = max(4, batch_cap // 2)
                ledger.add({"config": config, "fitness": None, "metrics": {},
                            "status": "void (OOM)", "run_dir": str(run_dir)})
                log(f"[auto-train] trial {trial_no}: void (OOM), "
                    f"batch cap lowered to {batch_cap}")
                continue
            consecutive_failures += 1
            ledger.add({"config": config, "fitness": None, "metrics": {},
                        "status": f"failed: {exc}", "run_dir": ""})
            trial_count += 1
            log(f"[auto-train] trial {trial_no}: FAILED: {exc}")
            if consecutive_failures >= 3:
                log("[auto-train] too many consecutive failures, stopping")
                break
            continue
        finally:
            stop_monitor.set()

        best = ledger.best()
        if best and best["fitness"] >= target:
            log(f"[auto-train] TARGET REACHED: {best['fitness']} >= {target}")
            break
        recent = [t for t in ledger.trials if t.get("fitness") is not None][-cfg["plateau_trials"]:]
        if len(recent) >= cfg["plateau_trials"]:
            improved = max(t["fitness"] for t in recent) - min(t["fitness"] for t in recent)
            if improved < cfg["min_improvement"] and best and best["fitness"] <= best_fitness + 1e-9:
                log("[auto-train] plateau detected, stopping")
                break

    best = ledger.best()
    log("[auto-train] done")
    if best:
        log(f"best trial {best['id']}: fitness={best['fitness']} config={best['config']}")
    else:
        log("no successful trial")


if __name__ == "__main__":
    main()

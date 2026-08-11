"""Training executor: launches train_cubit.py, or writes a mock results.csv."""

import csv
import json
import subprocess
from pathlib import Path


class Trainer:
    def __init__(self, training, python, train_script):
        self.training = training
        self.python = python
        self.train_script = train_script

    def run(self, config, trial_id, mock=False, resume=False):
        name = f"auto_trial_{trial_id:03d}"
        project = Path(self.training["project"])
        run_dir = project / name
        if mock:
            self._write_mock_results(run_dir, trial_id)
            return run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        batch = config.get("batch", self.training.get("batch", 16))
        oom_seen = False
        while True:
            cmd = self._build_cmd(config, name, project, batch, resume=resume)
            with log_path.open("ab") as f:
                try:
                    subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
                    break
                except subprocess.CalledProcessError:
                    text = ""
                    if log_path.exists():
                        text = log_path.read_text(encoding="utf-8", errors="ignore")
                    if batch > 4 and ("OutOfMemoryError" in text
                                      or "out of memory" in text):
                        oom_seen = True
                        batch //= 2
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(f"\n[auto-train] OOM detected, retrying with batch={batch}\n")
                        continue
                    if oom_seen:
                        raise RuntimeError("OOM_RETRIES_EXHAUSTED")
                    raise
        (run_dir / "args.json").write_text(
            json.dumps({"batch_used": batch, "config": config}), encoding="utf-8"
        )
        return run_dir

    def _build_cmd(self, config, name, project, batch, resume=False):
        if resume:
            return [
                self.python,
                str(self.train_script),
                "--project", str(project),
                "--name", name,
                "--resume",
            ]
        cmd = [
            self.python,
            str(self.train_script),
            "--data", str(self.training["data"]),
            "--weights", str(self.training["weights"]),
            "--project", str(project),
            "--name", name,
            "--epochs", str(config["epochs"]),
            "--batch", str(batch),
            "--imgsz", str(config["imgsz"]),
            "--lr0", str(config["lr0"]),
            "--warmup", str(config["warmup"]),
            "--patience", str(config["patience"]),
            "--workers", str(config["workers"]),
            "--device", str(self.training["device"]),
            "--fraction", str(self.training.get("fraction", 1.0)),
        ]
        if not config["amp"]:
            cmd.append("--no-amp")
        if self.training.get("cache"):
            cmd.append("--cache")
        if self.training.get("distill_model"):
            cmd += ["--distill-model", str(self.training["distill_model"]),
                    "--dis", str(self.training.get("dis", 6.0))]
        return cmd

    @staticmethod
    def _write_mock_results(run_dir, trial_id):
        run_dir.mkdir(parents=True, exist_ok=True)
        headers = [
            "epoch", "time", "train/box_loss", "train/seg_loss", "train/cls_loss",
            "train/dfl_loss", "train/sem_loss", "train/dis_loss",
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
            "metrics/mAP50-95(B)", "metrics/precision(M)", "metrics/recall(M)",
            "metrics/mAP50(M)", "metrics/mAP50-95(M)", "val/box_loss",
            "val/seg_loss", "val/cls_loss", "val/dfl_loss", "val/sem_loss",
            "val/dis_loss", "lr/pg0", "lr/pg1", "lr/pg2",
        ]
        fitness = min(0.75 + 0.02 * trial_id, 0.95)
        row = {
            "epoch": "1",
            "time": "60",
            "metrics/mAP50(M)": f"{fitness:.4f}",
            "metrics/mAP50-95(M)": f"{fitness * 0.62:.4f}",
            "metrics/mAP50(B)": f"{fitness + 0.02:.4f}",
            "metrics/mAP50-95(B)": f"{fitness * 0.82:.4f}",
        }
        with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow({h: row.get(h, "0") for h in headers})
        (run_dir / "args.json").write_text(
            json.dumps({"mock": True, "trial_id": trial_id}), encoding="utf-8"
        )

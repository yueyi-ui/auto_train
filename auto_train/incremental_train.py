"""Incremental training: merge new labeled material into a base dataset,
fine-tune from the current best weights, evaluate and record the result.

Usage:
    python incremental_train.py --base-data F:/cubit_dedup/data.yaml \
        --new-data D:/incoming --out F:/cubit_incr \
        --weights C:/temp/auto_train_runs/auto_trial_003/weights/best.pt

--dry-run only merges the dataset and prints a report without training.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from evaluator import evaluate
from ledger import Ledger
from trainer import Trainer

BASE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_data_yaml(path):
    text = Path(path).read_text(encoding="utf-8")
    base = Path(path).resolve().parent

    def resolve(p):
        p = Path(str(p).strip().replace("\\", "/"))
        return (base / p).resolve() if not p.is_absolute() else p.resolve()

    data = {"path": None, "train": None, "val": None, "nc": 1, "names": {}}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("path:"):
            data["path"] = resolve(s.split(":", 1)[1])
        elif s.startswith("train:"):
            data["train"] = resolve(s.split(":", 1)[1])
        elif s.startswith("val:"):
            data["val"] = resolve(s.split(":", 1)[1])
        elif s.startswith("nc:"):
            data["nc"] = int(s.split(":", 1)[1].strip())
    in_names = False
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        s = line.strip()
        if not s or ":" not in s:
            break
        k, v = s.split(":", 1)
        try:
            data["names"][int(k)] = v.strip().strip('"').strip("'")
        except ValueError:
            break
    return data


def find_labels_dir(img_dir):
    candidates = [
        img_dir.parents[1] / "labels" / img_dir.name,
        img_dir.parent / "labels",
        img_dir.parents[1] / "labels",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))
    return True


def scan_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def prepare_weights(weights, out_dir):
    """Strip DistillationModel prefixes so the checkpoint loads as a plain YOLO model."""
    try:
        import torch
    except ImportError:
        return weights
    try:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception:
        return weights
    if not isinstance(ck, dict) or not ck.get("ema"):
        return weights
    ema = ck["ema"]
    if not hasattr(ema, "student_model"):
        return weights
    clean = dict(ck)
    clean["model"] = ema.student_model
    clean["ema"] = None
    clean["optimizer"] = None
    clean["scaler"] = None
    clean["epoch"] = 0
    clean_path = out_dir / "student_clean.pt"
    torch.save(clean, str(clean_path))
    return str(clean_path)


def copy_images(images, labels_dir, dst_images, dst_labels, count):
    copied = 0
    for img in images:
        lbl = (labels_dir / (img.stem + ".txt")) if labels_dir else None
        if lbl is not None and lbl.exists():
            link_or_copy(img, dst_images / img.name)
            link_or_copy(lbl, dst_labels / (img.stem + ".txt"))
            copied += 1
            continue
        if lbl is None or not lbl.exists():
            if labels_dir is None:
                # Background images without any label dir are still usable.
                link_or_copy(img, dst_images / img.name)
                copied += 1
    count["images"] += copied
    return copied


def merge(base_yaml, new_images, new_labels_dir, out, val_frac, force, rng):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()) and not force:
        sys.exit(f"refusing to overwrite non-empty {out}; use --force to override")
    out.mkdir(parents=True, exist_ok=True)
    dirs = {
        "train_images": out / "images" / "train",
        "val_images": out / "images" / "val",
        "train_labels": out / "labels" / "train",
        "val_labels": out / "labels" / "val",
        "unlabeled": out / "unlabeled",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    base = parse_data_yaml(base_yaml)
    count = {"base_train": 0, "base_val": 0, "new_train": 0, "new_val": 0,
             "unlabeled": 0, "images": 0}

    base_labels_train = find_labels_dir(base["train"])
    base_labels_val = find_labels_dir(base["val"])
    count["base_train"] += copy_images(scan_images(base["train"]), base_labels_train,
                                       dirs["train_images"], dirs["train_labels"], count)
    count["base_val"] += copy_images(scan_images(base["val"]), base_labels_val,
                                     dirs["val_images"], dirs["val_labels"], count)

    labeled = []
    for img in new_images:
        lbl = (new_labels_dir / (img.stem + ".txt")) if new_labels_dir else None
        if lbl is not None and lbl.exists():
            labeled.append((img, lbl))
        else:
            count["unlabeled"] += 1
            shutil.copy2(str(img), str(dirs["unlabeled"] / img.name))

    rng.shuffle(labeled)
    n_val = max(1, round(len(labeled) * val_frac)) if labeled else 0
    for idx, (img, lbl) in enumerate(labeled):
        if idx < n_val:
            link_or_copy(img, dirs["val_images"] / img.name)
            link_or_copy(lbl, dirs["val_labels"] / (img.stem + ".txt"))
            count["new_val"] += 1
        else:
            link_or_copy(img, dirs["train_images"] / img.name)
            link_or_copy(lbl, dirs["train_labels"] / (img.stem + ".txt"))
            count["new_train"] += 1

    names = base["names"] or {0: "crack"}
    nc = base["nc"] if base["nc"] else len(names)
    yaml_text = (
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {nc}\n"
        "names:\n"
    )
    for k in sorted(names):
        yaml_text += f"  {k}: {names[k]}\n"
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (out / "merge_report.json").write_text(
        json.dumps(count, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", required=True)
    ap.add_argument("--new-data", required=True)
    ap.add_argument("--out", default="F:/cubit_incr")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr0", type=float, default=0.0001)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--project", default="C:/temp/auto_train_runs")
    ap.add_argument("--name", default="incr_trial")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    new_data = Path(args.new_data)
    new_images_dir = new_data / "images" if (new_data / "images").is_dir() else new_data
    new_labels_dir = new_data / "labels" if (new_data / "labels").is_dir() else new_data
    new_images = scan_images(new_images_dir)
    if not new_images:
        sys.exit(f"no images found in {new_images_dir}")

    rng = random.Random(0)
    out, count = merge(args.base_data, new_images, new_labels_dir,
                       args.out, args.val_frac, args.force, rng)
    print(f"[incr] merged to {out}")
    print(f"[incr] base_train={count['base_train']} base_val={count['base_val']} "
          f"new_train={count['new_train']} new_val={count['new_val']} "
          f"unlabeled={count['unlabeled']}")
    if args.dry_run:
        print("[incr] dry-run done, no training")
        return

    weights = prepare_weights(args.weights, out)
    if weights != args.weights:
        print(f"[incr] converted distill checkpoint -> {weights}")

    training = {
        "data": str(out / "data.yaml"),
        "weights": weights,
        "project": args.project,
        "device": args.device,
        "workers": args.workers,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "warmup": args.warmup,
        "patience": args.patience,
        "amp": True,
        "fraction": 1.0,
        "distill_model": None,
        "dis": 6.0,
    }
    config = {k: training[k] for k in
              ("epochs", "batch", "imgsz", "lr0", "warmup", "patience", "workers", "amp")}
    trainer = Trainer(training, sys.executable, BASE.parent / "train_cubit.py")
    run_dir = trainer.run(config, 1, mock=False)
    result = evaluate(run_dir, "metrics/mAP50(M)")
    ledger = Ledger(BASE / "experiments.json")
    ledger.add({
        "config": {"weights": weights, **config},
        "fitness": result["fitness"] if result else None,
        "metrics": result["metrics"] if result else {},
        "status": "ok" if result else "nan",
        "run_dir": str(run_dir),
        "kind": "incremental",
    })
    print(f"[incr] fitness={result['fitness'] if result else None} run_dir={run_dir}")


if __name__ == "__main__":
    main()

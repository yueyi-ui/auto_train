# -*- coding: utf-8 -*-
"""Generate data.yaml and classes.txt by scanning YOLO label files.

Class names are resolved in this order:
    1. --names argument (explicit comma-separated list)
    2. existing classes.txt next to the labels
    3. existing data.yaml names
    4. placeholder class_<id> names

Supported layouts (relative to --data):
    images/train + labels/train
    images/val   + labels/val
    images       + labels
    <flat images> + labels

Usage:
    python make_yaml_from_labels.py --data F:/new_dataset
    python make_yaml_from_labels.py --data F:/new_dataset --names tissue,phone,gray_cube
"""

import argparse
import sys
from collections import Counter
from pathlib import Path


def collect_splits(data):
    """Return [(split_name, images_dir, labels_dir_or_None), ...]."""
    splits = []
    img_root = data / "images"
    lbl_root = data / "labels"
    if img_root.is_dir():
        if (img_root / "train").is_dir():
            for name in ("train", "val", "test"):
                img_dir = img_root / name
                if img_dir.is_dir():
                    lbl_dir = lbl_root / name if (lbl_root / name).is_dir() else None
                    splits.append((name, img_dir, lbl_dir))
        else:
            lbl_dir = lbl_root if lbl_root.is_dir() else None
            splits.append(("train", img_root, lbl_dir))
    else:
        lbl_dir = lbl_root if lbl_root.is_dir() else None
        splits.append(("train", data, lbl_dir))
    return splits


def scan_classes(labels_dir):
    """Count class ids appearing in every label txt under labels_dir."""
    counts = Counter()
    if not labels_dir or not labels_dir.is_dir():
        return counts
    for txt in sorted(labels_dir.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        try:
            lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if cls >= 0:
                counts[cls] += 1
    return counts


def read_classes_txt(candidates):
    """Return the first non-empty classes.txt content as a list of names."""
    for path in candidates:
        if path and path.is_file():
            names = [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
            if names:
                return names
    return None


def read_yaml_names(path):
    """Parse the names block of a data.yaml into {class_id: name}."""
    names = {}
    if not path or not path.is_file():
        return names
    in_names = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if not stripped or ":" not in stripped:
            break
        key, value = stripped.split(":", 1)
        try:
            names[int(key)] = value.strip().strip("\"'")
        except ValueError:
            break
    return names


def build_names(class_ids, explicit, classes_txt, yaml_names):
    """Merge the available name sources with class_<id> placeholders."""
    names = {}
    if explicit:
        for idx, name in enumerate(explicit):
            names[idx] = name
    elif classes_txt:
        for idx, name in enumerate(classes_txt):
            names[idx] = name
    elif yaml_names:
        names = dict(yaml_names)
    for cls_id in class_ids:
        names.setdefault(cls_id, f"class_{cls_id}")
    return dict(sorted(names.items()))


def build_yaml_text(data, splits, names):
    train = next((d for s, d, _ in splits if s == "train"), None)
    val = next((d for s, d, _ in splits if s == "val"), None)
    test = next((d for s, d, _ in splits if s == "test"), None)
    lines = [f"path: {data.as_posix()}"]
    for split, img_dir in (("train", train), ("val", val), ("test", test)):
        if img_dir:
            lines.append(f"{split}: {img_dir.relative_to(data).as_posix()}")
    lines.append(f"nc: {max(names) + 1}")
    lines.append("names:")
    for cls_id in sorted(names):
        lines.append(f"  {cls_id}: {names[cls_id]}")
    return "\n".join(lines) + "\n"


def write_yaml(path, text, force):
    if path.exists() and path.read_text(encoding="utf-8") == text:
        print(f"unchanged {path}")
        return
    if path.exists() and not force:
        sys.exit(
            f"{path} already exists and differs; re-run with --force to overwrite"
        )
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def write_classes_txt(splits, names, force):
    lines = [names[idx] for idx in range(max(names) + 1)]
    text = "\n".join(lines) + "\n"
    for _, _, labels_dir in splits:
        if not labels_dir or not labels_dir.is_dir():
            continue
        path = labels_dir / "classes.txt"
        if path.exists():
            if path.read_text(encoding="utf-8", errors="ignore") == text:
                print(f"unchanged {path}")
                continue
            if not force:
                print(f"exists, skip {path} (use --force to overwrite)")
                continue
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate data.yaml and classes.txt from YOLO labels."
    )
    parser.add_argument("--data", required=True, help="dataset root directory")
    parser.add_argument(
        "--names",
        default=None,
        help="comma-separated class names in ascending class-id order "
             "(overrides classes.txt / data.yaml)",
    )
    parser.add_argument("--out", default=None, help="data.yaml output path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing data.yaml / classes.txt",
    )
    args = parser.parse_args()

    data = Path(args.data).resolve()
    if not data.is_dir():
        sys.exit(f"data dir not found: {data}")

    splits = collect_splits(data)
    if not splits:
        sys.exit(f"no images/labels layout found under {data}")

    counts = Counter()
    for split, _, labels_dir in splits:
        if labels_dir:
            n_files = sum(
                1 for p in labels_dir.glob("*.txt") if p.name != "classes.txt"
            )
            print(f"[{split}] {n_files} label files in {labels_dir}")
        split_counts = scan_classes(labels_dir)
        if split_counts:
            print(f"[{split}] {sum(split_counts.values())} instances, "
                  f"class ids={dict(split_counts)}")
        counts.update(split_counts)

    if not counts:
        sys.exit("no class ids found in any label file")

    explicit = None
    if args.names:
        explicit = [name.strip() for name in args.names.split(",")]
        if any(not name for name in explicit):
            sys.exit("--names must be a comma-separated list without empty entries")

    labels_dirs = [labels_dir for _, _, labels_dir in splits if labels_dir]
    classes_txt = read_classes_txt(
        [labels_dir / "classes.txt" for labels_dir in labels_dirs]
        + [data / "classes.txt"]
    )
    yaml_names = read_yaml_names(data / "data.yaml")

    if explicit:
        print(f"names from --names ({len(explicit)} entries)")
    elif classes_txt:
        print(f"names from classes.txt ({len(classes_txt)} entries)")
    elif yaml_names:
        print(f"names from existing data.yaml ({len(yaml_names)} entries)")
    else:
        print("no name source; using class_<id> placeholders")

    names = build_names(sorted(counts), explicit, classes_txt, yaml_names)
    for cls_id in sorted(names):
        count = counts.get(cls_id, 0)
        suffix = f" ({count} instances)" if count else " (no labels yet)"
        print(f"  class {cls_id}: {names[cls_id]}{suffix}")

    out = Path(args.out).resolve() if args.out else data / "data.yaml"
    write_yaml(out, build_yaml_text(data, splits, names), args.force)
    write_classes_txt(splits, names, args.force)
    print("done")


if __name__ == "__main__":
    main()

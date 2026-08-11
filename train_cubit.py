# -*- coding: utf-8 -*-
"""yolo11x-seg high-resolution training on CUBIT tiles + merged_hrcds.

Server usage (my-yolo-train env, GPU 3):
    python train_cubit.py --data /data/combined_896.yaml

For CUBIT-only training:
    python train_cubit.py --data /data/cubit_896/data.yaml --name yolo11x_896_cubit

Note: work log mentions a "v4 parameter plan" without listing it; defaults below
mirror the proven v3 run with erasing=0.0 for fine cracks (7/29 lesson).
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/combined_896.yaml")
    ap.add_argument("--name", default="yolo11x_896_hr")
    ap.add_argument("--weights", default="yolo11x-seg.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=0.0005)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="enable AMP (default); use --no-amp for FP32 training")
    ap.add_argument("--bf16", action="store_true",
                    help="experimental: force bfloat16 autocast (Ampere+ GPU); "
                         "aims for FP16-like speed while avoiding FP16 overflow")
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 matmul on Ampere+ (FP32 speedup, slight precision loss)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup epochs; use 0 when fine-tuning from a "
                         "trained checkpoint")
    ap.add_argument("--device", default="3",
                    help='e.g. "3" or "0,3" for DDP across GPU 0 and 3')
    ap.add_argument("--project", default="/data/runs")
    ap.add_argument("--distill-model", default=None,
                    help="teacher weights path for knowledge distillation")
    ap.add_argument("--dis", type=float, default=6.0,
                    help="distillation loss weight (default 6.0)")
    ap.add_argument("--cache", action="store_true",
                    help="cache images in RAM for faster epochs (needs free RAM)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of dataset to use per split (0-1), for quick smoke runs")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from <project>/<name>/weights/last.pt")
    args = ap.parse_args()

    if args.resume:
        from pathlib import Path
        from ultralytics import YOLO

        last = Path(args.project) / args.name / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"resume checkpoint not found: {last}")
        YOLO(str(last)).train(resume=True)
        print("Resume complete")
        return

    from ultralytics import YOLO

    amp = args.amp
    if args.bf16:
        import torch
        from ultralytics.engine import trainer as trainer_module

        def bf16_autocast(enabled, device="cuda"):
            return torch.amp.autocast(device, dtype=torch.bfloat16)

        trainer_module.autocast = bf16_autocast
        amp = False  # bf16 has FP32-like exponent range, no GradScaler needed
    elif args.tf32:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = YOLO(args.weights)
    train_kwargs = dict(
        data=args.data,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.005,
        cos_lr=True,
        warmup_epochs=args.warmup,
        patience=args.patience,
        close_mosaic=15,
        fliplr=0.5,
        flipud=0.0,
        degrees=10,
        shear=5,
        scale=0.5,
        translate=0.2,
        mosaic=1.0,
        copy_paste=0.3,
        hsv_h=0.05,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        workers=args.workers,
        cache=args.cache,
        device=args.device,
        amp=amp,
        fraction=args.fraction,
        exist_ok=True,
        verbose=True,
    )
    if args.distill_model:
        train_kwargs["distill_model"] = args.distill_model
        train_kwargs["dis"] = args.dis
    model.train(**train_kwargs)
    print("Training complete")


if __name__ == "__main__":
    main()

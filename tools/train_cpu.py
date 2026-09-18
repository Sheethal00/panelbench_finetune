#!/usr/bin/env python3
"""
Full CPU training loop for the panel-device fine-tune.

YOLOX's own `Trainer` (yolox/core/trainer.py) hardcodes CUDA in three
places that aren't optional: the device string "cuda:{local_rank}",
DataPrefetcher's `torch.cuda.Stream()`, and Exp.random_resize's
`.cuda()` tensor. Its COCOEvaluator.evaluate() does too (torch.cuda.
HalfTensor, explicit .cuda() calls). None of that can be reused as-is
for a CPU run -- it fails immediately, not slowly.

Given how small this dataset is (15 images), a full CPU run is actually
a reasonable choice here, not just a smoke test -- likely minutes, not
hours. This script reimplements the same epoch loop Trainer runs (warmup
-> cosine LR, mosaic on -> mosaic off + L1 loss for the final
no_aug_epochs, EMA, checkpointing, MLflow logging) using the *same* Exp
methods (get_model, get_data_loader, get_optimizer, get_lr_scheduler) so
training behavior matches what a real GPU run would use -- just on CPU,
with the CUDA-only pieces removed.

What's intentionally left out, and why:
  - No held-out evaluation loop. COCOEvaluator.evaluate() is CUDA-only,
    and per the exp file's own reasoning, an AP computed with val==train
    on 15 images isn't a meaningful metric anyway. This script
    checkpoints on training loss instead: "latest_ckpt.pth" every epoch,
    "best_ckpt.pth" whenever mean epoch loss improves.
  - No multiscale resizing (that's Exp.random_resize's `.cuda()` tensor)
    -- trains at the fixed exp.input_size for the whole run. Multiscale
    robustness matters much more for varied real-world deployment
    conditions than for getting a first fine-tune off the ground on 15
    images.

Usage:
    python tools/train_cpu.py \
        --exp-file exps/yolox_nano_panel_finetune.py \
        --ckpt yolox_nano.pth \
        --batch-size 4 \
        --experiment-name panel_v1
"""
import argparse
import os
import time

import torch
from loguru import logger

from yolox.exp import get_exp, check_exp_value
from yolox.utils import ModelEMA, load_ckpt, save_checkpoint, MlflowLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-file", required=True)
    ap.add_argument("--ckpt", default=None,
                     help="COCO-pretrained checkpoint to fine-tune from (e.g. yolox_nano.pth)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--save-every", type=int, default=10,
                     help="Also save a numbered checkpoint every N epochs, "
                          "beyond the always-updated latest/best")
    ap.add_argument("--output-dir", default=None,
                     help="Defaults to exp.output_dir/<experiment-name>, "
                          "matching YOLOX's own layout")
    ap.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging entirely")
    ap.add_argument("--resume", action="store_true",
                     help="Resume from <output-dir>/latest_ckpt.pth if it exists "
                          "(e.g. after a power loss or Ctrl+C). Ignored if no such "
                          "checkpoint is found -- starts fresh instead. Use the same "
                          "--batch-size as the run being resumed; a mismatch throws off "
                          "the LR schedule's iteration math.")
    args = ap.parse_args()

    exp = get_exp(args.exp_file, None)
    check_exp_value(exp)
    experiment_name = args.experiment_name or exp.exp_name
    output_dir = args.output_dir or os.path.join(exp.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")
    logger.info(f"Exp '{exp.exp_name}': num_classes={exp.num_classes}, "
                f"max_epoch={exp.max_epoch}, input_size={exp.input_size}, "
                f"output_dir={output_dir}")

    # ---- data ----
    train_loader = exp.get_data_loader(
        batch_size=args.batch_size, is_distributed=False,
        no_aug=False, cache_img=None,
    )
    max_iter = len(train_loader)
    logger.info(f"{max_iter} iteration(s)/epoch at batch_size={args.batch_size}")
    if max_iter == 0:
        raise SystemExit("Dataloader produced 0 batches -- batch_size is "
                          "probably larger than the dataset. Reduce --batch-size.")

    # ---- model ----
    model = exp.get_model().to(device)
    model.train()
    exp.reapply_stem_freeze(model)  # model.train() just re-enabled BN training on the frozen stem too -- undo that
    if args.ckpt:
        logger.info(f"Loading checkpoint for fine-tuning: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)["model"]
        model = load_ckpt(model, ckpt)  # shape-mismatched layers (the head) are skipped automatically

    optimizer = exp.get_optimizer(args.batch_size)
    lr_scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * args.batch_size, max_iter)

    no_aug_start_epoch = exp.max_epoch - exp.no_aug_epochs
    start_epoch = 0
    best_loss = float("inf")
    mosaic_closed = False

    # ---- resume, if requested and a checkpoint exists ----
    # Deliberately checked *after* the --ckpt (pretrained COCO weights)
    # load above and *before* EMA construction below: resuming should
    # pick up exactly where a previous run left off, not re-apply the
    # original pretrained weights on top, and the EMA average should
    # start from the resumed weights, not from scratch.
    latest_ckpt_path = os.path.join(output_dir, "latest_ckpt.pth")
    if args.resume:
        if os.path.isfile(latest_ckpt_path):
            logger.info(f"Resuming from {latest_ckpt_path}")
            resume_ckpt = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(resume_ckpt["model"])
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            start_epoch = resume_ckpt["start_epoch"]
            best_loss = resume_ckpt.get("training_best_loss", float("inf"))
            saved_batch_size = resume_ckpt.get("batch_size")
            if saved_batch_size is not None and saved_batch_size != args.batch_size:
                logger.warning(
                    f"Resuming with --batch-size {args.batch_size}, but the checkpoint "
                    f"was saved with batch_size {saved_batch_size} -- the LR schedule's "
                    f"iteration math will be off. Use --batch-size {saved_batch_size} instead."
                )
            exp.reapply_stem_freeze(model)  # load_state_dict doesn't touch train/eval mode, but no harm being explicit
            if start_epoch >= no_aug_start_epoch:
                train_loader.close_mosaic()
                model.head.use_l1 = True
                mosaic_closed = True
            logger.info(f"Resumed at epoch {start_epoch}/{exp.max_epoch}, "
                        f"best training loss so far: {best_loss:.3f}")
        else:
            logger.warning(f"--resume passed but no checkpoint found at "
                            f"{latest_ckpt_path} -- starting fresh")

    ema_model = ModelEMA(model, 0.9998) if exp.ema else None

    # ---- mlflow (reuses YOLOX's own built-in logger, same one `-l mlflow`
    # uses on GPU, so metrics/params land in the same schema either way) ----
    mlflow_logger, mlflow_args = None, None
    if not args.no_mlflow:
        mlflow_logger = MlflowLogger()
        mlflow_args = argparse.Namespace(
            experiment_name=experiment_name, batch_size=args.batch_size,
            exp_file=args.exp_file, resume=False, ckpt=args.ckpt,
            start_epoch=0, num_machines=1, fp16=False, logger="mlflow",
        )
        mlflow_logger.setup(args=mlflow_args, exp=exp)

    global_step = start_epoch * max_iter

    if start_epoch >= exp.max_epoch:
        raise SystemExit(
            f"Resumed checkpoint is already at epoch {start_epoch}/{exp.max_epoch} -- "
            f"training was already complete. Nothing to do. (Raise max_epoch in the "
            f"exp file first if you want to keep training this run further.)"
        )

    logger.info("Training start (CPU)...")
    train_start = time.time()

    # YOLOX's dataloader uses an InfiniteSampler -- it never raises
    # StopIteration on its own. The real Trainer handles this by creating
    # ONE iterator for the whole run and manually pulling exactly max_iter
    # batches per "epoch" (see yolox/core/trainer.py's train_in_iter). A
    # plain `for batch in train_loader:` loop here would just run forever
    # within a single epoch -- create the iterator once, outside the epoch
    # loop, same as the real Trainer does.
    train_iter = iter(train_loader)

    for epoch in range(start_epoch, exp.max_epoch):
        if epoch >= no_aug_start_epoch and not mosaic_closed:
            logger.info("--- closing mosaic, enabling L1 loss for the remaining epochs ---")
            train_loader.close_mosaic()
            model.head.use_l1 = True
            mosaic_closed = True

        epoch_losses = []
        epoch_start = time.time()
        for it in range(max_iter):
            inps, targets, img_info, ids = next(train_iter)
            inps = inps.to(device).float()
            targets = targets.to(device).float()
            targets.requires_grad = False

            outputs = model(inps, targets)
            loss = outputs["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_model is not None:
                ema_model.update(model)

            lr = lr_scheduler.update_lr(global_step + 1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss_val = loss.item()
            epoch_losses.append(loss_val)

            if mlflow_logger is not None:
                metrics = {k: (v.item() if hasattr(v, "item") else v)
                           for k, v in outputs.items() if "loss" in k}
                metrics["lr"] = lr
                mlflow_logger.on_log(mlflow_args, exp, global_step, metrics)

            global_step += 1

            if (it + 1) % exp.print_interval == 0 or it == max_iter - 1:
                logger.info(f"epoch {epoch + 1}/{exp.max_epoch}  iter {it + 1}/{max_iter}  "
                            f"total_loss={loss_val:.3f}  lr={lr:.2e}")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start
        logger.info(f"epoch {epoch + 1} done in {epoch_time:.1f}s -- mean total_loss={mean_loss:.3f}")

        is_best = mean_loss < best_loss
        best_loss = min(best_loss, mean_loss)

        save_model = ema_model.ema if ema_model is not None else model
        ckpt_state = {
            "start_epoch": epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_ap": 0.0,   # no real eval AP computed here -- see module docstring
            "curr_ap": None,
            "training_best_loss": best_loss,   # for --resume
            "batch_size": args.batch_size,     # for --resume's LR-schedule sanity check
        }
        save_checkpoint(ckpt_state, is_best, output_dir, "latest")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == exp.max_epoch:
            save_checkpoint(ckpt_state, False, output_dir, f"epoch_{epoch + 1}")

    total_time = time.time() - train_start
    logger.info(f"Training finished in {total_time / 60:.1f} min. "
                f"Checkpoints in {output_dir}/ "
                f"(best_ckpt.pth = lowest mean-training-loss epoch).")

    if mlflow_logger is not None:
        best_ckpt_path = os.path.join(output_dir, "best_ckpt.pth")
        # Log the checkpoint as an artifact directly, rather than going
        # through MlflowLogger.on_train_end() -- that method also tries to
        # upload a train_log.txt that only the real Trainer's file-based
        # logger setup ever creates, which we don't replicate here, and
        # would raise a file-not-found error at the very end of a
        # successful run.
        if os.path.exists(best_ckpt_path):
            mlflow_logger._ml_flow.log_artifact(best_ckpt_path, artifact_path=experiment_name)
            logger.info("Logged best_ckpt.pth to MLflow as an artifact.")
        if mlflow_logger._ml_flow.active_run() is not None:
            mlflow_logger._ml_flow.end_run()


if __name__ == "__main__":
    main()
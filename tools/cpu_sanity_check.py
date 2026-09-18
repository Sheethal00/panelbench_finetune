#!/usr/bin/env python3
"""
Pipeline smoke test for the panel-detector fine-tune, meant to run on the
CPU dev machine before handing off to GPU for the real run.

This deliberately does NOT use YOLOX's own `Trainer`
(`yolox.core.trainer.Trainer`) -- that class is CUDA-only in a few places
that aren't optional:
  - `self.device = "cuda:{}".format(self.local_rank)`, then `model.to(self.device)`
  - `DataPrefetcher` opens a `torch.cuda.Stream()` unconditionally
  - `Exp.random_resize()` allocates a `torch.LongTensor(...).cuda()`
Trying to run the real Trainer on a CPU-only box fails immediately, it's
not a matter of passing a "cpu" flag somewhere.

So instead this script builds the dataset, model, and optimizer the same
way the Trainer would (calling the same `Exp` methods), and runs a few
plain forward/backward iterations directly. What it verifies:
  - annotations parse, images load, category mapping is sane
  - image/target tensor shapes match what the model expects
  - a forward + backward pass runs and produces a finite, non-NaN loss
What it does NOT verify: whether the run will actually converge to a good
model. That's what the real GPU run is for.

Usage:
    python cpu_sanity_check.py --exp-file exps/yolox_nano_panel_cpu_sanity.py --iters 3
"""
import argparse
import math
import sys

import torch

from yolox.exp import get_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-file", required=True)
    ap.add_argument("--batch-size", type=int, default=2,
                     help="Small on purpose -- this is a plumbing check, not a real run")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    exp = get_exp(args.exp_file, None)
    print(f"Loaded exp '{exp.exp_name}': num_classes={exp.num_classes}, "
          f"data_dir={exp.data_dir}, input_size={exp.input_size}")

    print("Building dataloader (this reads annotations + resolves image paths)...")
    train_loader = exp.get_data_loader(
        batch_size=args.batch_size, is_distributed=False,
        no_aug=True,  # skip mosaic explicitly regardless of exp setting -- fast smoke test
        cache_img=None,
    )
    print(f"{len(train_loader)} iteration(s)/epoch at batch_size={args.batch_size}")
    if len(train_loader) == 0:
        sys.exit("Dataloader produced 0 batches -- check --batch-size vs. dataset size "
                  "and that instances_train.json actually has entries.")

    print("Building model on CPU...")
    model = exp.get_model()
    model.train()
    exp.reapply_stem_freeze(model)  # re-apply eval() to the frozen stem after model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable:,}  |  Frozen params: {frozen:,}")

    optimizer = exp.get_optimizer(args.batch_size)

    it = iter(train_loader)
    ran = 0
    for i in range(args.iters):
        try:
            inps, targets, img_info, ids = next(it)
        except StopIteration:
            print(f"Dataloader exhausted after {ran} iteration(s) "
                  f"(dataset smaller than --iters * --batch-size, that's fine)")
            break

        inps = inps.float()
        targets = targets.float()
        targets.requires_grad = False

        outputs = model(inps, targets)
        loss = outputs["total_loss"]

        if not torch.isfinite(loss):
            sys.exit(f"Iteration {i}: loss is not finite ({loss.item()}) -- "
                     f"check annotations/box coordinates for this batch "
                     f"(image ids: {ids})")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_parts = {k: v.item() if hasattr(v, "item") else v
                      for k, v in outputs.items() if "loss" in k}
        print(f"[iter {i}] image ids={ids}  {loss_parts}")
        ran += 1

    if ran == 0:
        sys.exit("Ran 0 iterations -- something above should have raised first; "
                  "please report this as a bug in this script.")

    print(f"\nOK: {ran} forward/backward iteration(s) completed on CPU with finite loss.")
    print("This confirms the data pipeline and model wiring are sound. "
          "Proceed to the real GPU run (see README section 4).")


if __name__ == "__main__":
    main()
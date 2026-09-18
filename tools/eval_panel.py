#!/usr/bin/env python3
"""
Benchmark the fine-tuned panel-device model with real COCO metrics
(AP, AP50, AP75, per-class AP, AR) against all 15 annotated images.

IMPORTANT CAVEAT, same one as the rest of this pipeline: with only 15
images total and no real held-out split (val2017 is a duplicate of
train2017 -- see the converter's own warning), this evaluates the model
on images it was trained on. Read these numbers as a memorization /
sanity check ("did it learn the training set at all"), not as an
estimate of how it'll perform on a panel it's never seen. This script
prints that caveat again before the results so it isn't missed.

Why this doesn't use YOLOX's own tools/eval.py or COCOEvaluator: same
reason as everywhere else in this pipeline -- COCOEvaluator.evaluate()
hardcodes CUDA (torch.cuda.HalfTensor, explicit .cuda() calls). The
actual mAP *math* (pycocotools' COCOeval) has no such dependency --
it's plain numpy -- so this script does its own CPU inference loop
(same pattern as demo_panel.py) and feeds the results into COCOeval
directly.

Usage:
    python tools/eval_panel.py \
        --exp-file exps/yolox_nano_panel_finetune.py \
        --ckpt YOLOX_outputs/panel_v1/best_ckpt.pth \
        --data-dir data/panel_coco \
        --split train \
        --conf 0.001 --nms 0.45
"""
import argparse
import contextlib
import io
import json
import os
import time

import cv2
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True,
                     help="The --output-dir you passed to convert_roboflow_coco.py")
    ap.add_argument("--split", choices=["train", "val"], default="train",
                     help="Which split to evaluate on. At 15 images with no real "
                          "held-out set, val is just a copy of train -- 'train' is "
                          "the more honest choice to evaluate against.")
    ap.add_argument("--conf", type=float, default=0.001,
                     help="Low on purpose: COCOeval needs the full score "
                          "distribution to compute precision/recall curves, "
                          "not just confident detections. Don't raise this for "
                          "benchmarking (use demo_panel.py with a normal "
                          "threshold like 0.25 to look at actual boxes instead).")
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--tsize", type=int, default=None)
    args = ap.parse_args()

    ann_file = os.path.join(args.data_dir, "annotations",
                             f"instances_{args.split}.json")
    img_dir = os.path.join(args.data_dir, f"{args.split}2017")
    if not os.path.isfile(ann_file):
        raise SystemExit(f"Annotation file not found: {ann_file}")

    coco_gt = COCO(ann_file)
    class_ids = sorted(coco_gt.getCatIds())  # index N here == model's class N, same convention COCODataset uses
    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(class_ids)}

    exp = get_exp(args.exp_file, None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    model.eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint: {args.ckpt}")

    preproc = ValTransform(legacy=False)
    img_ids = coco_gt.getImgIds()
    logger.info(f"Evaluating on {len(img_ids)} image(s) from {ann_file}")

    detections = []
    t_start = time.time()
    for img_id in img_ids:
        img_meta = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_meta["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read {img_path}, skipping")
            continue

        height, width = img.shape[:2]
        ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
        img_t, _ = preproc(img, None, exp.test_size)
        img_t = torch.from_numpy(img_t).unsqueeze(0).float()

        with torch.no_grad():
            outputs = model(img_t)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf,
                                   exp.nmsthre, class_agnostic=True)

        output = outputs[0]
        if output is None:
            continue
        output = output.cpu()
        bboxes = output[:, 0:4] / ratio  # x1,y1,x2,y2 in original-image pixels
        scores = output[:, 4] * output[:, 5]
        cls = output[:, 6]

        for box, score, c in zip(bboxes.tolist(), scores.tolist(), cls.tolist()):
            x1, y1, x2, y2 = box
            category_id = class_ids[int(c)]  # map model's 0-indexed class back to the original COCO category id
            detections.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO wants x,y,w,h
                "score": score,
            })

    elapsed = time.time() - t_start
    logger.info(f"Inference on {len(img_ids)} image(s) took {elapsed:.1f}s "
                f"({len(detections)} raw detection(s) above conf={args.conf})")

    print()
    print("=" * 70)
    print(f"BENCHMARKING ON THE '{args.split}' SPLIT -- {len(img_ids)} IMAGE(S)")
    print("This is the same set of images the model was trained on (no real")
    print("held-out split exists at this dataset size). These numbers show")
    print("whether the model learned the training data, NOT how well it will")
    print("generalize to panels it hasn't seen. Don't quote these as real mAP.")
    print("=" * 70)

    if not detections:
        print("\nNo detections above conf={:.4f} on any image -- nothing to score.".format(args.conf))
        return

    coco_dt = coco_gt.loadRes(detections)

    print("\n--- Overall (all classes) ---")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n--- Per class ---")
    for cat_id in class_ids:
        name = id_to_name[cat_id]
        n_gt = len(coco_gt.getAnnIds(catIds=[cat_id]))
        print(f"\n[{name}]  ({n_gt} ground-truth instance(s) in this split)")
        if n_gt == 0:
            print("  No ground-truth instances -- AP undefined, skipping.")
            continue
        # Suppress pycocotools' own verbose per-call prints and just show
        # the final summary table for this one class.
        cat_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        cat_eval.params.catIds = [cat_id]
        with contextlib.redirect_stdout(io.StringIO()):
            cat_eval.evaluate()
            cat_eval.accumulate()
        cat_eval.summarize()


if __name__ == "__main__":
    main()

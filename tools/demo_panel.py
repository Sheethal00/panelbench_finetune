#!/usr/bin/env python3
"""
Run inference on one image with the fine-tuned panel-device model and draw
boxes labeled with YOUR class names, not COCO's.

Why this script exists: YOLOX's own tools/demo.py hardcodes
`cls_names=COCO_CLASSES` for the labels it draws (see Predictor.__init__ in
that file) -- it has no way to know about your labels.txt. Your model's
predictions are almost certainly fine; demo.py was just printing "person"
for class index 0 because that's COCO's class 0, when your class 0 is
"MCB" (or whatever your labels.txt says). The detection logic below is
otherwise identical to demo.py's Predictor -- just with the right names
wired in.

Usage:
    python tools/demo_panel.py \
        --exp-file exps/yolox_nano_panel_finetune.py \
        --ckpt YOLOX_outputs/panel_v1/best_ckpt.pth \
        --labels data/panel_coco/labels.txt \
        --path path/to/image.jpg \
        --conf 0.25 --nms 0.45 --tsize 416
"""
import argparse
import os
import time

import cv2
import torch
from loguru import logger

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess, vis


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels", required=True,
                     help="labels.txt from the converter -- order must match "
                          "the model's class indices (it does, if this is the "
                          "same labels.txt the converter wrote)")
    ap.add_argument("--path", required=True, help="path to a single image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--tsize", type=int, default=None)
    ap.add_argument("--output-dir", default="./demo_output")
    args = ap.parse_args()

    cls_names = load_labels(args.labels)
    logger.info(f"Using class names from {args.labels}: {cls_names}")

    exp = get_exp(args.exp_file, None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    model.eval()

    logger.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint.")

    preproc = ValTransform(legacy=False)
    img = cv2.imread(args.path)
    if img is None:
        raise SystemExit(f"Could not read image: {args.path}")
    height, width = img.shape[:2]
    ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)

    img_t, _ = preproc(img, None, exp.test_size)
    img_t = torch.from_numpy(img_t).unsqueeze(0).float()

    with torch.no_grad():
        t0 = time.time()
        outputs = model(img_t)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf,
                               exp.nmsthre, class_agnostic=True)
        logger.info(f"Infer time: {time.time() - t0:.3f}s")

    output = outputs[0]
    if output is None:
        logger.info("No detections above the confidence threshold.")
        result_img = img
    else:
        output = output.cpu()
        bboxes = output[:, 0:4] / ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        for box, score, c in zip(bboxes.tolist(), scores.tolist(), cls.tolist()):
            name = cls_names[int(c)] if int(c) < len(cls_names) else f"class_{int(c)}"
            logger.info(f"  {name}: {score:.2f}  box={[round(x, 1) for x in box]}")
        result_img = vis(img, bboxes, scores, cls, exp.test_conf, cls_names)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, os.path.basename(args.path))
    cv2.imwrite(out_path, result_img)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

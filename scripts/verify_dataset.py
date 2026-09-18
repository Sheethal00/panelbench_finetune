#!/usr/bin/env python3
"""
Sanity-check a converted COCO-layout dataset before spending a training run
on it. On a ~15-image dataset, one bad annotation or a class with a single
instance is worth knowing about up front, not discovering after a
confusing training curve.

Usage:
    python verify_dataset.py --data-dir /data/panel_coco [--split train]
"""
import argparse
import json
import os
from collections import Counter

from PIL import Image


def check_split(data_dir, split, ann_name):
    ann_path = os.path.join(data_dir, "annotations", ann_name)
    img_dir = os.path.join(data_dir, split)
    if not os.path.isfile(ann_path):
        print(f"[{split}] SKIP -- {ann_path} not found")
        return

    with open(ann_path) as f:
        data = json.load(f)

    images_by_id = {im["id"]: im for im in data["images"]}
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    class_ids_sorted = sorted(cat_names)  # this is the index order YOLOX uses

    print(f"\n=== {split} ({ann_name}) ===")
    print(f"{len(data['images'])} image(s), {len(data['annotations'])} annotation(s), "
          f"{len(cat_names)} class(es)")

    missing_files, bad_boxes, no_annotation_images = [], [], []
    instances_per_class = Counter()
    imgs_with_annotations = set()

    for img in data["images"]:
        path = os.path.join(img_dir, img["file_name"])
        if not os.path.isfile(path):
            missing_files.append(img["file_name"])
            continue
        try:
            with Image.open(path) as im:
                actual_w, actual_h = im.size
            if (img.get("width"), img.get("height")) != (actual_w, actual_h):
                print(f"  WARNING: {img['file_name']} JSON size "
                      f"{img.get('width')}x{img.get('height')} != actual "
                      f"{actual_w}x{actual_h} -- boxes will be misaligned")
        except Exception as e:
            print(f"  WARNING: could not open {img['file_name']}: {e}")

    for ann in data["annotations"]:
        imgs_with_annotations.add(ann["image_id"])
        x, y, w, h = ann["bbox"]
        img = images_by_id.get(ann["image_id"])
        instances_per_class[ann["category_id"]] += 1
        if w <= 1 or h <= 1:
            bad_boxes.append((img["file_name"] if img else ann["image_id"], w, h))
        if img and (x < 0 or y < 0 or x + w > img["width"] + 1 or y + h > img["height"] + 1):
            bad_boxes.append((img["file_name"], "out-of-bounds bbox"))

    for img in data["images"]:
        if img["id"] not in imgs_with_annotations:
            no_annotation_images.append(img["file_name"])

    print("\nPer-class instance counts (index order = model output order):")
    for idx, cid in enumerate(class_ids_sorted):
        n = instances_per_class.get(cid, 0)
        flag = "  <-- only 1-2 instances, expect this class to be unreliable" if n <= 2 else ""
        print(f"  [{idx}] {cat_names[cid]:<24} {n} instance(s){flag}")

    if missing_files:
        print(f"\nMISSING FILES ({len(missing_files)}): {missing_files[:5]}"
              f"{' ...' if len(missing_files) > 5 else ''}")
    if bad_boxes:
        print(f"\nDEGENERATE/OUT-OF-BOUNDS BOXES ({len(bad_boxes)}): {bad_boxes[:5]}"
              f"{' ...' if len(bad_boxes) > 5 else ''}")
    if no_annotation_images:
        print(f"\nIMAGES WITH NO ANNOTATIONS ({len(no_annotation_images)}): "
              f"{no_annotation_images}")

    if not (missing_files or bad_boxes):
        print("\nNo missing files or malformed boxes found.")

    return len(cat_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    n_classes_train = check_split(args.data_dir, "train2017", "instances_train.json")
    check_split(args.data_dir, "val2017", "instances_val.json")

    if n_classes_train is not None:
        print(f"\nNUM_CLASSES = {n_classes_train}   <-- paste this into "
              f"exps/yolox_nano_panel_finetune.py")


if __name__ == "__main__":
    main()

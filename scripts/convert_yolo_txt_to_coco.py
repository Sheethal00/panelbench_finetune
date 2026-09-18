#!/usr/bin/env python3
"""
Fallback converter: if Roboflow ended up exporting "YOLOv8" / "YOLOv5 PyTorch"
txt format instead of COCO JSON, use this to produce a COCO JSON for one
split. Run it once per split (train, valid), then point
convert_roboflow_coco.py-style handling at the results -- or just use the
paths this prints directly as --data-dir input, since it already writes
the train2017/val2017 + annotations/ layout YOLOX expects.

Expected input layout (standard Ultralytics-style Roboflow YOLO export):

    <root>/
        data.yaml          # must contain a `names:` list, e.g. names: [breaker, meter]
        train/
            images/*.jpg
            labels/*.txt   # normalized "class cx cy w h" per line
        valid/
            images/*.jpg
            labels/*.txt

Usage:
    python convert_yolo_txt_to_coco.py --root /path/to/export \
        --output-dir /data/panel_coco
"""
import argparse
import json
import os
import shutil
import sys

from PIL import Image


def load_class_names(root):
    yaml_path = os.path.join(root, "data.yaml")
    if not os.path.isfile(yaml_path):
        sys.exit(f"Expected {yaml_path} with a `names:` list; not found.")
    # Avoid a hard pyyaml dependency for one field: parse the `names:` line(s)
    # by hand. Handles both `names: [a, b, c]` and the multi-line `- a` form.
    with open(yaml_path) as f:
        text = f.read()
    try:
        import yaml
        data = yaml.safe_load(text)
        names = data["names"]
        if isinstance(names, dict):  # some exports use {0: 'a', 1: 'b'}
            names = [names[i] for i in sorted(names)]
        return list(names)
    except ImportError:
        sys.exit("pyyaml not installed -- run `pip install pyyaml` "
                  "(needed just to parse data.yaml's `names:` list).")


def convert_split(images_dir, labels_dir, class_names):
    images, annotations = [], []
    ann_id = 1
    img_id = 1
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(images_dir, fname)
        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": width,
            "height": height,
        })

        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        if os.path.isfile(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, w, h = (float(x) for x in parts[:5])
                    cls = int(cls)
                    box_w, box_h = w * width, h * height
                    x = (cx * width) - box_w / 2
                    y = (cy * height) - box_h / 2
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,  # 0-indexed; matches categories below
                        "bbox": [x, y, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    })
                    ann_id += 1
        img_id += 1

    categories = [{"id": i, "name": n, "supercategory": "none"}
                  for i, n in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-split", default="valid")
    args = ap.parse_args()

    class_names = load_class_names(args.root)
    print(f"Classes from data.yaml: {class_names}")

    os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)

    split_map = {"train": "train2017", args.val_split: "val2017"}
    found_val = os.path.isdir(os.path.join(args.root, args.val_split))
    if not found_val:
        print(f"No '{args.val_split}/' split found -- will duplicate train "
              f"into val (see README: a real val split isn't meaningful at "
              f"this dataset size anyway).")

    train_data = None
    for split_name, out_name in split_map.items():
        images_dir = os.path.join(args.root, split_name, "images")
        labels_dir = os.path.join(args.root, split_name, "labels")
        if not os.path.isdir(images_dir):
            continue
        data = convert_split(images_dir, labels_dir, class_names)
        if split_name == "train":
            train_data = data

        img_out_dir = os.path.join(args.output_dir, out_name)
        os.makedirs(img_out_dir, exist_ok=True)
        for img in data["images"]:
            shutil.copy2(os.path.join(images_dir, img["file_name"]),
                         os.path.join(img_out_dir, img["file_name"]))

        ann_name = f"instances_{'train' if out_name == 'train2017' else 'val'}.json"
        with open(os.path.join(args.output_dir, "annotations", ann_name), "w") as f:
            json.dump(data, f)
        print(f"Wrote {len(data['images'])} image(s) to {img_out_dir}")

    if not found_val and train_data is not None:
        val_dir = os.path.join(args.output_dir, "val2017")
        os.makedirs(val_dir, exist_ok=True)
        for img in train_data["images"]:
            shutil.copy2(os.path.join(args.root, "train", "images", img["file_name"]),
                         os.path.join(val_dir, img["file_name"]))
        with open(os.path.join(args.output_dir, "annotations", "instances_val.json"), "w") as f:
            json.dump(train_data, f)

    labels_path = os.path.join(args.output_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for n in class_names:
            f.write(n + "\n")

    print(f"\nlabels.txt written to {labels_path}")
    print(f"NUM_CLASSES = {len(class_names)}   <-- paste this into "
          f"exps/yolox_nano_panel_finetune.py")


if __name__ == "__main__":
    main()

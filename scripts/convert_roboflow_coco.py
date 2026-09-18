#!/usr/bin/env python3
"""
Convert a Roboflow "COCO JSON" export into the directory layout YOLOX's
COCODataset expects:

    <output-dir>/
        annotations/
            instances_train.json
            instances_val.json
        train2017/   <- images referenced by instances_train.json
        val2017/     <- images referenced by instances_val.json
        labels.txt

Why this reshaping is needed: YOLOX's COCODataset (see yolox/data/datasets/
coco.py in Megvii-BaseDetection/YOLOX) loads images from
`<data_dir>/<name>/<file_name>` where `name` defaults to "train2017" for
training and "val2017" for eval, and reads categories independently from
each split's own JSON via `sorted(coco.getCatIds())`. Roboflow's raw export
(train/, valid/, test/ folders each with their own _annotations.coco.json)
doesn't match that layout, and can carry a "phantom" category with zero
annotations that would otherwise silently become an extra class.

Usage:
    python convert_roboflow_coco.py \
        --roboflow-root /path/to/roboflow_export \
        --output-dir /data/panel_coco \
        [--val-split valid]   # name of the Roboflow folder to use as val;
                               # if it doesn't exist, train is duplicated
                               # into val with a loud warning (see README:
                               # a real val set is not meaningful at 15
                               # images anyway).
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter


def load_split(root, split_name):
    split_dir = os.path.join(root, split_name)
    json_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    return split_dir, data


def used_category_ids(splits):
    used = set()
    for _, data in splits.values():
        for ann in data.get("annotations", []):
            used.add(ann["category_id"])
    return used


def build_filtered_categories(splits):
    """Return the categories list, sorted by id, restricted to ids that
    actually appear on at least one annotation in ANY split. Every output
    JSON gets this exact same list so class-index order matches across
    train/val."""
    used_ids = used_category_ids(splits)
    all_cats = {}
    for _, data in splits.values():
        for cat in data.get("categories", []):
            all_cats[cat["id"]] = cat

    dropped = [c for cid, c in all_cats.items() if cid not in used_ids]
    if dropped:
        print("Dropping categories with zero annotations across all splits "
              "(this is the Roboflow phantom-category case mentioned in the "
              "README):")
        for c in dropped:
            print(f"    id={c['id']} name={c.get('name')!r}")

    kept = sorted((c for cid, c in all_cats.items() if cid in used_ids),
                  key=lambda c: c["id"])
    if not kept:
        sys.exit("No categories with annotations found in any split -- "
                  "check the Roboflow export.")
    return kept


def write_split(output_dir, out_name, src_dir, data, categories):
    img_out_dir = os.path.join(output_dir, out_name)
    os.makedirs(img_out_dir, exist_ok=True)

    copied, missing = 0, []
    for img in data["images"]:
        src = os.path.join(src_dir, img["file_name"])
        dst = os.path.join(img_out_dir, img["file_name"])
        if not os.path.isfile(src):
            missing.append(src)
            continue
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
        copied += 1
    if missing:
        print(f"WARNING: {len(missing)} image(s) listed in JSON but not "
              f"found on disk for split '{out_name}', e.g. {missing[0]}")

    out_json = {
        "images": data["images"],
        "annotations": data["annotations"],
        "categories": categories,
    }
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    ann_name = f"instances_{'train' if out_name == 'train2017' else 'val'}.json"
    with open(os.path.join(ann_dir, ann_name), "w") as f:
        json.dump(out_json, f)

    return copied, ann_name


def write_labels_txt(output_dir, categories):
    # Order must match sorted(category_id) -- the same order COCODataset
    # uses internally for class_ids -- so index N here == model output
    # index N after training.
    path = os.path.join(output_dir, "labels.txt")
    with open(path, "w") as f:
        for cat in categories:
            f.write(cat["name"] + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow-root", required=True,
                     help="Directory containing train/, valid/, (test/) "
                          "subfolders, each with _annotations.coco.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-split", default="valid",
                     help="Roboflow folder name to use as the val split "
                          "(default: 'valid')")
    args = ap.parse_args()

    train = load_split(args.roboflow_root, "train")
    if train is None:
        sys.exit(f"Could not find train/_annotations.coco.json under "
                  f"{args.roboflow_root}")

    splits = {"train": train}
    val = load_split(args.roboflow_root, args.val_split)
    duplicated_val = False
    if val is not None:
        splits["val"] = val
    else:
        print(f"No '{args.val_split}/' split found -- with only "
              f"{len(train[1]['images'])} training images total, a proper "
              f"held-out val split isn't very meaningful anyway. Duplicating "
              f"train into val so the pipeline (which expects a val set to "
              f"exist) still runs. Treat any eval AP from this as a sanity "
              f"check only, not a real metric -- see README section on "
              f"eval_interval.")
        splits["val"] = train
        duplicated_val = True

    categories = build_filtered_categories({k: v for k, v in splits.items()
                                             if not (k == "val" and duplicated_val)})

    os.makedirs(args.output_dir, exist_ok=True)
    train_dir, train_data = splits["train"]
    val_dir, val_data = splits["val"]

    n_train, _ = write_split(args.output_dir, "train2017", train_dir, train_data, categories)
    n_val, _ = write_split(args.output_dir, "val2017", val_dir, val_data, categories)
    labels_path = write_labels_txt(args.output_dir, categories)

    counts = Counter()
    for ann in train_data["annotations"]:
        counts[ann["category_id"]] += 1
    id_to_name = {c["id"]: c["name"] for c in categories}

    print()
    print(f"Wrote {n_train} train image(s), {n_val} val image(s) to {args.output_dir}")
    if duplicated_val:
        print("  (val2017 is a copy of train2017 -- see warning above)")
    print(f"labels.txt written to {labels_path} ({len(categories)} classes):")
    for cat in categories:
        print(f"    {cat['id']:>4}  {cat['name']:<24}  "
              f"{counts.get(cat['id'], 0)} instance(s) in train")
    print()
    print(f"NUM_CLASSES = {len(categories)}   <-- paste this into "
          f"exps/yolox_nano_panel_finetune.py")


if __name__ == "__main__":
    main()

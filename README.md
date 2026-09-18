# PanelBenchmark → YOLOX-Nano fine-tuning pipeline

This is scaffolding for fine-tuning YOLOX-Nano on your ~15-image panel-device
dataset, with MLflow tracking, a CPU pipeline check, and a GPU training
command. It's built against the actual `Megvii-BaseDetection/YOLOX` source
(checked directly against the repo while writing this — versions/APIs do
drift, so if you hit an `AttributeError`, diff against your installed
version's `yolox/exp/yolox_base.py` and `yolox/core/trainer.py`).

**Key finding that simplifies this a lot: YOLOX already ships a built-in
MLflow logger.** You don't need a custom Trainer subclass — `tools/train.py`
(YOLOX's own script) takes a `--logger mlflow` flag and does the rest. This
repo only adds: dataset conversion, a fine-tuning-tuned exp config, and a
CPU sanity script (see below for *why* a separate CPU script exists — the
short version is that YOLOX's real `Trainer` hardcodes CUDA and will not
run on a CPU-only box at all).

## 0. One-time setup

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -e .            # in a venv; add --break-system-packages only for system python
pip install mlflow python-dotenv   # required by YOLOX's built-in MlflowLogger
wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth
```

This mirrors the same torch/torchvision-from-one-index-url and
`--break-system-packages`-only-outside-venv caveats already noted in your
`export_yolox.sh`. Pin `torch`/`torchvision` the same way here.

## 1. Confirm the Roboflow export format

Your note-to-self was right to flag this — it matters. **Re-export from
Roboflow as "COCO JSON"**, not YOLO txt, if you have the choice: YOLOX's
own dataset loader (`COCODataset`) expects COCO JSON natively, and skipping
a conversion step removes one whole class of bugs on a dataset this small
where every image counts.

- Got a COCO JSON export → use `scripts/convert_roboflow_coco.py`.
- Only have a YOLO-txt export (images/ + labels/ + data.yaml) → use
  `scripts/convert_yolo_txt_to_coco.py` first to get COCO JSON, then treat
  its output the same way.

### The Roboflow "phantom category" gotcha

Roboflow's COCO exporter sometimes includes a category that has zero actual
annotations (an artifact of how they build the categories list). YOLOX's
`COCODataset` does `self.class_ids = sorted(coco.getCatIds())` — it reads
**every category in the JSON**, whether or not it's used, and turns each
into a class index. An unused phantom category silently becomes a real
extra class your model has to (fail to) predict. `convert_roboflow_coco.py`
filters categories down to ones that actually have at least one annotation,
and — just as important — writes the **same** category list into every
split's JSON, because `COCODataset` computes class indices independently
per split; if train and val ever disagreed on category order, label 2 in
training and label 2 in eval could be different classes.

```bash
python scripts/convert_roboflow_coco.py \
  --roboflow-root /path/to/roboflow_export \
  --output-dir /data/panel_coco
```

Run this, then:

```bash
python scripts/verify_dataset.py --data-dir /data/panel_coco
```

`verify_dataset.py` prints per-class instance counts, flags any degenerate
boxes, and prints the exact `num_classes` value to paste into the exp file
below. **Read the per-class counts carefully** — with 15 images total,
it's easy for one class to have 1-2 instances, which is worth knowing
about before you interpret any training curve.

## 2. The exp config

Edit `exps/yolox_nano_panel_finetune.py`:

- `NUM_CLASSES` — set to whatever `verify_dataset.py` printed.
- `DATA_DIR` — set to your `--output-dir` from step 1.

Everything else is pre-set for a ~15-image fine-tune off the COCO
checkpoint. The comments in the file explain the reasoning for each
non-default choice (LR, augmentation, freezing, eval). Read them before a
real run — several of these are judgment calls for extreme low-data
fine-tuning, not settled best practice, and you may want to try it both
ways.

## 3. CPU sanity check (dev machine)

YOLOX's real `Trainer` calls `.cuda()` unconditionally in several places
(`self.device = "cuda:{}"`, `DataPrefetcher`'s `torch.cuda.Stream()`,
`Exp.random_resize`'s `.cuda()` tensor) — it will not run at all on a
CPU-only machine, full stop. So this doesn't try to force the real
`Trainer` onto CPU; instead `tools/cpu_sanity_check.py` builds the dataset,
dataloader, and model directly and runs a handful of plain forward/backward
iterations on CPU. It's a pipeline smoke test (does everything load, do
the shapes line up, does loss compute and backprop without NaN), not a
real training run.

```bash
python tools/cpu_sanity_check.py \
  --exp-file exps/yolox_nano_panel_cpu_sanity.py \
  --iters 3
```

If this passes, the annotation format, image paths, category mapping, and
model/loss wiring are all sound — the only thing left to find out on GPU
is whether the actual training run converges sensibly.

## 4. Real training run (GPU machine)

```bash
export MLFLOW_TRACKING_URI=./mlruns          # or a remote server URI
export MLFLOW_EXPERIMENT_NAME=panelbench-yolox-nano
export YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS=True # also upload best_ckpt.pth to MLflow

python YOLOX/tools/train.py \
  -f exps/yolox_nano_panel_finetune.py \
  -d 1 -b 8 \
  -c yolox_nano.pth \
  -expn panel_v1 \
  -l mlflow
```

- `-d 1` = 1 GPU. `-b 8` = batch size; with 15 images and mosaic on, even
  batch 4-8 gives you several iterations per epoch — don't feel obligated
  to go bigger just because the GPU has room.
- `-c yolox_nano.pth` loads the COCO-pretrained weights. YOLOX's own
  `load_ckpt` silently skips any layer whose shape doesn't match (i.e. the
  final classification layers, since your class count ≠ 80) and loads
  everything else — this is exactly what you want for fine-tuning to a new
  class set, no extra flags needed.
- `-l mlflow` turns on YOLOX's built-in MLflow logger. It logs exp
  hyperparameters, per-iteration losses, and (with the env var above) the
  best checkpoint as an artifact. Run `mlflow ui --backend-store-uri
  ./mlruns` to view it.
- Do **not** add `--fp16` on a fine-tune this small; the numerical noise
  isn't worth it for a few-minute training run.

## 5. Back to the export pipeline

Once training finishes, `YOLOX_outputs/panel_v1/best_ckpt.pth` is your
fine-tuned checkpoint — hand it to the existing
`conversion/export_yolox.sh` exactly like `runs/train/exp/weights/best.pt`
was passed before. Two things from your own notes still apply and are easy
to forget:

- Replace the placeholder COCO-80 `labels.txt` with the one
  `convert_roboflow_coco.py` generated (`<output-dir>/labels.txt`) — order
  matters, it must match `sorted(category_id)` order, which is what the
  converter produces.
- Add a fresh `benchmark_config.json` entry pointing at the new ONNX/TFLite
  files; the existing entries still point at the COCO-pretrained model.
- The NMS post-processing step is still unimplemented in the benchmark
  harness (per your notes, YOLOX is not NMS-free) — fine-tuning doesn't
  change that; it's still open work for the Android integration.

## Files

```
scripts/convert_roboflow_coco.py     Roboflow COCO-JSON export -> YOLOX layout
scripts/convert_yolo_txt_to_coco.py  Fallback: Roboflow YOLO-txt export -> COCO JSON
scripts/verify_dataset.py            Sanity-check counts/classes/boxes before training
exps/yolox_nano_panel_finetune.py    Real fine-tuning exp config (edit NUM_CLASSES/DATA_DIR)
exps/yolox_nano_panel_cpu_sanity.py  Same, tuned for the CPU smoke test
tools/cpu_sanity_check.py            Bypasses YOLOX's CUDA-only Trainer for a CPU smoke test
```

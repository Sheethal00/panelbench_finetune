#!/usr/bin/env python3
"""
YOLOX-Nano fine-tuning exp for the panel-device dataset (~15 annotated
images). This is a YOLOX "exp file" -- pass its path directly to
`tools/train.py -f <this file>`; it does not need to live inside the
YOLOX repo.

EDIT THESE TWO BEFORE RUNNING:
"""
NUM_CLASSES = 0          # <-- set from verify_dataset.py / convert script output
DATA_DIR = "/data/panel_coco"   # <-- set to your --output-dir from the converter

import os  # noqa: E402

import torch.nn as nn  # noqa: E402
from loguru import logger  # noqa: E402

from yolox.exp import Exp as MyExp  # noqa: E402


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        if NUM_CLASSES <= 0:
            raise ValueError(
                "Set NUM_CLASSES at the top of this file before training "
                "-- run scripts/verify_dataset.py on your converted dataset "
                "to get the right value."
            )

        # ---------------- model: YOLOX-Nano architecture ---------------- #
        # depth/width/depthwise=True is what makes this "nano" rather than
        # the larger YOLOX variants -- matches exps/default/yolox_nano.py
        # in the upstream repo, since we're fine-tuning the nano checkpoint,
        # not a different-sized one.
        self.depth = 0.33
        self.width = 0.25
        self.act = "silu"
        self.num_classes = NUM_CLASSES
        self.input_size = (416, 416)
        self.test_size = (416, 416)

        # ---------------- data ---------------- #
        self.data_dir = DATA_DIR
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.data_num_workers = 2

        # ---------------- augmentation ---------------- #
        # Mosaic (compositing 4 sampled images together) is one of the only
        # sources of *compositional* diversity available with this few
        # source images, so it stays on -- but kept at nano's already-modest
        # 0.5 probability rather than pushed higher, since with ~12-13
        # training images mosaic will start recombining the same handful of
        # images repeatedly, and more repetition isn't obviously better.
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        # Mixup (alpha-blending two images' pixels together) needs even more
        # source diversity than mosaic to be a good signal rather than noise
        # -- off, same as the nano default.
        self.enable_mixup = False
        # HSV/flip jitter is "free" diversity that doesn't depend on having
        # many source images -- kept aggressive, it's doing more relative
        # work here than in a normal-sized-dataset run.
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0

        # ---------------- training schedule ---------------- #
        # Short run: with a handful of images (few iterations/epoch even
        # with mosaic), there isn't 300 epochs' worth of new information to
        # learn -- past some point you're just re-fitting the same augmented
        # views over and over. 80 is a starting point to inspect loss curves
        # against, not a validated number.
        self.max_epoch = 80
        self.warmup_epochs = 2
        # Base YOLOX-from-scratch LR (0.01/64 per image) assumes randomly
        # initialized weights that need to move a lot. Fine-tuning from the
        # COCO checkpoint on 15 images needs a much gentler LR, or a couple
        # of aggressive gradient steps on a handful of images can wreck the
        # pretrained features before anything useful is learned.
        self.basic_lr_per_img = 0.0005 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05
        # Turn mosaic off (and switch on the extra L1 loss term) for the
        # last quarter of training rather than upstream's ~5% -- the idea
        # being that on a tiny dataset you want a longer final stretch of
        # training on *real, uncomposited* images so the loss you're
        # watching reflects the actual data, not mosaic'd composites.
        self.no_aug_epochs = 20
        self.ema = True  # smooths noisy gradients from tiny/uneven batches
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 1   # few iterations/epoch -- want feedback on all of them
        self.save_history_ckpt = False
        self.seed = 42  # tiny-dataset runs are noisy; fix this to compare changes fairly

        # A real held-out val set isn't statistically meaningful at this
        # dataset size (see README / your own note about k-fold vs
        # loss-only monitoring). Rather than disable eval outright --
        # get_evaluator() still needs a val_ann file to exist -- eval is
        # only run once, at the very end, so "best_ckpt" just becomes the
        # final-epoch checkpoint instead of an early-stopping decision
        # driven by noisy small-sample AP.
        self.eval_interval = self.max_epoch

        # ---------------- freezing ---------------- #
        # Freeze the CSPDarknet stem (the generic low-level feature
        # extractor) and let the PAFPN neck + head adapt. With this little
        # data, letting every layer move risks overwriting the general
        # visual features from COCO pretraining before the model has seen
        # enough panel-device examples to justify moving them. Flip to
        # False for a later experiment once you have more data or want to
        # compare against full fine-tuning.
        self.freeze_stem = True

        self.exp_name = "panel_yolox_nano_v1"

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        if self.freeze_stem:
            self._freeze_stem()

        return self.model

    def _freeze_stem(self):
        stem = self.model.backbone.backbone  # CSPDarknet inside YOLOPAFPN
        n_frozen = 0
        for p in stem.parameters():
            p.requires_grad = False
            n_frozen += 1
        stem.eval()

        # requires_grad=False stops weight updates, but nn.Module.train()
        # (called by the Trainer every epoch) would still flip this
        # submodule's BatchNorm layers back into training mode and let
        # their running_mean/running_var keep updating off tiny, unusual
        # batches -- which defeats the point of "frozen" for a submodule
        # meant to keep its pretrained statistics. Monkey-patch train() on
        # this model instance so the stem is always forced back to eval()
        # regardless of who calls .train().
        model = self.model
        original_train = model.train

        def train_keep_stem_frozen(mode=True):
            original_train(mode)
            stem.eval()
            return model

        model.train = train_keep_stem_frozen

        logger.info(f"[panel-finetune] froze {n_frozen} backbone-stem "
                    f"parameter tensor(s); stem BatchNorm forced to eval() "
                    f"to keep pretrained running stats")

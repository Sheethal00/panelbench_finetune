#!/usr/bin/env python3
"""
Same model/data config as yolox_nano_panel_finetune.py, but tuned for a
quick CPU pipeline smoke test rather than a real training run:
  - mosaic/mixup off (faster, and irrelevant for a shape/plumbing check)
  - 0 dataloader workers (avoids multiprocessing overhead/quirks for a
    3-iteration check)
Used by tools/cpu_sanity_check.py, which does NOT use YOLOX's real Trainer
(that hardcodes CUDA) -- see that script and the README for why.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolox_nano_panel_finetune import Exp as FinetuneExp  # noqa: E402


class Exp(FinetuneExp):
    def __init__(self):
        super().__init__()
        self.mosaic_prob = 0.0
        self.enable_mixup = False
        self.data_num_workers = 0
        self.print_interval = 1
        self.exp_name = "panel_yolox_nano_cpu_sanity"

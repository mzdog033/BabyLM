# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import os
import platform
import shutil
import time
import warnings
import numpy as np
from stat import S_IREAD, S_IRGRP, S_IROTH

import torch
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm

from .dist_utils import use_tpu
from .epoch_based_runner import EpochBasedRunner
from .hooks.optimizer import OptimizerHook, DistOptimizerHook, TPUOptimizerHook
from .hooks.checkpoint import CheckpointHook, TPUCheckpointHook

class TPUEpochBasedRunner(EpochBasedRunner):
    def __init__(self, rec_save_freq=50, *args, **kwargs):
        assert use_tpu(), "Must use TPU!"
        super().__init__(*args, **kwargs)
        self.rec_save_freq = rec_save_freq

    def run_epoch(self):
        self.model.train()
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            if (i+1) % self.rec_save_freq == 0:
                self.record_saver.save(
                        {'train_results': self.iter_outputs})
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def build_data_loader(self):
        super().build_data_loader()
        device = xm.xla_device()
        self.data_loader = pl.MpDeviceLoader(self.data_loader, device)

    def register_optimizer_hook(self):
        opt_hook_builder = self.optimizer_hook_params['builder']
        if opt_hook_builder == OptimizerHook or opt_hook_builder == DistOptimizerHook:
            opt_hook_builder = TPUOptimizerHook

        optimizer_hook = opt_hook_builder(
                **self.optimizer_hook_params['builder_kwargs'])
        assert isinstance(optimizer_hook, TPUOptimizerHook),\
                "Please Use TPU Optimizer Hook!"
        self.register_hook(optimizer_hook)

    def register_save_ckpt_hook(self):
        ckpt_hook_builder = self.save_params['ckpt_hook_builder']
        if ckpt_hook_builder == CheckpointHook:
            ckpt_hook_builder = TPUCheckpointHook
        save_ckpt_hook = ckpt_hook_builder(
                **self.save_params['ckpt_hook_kwargs'])
        assert isinstance(save_ckpt_hook, TPUCheckpointHook), \
                "Please Use TPU Checkpoint Hook!"
        self.register_hook(save_ckpt_hook, priority='LOWEST')
        self.save_ckpt_hook = save_ckpt_hook # for later loading purpose

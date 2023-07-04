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

from .base_runner import BaseRunner
from .checkpoint import symlink, save_checkpoint
from .dist_utils import get_dist_info


class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def run_epoch(self):
        self.model.train()
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.record_saver.save(
                    {'train_results': self.iter_outputs})
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, data_batch, train_mode):
        iter_outputs = self.batch_processor_params['func'](
            self.model, self.loss_func, data_batch,
            **self.batch_processor_params['func_kwargs'])

        if not isinstance(iter_outputs, dict):
            raise TypeError('"batch_processor()" must return a dict')
        assert 'loss' in iter_outputs, "Loss must be included in the outputs"
        self.iter_outputs = iter_outputs

    def train(self, train_upto_epoch=None):
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            if (train_upto_epoch is not None) \
                    and (self.epoch >= train_upto_epoch):
                break
            self.run_epoch()

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def test(self):
        pass

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        only_as_cache=False):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            only_as_cache (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)

        # Mainly used for TPU cases, where this save_checkpoint function 
        # needs to be called in all devices, but we don't want the following things
        # to be called in all devices.
        rank, _ = get_dist_info()
        if rank != 0:
            return
        dst_file = osp.join(out_dir, self.LATEST_CKPT_NAME)

        now_cache = dst_file
        move_list = []
        for idx in range(self.save_params['cache_ckpt_keep_nums']):
            new_cache = osp.join(
                    out_dir, self.PREV_CACHE_CKPT_NAME.format(idx+1))
            if os.path.exists(now_cache):
                move_list.append([now_cache, new_cache])
            now_cache = new_cache
        move_list.reverse()
        for now_cache, new_cache in move_list:
            shutil.move(now_cache, new_cache)

        if not only_as_cache:
            if platform.system() != 'Windows':
                symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
            os.chmod(filepath, S_IREAD|S_IRGRP|S_IROTH)
        else:
            if osp.exists(dst_file):
                os.remove(dst_file)
            shutil.move(filepath, dst_file)

    def cleanup_cached_models(self, out_dir):
        for idx in range(self.save_params['cache_ckpt_keep_nums']):
            new_cache = osp.join(
                    out_dir, self.PREV_CACHE_CKPT_NAME.format(idx+1))
            if os.path.exists(new_cache):
                os.remove(new_cache)


class MultiEpochBasedRunner:
    """Multiple Epoch-based Runners.

    One data provider. Multiple models.
    """
    def __init__(
            self,
            train_data_params,
            other_params,
            dev_mapping=None,
            ):
        self.other_params = other_params
        self.dev_mapping = dev_mapping

        self.check_params()

        self.setup()

    def check_params(self):
        self.check_other_params()
        
    def check_other_params(self):
        message = "Other_params should be a list of dicts"
        assert isinstance(self.other_params, list), message
        assert np.all(
                [isinstance(_other_param, dict) \
                 for _other_param in self.other_params]), message

    def setup(self):
        pass

    def train(self):
        pass

    def test(self):
        raise NotImplementedError

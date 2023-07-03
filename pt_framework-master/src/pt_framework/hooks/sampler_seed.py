from .hook import Hook
from ..dist_utils import use_tpu


class DistSamplerSeedHook(Hook):
    def before_epoch(self, runner):
        if use_tpu():
            loader = runner.data_loader._loader
        else:
            loader = runner.data_loader

        if hasattr(loader.sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            loader.sampler.set_epoch(runner.epoch)
        elif hasattr(loader.batch_sampler.sampler, 'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            loader.batch_sampler.sampler.set_epoch(runner.epoch)

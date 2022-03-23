import torch
from torchvision import transforms


class Jitter(object):
    def __init__(self, noise_range):
        self.noise_range = noise_range

    def __call__(self, sample):

        noise = self.noise_range * (torch.rand(sample.shape) - 0.5)
        sample = sample + noise

        return sample


class DownSample(object):
    def __init__(self, skip_nth_step):
        assert isinstance(skip_nth_step, int)
        self.skip_nth_step = skip_nth_step + 1

    def __call__(self, sample):

        nth_range = torch.arange(0, sample.shape[0], step=self.skip_nth_step)
        down_sample = torch.index_select(sample, dim=0, index=nth_range)

        return down_sample

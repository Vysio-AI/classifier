import torch
from torchvision import transforms


class Jitter(object):
    def __init__(self, noise_range):
        self.noise_range = noise_range

    def __call__(self, sample):

        noise = self.noise_range * (torch.rand(sample.shape) - 0.5)
        sample = sample + noise

        return sample

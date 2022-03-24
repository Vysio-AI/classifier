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


class Rotate(object):
    def __init__(self):

        self.rotation_methods = [
            self.z_180,
            self.z_pos_90,
            self.z_neg_90,
            self.y_180,
            self.x_180,
        ]

    def __call__(self, sample):

        # choose an index for one of the methods
        rand_int = torch.randint(0, high=len(self.rotation_methods) + 1, size=(1,))[0]

        # choice of 0 does nothing
        if rand_int > 0:
            sample = self.rotation_methods[rand_int](sample)

        return sample

    def z_pos_90(self, sample: torch.Tensor):
        # a_y = -a_x
        # a_x = a_y
        # w_x = w_y
        # w_y = -w_x
        idx_swap = torch.IntTensor([1, 0, 2, 4, 3, 5])
        sample = sample.index_select(0, idx_swap)
        sample[1] = -1 * sample[1]
        sample[4] = -1 * sample[4]
        return sample

    def z_neg_90(self, sample):
        # a_y = a_x
        # a_x = -a_y
        # w_x = -w_y
        # w_y = w_x
        idx_swap = torch.IntTensor([1, 0, 2, 4, 3, 5])
        sample = sample.index_select(0, idx_swap)
        sample[0] = -1 * sample[0]
        sample[3] = -1 * sample[3]
        return sample

    def z_180(self, sample):
        # a_y = -a_y
        # a_x = -a_x
        # w_y = -w_y
        # w_x = -w_x
        sample[0] = -1 * sample[0]
        sample[1] = -1 * sample[1]
        sample[3] = -1 * sample[3]
        sample[4] = -1 * sample[4]
        return sample

    def y_180(self, sample):
        # a_z = -a_z
        # a_x = -a_x
        # w_z = -w_z
        # w_x = -w_x
        sample[0] = -1 * sample[0]
        sample[2] = -1 * sample[2]
        sample[3] = -1 * sample[3]
        sample[5] = -1 * sample[5]
        return sample

    def x_180(self, sample):
        # a_z = -a_z
        # a_y = -a_y
        # w_z = -w_z
        # w_y = -w_y
        sample[1] = -1 * sample[1]
        sample[2] = -1 * sample[2]
        sample[4] = -1 * sample[4]
        sample[5] = -1 * sample[5]
        return sample

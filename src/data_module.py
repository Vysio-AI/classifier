from enum import Enum
import json
import os
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class LearningPhase(Enum):
    """Class enumerating learning phases"""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

class SparDataset(Dataset):
    """Spar Dataset"""

    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None


class ShoulderExerciseDataModule(pl.LightningDataModule):
    """Shoulder Exercise Data Module"""

    def __init__(self, **kwargs):
        super().__init__()

        self.args = kwargs

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

import json
import math
import os
import pathlib
from enum import Enum
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGURATIONS = EasyDict(yaml.load(f, yaml.FullLoader))
    DATA_CONFIG = CONFIGURATIONS["data_generation"]


class LearningPhase(Enum):
    """Class enumerating learning phases"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SparDataset(Dataset):
    """Spar Dataset"""

    def __init__(self, data_type: LearningPhase, data_dir, window_size, window_stride):

        assert isinstance(data_type, LearningPhase)

        self.data = []
        self.classes = ["PEL", "ABD", "FEL", "IR", "ER", "TRAP", "ROW"]
        self.csv_data_columns = ["ax", "ay", "az", "wx", "wy", "wz"]
        self.window_size = window_size
        self.window_stride = window_stride

        # Grab the list of file patterns for relevant csv files
        csv_file_patterns = DATA_CONFIG[f"{data_type.value}_csv_file_pattern"]
        assert isinstance(csv_file_patterns, list)
        csv_file_list = []
        # Generate a list of all matching csv file paths
        for pattern in csv_file_patterns:
            csv_file_list.extend(glob(pattern, recursive=True))

        print(f"[info] sourcing {len(csv_file_list)} {data_type.value} csv files")
        print(f"[info] storing windows to {data_dir}")
        # Checks if destination folder already exists since it will
        os.makedirs(data_dir, exist_ok=True)

        # store paths for generated data windows
        for csv_path in tqdm(csv_file_list):
            csv_filename = pathlib.Path(csv_path).stem
            csv_directory_name = pathlib.Path(csv_path).parent.stem
            # Grab class from csv filename of format 'S1_E0_R'
            y_class = int(csv_filename.split("_")[1][1])

            # Generate the appropriate onehot label
            y_onehot = np.zeros(len(self.classes), int)
            y_onehot[y_class] = 1

            # read csv as pandas dataframe
            csv_data = pd.read_csv(csv_path)
            # extract the relevant columns
            csv_data = csv_data[self.csv_data_columns]
            assert not csv_data.empty
            assert not csv_data.isnull().values.any()

            # Number of sequences expected to be generated
            # Incomplete sequences are ignored
            number_of_sequences = math.floor(
                (csv_data.shape[0] - self.window_size) / self.window_stride + 1
            )

            # Store the generated sequences from the sliding window
            sequence_list = []
            for count in range(number_of_sequences):
                start = count * self.window_stride
                end = count * self.window_stride + self.window_size
                sequence = csv_data[start:end].to_numpy(dtype="float32")
                sequence_list.append(sequence)

                # Create the window data file path
                csv_window_path = os.path.join(
                    data_dir,
                    "{}_{}_sequence_{}_size_{}_shift_{}.tfrecord".format(
                        csv_directory_name,
                        csv_filename,
                        count,
                        self.window_size,
                        self.window_stride,
                    ),
                )

                # Write window data to csv file
                csv_data[start:end].to_csv(csv_window_path)
                # Store file path and label
                self.data.append([csv_window_path, y_onehot])

            # Assert the overlaps in the sequence list match
            sequence_list = np.array(sequence_list)
            overlap = self.window_size - self.window_stride
            if overlap > 0:
                for idx in range(sequence_list.shape[0] - 1):
                    assert np.all(
                        sequence_list[idx, -overlap:]
                        == sequence_list[idx + 1, :overlap]
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_path, y_onehot = self.data[idx]

        # read csv as pandas dataframe
        csv_data = pd.read_csv(x_path)
        # extract the relevant columns
        csv_data = csv_data[self.csv_data_columns]
        # store x and y as tensors
        x_tensor = torch.from_numpy(csv_data.to_numpy(dtype="float32"))
        label = torch.from_numpy(y_onehot)
        return {
            "timeseries": x_tensor,
            "label": label,
            "class": self.classes[np.argmax(label)],
        }


class ShoulderExerciseDataModule(pl.LightningDataModule):
    """Shoulder Exercise Data Module"""

    def __init__(self, **kwargs):
        super().__init__()

        self.args = kwargs
        self.window_size = self.args["window_size"]
        self.window_stride = self.args["window_stride"]
        self.data_dir = self.args["data_dir"]
        self.batch_size = self.args["batch_size"]

    def train_dataloader(self):
        train_dataset = SparDataset(
            LearningPhase.TRAIN,
            data_dir=self.data_dir,
            window_size=self.window_size,
            window_stride=self.window_stride,
        )
        print("[info] sourced {} training windows".format(len(train_dataset)))
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = SparDataset(
            LearningPhase.VALIDATION,
            data_dir=self.data_dir,
            window_size=self.window_size,
            window_stride=self.window_stride,
        )
        print("[info] sourced {} validation windows".format(len(val_dataset)))
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_dataset = SparDataset(
            LearningPhase.TEST,
            data_dir=self.data_dir,
            window_size=self.window_size,
            window_stride=self.window_stride,
        )
        print("[info] sourced {} test windows".format(len(test_dataset)))
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__":

    dataset = SparDataset(
        LearningPhase.VALIDATION,
        data_dir="./tmp/spardata",
        window_size=100,
        window_stride=50,
    )
    data_loader = DataLoader(dataset, batch_size=128)

    for item in data_loader:
        # import pdb; pdb.set_trace()
        print("x shape: ", item.get("timeseries").shape)
        print("y shape: ", item.get("label").shape)
        print("sample_0 label: ", item.get("label")[0])
        print("sample_0 class: ", item.get("class")[0])
        break

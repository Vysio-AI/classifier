import json
import math
import os
import pathlib
from enum import Enum
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.io
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
        csv_file_patterns = DATA_CONFIG[f"spar_{data_type.value}_csv_file_pattern"]
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
                    "{}_{}_sequence_{}_size_{}_shift_{}.csv".format(
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

        self.dataset_dict = {"spar": SparDataset, "utd_mhad": UTDMHADDataset}

        self.args = kwargs
        self.window_size = self.args["window_size"]
        self.window_stride = self.args["window_stride"]
        self.data_dir = self.args["data_dir"]
        self.batch_size = self.args["batch_size"]
        self.num_workers = self.args["num_workers"]
        self.dataset = self.dataset_dict[self.args["dataset"]]

    def train_dataloader(self):
        train_dataset = self.dataset(
            LearningPhase.TRAIN,
            data_dir=self.data_dir,
            window_size=self.window_size,
            window_stride=self.window_stride,
        )
        print("[info] sourced {} training windows".format(len(train_dataset)))
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        val_dataset = self.dataset(
            LearningPhase.VALIDATION,
            data_dir=self.data_dir,
            window_size=self.window_size,
            window_stride=self.window_stride,
        )
        print("[info] sourced {} validation windows".format(len(val_dataset)))
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        test_dataset = self.dataset(
            LearningPhase.TEST,
            data_dir=self.data_dir,
            window_size=self.window_size,
            window_stride=self.window_stride,
        )
        print("[info] sourced {} test windows".format(len(test_dataset)))
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class UTDMHADDataset(Dataset):
    """
    Dataset class for UTD Multimodal Human Action Dataset

    Example expected file path:
    sub1/left_45_degree/catch/a_s_t2_skel_*.mat
    Therefore:
    - subject = 1
    - orientation = left_45_degree (ignore)
    - activity = catch
    - trial = 2
    - data type = skel
    """

    def __init__(self, data_type: LearningPhase, data_dir, window_size, window_stride):
        assert isinstance(data_type, LearningPhase)

        self.data = []
        self.csv_data_columns = ["ax", "ay", "az", "wx", "wy", "wz"]
        self.window_size = window_size
        self.window_stride = window_stride
        self.data_dir = data_dir

        self.class_to_num = {
            "catch": 0,
            "draw_circle": 1,
            "draw_tick": 2,
            "draw_triangle": 3,
            "knock": 4,
            "throw": 5,
        }

        # Grab the list of file patterns for relevant data files
        file_patterns = DATA_CONFIG[f"mhad_{data_type.value}_csv_file_pattern"]
        assert isinstance(file_patterns, list)
        file_list = []
        # Generate a list of all matching csv file paths
        for pattern in file_patterns:
            file_list.extend(glob(pattern, recursive=True))

        # Checks if destination folder already exists since it will
        os.makedirs(self.data_dir, exist_ok=True)

        for file_path in tqdm(file_list):
            file_path = pathlib.Path(file_path)
            file_name = file_path.stem

            file_type = file_name.split("_")[3]
            assert file_type in ["depth", "inertial", "skel"]

            activity = file_path.parents[0].stem
            assert activity in self.class_to_num

            trial_num = int(file_name.split("_")[2][1])
            orientation = file_path.parents[1].stem
            subject_num = int(file_path.parents[2].stem[-1])

            csv_filename = "S{}_E{}_T{}_{}_{}".format(
                subject_num,
                self.class_to_num[activity],
                trial_num,
                orientation,
                file_type,
            )

            if file_type == "inertial":
                mat_data = scipy.io.loadmat(file_path)["d_iner"]
                data_frame = pd.DataFrame(
                    mat_data, columns=["ax", "ay", "az", "wx", "wy", "wz"]
                )

    def window_data(self, csv_data: pd.DataFrame, csv_filename):
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
                self.data_dir,
                "{}_{}.csv".format(csv_filename, count),
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
                    sequence_list[idx, -overlap:] == sequence_list[idx + 1, :overlap]
                )


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

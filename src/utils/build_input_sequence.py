import math
import os
import pathlib
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from easydict import EasyDict
from tqdm import tqdm

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGURATIONS = EasyDict(yaml.load(f, yaml.FullLoader))
    DATA_CONFIG = CONFIGURATIONS["data_generation"]
    DEMO_INPUT = CONFIGURATIONS["demo_input"]
    class_config = DEMO_INPUT["class_config"]


def generate_demo_input(action_sequence_path_list=DEMO_INPUT["action_sequences"]):
    """
    Create a 'realistic' session IMU output from the action_sequence_path_list
    """

    print(f"[info] action_sequence_path_list: {action_sequence_path_list}")

    action_duration = DEMO_INPUT["action_duration"]
    overlap_length = DEMO_INPUT["overlap_length"]
    none_class_val = DEMO_INPUT["none_class_val"]

    num_actions = len(action_sequence_path_list)
    num_steps = num_actions * action_duration - overlap_length * (num_actions - 1)
    input_sequence_X = np.zeros((num_steps, 6))
    input_sequence_Y = np.zeros(num_steps, dtype=int)

    print(f"[info] input_sequence_X.shape: {input_sequence_X.shape}")

    start_index = 0

    # Process each timeseries for each exercise performed by each subject
    for csv_path in action_sequence_path_list:

        print(csv_path)

        csv_filename = pathlib.Path(csv_path).stem
        csv_directory_name = pathlib.Path(csv_path).parent.stem
        # Grab class from csv filename of format 'S1_E0_R'
        y_class = int(csv_filename.split("_")[1][1])

        # read csv as pandas dataframe
        csv_data = pd.read_csv(csv_path)
        # extract the relevant columns
        csv_data = csv_data[["ax", "ay", "az", "wx", "wy", "wz"]]
        print("csv_data.shape = {}".format(csv_data.shape))
        assert not csv_data.empty
        assert not csv_data.isnull().values.any()
        # TODO Might be helpful to make sure action duration isn't too long

        overlap_end = start_index + overlap_length
        end = start_index + action_duration
        print(start_index, overlap_end, end)

        # add non overlap portion
        assert overlap_end - end == overlap_length - action_duration
        input_sequence_X[overlap_end:end] = csv_data[overlap_length:action_duration]
        input_sequence_Y[overlap_end:end] = [y_class] * (
            action_duration - overlap_length
        )
        # additive overlap
        input_sequence_X[start_index:overlap_end] += csv_data[:overlap_length]
        # TODO if start don't use non class val
        input_sequence_Y[start_index:overlap_end] = [none_class_val] * overlap_length

        if start_index == 0:
            input_sequence_Y[start_index:overlap_end] = [y_class] * overlap_length

        start_index += action_duration - overlap_length

    unique_classes, class_counts = np.unique(input_sequence_Y, return_counts=True)
    unique_classes = dict(zip(unique_classes, class_counts))
    print("[info] unique_classes = {}".format(unique_classes))
    assert unique_classes[none_class_val] == (num_actions - 1) * overlap_length
    return input_sequence_X, input_sequence_Y


def plot_demo_input(input_X, input_Y):
    fig, ax = plt.subplots(6, sharex=True)

    for count, point in enumerate(zip(input_X, input_Y)):
        x = point[0]
        y = point[1]
        time = count /50 # sample / 50 Hz = seconds
        for dim in range(6):
            ax[dim].plot(time, x[dim], marker="o", markersize=1, color=class_config[str(y)]['color'])

    plt.show()


if __name__ == "__main__":
    X, Y = generate_demo_input()
    plot_demo_input(X, Y)

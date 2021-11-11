import math
import os
import pathlib

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

FEATURE_MAP = {
    "n_steps": tf.io.FixedLenFeature([], tf.int64),
    "n_features": tf.io.FixedLenFeature([], tf.int64),
    # "subject": tf.io.FixedLenFeature([], tf.int64),
    # "side": tf.io.FixedLenFeature([], tf.int64),
    # "y": tf.io.FixedLenFeature([], tf.int64),
    "y_onehot": tf.io.VarLenFeature(tf.int64),
    "X": tf.io.VarLenFeature(tf.float32),
    # "y_label": tf.io.FixedLenFeature([], tf.string),
    # "x_labels": tf.io.FixedLenFeature([], tf.string),
}


def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _bytes_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64 from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _get_tfrecord_features(X, y):

    # Generate the appropriate onehot label
    y_onehot = [0, 0, 0, 0, 0, 0, 0]
    y_onehot[y] = 1

    feature = {
        "n_steps": _int64_feature(X.shape[0]),
        "n_features": _int64_feature(X.shape[1]),
        "y_onehot": _int64_list_feature(y_onehot),
        "X": _float_list_feature(X.ravel().tolist()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_window_tfrecords(
    csv_path_list,
    tfrecord_windows_destination=DATA_CONFIG["tfrecord_windows_destination"],
):
    """
    Generate tfrecords holding a window of time series steps from each csv.
    Filename format: S1_E0_R = subject 1, activity 0 (PEN), right side
    Expected columns: ax, ay, az, wx, wy, wz
    Returns: A list of generated tfrecord locations
    """

    print(f"[info] storing tfrecords to {tfrecord_windows_destination}")

    # Checks if destination folder already exists since it will
    os.makedirs(tfrecord_windows_destination, exist_ok=True)

    # store paths for generated tfrecords
    tfrecord_path_list = []
    # Process each timeseries for each exercise performed by each subject
    for csv_path in tqdm(csv_path_list):

        csv_filename = pathlib.Path(csv_path).stem
        csv_directory_name = pathlib.Path(csv_path).parent.stem
        # Grab class from csv filename of format 'S1_E0_R'
        y_class = int(csv_filename.split("_")[1][1])

        # read csv as pandas dataframe
        csv_data = pd.read_csv(csv_path)
        # extract the relevant columns
        csv_data = csv_data[["ax", "ay", "az", "wx", "wy", "wz"]]
        assert not csv_data.empty
        assert not csv_data.isnull().values.any()

        # get windowing parameters
        window_size = DATA_CONFIG["window_size"]
        window_shift_length = DATA_CONFIG["window_shift_length"]

        # The number of sequences expected to be generated from this time series
        # Incomplete sequences are ignored
        number_of_sequences = math.floor(
            (csv_data.shape[0] - window_size) / window_shift_length + 1
        )

        # Store the generated sequences from the sliding window
        sequence_list = []
        for count in range(number_of_sequences):
            start = count * window_shift_length
            end = count * window_shift_length + window_size
            sequence = csv_data[start:end].to_numpy(dtype="float32")
            sequence_list.append(sequence)

            tf_features = _get_tfrecord_features(X=sequence, y=y_class)

            # Create the tfrecord file path
            tfrecord_path = os.path.join(
                tfrecord_windows_destination,
                "{}_{}_sequence_{}_size_{}_shift_{}.tfrecord".format(
                    csv_directory_name,
                    csv_filename,
                    count,
                    window_size,
                    window_shift_length,
                ),
            )

            # Write tfrecord to memory
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                writer.write(tf_features.SerializeToString())

            # Test the first tfrecord generation
            if count == 0:
                _test_tfrecord_generation(tfrecord_path, X0=sequence, y0=y_class)

            tfrecord_path_list.append(tfrecord_path)

        # Assert the overlaps in the sequence list match
        sequence_list = np.array(sequence_list)
        for idx in range(sequence_list.shape[0] - 1):
            overlap = window_size - window_shift_length
            assert np.all(
                sequence_list[idx, -overlap:] == sequence_list[idx + 1, :overlap]
            )

    return tfrecord_path_list


def _test_tfrecord_generation(tfrecord_path, X0, y0):
    def _parse_exercise_example(tfrecord):
        """Get exercise data from tfrecord"""
        parsed_example = tf.io.parse_single_example(tfrecord, FEATURE_MAP)
        return parsed_example

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_exercise_example)

    for example in parsed_dataset.take(-1):
        # Generate the appropriate onehot label
        y0_onehot = np.zeros(7, dtype="uint64")
        y0_onehot[y0] = 1
        y_onehot = tf.sparse.to_dense(example["y_onehot"])
        assert np.all(y0_onehot == y_onehot.numpy())
        assert X0.shape[0] == example["n_steps"].numpy()
        assert X0.shape[1] == example["n_features"].numpy()
        X_flat = tf.sparse.to_dense(example["X"])
        X = tf.reshape(X_flat, [example["n_steps"], example["n_features"]])
        assert np.all(X0 == X.numpy())

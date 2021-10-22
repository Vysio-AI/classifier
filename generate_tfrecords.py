import math
import os
import pathlib
import pdb

import numpy as np
import seglearn
import tensorflow as tf
import yaml
from easydict import EasyDict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGURATIONS = EasyDict(yaml.load(f, yaml.FullLoader))
    DATA_CONFIG = CONFIGURATIONS["data_generation"]

FEATURE_MAP = {
    "n_steps": tf.io.FixedLenFeature([], tf.int64),
    "n_features": tf.io.FixedLenFeature([], tf.int64),
    "subject": tf.io.FixedLenFeature([], tf.int64),
    "side": tf.io.FixedLenFeature([], tf.int64),
    "y": tf.io.FixedLenFeature([], tf.int64),
    "y_onehot": tf.io.VarLenFeature(tf.int64),
    "X": tf.io.VarLenFeature(tf.float32),
    "y_label": tf.io.FixedLenFeature([], tf.string),
    "x_labels": tf.io.FixedLenFeature([], tf.string),
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


def _int_to_onehot(value):
    """Convert an integer in the range [0,7] to an 7 element onehot array"""


def _get_tfrecord_features(X, y, y_label, x_labels, side, subject):
    """Generate the features for the TFRecord file"""

    # Generate the appropriate onehot label
    y_onehot = [0, 0, 0, 0, 0, 0, 0]
    y_onehot[y] = 1

    feature = {
        "n_steps": _int64_feature(X.shape[0]),
        "n_features": _int64_feature(X.shape[1]),
        "subject": _int64_feature(subject),
        "side": _int64_feature(side),
        "y": _int64_feature(y),
        "X": _float_list_feature(X.ravel().tolist()),
        "y_label": _bytes_feature(str.encode(y_label)),
        "y_onehot": _int64_list_feature(y_onehot),
        "x_labels": _bytes_feature(np.array(x_labels).tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_seglearn_tfrecords(
    tfrecord_destination=DATA_CONFIG["tfrecord_destination"],
    spar_dataset_path=DATA_CONFIG["spar_dataset_path"],
):
    """Generate and store tfrecords for each subject-exercise-side time series
    data from the seglearn module
    """

    seglearn_watch_data = seglearn.datasets.load_watch()

    # Create a directory for tfrecords
    os.makedirs(tfrecord_destination, exist_ok=True)

    number_of_examples = len(seglearn_watch_data["X"])

    # Store each set of time-series as a tfrecord
    for idx in tqdm(range(number_of_examples)):
        # Extract relavent data to record
        # TODO: Check that float32 conversion is accurate
        X = np.array(seglearn_watch_data["X"][idx], dtype="float32")
        y = int(seglearn_watch_data["y"][idx])
        y_label = seglearn_watch_data["y_labels"][y]
        subject = seglearn_watch_data["subject"][idx]
        side = int(seglearn_watch_data["side"][idx])
        x_labels = seglearn_watch_data["X_labels"]

        # Generate tfrecord example
        tf_example = _get_tfrecord_features(
            X=X, y=y, y_label=y_label, subject=subject, side=side, x_labels=x_labels
        )

        # Create the tfrecord file path
        tfrecord_path = os.path.join(
            tfrecord_destination,
            "seglearn_S{}_E{}_{}.tfrecord".format(subject, y, side),
        )

        # Write tfrecord to memory
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            writer.write(tf_example.SerializeToString())

        # Test the first tfrecord generation
        if idx == 0:
            _test_tfrecord_generation(
                tfrecord_path, X, y, y_label, subject, side, x_labels
            )


def generate_spar_tfrecords(
    tfrecord_destination=DATA_CONFIG["tfrecord_destination"],
    spar_dataset_path=DATA_CONFIG["spar_dataset_path"],
):
    """
    Generate and store tfrecords for each subject-exercise-side time series
    data from the SPAR (github.com/dmbee/SPAR-dataset)
    Consists of 6-axis inertial sensor data (accelerometer and gyroscope)
    collected using an Apple Watch 2 and Apple Watch 3 from 20 healthy
    subjects (40 shoulders), as they perform 7 shoulder physiotherapy exercises.

    The activities are:

    1. Pendulum (PEN)
    2. Abduction (ABD)
    3. Forward elevation (FEL)
    4. Internal rotation with resistance band (IR)
    5. External rotation with resistance band (ER)
    6. Lower trapezius row with resistance band (TRAP)
    7. Bent over row with 3 lb dumbell (ROW)
    The subjects repeat each activity 20 times on each side (left and right).

    The data is available in csv format in the csv folder. Each file represents
    a single activity being repeated 20 times. The files are named to convey:

    S1_E0_R
    indicated subject 1, activity 0 (PEN), right side
    """

    spar_dataset = np.load(spar_dataset_path, allow_pickle=True).item()

    # Create a directory for tfrecords
    os.makedirs(tfrecord_destination, exist_ok=True)

    number_of_examples = len(spar_dataset["X"])

    # Store each set of time-series as a tfrecord
    for idx in tqdm(range(number_of_examples)):
        # Extract relavent data to record
        # TODO: Check that float32 conversion is accurate
        X = np.array(spar_dataset["X"][idx], dtype="float32")
        y = int(spar_dataset["y"][idx])
        y_label = spar_dataset["y_labels"][y]
        subject = spar_dataset["subject"][idx]
        side = int(spar_dataset["side"][idx])
        x_labels = spar_dataset["X_labels"]

        # Generate tfrecord example
        tf_example = _get_tfrecord_features(
            X=X, y=y, y_label=y_label, subject=subject, side=side, x_labels=x_labels
        )

        # Create the tfrecord file path
        tfrecord_path = os.path.join(
            tfrecord_destination, "spar_S{}_E{}_{}.tfrecord".format(subject, y, side)
        )

        # Write tfrecord to memory
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            writer.write(tf_example.SerializeToString())

        # Test the first tfrecord generation
        if idx == 0:
            _test_tfrecord_generation(
                tfrecord_path, X, y, y_label, subject, side, x_labels
            )


def generate_windowed_tfrecords(
    tfrecord_windows_destination=DATA_CONFIG["tfrecord_windows_destination"],
    tfrecord_source=DATA_CONFIG["tfrecord_destination"],
):
    """Convert tfrecords generated by `generate_seglearn_tfrecords`,
    segment them into windows, and save them as individual tfrecords"""
    print(f"[info] sourcing tfrecords from {tfrecord_source}")
    print(f"[info] storing tfrecords to {tfrecord_windows_destination}")
    tfrecord_source = pathlib.Path(tfrecord_source)
    # Grab a list of all the tfrecords from the source directory
    file_list = list(tfrecord_source.glob("**/*.tfrecord"))
    # Checks if destination folder already exists since it will
    os.makedirs(tfrecord_windows_destination, exist_ok=True)

    # Process each timeseries for each exercise performed by each subject
    for tfrecord_path in tqdm(file_list):
        tfrecord_path_stem = tfrecord_path.stem
        tfrecord_path = str(tfrecord_path)
        # Extract data from the tfrecord
        tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_path)
        tfrecord_dataset = tfrecord_dataset.map(
            lambda x: tf.io.parse_single_example(x, FEATURE_MAP)
        )
        example = {}
        for item in tfrecord_dataset.take(1):
            example = item
        # Convert the 1D data into its original dimensions
        X_flat_tensor = tf.sparse.to_dense(example["X"])
        X_tensor = tf.reshape(
            X_flat_tensor, [example["n_steps"], example["n_features"]]
        )
        X = X_tensor.numpy()
        # Converted the byte-converted numpy array back to an array of strings
        x_labels = np.frombuffer(example["x_labels"].numpy(), dtype="<U2")
        y_label = example["y_label"].numpy().decode("utf-8")
        y = example["y"].numpy()
        subject = example["subject"].numpy()
        side = example["side"].numpy()
        window_size = DATA_CONFIG["window_size"]
        window_shift_length = DATA_CONFIG["window_shift_length"]
        # The number of sequences expected to be generated from this time series
        number_of_sequences = math.floor(
            (X.shape[0] - window_size) / window_shift_length + 1
        )
        # Store the generated sequences from the sliding window
        sequence_list = []
        # Iterate through each sequence and generate a tfrecord
        # while sequences are ignored
        for count in range(number_of_sequences):
            start = count * window_shift_length
            end = count * window_shift_length + window_size
            sequence = X[start:end]
            sequence_list.append(sequence)

            tf_features = _get_tfrecord_features(
                X=sequence,
                x_labels=x_labels,
                y_label=y_label,
                y=y,
                subject=subject,
                side=side,
            )

            # Create the tfrecord file path
            tfrecord_path = os.path.join(
                tfrecord_windows_destination,
                "{}_sequence_{}_size_{}_shift_{}.tfrecord".format(
                    tfrecord_path_stem, count, window_size, window_shift_length
                ),
            )

            # Write tfrecord to memory
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                writer.write(tf_features.SerializeToString())

            # Test the first tfrecord generation
            if count == 0:
                _test_tfrecord_generation(
                    tfrecord_path,
                    X0=sequence,
                    y0=y,
                    y_label0=y_label,
                    subject0=subject,
                    side0=side,
                    x_labels0=x_labels,
                )

        # Assert the overlaps in the sequence list match
        sequence_list = np.array(sequence_list)
        for idx in range(sequence_list.shape[0] - 1):
            overlap = window_size - window_shift_length
            assert np.all(
                sequence_list[idx, -overlap:] == sequence_list[idx + 1, :overlap]
            )


def _test_tfrecord_generation(
    tfrecord_path, X0, y0, y_label0, subject0, side0, x_labels0
):
    """Assert that the data stored in the tfrecords is consistent and
    retrievable"""

    def _parse_exercise_example(tfrecord):
        """Get exercise data from tfrecord"""
        parsed_example = tf.io.parse_single_example(tfrecord, FEATURE_MAP)
        # X_flat = tf.sparse.to_dense(parsed_example["X"])
        # X = tf.reshape(
        #     X_flat, [parsed_example["n_steps"], parsed_example["n_features"]]
        # )
        # X_flat = tf.sparse.to_dense(parsed_example["X"])
        # x_labels = tf.sparse.to_dense(parsed_example['x_labels'])
        # X = tf.sparse.to_dense(parsed_example["X"])
        # subject = tf.sparse.to_dense(parsed_example["subject"])
        # x_labels = tf.sparse.to_dense(parsed_example["x_labels"])
        # y_labels = tf.sparse.to_dense(parsed_example["y_labels"])
        # decode_feat = tf.io.decode_raw(f)
        return parsed_example
        # return X, parsed_example['subject'], parsed_example['y'], parsed_example['y_label']
        # return x_labels

    # print(f"[info] Testing: {tfrecord_path}")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_exercise_example)
    # print(f"[info] parsed_dataset:\n{parsed_dataset}")

    for example in parsed_dataset.take(-1):
        # print("[info] example:\n{}".format(example))
        # Generate the appropriate onehot label
        y0_onehot = np.zeros(7, dtype="uint64")
        y0_onehot[y0] = 1
        y_onehot = tf.sparse.to_dense(example["y_onehot"])
        assert np.all(y0_onehot == y_onehot.numpy())
        assert y0 == example["y"].numpy()
        assert X0.shape[0] == example["n_steps"].numpy()
        assert X0.shape[1] == example["n_features"].numpy()
        assert subject0 == example["subject"].numpy()
        assert side0 == example["side"].numpy()
        assert str.encode(y_label0) == example["y_label"].numpy()
        x_labels = np.frombuffer(example["x_labels"].numpy(), dtype="<U2")
        assert np.all(x_labels0 == x_labels)

        # Compare X data
        X_flat = tf.sparse.to_dense(example["X"])
        X = tf.reshape(X_flat, [example["n_steps"], example["n_features"]])
        assert np.all(X0 == X.numpy())

if __name__ == "__main__":
    # generate_seglearn_tfrecords()
    # generate_spar_tfrecords()
    generate_windowed_tfrecords()

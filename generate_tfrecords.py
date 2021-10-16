import math
import os
import pathlib
import pdb

import numpy as np
import seglearn
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
    "subject": tf.io.FixedLenFeature([], tf.int64),
    "side": tf.io.FixedLenFeature([], tf.int64),
    "y": tf.io.FixedLenFeature([], tf.int64),
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


def _get_tfrecord_features(X, y, y_label, x_labels, side, subject):
    """Generate the features for the TFRecord file"""

    feature = {
        "n_steps": _int64_feature(X.shape[0]),
        "n_features": _int64_feature(X.shape[1]),
        "subject": _int64_feature(subject),
        "side": _int64_feature(side),
        "y": _int64_feature(y),
        "X": _float_list_feature(X.ravel().tolist()),
        "y_label": _bytes_feature(str.encode(y_label)),
        "x_labels": _bytes_feature(np.array(x_labels).tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_seglearn_tfrecords(
    tfrecord_destination=DATA_CONFIG["tfrecord_destination"],
):
    """Generate and store tfrecords for each subject-exercise time series
    data from the seglearn module"""

    seglearn_watch_data = seglearn.datasets.load_watch()
    print(f"[info] seglearn_watch_data.keys() = {seglearn_watch_data.keys()}")

    # Create a directory for tfrecords
    os.makedirs(tfrecord_destination, exist_ok=True)

    number_of_examples = len(seglearn_watch_data["X"])

    # Store each set of time-series as a tfrecord
    for idx in tqdm(range(number_of_examples)):
        # Extract relavent data to record
        # TODO: Check that float32 conversion is accurate
        X = np.array(seglearn_watch_data["X"][idx], dtype="float32")
        y = seglearn_watch_data["y"][idx]
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
            tfrecord_destination, "{}_{}.tfrecord".format(subject, y_label)
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
    os.makedirs(tfrecord_windows_destination, exist_ok=False)

    # Process each timeseries for each exercise performed by each subject
    for tfrecord_path in tqdm(file_list):
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
                "subject_{}_label_{}_window_{}.tfrecord".format(
                    subject, y_label, count
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
        X_flat = tf.sparse.to_dense(parsed_example["X"])
        X = tf.reshape(
            X_flat, [parsed_example["n_steps"], parsed_example["n_features"]]
        )
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


# def _test_tfrecord_windowing(
#     tfrecord_path_1="./data/tfrecords/10_ABD.tfrecord",
#     tfrecord_path_2="./data/tfrecords/10_ER.tfrecord",
# ):
#     def _parse_exercise_example(tfrecord):
#         """Get exercise data from tfrecord"""
#         parsed_example = tf.io.parse_single_example(tfrecord, FEATURE_MAP)
#         X_flat = tf.sparse.to_dense(parsed_example["X"])
#         X = tf.reshape(
#             X_flat, [parsed_example["n_steps"], parsed_example["n_features"]]
#         )
#         # x_labels = tf.sparse.to_dense(parsed_example['x_labels'])
#         # X = tf.sparse.to_dense(parsed_example["X"])
#         # subject = tf.sparse.to_dense(parsed_example["subject"])
#         # x_labels = tf.sparse.to_dense(parsed_example["x_labels"])
#         # y_labels = tf.sparse.to_dense(parsed_example["y_labels"])
#         # decode_feat = tf.io.decode_raw(f)
#         # if 1:
#         #     # if using flat_map
#         #     X_dataset = tf.data.Dataset.from_tensors(X)
#         #     return tf.data.Dataset.zip((X_dataset, X_dataset))
#         # else:
#         #     return (X, X)
#         X_window =  tf.data.Dataset.from_tensors(X).window(100)
#         # X = tf.data.Dataset.from_tensors(X)
#         return X_window
#         # return X, parsed_example['subject'], parsed_example['y'], parsed_example['y_label']
#         # return x_labels

#     SIZE = 100
#     SHIFT = 50
#     print(f"[info] Testing: {tfrecord_path_1}")
#     print(f"[info] Testing: {tfrecord_path_2}")

#     raw_dataset = tf.data.TFRecordDataset([tfrecord_path_1, tfrecord_path_2])
#     parsed_dataset = raw_dataset.map(_parse_exercise_example)
#     # parsed_dataset = raw_dataset.map(_parse_exercise_example).map(lambda x: x.window(2))
#     # parsed_dataset = raw_dataset.map(_parse_exercise_example)
#     # pdb.set_trace()
#     for y in parsed_dataset:
#         # pdb.set_trace()
#         # print(y.numpy().shape)
#         for item in y:
#             print(list(item.as_numpy_iterator()))
#         #     for x in item:
#         #         pdb.set_trace()
#         #         print(x.numpy().shape)
#     exit(0)
#     # parsed_dataset = (
#     #     raw_dataset.map(_parse_exercise_example)
#     #     .flat_map(lambda x: x.window(size=SIZE, shift=SHIFT))
#     #     # .unbatch()
#     # )
#     # parsed_dataset = raw_dataset.map(_parse_exercise_example).window(
#     #     size=SIZE, shift=SHIFT
#     # )

#     dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]))
#     dataset = dataset.window(2)

#     for window in dataset:
#         a = list(window[0].as_numpy_iterator())
#         b = list(window[1].as_numpy_iterator())
#         print(a, b)

#     print(f"[info] parsed_dataset:\n{parsed_dataset}")

#     for example in parsed_dataset.take(-1):
#         # print("[info] example:\n{}".format(example))
#         pdb.set_trace()
#         print("[info] dimensions_0: {}".format(example[0].shape))
#         print("[info] dimensions_1: {}".format(example[1].shape))


if __name__ == "__main__":
    # generate_seglearn_tfrecords()
    generate_windowed_tfrecords()

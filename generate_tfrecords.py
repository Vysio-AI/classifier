import os
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


def generate_seglearn_tfrecords(tfrecord_destination=DATA_CONFIG["tfrecord_destination"]):
    """Generate and store tfrecords for each subject-exercise time series
    data from the seglearn module"""

    def _get_exercise_example(X, y, y_label, x_labels, side, subject):
        """Generate the features for the TFRecord file"""

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

        feature = {
            "n_steps": _int64_feature(X.shape[0]),
            "n_features": _int64_feature(X.shape[1]),
            "subject": _int64_feature(subject),
            "side": _int64_feature(side),
            "y": _int64_feature(y),
            "X": _float_list_feature(X.ravel().tolist()),
            "y_label": _bytes_feature(str.encode(y_label)),
            "X_labels": _bytes_feature(np.array(x_labels).tobytes()),
            # "X_labels": _bytes_list_feature([str.encode(x) for x in x_labels]),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # TODO: Iterate through exercises, generate tf examples and store records
    seglearn_watch_data = seglearn.datasets.load_watch()
    print(f"[info] seglearn_watch_data.keys() = {seglearn_watch_data.keys()}")

    # Create a directory for tfrecords
    os.makedirs(tfrecord_destination, exist_ok=True)

    number_of_examples = len(seglearn_watch_data["X"])

    # Store each set of time-series as a tfrecord
    for idx in tqdm(range(number_of_examples)):
        # Extract relavent data to record
        X = seglearn_watch_data["X"][idx]
        y = seglearn_watch_data["y"][idx]
        y_label = seglearn_watch_data["y_labels"][y]
        subject = seglearn_watch_data["subject"][idx]
        side = int(seglearn_watch_data["side"][idx])
        x_labels = seglearn_watch_data["X_labels"]

        # Generate tfrecord example
        tf_example = _get_exercise_example(
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


def _test_tfrecord_generation(
    tfrecord_path, X0, y0, y_label0, subject0, side0, X_labels0
):
    """Assert that the data stored in the tfrecords is consistent and
    retrievable"""

    def _parse_exercise_example(tfrecord):
        """Get exercise data from tfrecord"""
        feature_map = {
            "n_steps": tf.io.FixedLenFeature([], tf.int64),
            "n_features": tf.io.FixedLenFeature([], tf.int64),
            "subject": tf.io.FixedLenFeature([], tf.int64),
            "side": tf.io.FixedLenFeature([], tf.int64),
            "y": tf.io.FixedLenFeature([], tf.int64),
            "X": tf.io.VarLenFeature(float),
            "y_label": tf.io.FixedLenFeature([], tf.string),
            "X_labels": tf.io.FixedLenFeature([], tf.string),
            # "X_labels": tf.io.VarLenFeature(tf.string),
        }
        parsed_example = tf.io.parse_single_example(tfrecord, feature_map)
        X_flat = tf.sparse.to_dense(parsed_example["X"])
        X = tf.reshape(
            X_flat, [parsed_example["n_steps"], parsed_example["n_features"]]
        )
        # X_labels = tf.sparse.to_dense(parsed_example['X_labels'])
        # X = tf.sparse.to_dense(parsed_example["X"])
        # subject = tf.sparse.to_dense(parsed_example["subject"])
        # X_labels = tf.sparse.to_dense(parsed_example["X_labels"])
        # y_labels = tf.sparse.to_dense(parsed_example["y_labels"])
        # decode_feat = tf.io.decode_raw(f)
        return parsed_example
        # return X, parsed_example['subject'], parsed_example['y'], parsed_example['y_label']
        # return X_labels

    print(f"[info] Testing: {tfrecord_path}")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_exercise_example)
    print(f"[info] parsed_dataset:\n{parsed_dataset}")

    for example in parsed_dataset.take(-1):
        print("[info] example:\n{}".format(example))
        assert y0 == example["y"].numpy()
        assert X0.shape[0] == example["n_steps"].numpy()
        assert X0.shape[1] == example["n_features"].numpy()
        assert subject0 == example["subject"].numpy()
        assert side0 == example["side"].numpy()
        assert str.encode(y_label0) == example["y_label"].numpy()
        X_labels = np.frombuffer(example["X_labels"].numpy(), dtype="<U2")
        comparison = X_labels0 == X_labels
        assert np.all(comparison)


if __name__ == "__main__":
    generate_seglearn_tfrecords()

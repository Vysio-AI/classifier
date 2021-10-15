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


def generate_lstm_tfrecords(tfrecord_destination=DATA_CONFIG["tfrecord_destination"]):
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


if __name__ == "__main__":
    generate_lstm_tfrecords()

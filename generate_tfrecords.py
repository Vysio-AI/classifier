import pdb

import numpy as np
import seglearn
import tensorflow as tf
import yaml
from easydict import EasyDict

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGURATIONS = EasyDict(yaml.load(f, yaml.FullLoader))
    DATA_CONFIG = CONFIGURATIONS["data_generation"]


def generate_lstm_tfrecords(tfrecord_destination=DATA_CONFIG["tfrecord_destination"]):
    """Generate and store tfrecords for each subject-exercise time series
    data from the seglearn module"""
    seglearn_watch_data = seglearn.datasets.load_watch()

    def _get_exercise_example(steps, y, y_label, x_labels, side, subject):
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
            "n_steps": _int64_feature(steps.shape[0]),
            "n_features": _int64_feature(steps.shape[1]),
            "subject": _int64_feature(subject),
            "side": _int64_feature(side),
            "y": _int64_feature(y),
            "X": _float_list_feature(steps.ravel().tolist()),
            "y_label": _bytes_list_feature(np.array(y_label).tobytes()),
            "X_labels": _bytes_list_feature(np.array(x_labels).tobytes()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # TODO: Iterate through exercises, generate tf examples and store records


if __name__ == "__main__":
    generate_lstm_tfrecords()

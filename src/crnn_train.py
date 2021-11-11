import os
from datetime import datetime
from enum import Enum
from glob import glob

import numpy as np
import tensorflow as tf
import yaml
from easydict import EasyDict

from crnn_model import get_crnn_model
from preprocessing import FEATURE_MAP, generate_window_tfrecords


class DataType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGS = EasyDict(yaml.load(f, yaml.FullLoader))
    CRNN_CONFIG = CONFIGS["crnn_config"]
    DATA_CONFIG = CONFIGS["data_generation"]
    EVALUATION_CONFIG = CONFIGS["evaluation"]


def get_tfrecord_data(data_type: DataType = DataType.TRAIN):
    """Generate train/test/validation tf.data.Datasets"""

    def _parse_example_function(tfrecord_proto):
        """Extract and formate tfrecord data"""
        example = tf.io.parse_single_example(tfrecord_proto, FEATURE_MAP)
        # Convert the 1D data into its original dimensions
        X_flat_tensor = tf.sparse.to_dense(example["X"])
        X_tensor = tf.reshape(
            X_flat_tensor, [example["n_steps"], example["n_features"]]
        )
        X_dataset = tf.data.Dataset.from_tensors(X_tensor)

        # Generate the appropriate onehot label
        y_onehot_tensor = tf.sparse.to_dense(example["y_onehot"])
        y_onehot_dataset = tf.data.Dataset.from_tensor_slices(y_onehot_tensor)
        # return tf.data.dataset.zip((x_dataset, y_onehot_dataset))
        return (X_tensor, y_onehot_tensor)

    assert isinstance(data_type, DataType)

    # Grab the list of file patterns for relevant csv files
    csv_file_patterns = DATA_CONFIG[f"{data_type.value}_csv_file_pattern"]
    assert isinstance(csv_file_patterns, list)
    csv_file_list = []
    # Generate a list of all matching csv file paths
    for pattern in csv_file_patterns:
        csv_file_list.extend(glob(pattern, recursive=True))

    csv_path_list_str = "\n".join(csv_file_list)
    print(f"[info] sourcing {data_type} csv: {len(csv_file_list)}")
    tfrecord_path_list = generate_window_tfrecords(csv_file_list)
    print(f"[info] number of windows created: {len(tfrecord_path_list)}")

    # Generate dataset from each tfrecord
    dataset = tf.data.TFRecordDataset(tfrecord_path_list)
    dataset = (
        dataset.map(
            _parse_example_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .shuffle(CRNN_CONFIG["shuffle_buffer_size"], reshuffle_each_iteration=True)
        .batch(CRNN_CONFIG["batch_size"])
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return dataset


def train_model():
    """Train the crnn model"""
    model = get_crnn_model()
    train_tf_dataset = get_tfrecord_data(DataType.TRAIN)
    validation_tf_dataset = get_tfrecord_data(DataType.VALIDATION)

    # Create the training tensorboard log directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logs_path = os.path.join(EVALUATION_CONFIG["log_dir"], "logs_{}".format(timestamp))

    # Define the tensorboard callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(logs_path, "checkpoint_{epoch}.tf"),
            save_weights_only=True,
            verbose=1,
            save_freq="epoch",
            save_best_only=True,
        ),
        # tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1, profile_batch='20,40'),
        tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1),
    ]

    model.fit(
        train_tf_dataset,
        epochs=CRNN_CONFIG["epochs"],
        validation_data=validation_tf_dataset,
        batch_size=CRNN_CONFIG["batch_size"],
        callbacks=callbacks,
    )

    return model


if __name__ == "__main__":
    # Set seed to make results replicable
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    train_model()

import os
import pathlib
import pdb
from datetime import datetime
from enum import Enum
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import yaml
from easydict import EasyDict

from model import get_crnn_model
from train import DataType, get_tfrecord_data

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGS = EasyDict(yaml.load(f, yaml.FullLoader))
    EVALUATION_CONFIG = CONFIGS["evaluation"]

if __name__ == "__main__":
    # Load model with weights
    model = get_crnn_model()
    weight_file_path = EVALUATION_CONFIG["weights_file"]
    model.load_weights(weight_file_path)
    # Get the validation dataset
    validation_tf_dataset = get_tfrecord_data(DataType.VALIDATION).unbatch()

    # Load the validation dataset in memory
    validation_Y = []
    validation_X = []
    for item in validation_tf_dataset.take(-1):
        validation_X.append(item[0].numpy())
        validation_Y.append(item[1].numpy())

    validation_X = np.array(validation_X[:])
    validation_Y = np.array(validation_Y[:])

    # Run predictions on the validation dataset
    import pdb; pdb.set_trace()
    pred_Y = model.predict(validation_X, batch_size=1)
    # Convert softmax values into predictions
    val_Y_argmax = np.argmax(validation_Y, axis=-1)
    pred_Y_argmax = np.argmax(pred_Y, axis=-1)

    # Build the confusion matrix
    confusion_matrix = tf.math.confusion_matrix(
        val_Y_argmax, pred_Y_argmax, num_classes=7
    ).numpy()
    # Normalize the confustion matrix
    confusion_matrix = (
        confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    )
    # Plot the confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, range(7), range(7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
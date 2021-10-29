
import os
import pdb
from datetime import datetime
from enum import Enum
from glob import glob
import pathlib

import pandas as pd
import seaborn as sn
import numpy as np
import yaml
from easydict import EasyDict
import tensorflow as tf
import matplotlib.pyplot as plt

from crnn_model import get_crnn_model
from crnn_train import get_tfrecord_data, DataType

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGS = EasyDict(yaml.load(f, yaml.FullLoader))
    CRNN_CONFIG = CONFIGS["crnn_config"]
    DATA_CONFIG = CONFIGS["data_generation"]
    EVALUATION_CONFIG = CONFIGS["evaluation"]

if __name__ == "__main__":
    model = get_crnn_model()
    weight_file_path = EVALUATION_CONFIG['weights_file']
    # assert pathlib.Path(weight_file_path).exists()
    model.load_weights(weight_file_path)
    validation_tf_dataset = get_tfrecord_data(DataType.VALIDATION).unbatch()

    validation_Y = []
    validation_X = []

    for item in validation_tf_dataset.take(-1):
        validation_X.append(item[0].numpy())
        validation_Y.append(item[1].numpy())

    validation_X = np.array(validation_X[:])
    validation_Y = np.array(validation_Y[:])

    pred_Y = model.predict(validation_X, batch_size=1)
    val_Y_argmax = np.argmax(validation_Y, axis=-1)
    pred_Y_argmax = np.argmax(pred_Y, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(val_Y_argmax, pred_Y_argmax, num_classes=7).numpy()
    df_cm = pd.DataFrame(confusion_matrix, range(7), range(7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

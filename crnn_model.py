import yaml
from easydict import EasyDict
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGS = EasyDict(yaml.load(f, yaml.FullLoader))
    CRNN_CONFIG = CONFIGS["crnn_config"]
 

def get_crnn_model():

    # Build the model
    model = Sequential()
    model.add(
        layers.Conv1D(
            filters=CRNN_CONFIG["conv_filters"],
            kernel_size=CRNN_CONFIG["conv_kernel_size"],
            padding=CRNN_CONFIG["conv_padding"],
            activation=CRNN_CONFIG["cnn_activations"],
            input_shape=tuple(CRNN_CONFIG["input_shape"]),
        )
    )
    model.add(layers.MaxPool1D(pool_size=CRNN_CONFIG["maxpooling_size"]))
    model.add(
        layers.Conv1D(
            filters=CRNN_CONFIG["conv_filters"],
            kernel_size=CRNN_CONFIG["conv_kernel_size"],
            padding=CRNN_CONFIG["conv_padding"],
            activation=CRNN_CONFIG["cnn_activations"],
        )
    )
    model.add(layers.MaxPool1D(pool_size=CRNN_CONFIG["maxpooling_size"]))
    model.add(
        layers.LSTM(
            units=CRNN_CONFIG["lstm_units"],
            dropout=CRNN_CONFIG["lstm_dropout"],
            recurrent_dropout=CRNN_CONFIG["lstm_recurrent_dropout"],
        )
    )
    # model.add(
    #     layers.LSTM(
    #         units=CRNN_CONFIG["lstm_units"],
    #         dropout=CRNN_CONFIG["lstm_dropout"],
    #         recurrent_dropout=CRNN_CONFIG["lstm_recurrent_dropout"],
    #     )
    # )
    model.add(
        layers.Dense(
            units=CRNN_CONFIG["output_shape"],
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2=CRNN_CONFIG["l2_reg"]),
        )
    )

    # Compile the model
    model.compile(
        loss=CRNN_CONFIG["loss"],
        optimizer=CRNN_CONFIG["optimizer"],
        metrics=["accuracy"],
    )

    model.summary()
    return model

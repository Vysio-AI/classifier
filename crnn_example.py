# Author: David Burns
# License: BSD

import pdb

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from easydict import EasyDict
from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import Segment
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGS = EasyDict(yaml.load(f, yaml.FullLoader))

def get_crnn_model(
    crnn_config = CONFIGS["crnn_config"]
):

    # Build the model
    model = Sequential()
    model.add(
        layers.Conv1D(
            filters=crnn_config["conv_filters"],
            kernel_size=crnn_config["conv_kernel_size"],
            padding=crnn_config["conv_padding"],
            activation=crnn_config["cnn_activations"],
            input_shape=tuple(crnn_config["input_shape"]),
        )
    )
    model.add(
        layers.Conv1D(
            filters=crnn_config["conv_filters"],
            kernel_size=crnn_config["conv_kernel_size"],
            padding=crnn_config["conv_padding"],
            activation=crnn_config["cnn_activations"],
        )
    )
    model.add(
        layers.LSTM(
            units=crnn_config["lstm_units"],
            dropout=crnn_config["lstm_dropout"],
            recurrent_dropout=crnn_config['lstm_recurrent_dropout'],
        )
    )
    model.add(
        layers.Dense(
            units=crnn_config['output_shape'],
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l2=crnn_config['l2_reg']),
        )
    )

    # Compile the model
    model.compile(
        loss=crnn_config['loss'], optimizer=crnn_config['optimizer'], metrics=["accuracy"]
    )

    model.summary()
    return model

# load the data
data = load_watch()
X = data["X"]
y = data["y"]

# create a segment learning pipeline
pipe = Pype(
    [
        ("seg", Segment(width=100, step=100, order="C")),
        (
            "crnn",
            KerasClassifier(
                build_fn=get_crnn_model, epochs=5, batch_size=256, verbose=0
            ),
        ),
    ]
)

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
# pdb.set_trace()
crnn_model = get_crnn_model()
pred_1 = crnn_model.predict(X_test[0][np.newaxis, :100])
print(f'[info] pred_1 = {pred_1}')
# prediction = pipe.predict_unsegmented(np.array(X_test[0][:100]))
# print(f'[info] prediction = {prediction}')

print("N series in train: ", len(X_train))
print("N series in test: ", len(X_test))
print("N segments in train: ", pipe.N_train)
print("N segments in test: ", pipe.N_test)
print("Accuracy score: ", score)

# img = mpimg.imread("segments.jpg")
# plt.imshow(img)

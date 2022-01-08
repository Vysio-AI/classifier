import pytorch_lightning as pl
import torch
import torch.nn as nn


class CRNNModel(pl.LightningModule):
    """
    Model class for CRNN
    """

    def __init__(self, **kwargs):
        """
        Initializes the network
        """

        super().__init__()

        # Initialize model architecture

        # self.crnn = model
        # self.args = kwargs

    def forward(self, x):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def test_step(self, batch):
        pass

# Build the model
# model = Sequential()
# model.add(
#     layers.Conv1D(
#         filters=CRNN_CONFIG["conv_filters"],
#         kernel_size=CRNN_CONFIG["conv_kernel_size"],
#         padding=CRNN_CONFIG["conv_padding"],
#         activation=CRNN_CONFIG["cnn_activations"],
#         input_shape=tuple(CRNN_CONFIG["input_shape"]),
#     )
# )
# model.add(layers.MaxPool1D(pool_size=CRNN_CONFIG["maxpooling_size"]))
# model.add(
#     layers.Conv1D(
#         filters=CRNN_CONFIG["conv_filters"],
#         kernel_size=CRNN_CONFIG["conv_kernel_size"],
#         padding=CRNN_CONFIG["conv_padding"],
#         activation=CRNN_CONFIG["cnn_activations"],
#     )
# )
# model.add(layers.MaxPool1D(pool_size=CRNN_CONFIG["maxpooling_size"]))
# model.add(
#     layers.LSTM(
#         units=CRNN_CONFIG["lstm_units"],
#         dropout=CRNN_CONFIG["lstm_dropout"],
#         recurrent_dropout=CRNN_CONFIG["lstm_recurrent_dropout"],
#         return_sequences=True
#     )
# )
# model.add(
#     layers.LSTM(
#         units=CRNN_CONFIG["lstm_units"],
#         dropout=CRNN_CONFIG["lstm_dropout"],
#         recurrent_dropout=CRNN_CONFIG["lstm_recurrent_dropout"],
#     )
# )
# model.add(
#     layers.Dense(
#         units=len(CRNN_CONFIG["class_names"]),
#         activation="softmax",
#         kernel_regularizer=tf.keras.regularizers.l2(l2=CRNN_CONFIG["l2_reg"]),
#     )
# )

# # Compile the model
# model.compile(
#     loss=CRNN_CONFIG["loss"],
#     optimizer=CRNN_CONFIG["optimizer"],
#     metrics=["accuracy"],
# )

# model.summary()
# return model

import typing

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
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

        # TODO: Initialize model architecture
        self.crnn = nn.Sequential()

        # Define loss function
        self.loss = nn.CrossEntropyLoss()

        # Define other model parameters
        self.lr = kwargs["learning_rate"]

    def forward(self, x: typing.Any) -> typing.Any:
        """
        Forward pass through the CRNN model.

        :param x: Input to CRNN model

        :return: Result of CRNN model inference on input
        """

        classification = self.crnn(x)
        return classification

    def configure_optimizers(self):
        """
        Choose what optimizer to use in optimization.
        """

        optimizer = torch.optim.Adam(
            params = self.crnn.parameters(),
            lr = self.lr
        )

        return optimizer

    def training_step(self, batch: typing.Any) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any]]:
        """
        Compute and return the training loss.

        :param batch: The output of the train Dataloader

        :return: Loss tensor or a dictionary
        """

        classifications = self.crnn(batch.get("timeseries"))
        loss = self.loss(classifications, batch.get("label"))

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch: typing.Any) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any], None]:
        """
        Compute and return the validation loss and accuracy.

        :param batch: The output of the validation Dataloader

        :return: Loss tensor, a dictionary, or None
        """

        classifications = self.crnn(batch.get("timeseries"))
        loss = self.loss(classifications, batch.get("label"))
        _, y_hat = torch.max(classifications, dim=1)
        acc = accuracy(y_hat, batch.get("label"))

        self.log("val_acc", acc)
        self.log("val_loss", loss)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch: typing.Any) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any], None]:
        """
        Compute and return the test accuracy.

        :param batch: The output of the test Dataloader

        :return: Loss tensor, a dictionary, or None
        """

        classifications = self.crnn(batch.get("timeseries"))
        _, y_hat = torch.max(classifications, dim=1)
        acc = accuracy(y_hat, batch.get("label"))

        self.log("test_acc", acc, on_epoch=True)

        return {"test_acc": acc}

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

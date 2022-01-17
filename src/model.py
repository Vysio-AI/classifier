import pdb
import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy


class CRNNModel(pl.LightningModule):
    """
    Model class for CRNN
    """

    def __init__(self, **kwargs):
        """
        Initializes the network
        """

        super().__init__()

        # Define loss function
        self.loss = nn.CrossEntropyLoss()

        # Define other model parameters
        self.lr = kwargs["learning_rate"]
        self.lstm_dropout = kwargs['lstm_dropout']
        self.lstm_hidden_size = 100
        self.num_classes = 7

        # TODO: Initialize model architecture
        self.cnn = nn.Sequential()
        # conv1
        self.cnn.add_module(
            "conv1", nn.Conv1d(in_channels=6, out_channels=128, kernel_size=7)
        )
        self.cnn.add_module("relu1", nn.ReLU(True))
        self.cnn.add_module("pooling1", nn.MaxPool1d(2))
        # conv2
        self.cnn.add_module(
            "conv2", nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7)
        )
        self.cnn.add_module("relu2", nn.ReLU(True))
        self.cnn.add_module("pooling2", nn.MaxPool1d(2))
        # rnn
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout = self.lstm_dropout
        )
        # out
        self.fc = nn.Linear(
            in_features=self.lstm_hidden_size * 2, out_features=self.num_classes
        )

    def forward(self, x: typing.Any) -> typing.Any:
        """
        Forward pass through the CRNN model.

        :param x: Input to CRNN model

        :return: Result of CRNN model inference on input
        """

        x_cnn = self.cnn(x.permute(0, 2, 1))
        x_permute = x_cnn.permute(0, 2, 1)
        x_rnn, _ = self.rnn(x_permute)
        return self.fc(x_rnn[:, -1, :])  # grab the last sequence

    def configure_optimizers(self):
        """
        Choose what optimizer to use in optimization.
        """

        optimizer = torch.optim.Adam(params=self.cnn.parameters(), lr=self.lr)

        return optimizer

    def training_step(
        self, batch: typing.Any, batch_idx: int
    ) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any]]:
        """
        Compute and return the training loss.

        :param batch: The output of the train Dataloader

        :return: Loss tensor or a dictionary
        """

        x = batch.get("timeseries")
        y = batch.get("label")

        # forward
        x_hat = self.cnn(x.permute(0, 2, 1))
        x_hat = x_hat.permute(0, 2, 1)
        x_hat, _ = self.rnn(x_hat)
        predictions = self.fc(x_hat[:, -1, :])

        loss = self.loss(predictions, y.float())
        _, y_hat = torch.max(predictions, dim=1)
        _, y = torch.max(y, dim=1)
        acc = accuracy(y_hat, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return {"loss": loss}

    def validation_step(
        self, batch: typing.Any, batch_idx: int
    ) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any], None]:
        """
        Compute and return the validation loss and accuracy.

        :param batch: The output of the validation Dataloader

        :return: Loss tensor, a dictionary, or None
        """
        x = batch.get("timeseries")
        y = batch.get("label")

        # forward
        x_hat = self.cnn(x.permute(0, 2, 1))
        x_hat = x_hat.permute(0, 2, 1)
        x_hat, _ = self.rnn(x_hat)
        predictions = self.fc(x_hat[:, -1, :])

        loss = self.loss(predictions, y.float())
        _, y_hat = torch.max(predictions, dim=1)
        _, y = torch.max(y, dim=1)
        acc = accuracy(y_hat, y)

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(
        self, batch: typing.Any
    ) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any], None]:
        """
        Compute and return the test accuracy.

        :param batch: The output of the test Dataloader

        :return: Loss tensor, a dictionary, or None
        """

        x = batch.get("timeseries")
        y = batch.get("label")

        # forward
        x_hat = self.cnn(x)
        x_hat = x_hat.permute(0, 2, 1)
        x_hat, _ = self.rnn(x_hat)
        predictions = self.fc(x_hat[:, -1, :])

        loss = self.loss(predictions, y)
        _, y_hat = torch.max(predictions, dim=1)
        _, y = torch.max(y, dim=1)
        acc = accuracy(y_hat, y)

        self.log("test_acc", acc, on_epoch=True)

        return {"test_acc": acc}

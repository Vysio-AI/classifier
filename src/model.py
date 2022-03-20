import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy


class CRNNModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("learning_rate")
        self.lstm_dropout = kwargs.get("lstm_dropout")
        self.num_classes = kwargs.get("num_classes")
        self.lstm_hidden_size = kwargs["hidden_size"]
        self.lstm_layers = kwargs["lstm_layers"]
        self.input_shape = kwargs["input_shape"]
        self.device = kwargs["device"]
        self.save_hyperparameters()

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, *self.input_shape))

        # Define loss function
        self.loss = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv1d(
            in_channels=self.input_shape[0], out_channels=128, kernel_size=7
        )
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.lstm_dropout,
        )

        self.lin1 = nn.Linear(
            in_features=self.lstm_hidden_size * self.lstm_layers,
            out_features=self.num_classes,
        )

        self.sm = nn.Softmax(dim=1)

    def forward(self, x: typing.Any) -> typing.Any:
        if x.shape[1] != self.input_shape[0]:
            x = x.permute(0, 2, 1)

        assert x.shape[1] == self.input_shape[0]

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = x[:, -1, :]  # grab the last sequence
        x = self.lin1(x)

        return x

    def get_class(self, x):
        softmax = self.sm(x)
        classification = torch.argmax(softmax, dim=1)
        return classification

    def spark_predict(self, x):
        # create batch dimension
        x = torch.unsqueeze(x, dim=0)

        x = self.forward(x)

        return self.get_class(x)

    def configure_optimizers(self):
        """
        Choose what optimizer to use in optimization.
        """

        optimizer = torch.optim.Adam(params=self.cnn.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch: typing.Any, batch_idx: int):
        """
        Compute and return the training loss.

        :param batch: The output of the train Dataloader

        :return: Loss tensor or a dictionary
        """

        x = batch.get("timeseries").to(self.device)
        y_gt = batch.get("label").to(self.device)

        assert batch.shape[1:] == self.input_shape

        # forward
        y_pred = self.forward(x)
        y_pred_softmax = self.get_class(y_pred)

        loss = self.loss(y_pred, y_gt.float())
        acc = accuracy(y_pred_softmax, y_gt)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch: typing.Any, batch_idx: int):
        """
        Compute and return the validation loss and accuracy.

        :param batch: The output of the validation Dataloader

        :return: Loss tensor, a dictionary, or None
        """
        x = batch.get("timeseries").to(self.device)
        y_gt = batch.get("label").to(self.device)

        assert batch.shape[1:] == self.input_shape

        # forward
        y_pred = self.forward(x)
        y_pred_softmax = self.get_class(y_pred)

        loss = self.loss(y_pred, y_gt.float())
        acc = accuracy(y_pred_softmax, y_gt)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

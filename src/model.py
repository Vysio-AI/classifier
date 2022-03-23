import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class CRNNModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("learning_rate")
        self.lstm_dropout = kwargs.get("lstm_dropout")
        self.num_classes = kwargs["num_classes"]
        self.lstm_hidden_size = kwargs["lstm_hidden_size"]
        self.lstm_layers = kwargs["lstm_layers"]
        self.channel_size = kwargs["channel_size"]
        self.weight_decay = kwargs["weight_decay"]
        self.save_hyperparameters()
        self.accuracy_top_k = kwargs.get("accuracy_top_k")

        # add validation metrics
        self.train_acc = torchmetrics.Accuracy(top_k=self.accuracy_top_k)
        self.train_f1 = torchmetrics.F1(num_classes=self.num_classes)
        self.train_auroc = torchmetrics.AUROC(num_classes=self.num_classes)
        # add validation metrics
        self.val_acc = torchmetrics.Accuracy(top_k=self.accuracy_top_k)
        self.val_f1 = torchmetrics.F1(num_classes=self.num_classes)
        self.val_auroc = torchmetrics.AUROC(num_classes=self.num_classes)

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, self.channel_size, 50))

        # Define loss function
        self.loss = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv1d(
            in_channels=self.channel_size, out_channels=128, kernel_size=7
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
        if x.shape[1] != self.channel_size:
            x = x.permute(0, 2, 1)

        assert x.shape[1] == self.channel_size

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

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
        )

        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     weight_decay=self.weight_decay,
        #     lr=self.lr,
        #     momentum=0.9,
        #     nesterov=True,
        # )

        return optimizer

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train/loss": 0,
                "train/acc": 0,
                "train/f1": 0,
                "train/auroc": 0,
                "val/loss": 0,
                "val/acc": 0,
                "val/f1": 0,
                "val/auroc": 0,
                "epoch_train_accuracy": 0,
                "epoch_train_f1": 0,
                "epoch_train_auroc": 0,
                "epoch_val_accuracy": 0,
                "epoch_val_f1": 0,
                "epoch_val_auroc": 0,
            },
        )

    def training_step(self, batch: typing.Any, batch_idx: int):
        """
        Compute and return the training loss.

        :param batch: The output of the train Dataloader

        :return: Loss tensor or a dictionary
        """

        x = batch.get("timeseries").cuda()
        y_gt = batch.get("label").cuda()
        y_class = batch.get("class").cuda()

        # forward
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y_gt.float())

        # accumulate and return metrics for logging
        acc = self.train_acc(y_pred, y_class)
        f1 = self.train_f1(y_pred, y_class)
        auroc = self.train_auroc(y_pred, y_class)

        self.log("train/loss", loss)
        self.log("train/acc", acc)
        self.log("train/f1", f1)
        self.log("train/auroc", auroc)

        return loss

    def validation_step(self, batch: typing.Any, batch_idx: int):
        """
        Compute and return the validation loss and accuracy.

        :param batch: The output of the validation Dataloader

        :return: Loss tensor, a dictionary, or None
        """
        x = batch.get("timeseries").cuda()
        y_gt = batch.get("label").cuda()
        y_class = batch.get("class").cuda()

        # forward
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y_gt.float())
        # accumulate and return metrics for logging
        acc = self.val_acc(y_pred, y_class)
        f1 = self.val_f1(y_pred, y_class)
        auroc = self.val_auroc(y_pred, y_class)

        self.log("val/loss", loss)
        self.log("val/acc", acc)
        self.log("val/f1", f1)
        self.log("val/auroc", auroc)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        # compute metrics
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        val_auroc = self.val_auroc.compute()

        # log metrics
        self.log("epoch_val_accuracy", val_acc)
        self.log("epoch_val_f1", val_f1)
        self.log("epoch_val_auroc", val_auroc)

        # reset all metrics
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

    def training_epoch_end(self, train_step_outputs):
        # compute metrics
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        train_auroc = self.train_auroc.compute()

        # log metrics
        self.log("epoch_train_accuracy", train_acc)
        self.log("epoch_train_f1", train_f1)
        self.log("epoch_train_auroc", train_auroc)

        # reset all metrics
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()

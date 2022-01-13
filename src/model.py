import pdb
import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


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
        self.lr = 0.001
        # self.lr = kwargs["learning_rate"]

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
            "conv2", nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7)
        )
        self.cnn.add_module("relu2", nn.ReLU(True))
        self.cnn.add_module("pooling2", nn.MaxPool1d(2))

        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(128, 100, 100), BidirectionalLSTM(100, 100, 7)
        # )
        hidden_size = 100
        num_classes = 7

        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: typing.Any) -> typing.Any:
        """
        Forward pass through the CRNN model.

        :param x: Input to CRNN model

        :return: Result of CRNN model inference on input
        """

        x_cnn = self.cnn(x)
        x_permute = x_cnn.permute(0, 2, 1)
        x_rnn, (hn, cn) = self.rnn(x_permute)
        return self.fc(x_rnn[:, -1, :])  # grab the last sequence

        # pdb.set_trace()
        # hn = hn.view(-1, 100)
        # x_1 = self.relu(hn)
        # x_2 = self.fc(x_1)
        # # return x, x_cnn, x_permute, x_rnn, x_out
        # # output = self.out(x)
        # return x_cnn

    def configure_optimizers(self):
        """
        Choose what optimizer to use in optimization.
        """

        optimizer = torch.optim.Adam(params=self.crnn.parameters(), lr=self.lr)

        return optimizer

    def training_step(
        self, batch: typing.Any
    ) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any]]:
        """
        Compute and return the training loss.

        :param batch: The output of the train Dataloader

        :return: Loss tensor or a dictionary
        """

        x = batch.get("timeseries")
        y = batch.get("label")

        # forward
        x_hat = self.cnn(x)
        x_hat = x_hat.permute(0, 2, 1)
        x_hat, _ = self.rnn(x_hat)
        predictions = self.fc(x_hat[:, -1, :])

        loss = self.loss(predictions, y)

        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(
        self, batch: typing.Any
    ) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any], None]:
        """
        Compute and return the validation loss and accuracy.

        :param batch: The output of the validation Dataloader

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
        acc = accuracy(y_hat, y)

        self.log("val_acc", acc)
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
        acc = accuracy(y_hat, y)

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

import pdb
import typing

import mlflow
import mlflow.pyfunc
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, collect_list, udf
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType
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
        self.lr = kwargs.get("learning_rate")
        self.lstm_dropout = kwargs.get("lstm_dropout")
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
            dropout=self.lstm_dropout,
        )
        # out
        self.fc = nn.Linear(
            in_features=self.lstm_hidden_size * 2, out_features=self.num_classes
        )
        self.sm = nn.Softmax()

    def forward(self, x: typing.Any) -> typing.Any:
        """
        Forward pass through the CRNN model.

        :param x: Input to CRNN model

        :return: Result of CRNN model inference on input
        """

        print(f"[p] shape: {len(x)}")
        print(f"[p] shape: {x.shape}")
        x = torch.unsqueeze(x, dim=0)
        print(f"[p] shape: {x.shape}")
        x_cnn = self.cnn(x)
        x_permute = x_cnn.permute(0, 2, 1)
        x_rnn, _ = self.rnn(x_permute)
        return self.sm(self.fc(x_rnn[:, -1, :]))  # grab the last sequence

    def configure_optimizers(self):
        """
        Choose what optimizer to use in optimization.
        """

        # TODO should be more than just cnn parameters
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        return optimizer

    def training_step(
        self, batch: typing.Any, batch_idx: int
    ) -> typing.Union[torch.Tensor, typing.Dict[str, typing.Any]]:
        """
        Compute and return the training loss.

        :param batch: The output of the train Dataloader

        :return: Loss tensor or a dictionary
        """

        x = batch.get("timeseries").cuda()
        y = batch.get("label").cuda()

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


if __name__ == "__main__":
    # deterministic
    pl.seed_everything(42)

    data = [
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    ]

    schema = StructType(
        [
            StructField("user_id", IntegerType(), True),
            StructField("session_id", IntegerType(), True),
            StructField("a_x", FloatType(), True),
            StructField("a_y", FloatType(), True),
            StructField("a_z", FloatType(), True),
            StructField("w_x", FloatType(), True),
            StructField("w_y", FloatType(), True),
            StructField("w_z", FloatType(), True),
        ]
    )

    spark = (
        SparkSession.builder.appName("Vysio")
        .config(
            "spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2"
        )
        .getOrCreate()
    )

    # Initialize CRNN model to train
    # TODO: remove/add the 'to(device='cuda')' method
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # crnn_model = CRNNModel(**{"lstm_dropout": 0.1}).to(device=device)
    crnn_model = CRNNModel(**{"lstm_dropout": 0.1})
    bc_model_state = spark.sparkContext.broadcast(crnn_model.state_dict())

    def get_model_for_eval():
        # Broadcast the model state_dict
        crnn_model.load_state_dict(bc_model_state.value)
        crnn_model.eval()
        return crnn_model

    def one_row_predict(x):
        model = get_model_for_eval()
        t = torch.tensor(x, dtype=torch.float32)
        prediction = model(t)
        print(f'[p] predicion:\n{prediction}')
        # pred = prediction.cpu().detach().item()
        # pred = prediction.detach().numpy()
        return 0.5

    df = spark.createDataFrame(data=data, schema=schema)
    df.show()

    dims = ["a_x", "a_y", "a_z", "w_x", "w_y", "w_z"]
    df_hat = df.groupBy("user_id").agg(
        collect_list("a_x").alias("a_x"),
        collect_list("a_y").alias("a_y"),
        collect_list("a_z").alias("a_z"),
        collect_list("w_x").alias("w_x"),
        collect_list("w_y").alias("w_y"),
        collect_list("w_z").alias("w_z"),
    )
    print("[p] grouped by user")
    df_hat.show()

    one_row_udf = udf(one_row_predict, FloatType())

    df_hat = df_hat.withColumn("input", array([col(dim) for dim in dims]))
    print("[p] added input column")
    df_hat.show()

    df_hat = df_hat.withColumn("classification", one_row_udf(col("input"))).show()

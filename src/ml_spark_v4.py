import pdb
import typing

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, collect_list, struct, udf
from pyspark.sql.types import (ArrayType, FloatType, IntegerType, StructField,
                               StructType)
from torchmetrics.functional import accuracy


class TestModel(pl.LightningModule):
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

    def predict(self, x: typing.Any) -> typing.Any:
        """
        Forward pass through the CRNN model.

        :param x: Input to CRNN model

        :return: Result of CRNN model inference on input
        """

        print(f"[p] shape: {x.shape}")
        x = torch.unsqueeze(x, dim=0)
        print(f"[p] shape: {x.shape}")
        x_cnn = self.cnn(x)
        x_permute = x_cnn.permute(0, 2, 1)
        x_rnn, _ = self.rnn(x_permute)
        x = self.fc(x_rnn[:, -1, :])
        print(f"[p] x.shape: {x.shape}")
        x = self.sm(x)
        print(f"[p] x.shape: {x.shape}")
        x = torch.argmax(x, dim=1)
        return x


if __name__ == "__main__":
    pl.seed_everything(42)

    # MODEL: Create model, save it and load it with MLflow
    model = TestModel(**{"lstm_dropout": 0.1})
    mlflow.pytorch.log_model(model, "ml_spark_model")  # log model
    model_path = mlflow.get_artifact_uri("ml_spark_model")
    print(f"[p] model_path: {model_path}")

    COL_NUM = 8

    data = np.random.rand(4, COL_NUM).astype(int)
    data[0:2, :2] = 1
    data[2:4, :2] = 2
    data = data.tolist()

    spark = (
        SparkSession.builder.appName("Vysio")
        .config(
            "spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2"
        )
        .getOrCreate()
    )
    df = spark.createDataFrame(data=data)
    df.show(vertical=True)
    dims = ["_" + str(i + 1) for i in range(0, COL_NUM)]

    df_hat = df.groupBy("_1").agg(
        collect_list("_3").alias("_3"),
        collect_list("_4").alias("_4"),
        collect_list("_5").alias("_5"),
        collect_list("_6").alias("_6"),
        collect_list("_7").alias("_7"),
        collect_list("_8").alias("_8"),
    )
    print("[p] grouped by user and aggregated")
    df_hat.show(vertical=True)
    print(df_hat)

    df_hat = df_hat.withColumn("input", array([col(f"_{i}") for i in range(3, 9)]))
    df_hat.show(vertical=True)

    def add_to(row):
        for item in row:
            print(type(item))
            print(item)

    # add_to = udf(lambda row: print(row), ArrayType(IntegerType()))
    add_to_udf = udf(lambda row: add_to(row))

    df_hat.withColumn("p2", add_to_udf(struct("_3"))).show()
    df_hat.withColumn("p1", add_to_udf(struct("_3", "_4"))).show()
    df_hat.withColumn("p3", add_to_udf(struct("input"))).show()

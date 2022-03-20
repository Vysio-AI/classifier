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
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType
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

    # SPARK: Create data, format it and start a spark session
    data = [(1, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)] * 10
    more_data = [(2, 2, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)] * 15
    data.extend(more_data)

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
    print("[p] grouped by user and aggregated")
    df_hat.show(vertical=True)

    df_hat = df_hat.withColumn("input", array([col(dim) for dim in dims]))
    print("[p] Created input array")
    df_hat.show(vertical=True)

    loaded_pytorch_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_path)
    print(f"[p] loaded_pytorch_model type: {type(loaded_pytorch_model)}")
    # loaded_pytorch_model = mlflow.pytorch.load_model(model_path)  # load model
    print(f"[p] loaded_pytorch_model:\n{loaded_pytorch_model}")

    # print("[p] {}".format(df_hat.select("input").rdd.flatMap(lambda x: x).collect()))
    def add_to(row):
        for item in row:
            print(type(item))
            print(item)

    # add_to = udf(lambda row: print(row), ArrayType(IntegerType()))
    add_to_udf = udf(lambda row: add_to(row))

    df_hat = df_hat.withColumn("classification", add_to_udf(struct(["input"])))
    print(f"[p] Show classification: {df_hat}")
    df_hat.show()

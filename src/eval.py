import mlflow
import mlflow.pyfunc
import pytorch_lightning as pl
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, collect_list, udf
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

from model import CRNNModel

# deterministic
pl.seed_everything(42)

if __name__ == "__main__":

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crnn_model = CRNNModel(**{"lstm_dropout": 0.1}).to(device=device)
    bc_model_state = spark.sparkContext.broadcast(crnn_model.state_dict())

    def get_model_for_eval():
        # Broadcast the model state_dict
        crnn_model.load_state_dict(bc_model_state.value)
        crnn_model.eval()
        return crnn_model

    def one_row_predict(x):
        model = get_model_for_eval()
        t = torch.tensor(x, dtype=torch.float32)
        prediction = model(t).cpu().detach().item()
        return prediction

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
    print(f"[p] grouped by user")
    df_hat.show()

    one_row_udf = udf(one_row_predict, FloatType())

    df_hat = df_hat.withColumn("input", array([col(dim) for dim in dims]))
    print(f'[p] added input column')
    df_hat.show()

    df_hat = df_hat.withColumn("classification", one_row_udf(col("input"))).show()

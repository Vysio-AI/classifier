import torch
import torch.nn as nn
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, udf
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, DoubleType
import pandas as pd
import numpy as np

spark = SparkSession.builder.master('local[*]') \
    .appName("model_training") \
    .getOrCreate()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Linear(5, 1)

    def forward(self, x):
        return self.w(x)

net = Net()
bc_model_state = spark.sparkContext.broadcast(net.state_dict())


df = spark.sparkContext.parallelize([[np.random.rand() for i in range(5)] for j in range(10)]).toDF()
df = df.withColumn('features', F.array([F.col(f"_{i}") for i in range(1, 6)]))

def get_model_for_eval():
  # Broadcast the model state_dict
  net.load_state_dict(bc_model_state.value)
  net.eval()
  return net

def one_row_predict(x):
    model = get_model_for_eval()
    t = torch.tensor(x, dtype=torch.float32)
    prediction = model(t).cpu().detach().item()
    return prediction

one_row_udf = udf(one_row_predict, FloatType())
df = df.withColumn('pred_one_row', one_row_udf(col('features')))
df.show()

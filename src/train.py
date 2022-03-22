import datetime
import os
from argparse import ArgumentParser

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler

from data_module import ShoulderExerciseDataModule
from model import CRNNModel

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Category Classifier")

    # General model paramters
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lstm_dropout", type=float, default=0.1)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--lstm_hidden_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=9)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--accuracy_top_k", type=int, default=1)

    # Dataset paramters
    parser.add_argument("--input_shape", type=tuple, default=(6, 100))
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--window_stride", type=int, default=50)
    parser.add_argument("--dataloader_source", type=str, default="./datasets")
    parser.add_argument("--dataloader_temp", type=str, default="./tmp/spardata")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--jitter_range", type=float, default=0)

    # Early stopping parameters
    parser.add_argument("--es_monitor", type=str, default="val_loss")
    parser.add_argument("--es_mode", type=str, default="min")
    parser.add_argument("--es_patience", type=int, default=7)

    # Other parameters
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=102, type=int)
    parser.add_argument("--logdir", default="./", type=str)

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    # Choose train and validation splits
    dict_args["load_csv_file_patterns"] = {
        "train": [
            "**/spar_csv/S[1-9]_*.csv",
            "**/spar_csv/S1[0-8]_*.csv",
        ],
        "validation": [
            "**/spar_csv/S1[8-9]_*.csv",
            "**/spar_csv/S20_*.csv",
        ],
    }

    pl.seed_everything(dict_args["seed"])

    # Initialize CRNN model to train
    model = CRNNModel(**dict_args)

    # Initialize data module
    data_module = ShoulderExerciseDataModule(**dict_args)

    # Callback: early stopping parameters
    early_stopping_callback = EarlyStopping(
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
        verbose=True,
        patience=dict_args["es_patience"],
    )

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
    )

    # Trainer: initialize training behaviour
    profiler = SimpleProfiler()
    now = datetime.datetime.now().strftime("%m_%a_%H_%M_%S")
    logger = TensorBoardLogger(
        save_dir=dict_args["logdir"],
        version=now,
        name="lightning_logs",
        log_graph=True,
    )
    trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        gpus=1 if dict_args["device"] == "cuda" else 0,
        deterministic=True,
        profiler=profiler,
        logger=logger,
    )

    # Trainer: train model
    trainer.fit(model, data_module)

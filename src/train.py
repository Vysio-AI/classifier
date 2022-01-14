from argparse import ArgumentParser
import os

from data_module import ShoulderExerciseDataModule
import mlflow
from model import CRNNModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Category Classifier")

    # General model paramters
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model optimization")
    parser.add_argument("--num-workers", type=int, help="Number of data loader workers")

    # Dataset paramters
    parser.add_argument("--window-size", type=int, help="Number of timesteps per window")
    parser.add_argument("--window-stride", type=int, help="Difference between consequtive windows")
    parser.add_argument("--data-dir", type=str, help="Location to store dataset samples")
    parser.add_argument("--batch-size", type=int, help="Training batch size")

    # Early stopping parameters
    parser.add_argument("--es-monitor", type=str, help="Early stopping monitor parameter")
    parser.add_argument("--es-mode", type=str, help="Early stopping mode parameter")
    parser.add_argument("--es-verbose", type=bool, help="Early stopping verbose parameter")
    parser.add_argument("--es-patience", type=int, help="Early stopping patience parameter")

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    # Autolog parameters, metrics and artifacts to MLflow
    mlflow.pytorch.autolog()

    # Initialize CRNN model to train
    model = CRNNModel(**dict_args)

    # Initialize data module
    data_module = ShoulderExerciseDataModule(**dict_args)

    early_stopping = EarlyStopping(
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
        verbose=dict_args["es_verbose"],
        patience=dict_args["es_patience"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping, checkpoint_callback], gpus=1
    )

    trainer.fit(model, data_module)
    # trainer.test(model, datamodule=data_module)

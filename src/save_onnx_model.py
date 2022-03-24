import os
from argparse import ArgumentParser
from glob import glob

import pytorch_lightning as pl
import yaml

from model import CRNNModel

if __name__ == "__main__":
    parser = ArgumentParser(description="Load model from log directory")

    parser.add_argument(
        "--model_log_dir",
        help="Specific model log directory holding checkpoints and params",
        type=str,
        required=True,
    )
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    # get first checkpoints file
    ckpt_pattern = os.path.join(dict_args["model_log_dir"], "**/*.ckpt")
    ckpt_path = glob(ckpt_pattern, recursive=True)
    assert len(ckpt_path) > 0
    ckpt_path = ckpt_path[0]

    # get hyper parameter yaml file to load
    hparams_path = os.path.join(dict_args["model_log_dir"], "hparams.yaml")
    assert os.path.exists(hparams_path)

    # load param file as dict
    with open(hparams_path, "r") as stream:
        try:
            hparams_dict = yaml.safe_load(stream)
            print(hparams_dict)
            dict_args.update(hparams_dict)
        except yaml.YAMLError as exc:
            print(exc)

    # Initialize CRNN model
    model = CRNNModel(**dict_args)
    print("[p] initialized model instance")

    # Load weights
    model = model.load_from_checkpoint(ckpt_path)
    print("[p] loaded model checkpoints")

    # save model to the source model log directory
    model_save_path = os.path.join(dict_args["model_log_dir"], "model.onnx")
    model.eval()  # set the model to inference mode
    model.to_onnx(
        model_save_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "step_dim"}},
    )
    print(f"[p] saved model to {model_save_path}")

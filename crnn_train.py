import pdb

import numpy as np
import tensorflow as tf
import yaml
from easydict import EasyDict


# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGS = EasyDict(yaml.load(f, yaml.FullLoader))

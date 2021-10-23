from abc import ABC, abstractmethod

import numpy as np
import yaml
from easydict import EasyDict


class TimeSeriesAugmentation(ABC):
    """Abstract class that defines a time series augmentation layer functionality"""

    @abstractmethod
    def augment(self, data: np.ndarray):
        """Method to augment inputted data."""


class WindowWarpAugmentation(TimeSeriesAugmentation):
    """Class that defines the functionality of window warp augmentations"""

    def augment(self, data: np.ndarray):
        print("Window warp augmentation not implemented")
        return data


class CroppingAugmentation(TimeSeriesAugmentation):
    """Class that defines the functionality of cropping augmentations"""

    def __init__(self, min_length: float, max_length: float=1.0):
        self.min_length = min_length
        self.max_length = max_length

    def augment(self, data: np.ndarray):
        print("Applying cropping augmentation")
        cropped_length = int(
            data.shape[1]*np.random.uniform(self.min_length, self.max_length)
        )
        start = int(np.random.uniform(0.0, data.shape[0] - cropped_length - 1))
        cropped_data = data[start:start+cropped_length, :]
        return cropped_data


class JitterAugmentation(TimeSeriesAugmentation):
    """Class that defines the functionality of jitter augmentations"""

    def __init__(self, additive_lb: float, additive_ub: float):
        self.additive_lb = additive_lb
        self.additive_ub = additive_ub

    def augment(self, data: np.ndarray):
        print("Applying jitter augmentation")
        jitter = self.additive_lb + (self.additive_ub -
                self.additive_lb) * np.random.rand(data.shape[0], data.shape[1])
        return data + jitter


class FlipAugmentation(TimeSeriesAugmentation):
    """Class that defines the functionality of flip augmentations"""

    def augment(self, data: np.ndarray):
        print("Applying flip augmentation")
        flipped_data = np.flip(data, axis=0)
        return flipped_data


class TimeSeriesAugmentationPipeline:
    """Class to control sequential execution of multiple time series data
    augmentation layers"""

    def __init__(self):
        self.pipeline = []

    def add(self, time_series_augmentation: TimeSeriesAugmentation):
        self.pipeline.append(time_series_augmentation)

    def run(self, data: np.ndarray) -> np.ndarray:
        for time_series_augmentation in self.pipeline:
            data = time_series_augmentation.augment(data)

        return data

with open("./config.yaml") as f:
    CONFIGS = EasyDict(yaml.load(f, yaml.FullLoader))
    DATA_AUGMENTATION_CONFIG = CONFIGS["data_augmentation"]

def get_data_augmentation_pipeline():
    """Function to return a data augmentation pipeline"""

    # Construct augmentation pipeline
    time_series_augmentation_pipeline = TimeSeriesAugmentationPipeline()

    # Add relevant augmentation layers
    time_series_augmentation_pipeline.add(
        CroppingAugmentation(
            min_length = DATA_AUGMENTATION_CONFIG["cropping_min_length"]
        )
    )
    time_series_augmentation_pipeline.add(
        WindowWarpAugmentation()
    )
    time_series_augmentation_pipeline.add(
        JitterAugmentation(
            additive_lb = DATA_AUGMENTATION_CONFIG["jitter_lb"],
            additive_ub = DATA_AUGMENTATION_CONFIG["jitter_ub"]
        )
    )
    time_series_augmentation_pipeline.add(
        FlipAugmentation()
    )

    # Return pipeline object
    return time_series_augmentation_pipeline

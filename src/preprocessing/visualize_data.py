from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from easydict import EasyDict
from scipy.integrate import cumtrapz

plt.style.use("seaborn")

# Store configuration file data in global object
with open("./config.yaml") as f:
    CONFIGURATIONS = EasyDict(yaml.load(f, yaml.FullLoader))
    DATA_CONFIG = CONFIGURATIONS["data_generation"]
    VIS_CONFIG = CONFIGURATIONS["visualization"]


def visualize_csv(
    csv_file_patterns=VIS_CONFIG["csv_file_patterns"],
    view_columns=VIS_CONFIG["view_columns"],
    view_trajectories=VIS_CONFIG["view_trajectories"],
):
    """
    Plot the timeseries data of csv's matching the path patterns.
    The csv is expected to have ['ax', 'ay', 'az', 'wx', 'wy', 'wz'] columns
    """
    assert isinstance(csv_file_patterns, list)
    csv_path_list = []
    # Generate a list of all matching csv file paths
    for pattern in csv_file_patterns:
        csv_path_list.extend(glob(pattern, recursive=True))
    print(f"[info] viewing {len(csv_path_list)} files")

    # maximum number of steps to view
    num_view_steps = VIS_CONFIG["num_view_steps"]

    if view_columns:
        for csv_path in csv_path_list:
            # import csv as data frame
            csv_data = pd.read_csv(csv_path)
            csv_data = csv_data[["ax", "ay", "az", "wx", "wy", "wz"]]
            # Take a look at all sensor outputs
            csv_data[:num_view_steps].plot(
                subplots=True,
                sharex=True,
                layout=(6, 1),
                title="File Path: {}".format(csv_path),
            )
        plt.show()

    if view_trajectories:
        for csv_path in csv_path_list:
            # import csv as data frame
            csv_data = pd.read_csv(csv_path)
            csv_data = csv_data[["ax", "ay", "az", "wx", "wy", "wz"]]
            csv_data = csv_data[:num_view_steps]

            # Double integrate accelerations to find positions
            dt = 1 / 50
            x = cumtrapz(cumtrapz(csv_data["ax"], dx=dt), dx=dt)
            y = cumtrapz(cumtrapz(csv_data["ay"], dx=dt), dx=dt)
            z = cumtrapz(cumtrapz(csv_data["az"], dx=dt), dx=dt)

            # Plot 3D Trajectory
            fig3, ax = plt.subplots()
            fig3.suptitle("Trajectory: {}".format(csv_path), fontsize=20)
            ax = plt.axes(projection="3d")
            ax.plot3D(x, y, z, c="red", label="trajectory")
            ax.set_xlabel("X position (m)")
            ax.set_ylabel("Y position (m)")
            ax.set_zlabel("Z position (m)")

        plt.show()


if __name__ == "__main__":
    visualize_csv()

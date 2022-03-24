from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import cumtrapz

plt.style.use("seaborn")

if __name__ == "__main__":
    """
    Plot the timeseries data of csv's matching the path patterns.
    The csv is expected to have ['ax', 'ay', 'az', 'wx', 'wy', 'wz'] columns
    """
    VIEW_NUM_STEPS = 1000
    VIEW_OFFSET = 250
    CSV_PATTERN_LIST = ["./datasets/spar_csv/S10_E[0-6]_L.csv"]

    # Generate a list of all matching csv file paths
    csv_path_list = []
    for pattern in CSV_PATTERN_LIST:
        csv_path_list.extend(glob(pattern, recursive=True))
    print(f"[p] viewing {len(csv_path_list)} files")

    for csv_path in csv_path_list:
        print(f"[p] viewing {csv_path}")
        # import csv as data frame
        csv_data = pd.read_csv(csv_path)
        csv_data = csv_data[["ax", "ay", "az", "wx", "wy", "wz"]]
        # Take a look at all sensor outputs
        csv_data[VIEW_OFFSET : VIEW_OFFSET + VIEW_NUM_STEPS].plot(
            subplots=True,
            sharex=True,
            layout=(6, 1),
            title="File Path: {}".format(csv_path),
        )
    plt.show()

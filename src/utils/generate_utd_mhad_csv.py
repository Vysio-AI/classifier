import os
import pathlib
from glob import glob

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

UTD_MHAD_LOCATION = "./tmp/MultiViewDataset/Data"
CSV_DESTINATION = "./datasets/utd_mhad_csv"
os.makedirs(CSV_DESTINATION, exist_ok=True)

# Example expected file path:
# sub1/left_45_degree/catch/a_s_t2_skel_*.mat
# Therefore:
# - subject = 1
# - orientation = left_45_degree (ignore)
# - activity = catch
# - trial = 2
# - data type = skel

file_pattern = os.path.join(UTD_MHAD_LOCATION, "**/*.mat")

file_list = glob(file_pattern, recursive=True)

class_names = {
    "catch": 0,
    "draw_circle": 1,
    "draw_tick": 2,
    "draw_triangle": 3,
    "knock": 4,
    "throw": 5,
}

skel_count = 0
inertial_count = 0

# store the inertial data
for file_path in tqdm(file_list):
    file_path = pathlib.Path(file_path)
    file_name = file_path.stem
    data_type = file_name.split("_")[3]
    trial_num = int(file_name.split("_")[2][1])
    activity = file_path.parents[0].stem
    orientation = file_path.parents[1].stem
    subject_num = int(file_path.parents[2].stem[-1])

    # grab data type
    assert data_type in ["depth", "inertial", "skel"]

    csv_filename = "S{}_E{}_T{}_{}_{}".format(
        subject_num, class_names[activity], trial_num, data_type, orientation
    )
    csv_path = os.path.join(CSV_DESTINATION, csv_filename)

    if data_type == "inertial":
        inertial_count += 1
        # print(file_path)
        # print(subject_num)
        # print(data_type)
        # print(trial_num)
        # print(orientation)
        # print(activity)

        mat_data = scipy.io.loadmat(file_path)["d_iner"]
        data_frame = pd.DataFrame(
            mat_data, columns=["ax", "ay", "az", "wx", "wy", "wz"]
        )
        data_frame.to_csv(csv_path+".csv")

    elif data_type == "skel":
        skel_count += 1
        mat_data = scipy.io.loadmat(file_path)["S_K2"][0, 0]["screen"]
        # Change dimensions to frame x joint x location (n x 25 x 3)
        mat_data = np.transpose(mat_data, [2, 0, 1])

        assert mat_data.shape[1:] == (25, 2)
        np.save(csv_path+".npy", mat_data)

print(f"[info] skel_count = {skel_count}")
print(f"[info] inertial_count = {inertial_count}")
assert skel_count == inertial_count

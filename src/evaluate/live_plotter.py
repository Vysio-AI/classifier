import pdb
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from build_input_sequence import generate_demo_input

# use ggplot style for more sophisticated visuals
plt.style.use("ggplot")

# max number of plotted points at a time
T_LIM = 100  # samples

plot_configs = {
    0: {"label": "a_x [m/s^2]", "ylim": [-1, 2]},
    1: {"label": "a_y [m/s^2]", "ylim": [-1, 2]},
    2: {"label": "a_z [m/s^2]", "ylim": [-2, 1]},
    3: {"label": "w_x [rad/s]", "ylim": [-5, 5]},
    4: {"label": "w_y [rad/s]", "ylim": [-5, 5]},
    5: {"label": "w_z [rad/s]", "ylim": [-3, 3]},
}

class_config = {
    0: {"label": "PEL", "color": "royalblue"},
    1: {"label": "ADB", "color": "darkviolet"},
    2: {"label": "FEL", "color": "plum"},
    3: {"label": "IR", "color": "firebrick"},
    4: {"label": "ER", "color": "orange"},
    5: {"label": "TRAP", "color": "darkkhaki"},
    6: {"label": "ROW", "color": "green"},
    69: {"label": "NULL", "color": "grey"},
}


def live_plotter(
    count,
    t_vec,
    x_data,
    y_data,
    plot_lines,
    ax_subplots,
    pause_time=0.02,
):
    x_data_subset = x_data[:, -T_LIM:]
    y_data_subset = y_data[-T_LIM:]
    # initialize the subplots
    if plot_lines[0] is None:
        plt.ion()
        fig, ax = plt.subplots(len(plot_lines), sharex=True)
        plt.suptitle("Live IMU (Accelerometer & Gyrosocipe) Output at 50Hz")
        for dim, _ in enumerate(plot_lines):
            (plot_lines[dim],) = ax[dim].plot(
                t_vec, x_data_subset[dim], "-", linewidth=2, alpha=0.8
            )
            ax_subplots[dim] = ax[dim]
            ax[dim].set_ylabel(plot_configs[dim]["label"])
            # ax[dim].set_ylim(plot_configs[dim]["ylim"])

        # update plot label/title
        plt.xlabel("Time (s)")
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    plt.suptitle(
        "Live IMU (Accelerometer & Gyrosocipe) Output at 50Hz\nClass = {}".format(
            class_config[y_data_subset[-1]]["label"]
        )
    )
    for dim, ax_line in enumerate(plot_lines):
        ax_line.set_ydata(x_data_subset[dim])
        ax_line.set_color(class_config[y_data_subset[-1]]["color"])
        # plot_line0.axes.set_xticklabels(["{:.2f}".format(x) for x in t_vec+count/50])

        # adjust limits if new data goes beyond bounds
        if (
            np.min(x_data_subset[dim]) <= ax_line.axes.get_ylim()[0]
            or np.max(x_data_subset[dim]) >= ax_line.axes.get_ylim()[1]
        ):
            ax_subplots[dim].set_ylim(
                [
                    np.min(x_data_subset[dim]) - np.std(x_data_subset[dim]),
                    np.max(x_data_subset[dim]) + np.std(x_data_subset[dim]),
                ]
            )
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return plot_lines


if __name__ == "__main__":
    # get the input sequence
    input_X, input_Y = generate_demo_input()

    sample_set = cycle(list(zip(input_X, input_Y)))

    # store plot points
    t_vec = np.linspace(0, 2, T_LIM + 1)[0:-1]
    x_vec = np.zeros((6, len(t_vec)))
    y_vec = np.zeros(len(t_vec))

    # store the line plots for each of the 6 axis
    plot_lines = [None, None, None, None, None]
    ax_subplots = [None, None, None, None, None]

    for count, sample in enumerate(sample_set):
        # pdb.set_trace()
        sample_x = sample[0][:, np.newaxis]
        sample_y = sample[1]

        assert sample_x.shape == (6, 1), f"{sample_x.shape}"
        x_vec = np.append(x_vec, sample_x, axis=-1)
        y_vec = np.append(y_vec, sample_y)
        assert x_vec.shape[0] == 6, f"{x_vec.shape}"
        assert x_vec.shape[1] == y_vec.shape[0], f"{x_vec.shape} {y_vec.shape}"

        plot_lines = live_plotter(
            count=count,
            t_vec=t_vec,
            x_data=x_vec,
            y_data=y_vec,
            plot_lines=plot_lines,
            ax_subplots=ax_subplots,
        )

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from build_input_sequence import generate_demo_input

# use ggplot style for more sophisticated visuals
plt.style.use("ggplot")

# max number of plotted points at a time
T_LIM = 100 # samples


def live_plotter(count, t_vec, x0_data, plot_line0, identifier="", x_nbins = 8, pause_time=0.02):
    x0_data_subset = x0_data[-T_LIM :]
    # initialize the subplots
    if plot_line0 == None:
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        (plot_line0,) = ax.plot(t_vec, x0_data_subset, "-", alpha=0.8)
        # update plot label/title
        plt.locator_params(axis="x", nbins=x_nbins)
        plt.ylabel("Y Label")
        plt.xlabel("Time (s)")
        plt.title("Title: {}".format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    plot_line0.set_ydata(x0_data_subset)
    # plot_line0.axes.set_xticklabels(["{:.2f}".format(x) for x in t_vec+count/50])

    # adjust limits if new data goes beyond bounds
    if (
        np.min(x0_data_subset) <= plot_line0.axes.get_ylim()[0]
        or np.max(x0_data_subset) >= plot_line0.axes.get_ylim()[1]
    ):
        plt.ylim(
            [
                np.min(x0_data_subset) - np.std(x0_data_subset),
                np.max(x0_data_subset) + np.std(x0_data_subset),
            ]
        )
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return plot_line0


if __name__ == "__main__":
    # get the input sequence
    input_X, input_Y = generate_demo_input()

    sample_set = cycle(list(zip(input_X, input_Y)))

    # store plot points
    t_vec = np.linspace(0, 2, T_LIM + 1)[0:-1]
    x0_vec = np.zeros(len(t_vec))

    # store the line plots for each of the 6 axis
    plot_lines = [None, None, None, None, None]

    for count, sample in enumerate(sample_set):
        sample_x0 = sample[0][0]
        x0_vec = np.append(x0_vec, sample_x0)

        plot_lines[0] = live_plotter(count, t_vec, x0_vec, plot_lines[0])

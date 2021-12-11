import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf


def plot_confusion_matrix(confusion_matrix, class_names):
    """
    Returns a matplotlib figure of the confusion_matrix
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(
        confusion_matrix,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    confusion_matrix = np.around(
        confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis],
        decimals=2,
    )
    # Use white text if squares are dark; otherwise black.
    threshold = confusion_matrix.max() / 2.0

    for i, j in itertools.product(
        range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
    ):
        color = "white" if confusion_matrix[i, j] > threshold else "black"
        plt.text(
            j, i, confusion_matrix[i, j], horizontalalignment="center", color=color
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

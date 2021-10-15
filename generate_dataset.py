import pdb

import numpy as np
import tensorflow as tf
from seglearn.datasets import load_watch

size = 3
shift = 1

data = load_watch()
X = data["X"]
print(f"[info] original 2:\n{X[2][:7]}")

a = tf.data.Dataset.range(0, 7)
b = tf.data.Dataset.range(7, 14)
c = tf.data.Dataset.range(14, 21)
combine = a.concatenate(b)
[print(x) for x in combine]
exit(0)

dataset_1 = tf.data.Dataset.from_tensors([a, b, c])
# dataset_1 = tf.data.Dataset.from_tensor_slices([ np.ones((7,6)), X[2][:7], X[3][:7] ])
dataset_1 = dataset_1.flat_map(
    lambda x: tf.data.Dataset.from_tensors(x).window(
        size=size, shift=shift, drop_remainder=True
    )
)
# dataset_1 = dataset_1.window(size=size, shift=shift, drop_remainder=True)
# dataset_1 = dataset_1.flat_map(lambda x: x.batch(size))
# dataset_1 = dataset_1.flat_map(lambda x: x)
print('lolg')
print(dataset_1)
[print(x) for x in dataset_1]

# for window in dataset_1:
#     print("lol")
#     print(window.numpy())

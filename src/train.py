from argparse import ArgumentParser
import os

from data_module import ShoulderExerciseDataModule
import mlflow
from model import CRNNModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Category Classifier")

    # General model paramters
    parser.add_argument("--learning-rate", type=float, help="Learning rate for model optimization")
    parser.add_argument("--num-workers", type=int, help="Number of data loader workers")

    # Early stopping parameters
    parser.add_argument("--es-monitor", type=str, help="Early stopping monitor parameter")
    parser.add_argument("--es-mode", type=str, help="Early stopping mode parameter")
    parser.add_argument("--es-verbose", type=bool, help="Early stopping verbose parameter")
    parser.add_argument("--es-patience", type=int, help="Early stopping patience parameter")

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    # Autolog parameters, metrics and artifacts to MLflow
    mlflow.pytorch.autolog()

    # Initialize CRNN model to train
    model = CRNNModel(**dict_args)

    # Initialize data module
    data_module = ShoulderExerciseDataModule(**dict_args)

    early_stopping = EarlyStopping(
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
        verbose=dict_args["es_verbose"],
        patience=dict_args["es_patience"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping, checkpoint_callback], gpus=1
    )

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

# def get_tfrecord_data(data_type: DataType = DataType.TRAIN):
#     """Generate train/test/validation tf.data.Datasets"""

#     def _parse_example_function(tfrecord_proto):
#         """Extract and formate tfrecord data"""
#         example = tf.io.parse_single_example(tfrecord_proto, FEATURE_MAP)
#         # Convert the 1D data into its original dimensions
#         X_flat_tensor = tf.sparse.to_dense(example["X"])
#         X_tensor = tf.reshape(
#             X_flat_tensor, [example["n_steps"], example["n_features"]]
#         )
#         X_dataset = tf.data.Dataset.from_tensors(X_tensor)

#         # Generate the appropriate onehot label
#         y_onehot_tensor = tf.sparse.to_dense(example["y_onehot"])
#         y_onehot_dataset = tf.data.Dataset.from_tensor_slices(y_onehot_tensor)
#         # return tf.data.dataset.zip((x_dataset, y_onehot_dataset))
#         return (X_tensor, y_onehot_tensor)

#     assert isinstance(data_type, DataType)

#     # Grab the list of file patterns for relevant csv files
#     csv_file_patterns = DATA_CONFIG[f"{data_type.value}_csv_file_pattern"]
#     assert isinstance(csv_file_patterns, list)
#     csv_file_list = []
#     # Generate a list of all matching csv file paths
#     for pattern in csv_file_patterns:
#         csv_file_list.extend(glob(pattern, recursive=True))

#     csv_path_list_str = "\n".join(csv_file_list)
#     print(f"[info] sourcing {data_type} csv: {len(csv_file_list)}")
#     tfrecord_path_list = generate_window_tfrecords(csv_file_list)
#     print(f"[info] number of windows created: {len(tfrecord_path_list)}")

#     # Generate dataset from each tfrecord
#     dataset = tf.data.TFRecordDataset(tfrecord_path_list)
#     dataset = (
#         dataset.map(
#             _parse_example_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
#         )
#         .shuffle(CRNN_CONFIG["shuffle_buffer_size"], reshuffle_each_iteration=True)
#         .batch(CRNN_CONFIG["batch_size"])
#         .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     )

#     return dataset


# def train_model():
#     """Train the crnn model"""
#     model = get_crnn_model()
#     train_tf_dataset = get_tfrecord_data(DataType.TRAIN)
#     validation_tf_dataset = get_tfrecord_data(DataType.VALIDATION)
#     # Load the validation dataset in memory
#     validation_Y = []
#     validation_X = []
#     for item in validation_tf_dataset.unbatch().take(-1):
#         validation_X.append(item[0].numpy())
#         validation_Y.append(item[1].numpy())

#     validation_X = np.array(validation_X[:])
#     validation_Y = np.array(validation_Y[:])

#     # Create the training tensorboard log directory
#     timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     logs_path = os.path.join(EVALUATION_CONFIG["log_dir"], "logs_{}".format(timestamp))

#     # Create a file writer to log confusion matrices
#     file_writer_confusion_matrix = tf.summary.create_file_writer(
#         logs_path + "/confusion_matrix"
#     )

#     def log_confusion_matrix(epoch, logs):
#         # Use the model to predict the values from the validation dataset.
#         pred_Y_softmax = model.predict(validation_X, batch_size=1)
#         pred_Y_argmax = np.argmax(pred_Y_softmax, axis=-1)
#         val_Y_argmax = np.argmax(validation_Y, axis=-1)

#         # Calculate the confusion matrix.
#         cm = sklearn.metrics.confusion_matrix(val_Y_argmax, pred_Y_argmax)
#         # Log the confusion matrix as an image summary.
#         figure = plot_confusion_matrix(cm, class_names=CRNN_CONFIG["class_names"])
#         cm_image = plot_to_image(figure)

#         # Log the confusion matrix as an image summary.
#         with file_writer_confusion_matrix.as_default():
#             tf.summary.image("Confusion Matrix: Validation", cm_image, step=epoch)

#     # Define the tensorboard callbacks
#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(
#             filepath=os.path.join(logs_path, "checkpoint_{epoch}.tf"),
#             save_weights_only=True,
#             verbose=1,
#             save_freq="epoch",
#             save_best_only=True,
#         ),
#         tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix),
#         # tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1, profile_batch='20,40'),
#         tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1),
#     ]

#     model.fit(
#         train_tf_dataset,
#         epochs=CRNN_CONFIG["epochs"],
#         validation_data=validation_tf_dataset,
#         batch_size=CRNN_CONFIG["batch_size"],
#         callbacks=callbacks,
#     )

#     return model


# if __name__ == "__main__":
#     # Set seed to make results replicable
#     seed = 0
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     train_model()

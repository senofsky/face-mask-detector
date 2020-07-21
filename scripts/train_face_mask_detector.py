"""Trains the face-mask-detector
"""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from compose import compose
from face_mask_detector.file_helper import (
    directory_is_not_readable,
    directory_is_not_writeable,
)
from imutils import paths
from PIL.Image import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from typing import List, Tuple


_LEARNING_RATE = 1e-4
_NUMBER_OF_EPOCHS = 20
_BATCH_SIZE = 32


def _parse_args() -> argparse.Namespace:
    """Parse the arguments given on the command line
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--dataset", "-d", type=str, required=True, help="path to the input dataset"
    )
    arg_parser.add_argument(
        "--plot",
        "-p",
        type=str,
        default="plot.png",
        help="path to output the loss/accuracy plot",
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="face_mask_detector.model",
        help="path to output the face mask detector model",
    )
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="enable verbose mode to print debug messages",
    )

    return arg_parser.parse_args()


def _configure_logging(verbosity: int) -> None:
    """Configures the log levels and log formats given the verbosity
    """
    if verbosity == 0:
        log_level = logging.WARNING
        log_format = "%(levelname)s:%(message)s"

    elif verbosity == 1:
        log_level = logging.INFO
        log_format = "%(levelname)s:%(message)s"

    else:
        log_level = logging.DEBUG
        log_format = "%(asctime)s:%(levelname)s:%(module)s:%(funcName)s%(message)s"

    logging.basicConfig(
        level=log_level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S",
    )


def _validate_args(args: argparse.Namespace) -> None:
    """Raises an exception if any argument is invalid
    """
    if directory_is_not_readable(args.dataset):
        logging.critical(f"dataset is not readable: {args.dataset}")
        raise IOError

    plot_parent_directory = os.path.dirname(args.plot)
    if plot_parent_directory == "":
        plot_parent_directory = "."

    directory_is_not_writeable_message = "directory is not writeable:"

    if directory_is_not_writeable(plot_parent_directory):
        logging.critical(f"{directory_is_not_writeable_message} {plot_parent_directory}")
        raise IOError

    model_parent_directory = os.path.dirname(args.model)
    if model_parent_directory == "":
        model_parent_directory = "."

    if directory_is_not_writeable(model_parent_directory):
        logging.critical(f"{directory_is_not_writeable_message} {model_parent_directory}")
        raise IOError


def _get_image_paths_from_dataset(dataset: str) -> List[str]:
    """Returns a list of image paths contained in the dataset
    """
    return list(paths.list_images(dataset))


def _generate_image_label(image_path: str) -> str:
    """Generates a label for the image given its path
    """
    return image_path.split(os.path.sep)[-2]


def _process_image(image_path: str) -> np.ndarray:
    """Processes an image via a pipeline that loads the image, converts it to
       an array, and preprocesses the image to the format required by the model
    """
    composite_function = compose(preprocess_input, img_to_array, load_img)

    return composite_function(image_path, target_size=(224, 224))


def _label_then_load_image(image_path: str) -> Tuple[str, Image]:
    """Loads the image and labels it
    """
    image_label = _generate_image_label(image_path)
    image = _process_image(image_path)

    return image_label, image


def _label_then_load_images(image_paths: List[str]) -> List[Tuple[str, Image]]:
    """Labels then loads all images
    """
    labeled_images = [_label_then_load_image(path) for path in image_paths]

    return labeled_images


def _get_labels_and_images(image_path: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Gets all images and their associative labels
    """
    labeled_images = _label_then_load_images(image_paths)

    image_labels, images = zip(*labeled_images)

    image_labels = np.array(image_labels)
    images = np.array(images, dtype="float32")

    return image_labels, images


def _construct_head_model(base_model: Model) -> Model:
    """Constructs the head of the model that will be placed on top of the base
       model
    """
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    return head_model


def _generate_plot(trained_head: Model, plot_path: str) -> None:
    """Generates a plot and saves it to the given path
    """
    plt.style.use("ggplot")
    plt.figure()
    x_range = np.arange(0, _NUMBER_OF_EPOCHS)
    plt.plot(x_range, trained_head.history["loss"], label="train_loss")
    plt.plot(x_range, trained_head.history["val_loss"], label="val_loss")
    plt.plot(x_range, trained_head.history["accuracy"], label="train_acc")
    plt.plot(x_range, trained_head.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)


if __name__ == "__main__":
    args = _parse_args()

    _configure_logging(args.verbose)

    logging.info("validating the dataset")
    try:
        _validate_args(args)
    except IOError:
        sys.exit(1)

    logging.info("Loading images from dataset...")
    image_paths = _get_image_paths_from_dataset(args.dataset)

    image_labels, images = _get_labels_and_images(image_paths)

    lb = LabelBinarizer()
    image_labels = lb.fit_transform(image_labels)
    image_labels = to_categorical(image_labels)

    # load the MobileNetV2 network, ensuring the head FC layer sets are left off
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
    )

    head_model = _construct_head_model(base_model)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in base_model.layers:
        layer.trainable = False

    # compile our model
    logging.info("compiling model...")

    optimizer = Adam(lr=_LEARNING_RATE, decay=(_LEARNING_RATE / _NUMBER_OF_EPOCHS))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # construct the training image generator for data augmentation
    image_data_generator = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Partition the data into training and testing groups
    # 75% for training, 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(
        images, image_labels, test_size=0.20, stratify=image_labels, random_state=42
    )

    # train the head of the network
    logging.info("training head...")
    trained_head = model.fit(
        image_data_generator.flow(trainX, trainY, batch_size=_BATCH_SIZE),
        steps_per_epoch=(len(trainX) // _BATCH_SIZE),
        validation_data=(testX, testY),
        validation_steps=(len(testX) // _BATCH_SIZE),
        epochs=_NUMBER_OF_EPOCHS,
    )

    # make predictions on the testing set
    logging.info("evaluating network...")
    predictions = model.predict(testX, batch_size=_BATCH_SIZE)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predictions = np.argmax(predictions, axis=1)

    # show a nicely formatted classification report
    print(
        classification_report(testY.argmax(axis=1), predictions, target_names=lb.classes_)
    )

    _generate_plot(trained_head, args.plot)

    # serialize the model to disk
    logging.info("saving face mask detector model...")
    model.save(args.model, save_format="h5")

    sys.exit(0)

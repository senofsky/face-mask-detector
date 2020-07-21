"""Detects face masks in images
"""

import argparse
import cv2
import numpy as np
import os
import sys

from face_mask_detector.file_helper import file_is_not_readable, directory_is_not_readable
from face_mask_detector.logger import generate_logger
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def _parse_args() -> argparse.Namespace:
    """Parse the arguments given on the command line
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    arg_parser.add_argument(
        "--image", "-i", type=str, required=True, help="path to the input image"
    )
    arg_parser.add_argument(
        "--face",
        "-f",
        type=str,
        default="face_detector",
        help="path to face detector model directory",
    )
    arg_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="face_mask_detector.model",
        help="path to the face mask detector model",
    )
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="enable verbose mode to print debug messages",
    )

    return arg_parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Raises an exception if any argument is invalid
    """
    if directory_is_not_readable(args.face):
        logger.error(f"face detector model is not readable: {args.face}")
        raise IOError

    if file_is_not_readable(args.model):
        logger.error(f"face detector model is not readable: {args.model}")
        raise IOError

    if file_is_not_readable(args.image):
        logger.error(f"image is not readable: {args.image}")
        raise IOError


def _generate_neural_net(model_directory: str):
    """Generates a neural net that detects faces
    """
    prototxtPath = os.path.sep.join([model_directory, "deploy.prototxt"])
    weightsPath = os.path.sep.join(
        [model_directory, "res10_300x300_ssd_iter_140000.caffemodel"]
    )
    neural_net = cv2.dnn.readNet(prototxtPath, weightsPath)

    return neural_net


def _get_box_dimensions(detection, width, height):
    """Computes the (x, y)-coordinates of the bounding box for the object
    """
    box = detection * np.array([width, height, width, height])
    (startX, startY, endX, endY) = box.astype("int")

    # Ensure the bounding boxes fall within the dimensions of the frame
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

    return startX, startY, endX, endY


def _process_detection(index, detections, confidence_threshold, image):
    """Processes a detection in an image
    """
    (height, width) = image.shape[:2]

    # extract the confidence (i.e., probability) associated with
    # the detection
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is
    # greater than the minimum confidence
    if confidence > confidence_threshold:

        detection = detections[0, 0, i, 3:7]
        (startX, startY, endX, endY) = _get_box_dimensions(detection, width, height)

        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # pass the face through the model to determine if the face
        # has a mask or not
        (mask, withoutMask) = model.predict(face)[0]

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        logger.info(f"Detected = {label}")

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(
            image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
        )
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)


if __name__ == "__main__":
    args = _parse_args()

    logger = generate_logger(__name__, args.verbose)

    try:
        _validate_args(args)
    except IOError:
        sys.exit(1)

    logger.info("loading serialized face detector model...")
    net = _generate_neural_net(args.face)

    logger.info("loading face mask detector model...")
    model = load_model(args.model)

    # load the input image, clone it, and grab the image spatial dimensions
    image = cv2.imread(args.image)

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    logger.info("computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        _process_detection(i, detections, args.confidence, image)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

"""Detects mask in webcam video stream
"""
import argparse
import cv2
import imutils
import logging
import numpy as np
import os
import sys
import time

from face_mask_detector.file_helper import file_is_not_readable, directory_is_not_readable
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from typing import Tuple


def _parse_args() -> argparse.Namespace:
    """Parse the arguments given on the command line
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "-f",
        "--face",
        type=str,
        default="face_detector",
        help="path to face detector model directory",
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="face_mask_detector.model",
        help="path to trained face mask detector model",
    )
    arg_parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="enable verbose mode to logging.info debug messages",
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
    if directory_is_not_readable(args.face):
        logging.error(f"face detector model is not readable: {args.face}")
        raise IOError

    if file_is_not_readable(args.model):
        logging.error(f"face detector model is not readable: {args.model}")
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


def _get_box_dimensions(detection, width, height) -> Tuple[int, int, int, int]:
    """Computes the (x, y)-coordinates of the bounding box for the object
    """
    box = detection * np.array([width, height, width, height])
    (startX, startY, endX, endY) = box.astype("int")

    # ensure the bounding boxes fall within the dimensions of
    # the frame
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

    return startX, startY, endX, endY


def _detect_and_predict_mask(frame, faceNet, maskNet, confidence_threshold):
    """Detects faces and predicts if a mask is present
    """
    # grab the dimensions of the frame and then construct a blob
    # from it
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locations = []
    predictions = []

    # loop over the detections
    for i in range(0, detections.shape[2]):

        # extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confidence_threshold:
            detection = detections[0, 0, i, 3:7]
            (startX, startY, endX, endY) = _get_box_dimensions(detection, width, height)

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=32)

    return (locations, predictions)


if __name__ == "__main__":
    args = _parse_args()

    _configure_logging(args.verbose)

    try:
        _validate_args(args)
    except IOError:
        sys.exit(1)

    logging.info("loading face detector model...")
    faceNet = _generate_neural_net(args.face)

    logging.info("loading face mask detector model...")
    maskNet = load_model(args.model)

    logging.info("starting video stream...")
    video_stream = VideoStream(src=0).start()

    # allow the camera sensor to warm up
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = video_stream.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locations, predictions) = _detect_and_predict_mask(
            frame, faceNet, maskNet, args.confidence
        )

        # loop over the detected face locations and their corresponding
        # prediction
        for (box, prediction) in zip(locations, predictions):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = prediction

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    video_stream.stop()

    sys.exit(0)

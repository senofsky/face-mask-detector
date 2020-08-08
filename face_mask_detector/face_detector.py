"""This module provides a function that detects faces in images
"""

import cv2
import logging

from . import face_detector_model
from importlib_resources import files, as_file


def _load_face_detecting_neural_net():
    """Loads a neural net that detects faces
    """
    logging.info("loading serialized face detector model...")

    prototxt_source = files(face_detector_model).joinpath("deploy.prototxt")
    weights_source = files(face_detector_model).joinpath(
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    with as_file(prototxt_source) as prototxt_path, as_file(
        weights_source
    ) as weights_path:
        neural_net = cv2.dnn.readNet(str(prototxt_path), str(weights_path))

    return neural_net


def detect_faces_in_image(image):
    """Detects faces in images using a neural net
    """
    logging.info("computing face detections...")

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    neural_net = _load_face_detecting_neural_net()
    neural_net.setInput(blob)

    detections = neural_net.forward()

    return detections

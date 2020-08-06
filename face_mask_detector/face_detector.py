"""This module provides a function that detects faces in images
"""

import cv2
import logging
import os


def _load_face_detecting_neural_net():
    """Loads a neural net that detects faces
    """
    logging.info("loading serialized face detector model...")

    face_detector_model_directory = "face_detector"

    prototxt_path = os.path.sep.join([face_detector_model_directory, "deploy.prototxt"])
    weights_path = os.path.sep.join(
        [face_detector_model_directory, "res10_300x300_ssd_iter_140000.caffemodel"]
    )
    neural_net = cv2.dnn.readNet(prototxt_path, weights_path)

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

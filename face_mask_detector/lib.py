"""This module provides functions to detect face masks in video streams and images
"""

import cv2
import numpy as np

from .face_detector import detect_faces_in_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from typing import List

_BGR_GREEN = (0, 255, 0)
_BGR_RED = (0, 0, 255)


def _get_box_dimensions(detection, width, height):
    """Computes the (x, y)-coordinates of the bounding box for the object
    """
    box = detection * np.array([width, height, width, height])
    (start_x, start_y, end_x, end_y) = box.astype("int")

    # Ensure the bounding boxes fall within the dimensions of the frame
    (start_x, start_y) = (max(0, start_x), max(0, start_y))
    (end_x, end_y) = (min(width - 1, end_x), min(height - 1, end_y))

    return start_x, start_y, end_x, end_y


def _process_region_of_interest(region_of_interest):
    """Extracts the region of interest, converts it from BGR to RGB channel
       ordering, resizes it to 224x224, and preprocesss it
    """
    face = region_of_interest
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    return face


def _load_face_mask_detector_model():
    """Loads the face mask detector model
    """
    face_mask_detector_model_path = "face_mask_detector.model"

    return load_model(face_mask_detector_model_path)


def _detect_face_mask_in_face_detection(face_detection, image):
    """Detects face mask in face detection
    """
    (height, width) = image.shape[:2]

    (start_x, start_y, end_x, end_x) = _get_box_dimensions(face_detection, width, height)

    face = _process_region_of_interest(image[start_y:end_x, start_x:end_x])

    face_mask_detector_model = _load_face_mask_detector_model()

    # pass the face through the model to determine if the face
    # has a mask or not
    (mask, withoutMask) = face_mask_detector_model.predict(face)[0]

    # determine the class label and color we'll use to draw
    # the bounding box and text
    label = "Mask" if mask > withoutMask else "No Mask"
    color = _BGR_GREEN if label == "Mask" else _BGR_RED

    # include the probability in the label
    probability = max(mask, withoutMask)
    label = "{}: {:.2f}%".format(label, probability * 100)

    # display the label and bounding box rectangle on the output frame
    cv2.putText(
        image, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
    )
    cv2.rectangle(image, (start_x, start_y), (end_x, end_x), color, 2)


def _get_face_detection(face_detections, index):
    """Returns the face detection
    """
    return face_detections[0, 0, index, 3:7]


def _get_face_detection_confidence(face_detections, index):
    """Returns the confidence (probability) that the detection contains a face
    """
    return face_detections[0, 0, index, 2]


def face_mask_in_image(image_path: str, confidence=0.5) -> bool:
    """This function returns True if a face mask is in an image

    Parameters:
        image_path (str): the path to the image

    Returns:
        bool: True if a face mask is in the picture, otherwise False
    """
    pass


def probability_of_face_mask_in_image(image_path: str) -> float:
    pass


def multiple_face_masks_in_image(image_path: str) -> bool:
    pass


def probabilities_of_face_masks_in_image(image_path: str) -> List[float]:
    pass


def display_image_with_face_mask_detections(
    image_path: str, confidence_threshold: float
) -> None:

    # load the input image, clone it, and grab the image spatial dimensions
    image = cv2.imread(image_path)

    face_detections = detect_faces_in_image(image)

    for index in range(0, face_detections.shape[2]):

        face_detection = _get_face_detection(face_detections, index)

        face_detection_confidence = _get_face_detection_confidence(face_detections, index)

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if face_detection_confidence > confidence_threshold:

            _detect_face_mask_in_face_detection(face_detection, image)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

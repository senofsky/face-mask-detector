"""This module provides functions to detect face masks in video streams and images
"""

import cv2
import face_mask_detector.face_mask_detector_model
import imutils
import numpy as np
import time

from importlib_resources import files, as_file
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from typing import List, Tuple

from .face_detector import detect_faces_in_image

_BGR_GREEN = (0, 255, 0)
_BGR_RED = (0, 0, 255)


def _get_box_dimensions(detection, width, height) -> Tuple[int, int, int, int]:
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
    # face = np.expand_dims(face, axis=0)

    return face


def _add_probability_to_label(probability, label):
    return "{}: {:.2f}%".format(label, probability * 100)


def _add_labeled_box_to_image(image, box_label, start_x, start_y, end_x, end_y):
    """Adds the label and bounding box rectangle on the output frame
    """

    (label, probability, color) = box_label
    label = _add_probability_to_label(probability, label)

    cv2.putText(
        image, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
    )
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)


def _get_face_detection(face_detections, index):
    """Returns the face detection
    """
    return face_detections[0, 0, index, 3:7]


def _get_face_detection_confidence(face_detections, index):
    """Returns the confidence (probability) that the detection contains a face
    """
    return face_detections[0, 0, index, 2]


def _generate_label_from_prediction(prediction):
    (mask, withoutMask) = prediction
    return "Mask" if mask > withoutMask else "No Mask"


def _generate_color_from_prediction(prediction):
    (mask, withoutMask) = prediction
    return _BGR_GREEN if mask > withoutMask else _BGR_RED


def _generate_probability_from_prediction(prediction):
    (mask, withoutMask) = prediction
    return max(mask, withoutMask)


def _get_box_label_from_prediction(prediction):
    return (
        _generate_label_from_prediction(prediction),
        _generate_probability_from_prediction(prediction),
        _generate_color_from_prediction(prediction),
    )


def _get_frame_from_video_stream(video_stream):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)

    return frame


def load_face_mask_detector_model():
    """Loads the face mask detector model
    """
    face_mask_detector_model_source = files(
        face_mask_detector.face_mask_detector_model
    ).joinpath("face_mask_detector.model")

    with as_file(face_mask_detector_model_source) as face_mask_detector_model_path:
        face_mask_detector_model = load_model(face_mask_detector_model_path)

    return face_mask_detector_model


def get_face_mask_detections(face_mask_detector_model, frame, confidence_threshold):
    """Detects faces and predicts if a mask is present
    """
    (height, width) = frame.shape[:2]

    face_detections = detect_faces_in_image(frame)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locations = []
    predictions = []

    for index in range(0, face_detections.shape[2]):

        face_detection_confidence = _get_face_detection_confidence(face_detections, index)

        if face_detection_confidence > confidence_threshold:

            face_detection = _get_face_detection(face_detections, index)
            (start_x, start_y, end_x, end_y) = _get_box_dimensions(
                face_detection, width, height
            )

            face = _process_region_of_interest(frame[start_y:end_y, start_x:end_x])

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locations.append((start_x, start_y, end_x, end_y))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:

        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        predictions = face_mask_detector_model.predict(faces, batch_size=32)

    return (locations, predictions)


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

    face_mask_detector_model = load_face_mask_detector_model()

    (locations, predictions) = get_face_mask_detections(
        face_mask_detector_model, image, confidence_threshold
    )

    # loop over the detected face locations and their corresponding
    # prediction
    for (box, prediction) in zip(locations, predictions):

        (start_x, start_y, end_x, end_y) = box

        box_label = _get_box_label_from_prediction(prediction)

        _add_labeled_box_to_image(
            image, box_label, start_x, start_y, end_x, end_y,
        )

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


def display_video_with_face_mask_detections(
    video_stream_source, confidence_threshold: float
) -> None:

    video_stream = VideoStream(src=video_stream_source).start()

    # allow the camera sensor to warm up
    time.sleep(2.0)

    face_mask_detector_model = load_face_mask_detector_model()
    # loop over the frames from the video stream
    while True:

        frame = _get_frame_from_video_stream(video_stream)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locations, predictions) = get_face_mask_detections(
            face_mask_detector_model, frame, confidence_threshold
        )

        # loop over the detected face locations and their corresponding
        # prediction
        for (box, prediction) in zip(locations, predictions):

            (start_x, start_y, end_x, end_y) = box

            box_label = _get_box_label_from_prediction(prediction)

            _add_labeled_box_to_image(
                frame, box_label, start_x, start_y, end_x, end_y,
            )

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    video_stream.stop()

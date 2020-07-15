"""Unit-tests for face_mask_detector
"""

import face_mask_detector
import os
import pytest


def _generate_image_path(image_name):
    return os.path.join("images/", image_name)


def test_that_face_mask_in_image_raises_exception_with_invalid_image_path():
    with pytest.raises(FileNotFoundError):
        invalid_image_path = "invalid-path"
        face_mask_detector.face_mask_in_image(invalid_image_path)


def test_that_face_mask_in_image_detects_face_mask_in_image():
    image_with_face_mask_path = _generate_image_path("person-wearing-face-mask.png")

    assert face_mask_detector.face_mask_in_image(image_with_face_mask_path)


def test_that_face_mask_in_image_detects_no_face_mask_in_image():
    image_without_face_mask_path = _generate_image_path(
        "person-not-wearing-face-mask.png"
    )

    assert face_mask_detector.face_mask_in_image(image_without_face_mask_path) == False

"""This module provides an command line interface to face-mask-detector which
   allows face-mask-detector to behave as a command line utility
"""

import argparse
import sys

from .file_helper import file_is_not_readable, directory_is_not_readable
from .lib import (
    display_image_with_face_mask_detections,
    display_video_with_face_mask_detections,
)


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
    arg_parser.add_argument("--image", "-i", type=str, help="path to the input image")
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="enable verbose mode to print debug messages",
    )

    return arg_parser.parse_args()


def main():

    args = _parse_args()

    try:
        if args.image:
            display_image_with_face_mask_detections(args.image, args.confidence)
        else:
            display_video_with_face_mask_detections(0, args.confidence)

    except IOError as error:
        print(error)
        sys.exit(1)

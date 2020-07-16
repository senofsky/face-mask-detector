#!/usr/bin/env python

import os
import re

from setuptools import setup, find_packages

long_description = open(
    os.path.join(
        os.path.dirname(__file__),
        "README.md"
    )
).read()

with open("face_mask_detector/__init__.py", encoding="utf8") as file:
    version = re.search(r'__version__ = "(.*?)"', file.read()).group(1)

setup(
    name="face-mask-detector",
    author="Mark V Senofsky",
    author_email="mark.v.senofsky@gmail.com",
    version=version,
    license="MIT",
    url="https://github.com/senofsky/face-mask-detector",
    description="A library and tool for detecting face masks in images and video streams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(".", exclude=["test"]),
    install_requires=[
        "imutils==0.5.3",
        "numpy==1.19.0",
        "opencv-python==4.3.0.36",
        "tensorflow==2.2.0",
        "pillow==7.2.0",
        "scikit-learn==0.23.1",
        "compose==1.1.1",
        "matplotlib==3.2.2"
    ],
    extras_require={
        "dev": [
            "black==19.10b0",
            "mypy==0.770",
            "pytest==5.4.3",
            "typing==3.7.4.1"
        ]
    }
)

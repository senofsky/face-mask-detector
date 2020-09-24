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
    description="A library and command line utility for detecting face masks in images and video streams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(".", exclude=["test"]),
    include_package_data=True,
    install_requires=[
        "imutils==0.5.3",
        "importlib-resources==3.0.0",
        "numpy==1.19.0",
        "opencv-python==4.3.0.36",
        "tensorflow==2.2.0",
        "pillow==7.2.0",
        "scikit-learn==0.23.1",
        "compose==1.1.1",
        "matplotlib==3.2.2",
        "scipy==1.4.1"
    ],
    extras_require={
        "dev": [
            "black==19.10b0",
            "mypy==0.770",
            "pytest==5.4.3",
            "typing==3.7.4.1"
        ]
    },
    entry_points={
        "console_scripts": [
            "face-mask-detector=face_mask_detector.command_line_interface:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=[
        "face-mask-detection",
        "face-mask",
        "covid-19",
        "machine-learning",
        "computer-vision",
    ],
    python_requires=">=3.6, <4",
)

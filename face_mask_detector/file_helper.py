"""Provides convenience functions for interacting with files and directories
"""
import os


def file_is_not_readable(file_path: str) -> bool:
    """Returns True if the given file is not readable
    """
    if os.access(file_path, os.R_OK):
        return False

    return True


def directory_is_not_readable(directory_path: str) -> bool:
    """Returns True if the given directory is not readable
    """
    if os.access(directory_path, os.R_OK):
        return False

    return True


def directory_is_not_writeable(directory_path: str) -> bool:
    """Returns True if the given directory is not writeable
    """
    if os.access(directory_path, os.W_OK):
        return False

    return True

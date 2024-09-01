# -*- coding: utf-8 -*-
import io


def _make_r_io_base(f, mode: str):
    """
    Ensure the input is a readable file-like object, otherwise open the file with the specified mode.

    Args:
        f (str or IO): The path or file-like object to be checked or opened.
        mode (str): The mode to open the file if needed.

    Returns:
        IO: A readable file-like object.

    Examples:
        # Ensure f is a readable file-like object
        file_obj = _make_r_io_base(f, mode="r")

    Note:
        If `f` is already a file-like object (`io.IOBase`), it is returned as is.
        If `f` is a string (file path), it opens the file in the specified mode.
    """
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

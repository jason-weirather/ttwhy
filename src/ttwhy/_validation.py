"""Small shared validation barriers for public numerical operations."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def image_array(image: ArrayLike) -> NDArray:
    value = np.asarray(image)
    if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[-1] != 3):
        raise ValueError(f"Expected a YX grayscale or YXS RGB image; got {value.shape}")
    if any(size == 0 for size in value.shape):
        raise ValueError("Image dimensions must be nonzero")
    if value.dtype.kind not in "buif":
        raise TypeError(f"Image must have a real numeric dtype; got {value.dtype}")
    return value

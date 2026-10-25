"""RAM-only display pyramids for already-decoded ordinary images.

Level zero is an independent, unchanged snapshot. Reduced pixels are rounded
finite-value box means in the same dtype. Unrounded sums and counts are carried
between reductions, so odd edges and repeated rounding do not bias later levels.
This is display preparation, not a scientific image conversion or a disk cache.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import image_array


def image_pyramid(pixels: ArrayLike) -> tuple[NDArray, ...]:
    """Own a YX/YXS image and its 2x, 4x, ... overviews, through 1x1.

    Every level has ceil(base_height / scale), ceil(base_width / scale) pixels.
    The final row/column covers only the remaining base pixels, never replicated
    or zero-padded edges. Nonfinite floating samples are omitted independently
    per component; entirely missing coarse components stay NaN. Base pixels,
    including nonfinite values, are retained verbatim. All arrays are read-only.

    The retained ordinary 2D pyramid is about 4/3 of the base raster's bytes;
    skinny images can approach twice the base. Temporary float64 sums/counts
    require additional RAM during construction. No file I/O occurs here.
    """
    base = np.array(image_array(pixels), copy=True, order="C")
    base.flags.writeable = False
    levels = [base]
    totals, counts = base, None
    while totals.shape[0] > 1 or totals.shape[1] > 1:
        shape = ((totals.shape[0] + 1) // 2, (totals.shape[1] + 1) // 2, *base.shape[2:])
        next_totals = np.zeros(shape, dtype=np.float64)
        next_counts = np.zeros(shape, dtype=np.float64)
        for dy in (0, 1):
            for dx in (0, 1):
                part = totals[dy::2, dx::2]
                target = (slice(0, part.shape[0]), slice(0, part.shape[1]))
                if counts is None:
                    valid = np.isfinite(part)
                    np.add(next_totals[target], part, out=next_totals[target], where=valid)
                    next_counts[target] += valid
                else:
                    next_totals[target] += part
                    next_counts[target] += counts[dy::2, dx::2]
        means = np.full(shape, np.nan, dtype=np.float64)
        np.divide(next_totals, next_counts, out=means, where=next_counts > 0)
        if base.dtype.kind in "bui":
            np.rint(means, out=means)
            limits = (0, 1) if base.dtype.kind == "b" else (
                np.iinfo(base.dtype).min, np.iinfo(base.dtype).max)
            np.clip(means, *limits, out=means)
        reduced = means.astype(base.dtype)
        reduced.flags.writeable = False
        levels.append(reduced)
        totals, counts = next_totals, next_counts
    return tuple(levels)

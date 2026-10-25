"""Display geometry derived from an Omeify image, not a second file reader."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omeify import Image


class ReadBudgetExceeded(ValueError):
    """An interactive viewport exceeds its permitted number of source samples."""


@dataclass(frozen=True, slots=True)
class PyramidLevel:
    """Level-to-camera mapping; extent-fit mappings are explicitly approximate."""

    index: int
    height: int
    width: int
    scale_y: float
    scale_x: float
    approximate: bool = False


@dataclass(frozen=True, slots=True)
class ViewInfo:
    """Owned display metadata; neither a path nor a complete image is required."""

    name: str
    shape: tuple[int, int]
    dtype: np.dtype
    channels: tuple[str, ...]
    is_rgb: bool
    levels: tuple[PyramidLevel, ...]
    pixel_aspect: float
    pixel_size: tuple[float, float, str] | None

    @classmethod
    def from_image(cls, image: "Image", *, name: str) -> "ViewInfo":
        """Use declared level scales, retaining labelled extent-fit display fallback.

        An unknown scale is never turned into physical calibration. The fallback
        preserves coarse browsing of third-party files, not exact registration.
        """
        height, width = image.levels[0].spatial_shape
        levels = []
        for item in image.levels:
            # A shifted pixel-edge grid cannot be mapped with scale alone.
            # Until the general affine camera path carries offsets, omit that
            # level from ALL display plans, including sampled/runtime overviews.
            # The aligned base still gives an accurate, if possibly slower, view.
            offset = item.offset_yx
            if item.index and offset is not None and any(abs(v) > 1e-9 for v in offset):
                continue
            ly, lx = item.spatial_shape
            scale = item.downsample_yx
            sy, sx = (height / ly, width / lx) if scale is None else scale
            levels.append(PyramidLevel(item.index, ly, lx, sy, sx, scale is None))
        size = image.pixel_size
        return cls(
            name, (height, width), image.dtype, image.channel_names,
            image.image_type == "rgb", tuple(levels),
            1.0 if size is None else size.y / size.x,
            None if size is None else size.to_tuple(),
        )

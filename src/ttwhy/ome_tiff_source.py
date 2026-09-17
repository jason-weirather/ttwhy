"""Omeify is the sole file decoder and metadata interpreter."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path
from threading import RLock
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from ._validation import positive_integer


class ReadBudgetExceeded(ValueError):
    """The requested viewport needs too many source samples for an interactive read."""


@dataclass(frozen=True, slots=True)
class PyramidLevel:
    """Raster dimensions and display-extent mapping, not inferred calibration."""

    index: int
    height: int
    width: int
    scale_y: float
    scale_x: float


@dataclass(frozen=True, slots=True)
class SlideInfo:
    """Owned metadata snapshot; no pixels are loaded to construct this record."""

    path: Path
    series: int
    series_name: str | None
    series_count: int
    shape: tuple[int, int]
    dtype: np.dtype
    channels: tuple[str, ...]
    is_rgb: bool
    levels: tuple[PyramidLevel, ...]
    pixel_aspect: float
    pixel_size: tuple[float, float, str] | None


class OmeTiffSource:
    """Context-managed, read-only Omeify adapter for one OME image series.

    Reads and close are serialized. Pixel results are independent arrays.
    Reader caches retain at most one active channel/level pair: switching pairs
    clears the previous channel through Omeify's public API. Omeify 0.16.1
    targets 64 MiB of decoded segments per plane, but one oversized source tile
    or strip can exceed that. This adapter does not claim a strict RSS limit.
    """

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        series: int = 0,
        max_read_pixels: int = 4_194_304,
    ) -> None:
        if isinstance(series, bool) or not isinstance(series, Integral) or series < 0:
            raise ValueError("series must be a nonnegative integer")
        self.path = Path(path)
        self.series = int(series)
        self.max_read_pixels = positive_integer(max_read_pixels, "max_read_pixels")
        self._reader = None
        self._info: SlideInfo | None = None
        self._cached_pair: tuple[int, int] | None = None
        self._lock = RLock()

    @property
    def is_open(self) -> bool:
        return self._reader is not None

    @property
    def info(self) -> SlideInfo:
        if self._info is None:
            raise RuntimeError("Open the source before accessing its metadata")
        return self._info

    def open(self) -> "OmeTiffSource":
        """Open the reader and copy display metadata, without reading source pixels."""
        from omeify import OMETiffReader

        with self._lock:
            if self.is_open:
                return self
            reader = OMETiffReader(self.path, series=self.series)
            try:
                reader.open()
                # This public Omeify property validates singleton nonspatial axes
                # and supported RGB/planar layouts, without materializing pixels.
                axes, shape = reader.output_axes, reader.output_shape
                height, width = shape[axes.index("Y")], shape[axes.index("X")]
                levels = []
                for index, level in enumerate(reader.levels):
                    dimensions = dict(zip(level.axes, level.shape, strict=True))
                    ly, lx = int(dimensions["Y"]), int(dimensions["X"])
                    if ly < 1 or lx < 1 or ly > height or lx > width:
                        raise ValueError(f"Invalid pyramid dimensions at level {index}")
                    levels.append(PyramidLevel(index, ly, lx, height / ly, width / lx))
                size = reader.pixel_size
                info = SlideInfo(
                    self.path, self.series, reader.series_name, reader.series_count,
                    (height, width), reader.dtype,
                    tuple(channel.name for channel in reader.channels), reader.is_rgb,
                    tuple(levels), 1.0 if size is None else size.y / size.x,
                    None if size is None else size.to_tuple(),
                )
            except BaseException:
                reader.close()
                raise
            self._reader, self._info = reader, info
            return self

    def select_channels(self, channels: Sequence[int]) -> tuple[int, ...]:
        """Validate a nonempty ordered selection of logical channel indices."""
        if len(channels) == 0:
            raise ValueError("Select at least one channel")
        result = []
        for index in channels:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("Channel indices must be integers")
            if not 0 <= index < len(self.info.channels):
                raise IndexError(f"Channel {index} is outside 0..{len(self.info.channels) - 1}")
            if index in result:
                raise ValueError(f"Channel {index} is selected more than once")
            result.append(int(index))
        return tuple(result)

    def read_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        channel: int = 0,
        level: int = 0,
    ) -> NDArray:
        """Read one logical YX or RGB YXS channel at level-pixel coordinates."""
        with self._lock:
            if self._reader is None:
                raise RuntimeError("OmeTiffSource is closed")
            self.select_channels([channel])
            if isinstance(level, bool) or not isinstance(level, Integral):
                raise TypeError("level must be an integer")
            if not 0 <= level < len(self.info.levels):
                raise IndexError("Pyramid level does not exist")
            dimensions = self.info.levels[level]
            if any(isinstance(x, bool) or not isinstance(x, Integral)
                   for x in (y0, y1, x0, x1)):
                raise TypeError("Region bounds must be integers")
            if not (0 <= y0 < y1 <= dimensions.height and 0 <= x0 < x1 <= dimensions.width):
                raise ValueError("Region must be nonempty and lie inside the selected level")
            pixels = (y1 - y0) * (x1 - x0) * (3 if self.info.is_rgb else 1)
            if pixels > self.max_read_pixels:
                raise ReadBudgetExceeded(f"Read needs {pixels:,} samples; limit is "
                                         f"{self.max_read_pixels:,}. Zoom in or use a pyramid.")
            pair = (int(channel), int(level))
            if self._cached_pair is not None and pair != self._cached_pair:
                self._reader[self._cached_pair[0]].clear_cache()
            self._cached_pair = pair
            # Logical RGB channel access keeps all three samples together.
            return self._reader[channel].read_region(y0, y1, x0, x1, level=level).copy()

    def clear_cache(self) -> None:
        """Release decoded source segments without closing the file."""
        with self._lock:
            if self._reader is not None:
                for channel in self._reader.channels:
                    channel.clear_cache()
            self._cached_pair = None

    def close(self) -> None:
        with self._lock:
            if self._reader is not None:
                self._reader.close()
                self._reader = None
            self._cached_pair = None

    def __enter__(self) -> "OmeTiffSource":
        return self.open()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"OmeTiffSource(path={str(self.path)!r}, series={self.series}, "
                f"open={self.is_open})")

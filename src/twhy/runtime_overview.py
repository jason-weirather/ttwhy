"""Bounded, full-coverage display reductions built incrementally from native levels.

These are session-local display caches, not new Omeify levels or calibration.
One step reads one disjoint native rectangle. The viewer schedules steps only
between foreground frames on the same worker; no concurrent source I/O occurs.
"""

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite, isqrt

import numpy as np
from numpy.typing import NDArray

from .resampling import resample_image
from .view_info import PyramidLevel, ViewInfo
from .viewport import Viewport

DEFAULT_OVERVIEW_CACHE_MIB = 64.0
DEFAULT_OVERVIEW_MAX_SOURCE_PIXELS = 134_217_728
# Both retained resolution and per-step work are bounded independently of a WSI.
MAX_OVERVIEW_PIXELS = 262_144
MAX_CHUNK_SAMPLES = 262_144


@dataclass(frozen=True, slots=True)
class OverviewSpec:
    level: PyramidLevel
    factor: int
    shape: tuple[int, int]
    coverage: tuple[float, float]  # Valid native pixel-edge extent.
    components: int

    @property
    def nbytes(self) -> int:
        # Weighted values and finite coverage, both float64. The builder becomes
        # the cache entry in place, with no second full output allocation.
        return self.shape[0] * self.shape[1] * self.components * 16


@dataclass(frozen=True, slots=True)
class OverviewProgress:
    pending: bool = False
    published: bool = False


@dataclass(frozen=True, slots=True)
class IntegratedOverview:
    spec: OverviewSpec
    weighted: NDArray
    support: NDArray

    def sample(self, viewport: Viewport, *, rows: int, columns: int) -> NDArray:
        """Area-resample the cached reduction, including finite support and edges.

        Box means cover every native pixel. Arbitrary subsequent camera footprints
        still approximate within each reduced pixel, as any display pyramid does.
        """
        spec = self.spec
        sy, sx = spec.level.scale_y * spec.factor, spec.level.scale_x * spec.factor
        bounds = (viewport.x / sx, viewport.y / sy, viewport.width / sx, viewport.height / sy)
        coverage = tuple(value / spec.factor for value in spec.coverage)
        numerator = resample_image(self.weighted, rows=rows, columns=columns,
                                   bounds=bounds, coverage_shape=coverage)
        denominator = resample_image(self.support, rows=rows, columns=columns,
                                     bounds=bounds, coverage_shape=coverage)
        np.divide(numerator, denominator, out=numerator, where=denominator > 0)
        numerator[~(denominator > 0)] = np.nan
        return numerator


class OverviewBuilder:
    """One channel's reduction; partial arrays are never presented as an image."""

    def __init__(self, spec: OverviewSpec, channel: int, max_read_pixels: int) -> None:
        self.spec, self.channel = spec, channel
        tail = (3,) if spec.components == 3 else ()
        self.weighted = np.zeros((*spec.shape, *tail), dtype=np.float64)
        self.support = np.zeros_like(self.weighted)
        pixels = min(MAX_CHUNK_SAMPLES, max_read_pixels) // spec.components
        if pixels < 1:
            raise ValueError("Overview chunk budget cannot hold one pixel")
        # Prefer compact rectangles, aligning reductions when possible. Very
        # large reduction blocks can span chunks; sums/support accumulate exactly.
        self.chunk_h = min(spec.level.height, max(1, isqrt(pixels)), 512)
        self.chunk_w = min(spec.level.width, pixels // self.chunk_h, 512)
        if spec.factor <= self.chunk_h:
            self.chunk_h = self.chunk_h // spec.factor * spec.factor
        if spec.factor <= self.chunk_w:
            self.chunk_w = self.chunk_w // spec.factor * spec.factor
        self.y = self.x = self.processed = 0

    @property
    def done(self) -> bool:
        return self.y >= self.spec.level.height

    def step(self, read_region) -> None:
        if self.done:
            return
        spec = self.spec
        y0, x0 = self.y, self.x
        y1 = min(spec.level.height, y0 + self.chunk_h)
        x1 = min(spec.level.width, x0 + self.chunk_w)
        pixels = np.asarray(read_region(self.channel, spec.level.index, (y0, y1, x0, x1)))
        tail = (3,) if spec.components == 3 else ()
        if pixels.shape != (y1 - y0, x1 - x0, *tail):
            raise ValueError("Overview reader returned an unexpected regional shape")
        finite = np.isfinite(pixels)
        wy = np.clip(spec.coverage[0] - np.arange(y0, y1), 0, 1)
        wx = np.clip(spec.coverage[1] - np.arange(x0, x1), 0, 1)
        weight = wy[:, None] * wx[None, :]
        if tail:
            weight = weight[..., None]
        values = np.where(finite, pixels, 0).astype(np.float64)
        values *= weight
        support = finite * weight
        # Partition this chunk at the global reduction-block boundaries. No
        # rounding of means and no equal weighting of partial edge blocks.
        ys = np.r_[0, np.arange((y0 // spec.factor + 1) * spec.factor, y1, spec.factor) - y0]
        xs = np.r_[0, np.arange((x0 // spec.factor + 1) * spec.factor, x1, spec.factor) - x0]
        target = (slice(y0 // spec.factor, (y1 - 1) // spec.factor + 1),
                  slice(x0 // spec.factor, (x1 - 1) // spec.factor + 1))
        for source, destination in ((values, self.weighted), (support, self.support)):
            destination[target] += np.add.reduceat(np.add.reduceat(source, xs, axis=1), ys, axis=0)
        self.processed += pixels.size
        self.x = x1
        if self.x == spec.level.width:
            self.x = 0
            self.y = y1

    def finish(self) -> IntegratedOverview:
        if not self.done:
            raise RuntimeError("Cannot publish an incomplete overview")
        spec = self.spec
        # Normalize by each output cell's actual geometric support. The resampler
        # clips the same edge support, so it is not counted a second time.
        wy = np.clip(spec.coverage[0] - np.arange(spec.shape[0]) * spec.factor, 0, spec.factor)
        wx = np.clip(spec.coverage[1] - np.arange(spec.shape[1]) * spec.factor, 0, spec.factor)
        for row, height in enumerate(wy):
            area = height * wx
            if spec.components == 3:
                area = area[:, None]
            for array in (self.weighted, self.support):
                np.divide(array[row], area, out=array[row], where=area > 0)
        self.weighted.flags.writeable = self.support.flags.writeable = False
        return IntegratedOverview(spec, self.weighted, self.support)


class OverviewCache:
    """Worker-owned LRU, sharing its byte cap with at most one unfinished build.

    No executor or implicit I/O: callers drive step() and may cancel between steps.
    The scan limit counts selected channels and RGB samples, not decoder tile bytes.
    """

    def __init__(self, *, cache_mib: float, max_source_pixels: int) -> None:
        if not isfinite(cache_mib) or cache_mib < 0:
            raise ValueError("overview_cache_mib must be finite and nonnegative")
        if (isinstance(max_source_pixels, bool) or not isinstance(max_source_pixels, int)
                or max_source_pixels < 0):
            raise ValueError("overview_max_source_pixels must be a nonnegative integer")
        self.budget = int(cache_mib * 1024**2)
        self.max_source_pixels = max_source_pixels
        self.entries: OrderedDict[tuple[int, OverviewSpec], IntegratedOverview] = OrderedDict()
        self.builder: OverviewBuilder | None = None
        self.failed: set[tuple[int, OverviewSpec]] = set()
        self.status = ""

    @property
    def nbytes(self) -> int:
        return (sum(item.spec.nbytes for item in self.entries.values())
                + (0 if self.builder is None else self.builder.spec.nbytes))

    def clear(self) -> None:
        self.entries.clear()
        self.failed.clear()
        self.cancel()

    def cancel(self) -> None:
        self.builder = None
        self.status = ""

    def spec(self, info: ViewInfo, channels: tuple[int, ...], max_read_pixels: int):
        components = 3 if info.is_rgb else 1
        level = max(info.levels, key=lambda item: item.scale_x * item.scale_y)
        total = level.height * level.width * components * len(channels)
        if not self.budget or not self.max_source_pixels or max_read_pixels < components:
            return None, ""
        if total <= max_read_pixels:
            return None, ""  # There is already an affordable native overview.
        if total > self.max_source_pixels:
            return None, "overview scan capped"
        target = min(MAX_OVERVIEW_PIXELS, self.budget // (16 * components * len(channels)))
        if target < 1:
            return None, "overview cache too small"
        factor = 2
        while ((level.height + factor - 1) // factor
               * ((level.width + factor - 1) // factor)) > target:
            factor *= 2
        shape = ((level.height + factor - 1) // factor,
                 (level.width + factor - 1) // factor)
        coverage = (min(float(level.height), info.shape[0] / level.scale_y),
                    min(float(level.width), info.shape[1] / level.scale_x))
        return OverviewSpec(level, factor, shape, coverage, components), ""

    def get(self, info: ViewInfo, channels: tuple[int, ...], max_read_pixels: int):
        spec, _ = self.spec(info, channels, max_read_pixels)
        if spec is None:
            return None
        keys = tuple((channel, spec) for channel in channels)
        if not all(key in self.entries for key in keys):
            return None
        for key in keys:
            self.entries.move_to_end(key)
        return tuple(self.entries[key] for key in keys)

    def step(self, info: ViewInfo, channels: tuple[int, ...], max_read_pixels: int,
             read_region) -> OverviewProgress:
        spec, reason = self.spec(info, channels, max_read_pixels)
        if spec is None:
            self.cancel()
            self.status = reason
            return OverviewProgress()
        wanted = {(channel, spec) for channel in channels}
        if self.builder is not None and (self.builder.channel, self.builder.spec) not in wanted:
            self.cancel()
        missing = [(channel, spec) for channel in channels if (channel, spec) not in self.entries]
        if not missing:
            self.status = "overview cached"
            return OverviewProgress()
        if any(key in self.failed for key in missing):
            self.cancel()
            self.status = "overview build failed; sampled fallback"
            return OverviewProgress()
        key = (self.builder.channel, spec) if self.builder is not None else missing[0]
        try:
            if self.builder is None:
                while self.entries and self.nbytes + spec.nbytes > self.budget:
                    victim = next(k for k in self.entries if k not in wanted)
                    self.entries.pop(victim)
                self.builder = OverviewBuilder(spec, key[0], max_read_pixels)
            self.builder.step(read_region)
            total = spec.level.height * spec.level.width * spec.components
            progress = self.builder.processed / total
            self.status = f"building overview ch {key[0]}: {progress:.0%}"
            published = self.builder.done
            if published:
                self.entries[key] = self.builder.finish()
                self.builder = None
                self.status = "overview cached" if len(missing) == 1 else "building overview"
            return OverviewProgress(pending=not published or len(missing) > 1, published=published)
        except Exception as error:
            self.failed.add(key)
            self.cancel()
            self.status = f"overview build failed: {type(error).__name__}: {error}"
            return OverviewProgress()

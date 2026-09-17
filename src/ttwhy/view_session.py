"""Public planning, sampling, and rendering stages for a borrowed open source."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite
from threading import RLock

import numpy as np
from numpy.typing import NDArray

from ._validation import positive_integer
from .compositing import CHANNEL_COLORS, ChannelStyle, compose_channels
from .intensity_window import IntensityWindow, estimate_window
from .ome_tiff_source import OmeTiffSource, PyramidLevel, ReadBudgetExceeded
from .resampling import resample_image
from .text_image import TextImage
from .text_renderer import TextRenderer
from .viewport import Viewport


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """A bounded source rectangle and its exact display sampling geometry."""

    viewport: Viewport
    level: PyramidLevel
    bounds: tuple[int, int, int, int]  # y0, y1, x0, x1 at selected level
    channels: tuple[int, ...]
    sample_shape: tuple[int, int]
    source_samples: int


@dataclass(frozen=True, slots=True)
class ViewSamples:
    """Independent, read-only preview intensities, in CYX or RGB YXS order.

    Values are area-averaged raw display samples, not quantitative whole-slide
    measurements. The mask is represented by NaN rather than an extra raster.
    """

    values: NDArray
    plan: ReadPlan
    is_rgb: bool

    def __post_init__(self) -> None:
        value = np.array(self.values, copy=True)
        rows, columns = self.plan.sample_shape
        expected = ((rows, columns, 3) if self.is_rgb else
                    (len(self.plan.channels), rows, columns))
        if value.shape != expected or value.dtype.kind != "f":
            raise ValueError(f"Preview values must be floating point with shape {expected}")
        value.flags.writeable = False
        object.__setattr__(self, "values", value)

    def __repr__(self) -> str:
        return (f"ViewSamples(shape={self.values.shape}, level={self.plan.level.index}, "
                f"channels={self.plan.channels}, rgb={self.is_rgb})")


class ViewSession:
    """Sample and render viewports while borrowing an open OmeTiffSource.

    A byte-bounded LRU retains resampled preview intensities. Thus contrast,
    gamma, density, and color changes can reuse pixels without TIFF I/O. It
    does not duplicate whole source tiles. Reader access is serialized.
    Closing the source ends the session's usable lifetime, even for cache hits.
    """

    def __init__(self, source: OmeTiffSource, *, cache_mib: float = 32.0) -> None:
        if not source.is_open:
            raise RuntimeError("ViewSession requires an open OmeTiffSource")
        if not isfinite(cache_mib) or cache_mib < 0:
            raise ValueError("cache_mib must be finite and nonnegative")
        self.source = source
        self._source_info = source.info
        self._budget = int(cache_mib * 1024 * 1024)
        self._cache: OrderedDict[ReadPlan, ViewSamples] = OrderedDict()
        self._cache_bytes = 0
        self._lock = RLock()

    def _require_source_open(self) -> None:
        if not self.source.is_open or self.source.info is not self._source_info:
            raise RuntimeError("The session's source lifetime has ended; create a new ViewSession")

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    def clear_cache(self) -> None:
        """Release preview rasters; source decoder caching remains source-owned."""
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0

    def plan(
        self,
        viewport: Viewport,
        *,
        rows: int,
        columns: int,
        channels: Sequence[int] = (0,),
    ) -> ReadPlan:
        """Choose a stored pyramid level and reject excessive reads before I/O.

        rows/columns describe the sample raster, so half-block renderers request
        twice the terminal row count. Level extent ratios are for visualization
        only and do not invent physical calibration for third-party pyramids.
        """
        self._require_source_open()
        rows, columns = positive_integer(rows, "rows"), positive_integer(columns, "columns")
        selected = self.source.select_channels(channels)
        candidates = [level for level in self.source.info.levels
                      if level.scale_x <= viewport.width / columns
                      and level.scale_y <= viewport.height / rows]
        level = max(candidates, key=lambda x: x.scale_x * x.scale_y,
                    default=self.source.info.levels[0])
        x0 = min(level.width, max(0, floor(viewport.x / level.scale_x)))
        y0 = min(level.height, max(0, floor(viewport.y / level.scale_y)))
        x1 = min(level.width, max(0, ceil((viewport.x + viewport.width) / level.scale_x)))
        y1 = min(level.height, max(0, ceil((viewport.y + viewport.height) / level.scale_y)))
        count = max(0, x1 - x0) * max(0, y1 - y0) * len(selected)
        count *= 3 if self.source.info.is_rgb else 1
        if count > self.source.max_read_pixels:
            raise ReadBudgetExceeded(
                f"Viewport needs {count:,} source samples at level {level.index}; "
                f"budget is {self.source.max_read_pixels:,}. Zoom in or supply a coarser pyramid."
            )
        return ReadPlan(viewport, level, (y0, y1, x0, x1), selected, (rows, columns), count)

    def fit(
        self,
        *,
        columns: int,
        rows: int,
        renderer: TextRenderer,
        channels: Sequence[int] = (0,),
    ) -> tuple[Viewport, bool]:
        """Return an image-fit viewport, or an explicitly flagged bounded center crop.

        A large file without a sufficiently coarse pyramid cannot have a cheap
        faithful overview. The boolean reports whether the read budget limited
        the fit. No conversion, sparse fake overview, or full-slide scan occurs.
        """
        viewport = Viewport.fit(self.source.info.shape, columns=columns, rows=rows,
                               cell_aspect=renderer.cell_aspect,
                               pixel_aspect=self.source.info.pixel_aspect)
        sample_rows, sample_columns = renderer.sample_shape(columns=columns, rows=rows)
        limited = False
        for _ in range(128):
            try:
                self.plan(viewport, rows=sample_rows, columns=sample_columns, channels=channels)
                return viewport, limited
            except ReadBudgetExceeded:
                viewport, limited = viewport.zoom(1.5), True
        raise ReadBudgetExceeded("Read budget is too small even for the minimum viewport")

    def sample(self, plan: ReadPlan) -> ViewSamples:
        """Execute a plan using regional Omeify reads and finite-value area averaging."""
        with self._lock:
            # Revalidate a public record; do not allow a constructed plan to
            # bypass budget, source, level, or channel validation.
            expected = self.plan(plan.viewport, rows=plan.sample_shape[0],
                                 columns=plan.sample_shape[1], channels=plan.channels)
            if plan != expected:
                raise ValueError("ReadPlan does not match this source and sampling geometry")
            if plan in self._cache:
                self._cache.move_to_end(plan)
                return self._cache[plan]
            y0, y1, x0, x1 = plan.bounds
            rows, columns = plan.sample_shape
            planes = []
            for channel in plan.channels:
                if x0 >= x1 or y0 >= y1:
                    shape = ((rows, columns, 3) if self.source.info.is_rgb else (rows, columns))
                    preview = np.full(shape, np.nan, dtype=np.float64)
                else:
                    pixels = self.source.read_region(y0, y1, x0, x1,
                                                     channel=channel, level=plan.level.index)
                    viewport, level = plan.viewport, plan.level
                    preview = resample_image(
                        pixels, rows=rows, columns=columns,
                        bounds=(viewport.x / level.scale_x - x0,
                                viewport.y / level.scale_y - y0,
                                viewport.width / level.scale_x,
                                viewport.height / level.scale_y),
                    )
                planes.append(preview)
            values = planes[0] if self.source.info.is_rgb else np.stack(planes)
            samples = ViewSamples(values, plan, self.source.info.is_rgb)
            size = samples.values.nbytes
            if size <= self._budget:
                while self._cache and self._cache_bytes + size > self._budget:
                    self._cache_bytes -= self._cache.popitem(last=False)[1].values.nbytes
                self._cache[plan] = samples
                self._cache_bytes += size
            return samples

    def prepare_styles(
        self,
        samples: ViewSamples,
        *,
        windows: Sequence[IntensityWindow | None] | None = None,
    ) -> tuple[ChannelStyle, ...]:
        """Resolve explicit windows or estimate initial display windows from a preview.

        windows follows the sampled channel order, not the full source channel
        order. Unsigned RGB defaults to its dtype range, preserving shared RGB
        balance. Other channels use finite preview percentiles. An entirely
        missing field receives a 0..1 display placeholder, not an inferred range.
        The caller may retain returned windows to keep contrast fixed while panning.
        """
        self._require_source_open()
        if windows is not None and len(windows) != len(samples.plan.channels):
            raise ValueError("One window is required for each sampled logical channel")
        styles = []
        for position, index in enumerate(samples.plan.channels):
            window = None if windows is None else windows[position]
            values = samples.values if samples.is_rgb else samples.values[position]
            if window is None:
                if samples.is_rgb and self.source.info.dtype.kind in "bu":
                    maximum = (1 if self.source.info.dtype.kind == "b" else
                               np.iinfo(self.source.info.dtype).max)
                    window = IntensityWindow(0, float(maximum))
                elif np.isfinite(values).any():
                    window = estimate_window(values)
                else:
                    window = IntensityWindow(0, 1)
            color = ((1.0, 1.0, 1.0) if samples.is_rgb or len(self.source.info.channels) == 1
                     else CHANNEL_COLORS[index % len(CHANNEL_COLORS)])
            styles.append(ChannelStyle(window, color))
        return tuple(styles)

    def render(
        self,
        samples: ViewSamples,
        *,
        styles: Sequence[ChannelStyle],
        renderer: TextRenderer,
        columns: int,
        rows: int,
    ) -> TextImage:
        """Compose independently windowed samples and render terminal cells.

        RGB uses one logical style and applies its window equally to all three
        RGB samples; its pseudocolor field is not used. Monochrome multi-marker
        output uses the peak composed signal rather than a color-weighted loss.
        """
        self._require_source_open()
        if len(styles) != len(samples.plan.channels):
            raise ValueError("One style is required for each sampled logical channel")
        if samples.is_rgb:
            if tuple(styles[0].color) != (1.0, 1.0, 1.0):
                raise ValueError("RGB samples require a white style; RGB is not pseudocolored")
            normalized = styles[0].window.apply(samples.values)
        else:
            normalized = compose_channels(samples.values, styles)
        if not renderer.color:
            normalized = np.max(normalized, axis=-1)
        return renderer.render(normalized, columns=columns, rows=rows)

    def __repr__(self) -> str:
        return (f"ViewSession(source={self.source!r}, "
                f"preview_cache_mib={self.cache_bytes / 1024**2:.2f})")

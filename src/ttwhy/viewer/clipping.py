"""Selection lifecycle and bounded, explicit copy/save work for the existing viewer."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..image_clip import clipboard_png, write_png_clip
from ..png_clip import render_clip_png
from ..pdf_text import extract_pdf_text
from ..text_clip import (
    CLIPBOARD_MODES, CLIP_SOURCES, ClipboardDelivery, clipboard_delivery,
    validate_clip_path, write_text_clip,
)
from .selection import RectangleSelection

if TYPE_CHECKING:
    from .viewer import Viewer


def file_stamp(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def clip_options(*, source_path=None, output_text_clip=None, output_ansi_clip=None,
                 output_png_clip=None, clipboard="auto", clip_source="auto",
                 clip_overwrite=True) -> None:
    """Preflight explicit destinations without creating or truncating any file."""
    if clipboard not in CLIPBOARD_MODES or clip_source not in CLIP_SOURCES:
        raise ValueError("Invalid clipboard or clip_source option")
    paths = [Path(p).expanduser() for p in (output_text_clip, output_ansi_clip,
                                             output_png_clip) if p is not None]
    source = None if source_path is None else Path(source_path)
    for path in paths:
        validate_clip_path(path, source=source, overwrite=clip_overwrite)
    for position, left in enumerate(paths):
        for right in paths[position + 1:]:
            if left.resolve() == right.resolve() or (
                left.exists() and right.exists() and left.samefile(right)
            ):
                raise ValueError("Text, ANSI and PNG clip destinations must be different files")


@dataclass(frozen=True)
class ClipResult:
    message: str
    delivery: ClipboardDelivery | None = None


class ClipController:
    """Freeze only the small text frame, not another PDF raster or pyramid.

    One copy/save can run at a time. Every operation captures its own immutable
    selection and page coordinates before leaving the UI thread. File errors are
    nonfatal and never silently fall back to exporting some other kind of text.
    """

    def __init__(self, viewer: Viewer, *, source_path=None, output_text_clip=None,
                 output_ansi_clip=None, output_png_clip=None, clipboard="auto",
                 clip_source="auto", clip_overwrite=True) -> None:
        if source_path is None and viewer.navigation is not None:
            source_path = viewer.navigation.document.path
        clip_options(source_path=source_path, output_text_clip=output_text_clip,
                     output_ansi_clip=output_ansi_clip, output_png_clip=output_png_clip,
                     clipboard=clipboard, clip_source=clip_source,
                     clip_overwrite=clip_overwrite)
        self.viewer = viewer
        self.source = None if source_path is None else Path(source_path).resolve()
        self.source_stamp = None if self.source is None else file_stamp(self.source)
        self.paths = {False: None if output_text_clip is None else Path(output_text_clip).expanduser(),
                      True: None if output_ansi_clip is None else Path(output_ansi_clip).expanduser(),
                      "png": None if output_png_clip is None else Path(output_png_clip).expanduser()}
        self.clipboard, self.clip_source, self.overwrite = clipboard, clip_source, clip_overwrite
        self.selection: RectangleSelection | None = None
        self.viewport = None
        self.raster_shape = None
        self.pdf_page: int | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._task: asyncio.Task | None = None
        self._closed = False
        self.pressed_button = None

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def source_label(self) -> str:
        nav = self.viewer.navigation
        return ("PDF text" if self.clip_source == "auto" and nav is not None
                and nav.document.kind == "pdf" else "screen text")

    def notify(self, message: str) -> None:
        self.viewer.message = message
        if self.viewer.application is not None:
            self.viewer.application.invalidate()

    def toggle(self) -> None:
        if self.selection is not None:
            self.end()
            return
        viewer, nav = self.viewer, self.viewer.navigation
        if (viewer.editing is not None or viewer.frame is None or viewer.busy
                or viewer.frame.shape != (viewer.canvas_size[1], viewer.canvas_size[0])
                or viewer.displayed_generation != viewer.generation
                or (nav is not None and (nav.pending is not None or nav.target != nav.active.index))):
            self.notify("Wait for the current frame before selecting.")
            return
        self.selection = RectangleSelection(viewer.frame, viewer.cell_lines)
        self.viewport, self.raster_shape = viewer.viewport, viewer.session.info.shape
        self.pdf_page = nav.active.index if self.source_label == "PDF text" else None
        viewer.canvas._drag = None
        viewer.application.layout.focus(viewer.canvas)
        self.notify("Select: drag a box, or arrows to position, Space to anchor, arrows to extend.")

    def end(self, *, message="Selection closed; navigation resumed.", resume=True) -> None:
        if self.selection is None:
            return
        self.selection = None
        self.viewport = self.raster_shape = self.pdf_page = None
        self.viewer.canvas._drag = None
        if self.viewer.editing in ("save-text", "save-ansi", "save-png"):
            self.viewer.cancel_edit()
        self.notify(message)
        if resume:
            self.viewer.request_frame()  # Also wakes a deferred PDF source handoff.

    def move(self, dx: int, dy: int, *, extend=False) -> None:
        if self.selection is not None:
            self.selection.move(dx, dy, extend=extend)
            self.viewer.application.invalidate()

    def mark(self) -> None:
        if self.selection is not None:
            self.selection.mark()
            self.notify("First corner set. Arrows resize; Tab swaps corners; y copies; w saves.")

    def swap_corner(self) -> None:
        if self.selection is not None:
            self.selection.swap_corner()
            self.viewer.application.invalidate()

    def edge(self, *, end=False, extend=False) -> None:
        if self.selection is not None:
            target = ((self.selection.frame.shape[1] - 1, self.selection.frame.shape[0] - 1)
                      if end else (0, 0))
            x, y = self.selection.cursor
            self.move(target[0] - x, target[1] - y, extend=extend)

    def save(self, *, ansi=False, image=False) -> None:
        if self.selection is None or self.selection.bounds is None:
            self.notify("Select a rectangle first.")
            return
        if self.busy:
            self.notify("A clip is already being prepared.")
            return
        key = "png" if image else ansi
        if self.paths[key] is not None:
            if image:
                self.submit_image(copy=False, path=self.paths[key])
            else:
                self.submit(ansi=ansi, copy=False, path=self.paths[key])
            return
        viewer = self.viewer
        viewer.editing = "save-png" if image else "save-ansi" if ansi else "save-text"
        stem = "selection" if self.source is None else self.source.stem
        suffix = ".clip.png" if image else ".clip.ansi" if ansi else ".clip.txt"
        viewer.entry.text = stem + suffix
        viewer.entry.buffer.cursor_position = len(viewer.entry.text)
        viewer.application.layout.focus(viewer.entry)
        viewer.application.invalidate()

    def accept_path(self, text: str, *, ansi=False, image=False) -> bool:
        key = "png" if image else ansi
        try:
            if not text.strip():
                raise ValueError("Enter a file path")
            path = Path(text).expanduser()
            options = {"output_text_clip": self.paths[False],
                       "output_ansi_clip": self.paths[True],
                       "output_png_clip": self.paths["png"]}
            options["output_png_clip" if image else
                    "output_ansi_clip" if ansi else "output_text_clip"] = path
            clip_options(source_path=self.source, clip_overwrite=self.overwrite, **options)
        except (ValueError, OSError) as error:
            self.notify(f"Cannot save: {error}")
            return False
        self.paths[key] = path
        self.viewer.cancel_edit()
        if image:
            self.submit_image(copy=False, path=path)
        else:
            self.submit(ansi=ansi, copy=False, path=path)
        return True

    def submit(self, *, ansi=False, copy=True, path=None) -> None:
        selection = self.selection
        if self._closed or self.busy:
            self.notify("A clip is already being prepared.")
            return
        if selection is None or selection.bounds is None:
            self.notify("Select a rectangle first.")
            return
        if path is None and copy:
            path = self.paths[ansi]
        if copy and self.clipboard == "off" and path is None:
            self.save(ansi=ansi)
            return
        # Copy from the frozen original, not the on-screen selection border.
        clip = selection.clip()
        page = None if ansi else self.pdf_page
        bounds = None
        if page is not None:
            y0, y1, x0, x1 = selection.bounds
            rows, columns = selection.frame.shape
            view = self.viewport
            bounds = (view.x + x0 / columns * view.width, view.y + y0 / rows * view.height,
                      view.x + x1 / columns * view.width, view.y + y1 / rows * view.height)
        shape = self.raster_shape
        # Navigable PDFs supply their document path even for Viewer(nav.session).
        source = self.source
        if page is not None:
            source = self.viewer.navigation.document.path
        stamp = self.source_stamp if source == self.source else None
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ttwhy-clip")

        def prepare() -> ClipResult:
            try:
                if page is not None:
                    if stamp is not None and file_stamp(source) != stamp:
                        raise ValueError("Input changed on disk; reopen it before copying PDF text")
                    text = extract_pdf_text(source, page, bounds=bounds, raster_shape=shape)
                    if not text.strip():
                        raise ValueError("No embedded PDF text in this box. No OCR was run; "
                                         "Y/W exports terminal artwork instead.")
                    label = "PDF text"
                else:
                    text, label = clip.to_text(ansi=ansi), "ANSI artwork" if ansi else "screen text"
                messages = []
                if path is not None:
                    write_text_clip(path, text, source=source, overwrite=self.overwrite)
                    messages.append(f"Saved {label}: {path}")
                delivery = None
                if copy and self.clipboard != "off":
                    try:
                        delivery = clipboard_delivery(text, self.clipboard)
                        messages.append(f"{delivery.message} ({label})")
                    except (ValueError, OSError, RuntimeError) as error:
                        messages.append(f"Clipboard unavailable: {error}")
                return ClipResult(" | ".join(messages), delivery)
            except Exception as error:
                return ClipResult(f"Clip failed: {error}")

        # Submit accepted file/local-copy work now, even when the next key is
        # Quit. Shutdown joins this worker; terminal OSC output is still forbidden
        # after the application stops. At most one clip operation is outstanding.
        future = asyncio.get_running_loop().run_in_executor(self._executor, prepare)

        async def deliver() -> None:
            result = await asyncio.shield(future)
            if self._closed or not self.viewer.application.is_running:
                return
            if result.delivery is not None and result.delivery.sequence:
                try:
                    # Only the UI thread writes terminal bytes; no repaint can
                    # split this sequence. Never send it to a redirected stdout.
                    self.viewer.application.output.write_raw(result.delivery.sequence)
                    self.viewer.application.output.flush()
                except (OSError, NotImplementedError) as error:
                    self.notify(f"{result.message} | Terminal write failed: {error}")
                    return
            self.notify(result.message)

        self._task = self.viewer.application.create_background_task(deliver())
        self.notify("Preparing selected text..." if page is not None else "Preparing selected artwork...")

    def submit_image(self, *, copy=True, path=None) -> None:
        """Export a real PNG, not the ANSI stream that chat applications cannot paint."""
        selection = self.selection
        if self._closed or self.busy:
            self.notify("A clip is already being prepared.")
            return
        if selection is None or selection.bounds is None:
            self.notify("Select a rectangle first.")
            return
        if path is None and copy:
            path = self.paths["png"]
        if copy and self.clipboard == "off" and path is None:
            self.save(image=True)
            return
        clip = selection.clip()
        source, aspect = self.source, self.viewer.renderer.cell_aspect
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ttwhy-clip")

        def prepare() -> ClipResult:
            try:
                # The screenshot is made from frozen terminal cells, including
                # exact RGB halves in blocks mode. No PDF raster is decoded.
                png = render_clip_png(clip, cell_height=round(12 * aspect))
                messages = []
                if path is not None:
                    write_png_clip(path, png, source=source, overwrite=self.overwrite)
                    messages.append(f"Saved PNG: {path}")
                if copy and self.clipboard != "off":
                    try:
                        delivery = clipboard_png(png, self.clipboard)
                        messages.append(delivery.message + " (paste into Discord)")
                    except (ValueError, OSError, RuntimeError) as error:
                        messages.append(f"Image clipboard unavailable: {error}")
                return ClipResult(" | ".join(messages))
            except Exception as error:
                return ClipResult(f"PNG clip failed: {error}")

        future = asyncio.get_running_loop().run_in_executor(self._executor, prepare)

        async def deliver() -> None:
            result = await asyncio.shield(future)
            if not self._closed and self.viewer.application.is_running:
                self.notify(result.message)

        self._task = self.viewer.application.create_background_task(deliver())
        self.notify("Preparing PNG artwork for sharing...")

    def status(self) -> str:
        selection = self.selection
        if selection is None:
            return ""
        position = f"col {selection.cursor[0]+1}, row {selection.cursor[1]+1}"
        if selection.bounds is None:
            return f"SELECT | {self.source_label} | {position} | Space: first corner"
        y0, y1, x0, x1 = selection.bounds
        return (f"SELECT | {self.source_label} | {x1-x0} cols x {y1-y0} rows "
                f"| {position} | frame frozen")

    def close(self) -> None:
        self._closed = True
        self.selection = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

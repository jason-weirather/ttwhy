"""Isolated PDF refinement jobs, with private pipes and no disk intermediates.

PDFium is not thread-safe, including across different documents. Each spawned
worker owns its document, bitmap and pyramid. Only owned NumPy arrays cross the
private pipe. Cancelling one job discards that pipe and process, never a shared
pool queue or a live image being sampled by the viewer.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from math import isfinite
from pathlib import Path

import numpy as np

from .image_input import DEFAULT_PDF_DPI, _pdf_pixels
from .image_pyramid import image_pyramid

DEFAULT_PDF_MAX_DPI = 600.0
DEFAULT_PDF_CACHE_PAGES = 3


def pdf_dpi_steps(max_dpi: float = DEFAULT_PDF_MAX_DPI) -> tuple[float, ...]:
    """Quick preview, useful intermediate resolutions, then the exact ceiling."""
    if isinstance(max_dpi, (bool, np.bool_)) or not isfinite(max_dpi) or max_dpi <= 0:
        raise ValueError("pdf_max_dpi must be positive and finite")
    return tuple(dpi for dpi in (DEFAULT_PDF_DPI, 300.0, 600.0) if dpi < max_dpi) + (
        float(max_dpi),
    )


def render_pdf_pyramid(path: str, page: int, dpi: float,
                       background: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    """Render genuine PDF content at this DPI, not an enlarged preview raster."""
    pixels = _pdf_pixels(Path(path), page, dpi, background)
    return image_pyramid(pixels)


def pdf_process(send, path: str, page: int, dpi: float,
                background: tuple[int, int, int]) -> None:
    """Spawn target: one result, no terminal input, no shared native handles."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            arrays = render_pdf_pyramid(path, page, dpi, background)
        except Exception as error:
            # Do not pickle native exception objects or retain their tracebacks.
            send.send((False, f"{type(error).__name__}: {error}"))
        else:
            send.send((True, arrays))
    except (BrokenPipeError, EOFError, OSError):
        pass  # The parent no longer wants this page/resolution.
    finally:
        send.close()


class PdfWorkers:
    """At most two live render processes; pipe reception never blocks the UI.

    The event loop owns this object. All run() tasks must finish or be cancelled
    and awaited before close(). A separate thread reads each private pipe so
    large raster transfers/unpickling cannot stall terminal event handling.
    """

    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._slots = asyncio.Semaphore(2)
        self._receivers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="twhy-pdf-ipc")
        self._processes: set = set()
        self._closed = False

    @property
    def active_count(self) -> int:
        return len(self._processes)

    async def render(self, path: Path, page: int, dpi: float,
                     background: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
        async with self._slots:
            if self._closed:
                raise RuntimeError("PDF workers are closed")
            receive, send = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=pdf_process, args=(send, str(path), page, dpi, background),
                name=f"twhy-pdf-{page}-{dpi:g}", daemon=True,
            )
            future = None
            started = False
            try:
                process.start()
                started = True
                self._processes.add(process)
                send.close()  # EOF is observable when the only sender exits.
                future = asyncio.get_running_loop().run_in_executor(self._receivers, receive.recv)
                try:
                    ok, result = await asyncio.shield(future)
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        f"PDF page {page} at {dpi:g} DPI: worker exited without a complete result "
                        "(possibly an allocation failure); lower --pdf-max-dpi"
                    ) from error
                if not ok:
                    raise RuntimeError(f"PDF page {page} at {dpi:g} DPI: {result}")
                for array in result:
                    array.flags.writeable = False
                return result
            finally:
                # Cancellation can arrive during a normal successful job's
                # finalizer, not just while waiting for its result. Shield the
                # ENTIRE process/pipe cleanup and await it even after repeated
                # cancellation; otherwise TaskGroup can finish with a live child.
                cleanup = asyncio.create_task(self._finish_job(
                    process, started, send, receive, future,
                ))
                interrupted = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        interrupted = True
                cleanup.result()
                if interrupted:
                    raise asyncio.CancelledError

    async def _finish_job(self, process, started, send, receive, future) -> None:
        """Private finalizer; its owner must shield it and wait until completion."""
        try:
            if started:
                if process.is_alive():
                    process.terminate()
                await asyncio.to_thread(process.join, 1.0)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join)
                self._processes.discard(process)
            send.close()
            if future is not None:
                with suppress(Exception):
                    await asyncio.shield(future)
        finally:
            receive.close()
            process.close()

    def close(self) -> None:
        if self._processes:
            raise RuntimeError("Await PDF jobs before closing their receivers")
        self._closed = True
        self._receivers.shutdown(wait=True, cancel_futures=True)

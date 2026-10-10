"""File-backed frontend entry point; borrowed-image Viewer use remains unchanged."""

from __future__ import annotations

from ..image_document import ImageDocument
from ..image_input import open_image
from ..pdf_workers import DEFAULT_PDF_CACHE_PAGES
from ..text_renderer import TextRenderer
from ..view_session import ViewSession
from .navigation import DocumentNavigation
from .clipping import clip_options
from .viewer import Viewer


async def view_file(path, *, series=0, page=None, frame=None, pdf_max_dpi=None,
                    pdf_cache_pages=None, background=None, channels=(0,), window=None,
                    mode="blocks", color=True, invert=None, cell_aspect=2.0,
                    viewport=None, cache_mib=32.0, max_read_pixels=4_194_304,
                    input=None, output=None, output_text_clip=None, output_ansi_clip=None,
                    output_png_clip=None, output_geojson_clip=None, clipboard="auto",
                    clip_source="auto", clip_overwrite=True) -> None:
    """Navigate a document with its normal terminal controls and renderer.

    mode must be a resolved TextRenderer mode (the CLI resolves 'auto'). Embedded
    Python callers using PDF processes must use the normal __main__ guard in
    executable scripts. PDFium's initial quick-preview work is serialized in the
    parent; all subsequent high-detail/prefetch native work is process-isolated.
    """
    clipping = dict(source_path=path, output_text_clip=output_text_clip,
                    output_ansi_clip=output_ansi_clip, output_png_clip=output_png_clip,
                    output_geojson_clip=output_geojson_clip, clipboard=clipboard,
                    clip_source=clip_source, clip_overwrite=clip_overwrite)
    clip_options(**clipping)
    document = ImageDocument(path, series=series, page=page, frame=frame,
                             pdf_max_dpi=pdf_max_dpi, background=background)
    if document.kind != "pdf" and pdf_cache_pages is not None:
        raise ValueError("--pdf-cache-pages applies only to PDF")
    if document.kind == "raster":
        name = document.path.name
        if frame is not None:
            name += f" | frame {frame} (0-based)"
        with (
            open_image(path, series=series, page=page, frame=frame, background=background) as image,
            ViewSession(image, name=name, cache_mib=cache_mib,
                        max_read_pixels=max_read_pixels) as session,
        ):
            renderer = TextRenderer(mode=mode, color=color,
                                    invert=session.info.is_rgb if invert is None else invert,
                                    cell_aspect=cell_aspect)
            await Viewer(session, channels=channels, window=window, renderer=renderer,
                         viewport=viewport, **clipping).run(input=input, output=output)
        return
    initial = document.initialize()
    try:
        navigation = DocumentNavigation(
            document, initial,
            cache_pages=DEFAULT_PDF_CACHE_PAGES if pdf_cache_pages is None else pdf_cache_pages,
            cache_mib=cache_mib, max_read_pixels=max_read_pixels, window=window,
            auto_invert=invert is None,
        )
    except BaseException:
        initial.release()
        raise
    try:
        renderer = TextRenderer(mode=mode, color=color,
                                invert=navigation.session.info.is_rgb if invert is None else invert,
                                cell_aspect=cell_aspect)
        viewer = Viewer(navigation.session, navigation=navigation, channels=channels,
                        window=window, renderer=renderer, viewport=viewport, **clipping)
        await viewer.run(input=input, output=output)
    finally:
        # Idempotent stop/close also covers invalid initial channel selection or
        # an exception before the terminal application has been constructed.
        await navigation.stop()
        navigation.close()
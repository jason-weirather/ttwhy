"""Terminal controls: cell presentation, selection tools, and mouse navigation."""

from typing import TYPE_CHECKING

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

if TYPE_CHECKING:
    from .viewer import Viewer


def safe_label(text: str) -> str:
    """Keep metadata literal and single-line, excluding terminal control characters."""
    return "".join(character if character.isprintable() else "?" for character in text)


class CanvasControl(UIControl):
    def __init__(self, viewer: "Viewer") -> None:
        self.viewer = viewer
        self._drag: Point | None = None

    def is_focusable(self) -> bool:
        return True

    def create_content(self, width: int, height: int) -> UIContent:
        self.viewer.resize_canvas(width, height)
        selection = self.viewer.clipping.selection
        if selection is not None:
            return UIContent(get_line=selection.row, line_count=max(1, height), show_cursor=False)
        lines = self.viewer.cell_lines
        return UIContent(
            get_line=lambda row: lines[row] if row < len(lines) else [],
            line_count=max(1, height), show_cursor=False,
        )

    def mouse_handler(self, event: MouseEvent) -> None:
        viewer = self.viewer
        if viewer.editing is not None:
            return None
        viewer.application.layout.focus(self)
        viewer.clipping.pressed_button = None
        selection = viewer.clipping.selection
        if selection is not None:
            self._drag = None
            x, y = event.position.x, event.position.y
            if event.event_type == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
                selection.start_drag(x, y)
            elif event.event_type == MouseEventType.MOUSE_MOVE and event.button == MouseButton.LEFT:
                selection.drag(x, y)
            elif event.event_type == MouseEventType.MOUSE_MOVE:
                selection.drag(x, y)
            elif event.event_type == MouseEventType.MOUSE_UP:
                selection.drag(x, y, finish=True)
            viewer.application.invalidate()
            return None
        columns, rows = viewer.canvas_size
        anchor = (min(1.0, max(0.0, (event.position.x + 0.5) / max(1, columns))),
                  min(1.0, max(0.0, (event.position.y + 0.5) / max(1, rows))))
        if event.event_type == MouseEventType.SCROLL_UP:
            viewer.zoom(1.25, anchor=anchor)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            viewer.zoom(1 / 1.25, anchor=anchor)
        elif event.event_type == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
            self._drag = event.position
        elif event.event_type == MouseEventType.MOUSE_UP:
            self._drag = None
        elif (event.event_type == MouseEventType.MOUSE_MOVE
              and event.button == MouseButton.LEFT and self._drag is not None):
            viewer.pan_cells(self._drag.x - event.position.x, self._drag.y - event.position.y)
            self._drag = event.position
        return None


class ChannelControl(UIControl):
    def __init__(self, viewer: "Viewer") -> None:
        self.viewer = viewer
        self.first_channel = 0
        self.visible_channels = 1

    def is_focusable(self) -> bool:
        return True

    def create_content(self, width: int, height: int) -> UIContent:
        viewer, info = self.viewer, self.viewer.session.info
        if viewer.selecting:
            selection = viewer.clipping.selection
            if selection is not None and selection.kind == "polygon":
                lines = [[("class:panel.title", " POLYGON / GEOJSON")]]
                lines += [[("class:panel", line)] for line in (
                    " Click: add polygon point",
                    " Arrows: move cursor",
                    " Space: add point",
                    " Shift+arrows: move and add points",
                    " Tab: remove last point",
                    " Home/End: canvas corners",
                    " n: name feature",
                    " j / Enter: copy GeoJSON",
                    " J: save GeoJSON",
                    " Esc / v: resume viewing",
                    "",
                    " GeoJSON is in image pixel coordinates.",
                    " Works for raster, OME-TIFF, and PDF page rasters.",
                    " Clipboard copies JSON text; save for files.",
                )]
            else:
                lines = [[("class:panel.title", " SELECT / COPY")]]
                lines += [[("class:panel", line)] for line in (
                    " Drag: rectangle",
                    " Arrows: move cursor/corner",
                    " Space: set first corner",
                    " Shift+arrows: extend",
                    " Tab: swap corners",
                    " Home/End: canvas corners",
                    " y / Enter: copy text",
                    " w: save text",
                    " i: copy PNG image",
                    " P: save PNG image",
                    " Y: copy ANSI artwork",
                    " W: save ANSI artwork",
                    " n: name GeoJSON",
                    " j: copy GeoJSON",
                    " J: save GeoJSON",
                    " Esc / s: resume viewing",
                    "",
                    " Text: " + viewer.clipping.source_label,
                    " PDF text requires text layer.",
                    " PNG/JPEG copy screen glyphs.",
                    " No OCR or invented text.",
                    " PNG is for Discord/share.",
                    " ANSI is for terminals only.",
                )]
            return UIContent(get_line=lambda row: lines[row] if row < len(lines) else [],
                             line_count=len(lines), show_cursor=False)
        self.visible_channels = max(1, height - 12)
        self.first_channel = min(self.first_channel, viewer.focused_channel)
        self.first_channel = max(self.first_channel,
                                 viewer.focused_channel - self.visible_channels + 1)
        lines = [[("class:panel.title", " CHANNELS  (0-based)")]]
        for index in range(self.first_channel,
                           min(len(info.channels), self.first_channel + self.visible_channels)):
            focus = ">" if index == viewer.focused_channel else " "
            selected = "x" if index in viewer.selected_channels else " "
            style = "class:panel.selected" if index == viewer.focused_channel else "class:panel"
            name = safe_label(info.channels[index])
            lines.append([(style, f"{focus}[{selected}] {index} {name}")])
        window = viewer.windows[viewer.focused_channel]
        limits = ["", " FOCUSED CHANNEL",
                  f" Lower: {window.lower:g}" if window else " Lower: auto pending",
                  f" Upper: {window.upper:g}" if window else " Upper: auto pending",
                  f" Gamma: {window.gamma:g}" if window else " Gamma: 1",
                  " l / u / g : edit limits", " a : auto visible field",
                  " space : toggle channel", " , / . : focus channel", " Tab : canvas / channels"]
        lines.extend([[("class:panel", line)] for line in limits])
        return UIContent(get_line=lambda row: lines[row] if row < len(lines) else [],
                         line_count=len(lines), show_cursor=False)

    def mouse_handler(self, event: MouseEvent) -> None:
        viewer = self.viewer
        if viewer.editing is not None or viewer.selecting:
            return None
        viewer.application.layout.focus(self)
        if event.event_type == MouseEventType.SCROLL_UP:
            viewer.focus_channel(-1)
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            viewer.focus_channel(1)
        elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
            row = event.position.y - 1
            index = self.first_channel + row
            if 0 <= row < self.visible_channels and index < len(viewer.windows):
                viewer.focused_channel = index
                viewer.toggle_channel()
        return None


def navigation_fragments(viewer: "Viewer") -> list:
    """Clickable, single-line page/series controls; image wheel remains zoom."""
    nav = viewer.navigation
    if nav is None:
        return []

    def previous(event):
        if (viewer.editing is None and not viewer.selecting
                and event.button == MouseButton.LEFT and event.event_type == MouseEventType.MOUSE_UP):
            viewer.step_destination(-1)

    def following(event):
        if (viewer.editing is None and not viewer.selecting
                and event.button == MouseButton.LEFT and event.event_type == MouseEventType.MOUSE_UP):
            viewer.step_destination(1)

    def jump(event):
        if (viewer.editing is None and not viewer.selecting
                and event.button == MouseButton.LEFT and event.event_type == MouseEventType.MOUSE_UP):
            viewer.begin_destination()

    return [("class:status", " [<] ", previous),
            ("class:status", "[>] ", following),
            ("class:status", "[p: go] ", jump),
            ("class:status", safe_label(nav.status()))]


def clip_fragments(viewer: "Viewer") -> list:
    """Clickable selection controls live in the existing footer, costing no canvas row."""
    def button(label, action):
        def clicked(event):
            if viewer.editing is not None or event.button != MouseButton.LEFT:
                return
            if event.event_type == MouseEventType.MOUSE_DOWN:
                viewer.clipping.pressed_button = label
            elif event.event_type == MouseEventType.MOUSE_UP:
                pressed, viewer.clipping.pressed_button = viewer.clipping.pressed_button, None
                # Ending a canvas drag over Copy is not a button click.
                if viewer.clipping.selection is not None:
                    viewer.clipping.selection.dragging = False
                if pressed == label:
                    action()
        return ("class:clip.button", label, clicked)

    clip = viewer.clipping
    if not viewer.selecting:
        return [button(" [s: Select] ", clip.toggle),
                button("[v: Polygon] ", clip.start_polygon)]
    if clip.selection is not None and clip.selection.kind == "polygon":
        return [button(" [j: GeoJSON] ", clip.submit_geojson),
                button("[J: Save GeoJSON] ", lambda: clip.save(geojson=True)),
                button("[n: Name] ", clip.begin_name),
                button("[v: Done] ", clip.end)]
    return [button(" [y: Text] ", clip.submit),
            button("[i: Copy PNG] ", clip.submit_image),
            button("[P: Save PNG] ", lambda: clip.save(image=True)),
            button("[w: Save] ", clip.save),
            button("[Y: ANSI] ", lambda: clip.submit(ansi=True)),
            button("[W: Save ANSI] ", lambda: clip.save(ansi=True)),
            button("[j: GeoJSON] ", clip.submit_geojson),
            button("[J: Save GeoJSON] ", lambda: clip.save(geojson=True)),
            button("[n: Name] ", clip.begin_name),
            button("[s: Done] ", clip.end)]

"""Terminal controls: cell presentation and mouse-to-camera translation only."""

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
        if viewer.editing is not None:
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

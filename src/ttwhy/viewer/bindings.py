"""Bindings are disabled during numeric entry, except cancel and emergency quit."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

if TYPE_CHECKING:
    from .viewer import Viewer


def create_bindings(viewer: "Viewer") -> KeyBindings:
    bindings = KeyBindings()
    browsing = Condition(lambda: viewer.editing is None)

    @bindings.add("c-q", eager=True)
    def quit_always(event: KeyPressEvent) -> None:
        event.app.exit()

    @bindings.add("c-c", eager=True)
    def interrupt(event: KeyPressEvent) -> None:
        if viewer.editing is not None:
            viewer.cancel_edit()
        else:
            event.app.exit()

    @bindings.add("q", filter=browsing)
    def quit_viewer(event: KeyPressEvent) -> None:
        event.app.exit()

    @bindings.add("escape", eager=True)
    def cancel(event: KeyPressEvent) -> None:
        viewer.cancel_edit()

    @bindings.add("c", filter=browsing)
    def panel(event: KeyPressEvent) -> None:
        viewer.panel_requested = not viewer.panel_requested
        event.app.layout.focus(viewer.canvas)
        event.app.invalidate()

    @bindings.add("tab", filter=browsing)
    def focus(event: KeyPressEvent) -> None:
        if viewer.panel_visible:
            target = (viewer.canvas if event.app.layout.has_focus(viewer.channel_control)
                      else viewer.channel_control)
            event.app.layout.focus(target)

    @bindings.add("+", filter=browsing)
    @bindings.add("=", filter=browsing)
    def zoom_in(event: KeyPressEvent) -> None:
        viewer.zoom(1.25)

    @bindings.add("-", filter=browsing)
    def zoom_out(event: KeyPressEvent) -> None:
        viewer.zoom(1 / 1.25)

    @bindings.add("home", filter=browsing)
    @bindings.add("r", filter=browsing)
    def fit(event: KeyPressEvent) -> None:
        viewer.fit()

    def arrow(dx: int, dy: int) -> Callable[[KeyPressEvent], None]:
        def move(event: KeyPressEvent) -> None:
            if event.app.layout.has_focus(viewer.channel_control) and dy:
                viewer.focus_channel(dy)
            else:
                viewer.pan_cells(dx * max(1, viewer.canvas_size[0] // 8),
                                 dy * max(1, viewer.canvas_size[1] // 8))
        return move

    for key, delta in (("left", (-1, 0)), ("right", (1, 0)),
                       ("up", (0, -1)), ("down", (0, 1))):
        bindings.add(key, filter=browsing)(arrow(*delta))

    @bindings.add(",", filter=browsing)
    def previous_channel(event: KeyPressEvent) -> None:
        viewer.focus_channel(-1)

    @bindings.add(".", filter=browsing)
    def next_channel(event: KeyPressEvent) -> None:
        viewer.focus_channel(1)

    @bindings.add(" ", filter=browsing)
    def toggle_channel(event: KeyPressEvent) -> None:
        viewer.toggle_channel()

    def select(index: int) -> Callable[[KeyPressEvent], None]:
        def select_one(event: KeyPressEvent) -> None:
            viewer.select_one(index)
        return select_one

    for number in range(1, 10):
        bindings.add(str(number), filter=browsing)(select(number - 1))

    def edit(field: str) -> Callable[[KeyPressEvent], None]:
        def begin_edit(event: KeyPressEvent) -> None:
            viewer.begin_edit(field)
        return begin_edit

    for key, field in (("l", "lower"), ("u", "upper"), ("g", "gamma")):
        bindings.add(key, filter=browsing)(edit(field))

    @bindings.add("a", filter=browsing)
    def auto(event: KeyPressEvent) -> None:
        viewer.auto_window()

    @bindings.add("i", filter=browsing)
    def invert(event: KeyPressEvent) -> None:
        viewer.change_renderer(invert=not viewer.renderer.invert)

    @bindings.add("t", filter=browsing)
    def color(event: KeyPressEvent) -> None:
        color = not viewer.renderer.color
        mode = (viewer.text_mode if not color and viewer.renderer.mode == "blocks"
                else viewer.renderer.mode)
        viewer.change_renderer(color=color, mode=mode)

    @bindings.add("m", filter=browsing)
    def mode(event: KeyPressEvent) -> None:
        viewer.cycle_mode()

    @bindings.add("b", filter=browsing)
    def blocks(event: KeyPressEvent) -> None:
        viewer.toggle_blocks()

    @bindings.add("?", filter=browsing)
    def help_message(event: KeyPressEvent) -> None:
        viewer.show_help = not viewer.show_help
        event.app.invalidate()

    return bindings

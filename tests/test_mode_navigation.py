"""Color/strict-ASCII mode families and symmetric selection keybindings."""

from types import SimpleNamespace

import numpy as np
import pytest

from ttwhy import TextRenderer
from ttwhy.viewer.bindings import create_bindings
from ttwhy.viewer.mode_navigation import ModeNavigation
from ttwhy.viewer.rendering import terminal_lines
from ttwhy.viewer.viewer import Viewer


def test_mode_navigation_roundtrip_and_strict_ascii():
    initial = TextRenderer(mode="blocks", color=True)
    state = ModeNavigation.from_renderer(initial)
    assert state.label == "blocks"
    assert state.family == "color"
    for label in ("ASCII classic", "ASCII fine", "ASCII bold", "ASCII classic"):
        state.advance_ascii()
        renderer = state.make_renderer(initial)
        assert state.label == label
        assert renderer.mode == "simple" and not renderer.color
        data = np.linspace(0, 1, 77).reshape(7, 11)
        frame = renderer.render(data, rows=7, columns=11)
        assert frame.to_text(ansi=True).isascii()
        assert "\x1b" not in frame.to_text(ansi=True)
    state.advance_color()
    assert state.label == "blocks"
    state.advance_color()
    assert state.label == "detail"
    state.advance_ascii()
    assert state.label == "ASCII classic"
    state.advance_ascii()
    assert state.label == "ASCII fine"
    state.advance_color()
    assert state.label == "detail"
    state.advance_color()
    assert state.label == "shape"


def test_custom_legacy_ramp_and_renderer_flags_are_preserved():
    initial = TextRenderer(mode="simple", color=False, invert=True, ramp=" .#",
                           cell_aspect=1.5)
    state = ModeNavigation.from_renderer(initial)
    # Even monochrome simple is initially an m style, not automatically M.
    assert state.label == "simple" and state.family == "color" and not state.color_enabled
    state.advance_ascii()
    assert state.label == "ASCII custom"
    state.advance_ascii()
    assert state.label == "ASCII fine"
    state.advance_color()
    restored = state.make_renderer(initial)
    assert (restored.mode, restored.color, restored.invert, restored.ramp,
            restored.cell_aspect) == ("simple", False, True, " .#", 1.5)
    state.advance_ascii()
    state.advance_ascii()
    state.advance_ascii()
    assert state.label == "ASCII custom"
    assert state.make_renderer(initial).ramp == " .#"


def test_viewer_methods_respect_independent_families_without_source_reads():
    class Session:
        info = SimpleNamespace(channels=("A",), is_rgb=False)

        def _select_channels(self, channels):
            return tuple(channels)

    viewer = Viewer(Session(), renderer=TextRenderer(mode="blocks"))
    viewer.application = SimpleNamespace(invalidate=lambda: None)
    changes = []
    viewer._set_viewport = lambda *args, **kwargs: changes.append(viewer.renderer.mode)
    viewer.cycle_ascii()
    assert viewer.mode_navigation.label == "ASCII classic"
    viewer.cycle_ascii()
    assert viewer.mode_navigation.label == "ASCII fine"
    viewer.cycle_mode()
    assert viewer.renderer.mode == "blocks" and viewer.renderer.color
    viewer.cycle_mode()
    assert viewer.renderer.mode == "detail" and viewer.renderer.color
    viewer.cycle_ascii()
    assert viewer.mode_navigation.label == "ASCII fine"
    viewer.cycle_mode()
    assert viewer.renderer.mode == "detail" and viewer.renderer.color
    assert len(changes) == 6




@pytest.mark.parametrize("initial_mode", ("detail", "shape", "simple"))
def test_monochrome_artwork_cycles_without_blocks_and_remembers_ascii(initial_mode):
    initial = TextRenderer(mode=initial_mode, color=False)
    state = ModeNavigation.from_renderer(initial)
    assert state.family == "color" and state.color_mode == initial_mode
    assert not state.color_enabled
    observed = []
    for _ in range(6):
        state.advance_color()
        renderer = state.make_renderer(initial)
        observed.append(renderer.mode)
        assert not renderer.color
        assert renderer.mode != "blocks"
        assert renderer.render(np.linspace(0, 1, 100).reshape(10, 10),
                               columns=10, rows=10).foreground is None
    expected = ("detail", "shape", "simple")
    assert tuple(observed[:3]) == tuple(expected[(expected.index(initial_mode) + i + 1) % 3]
                                       for i in range(3))
    state.advance_ascii()
    assert state.label == "ASCII classic" and not state.make_renderer(initial).color
    state.advance_ascii()
    assert state.label == "ASCII fine"
    state.advance_color()
    assert state.color_mode == initial_mode and not state.color_enabled
    assert state.make_renderer(initial).mode == initial_mode
    state.advance_ascii()
    assert state.label == "ASCII fine"


def test_color_preference_is_separate_from_ascii_and_blocks_is_color_only():
    initial = TextRenderer(mode="blocks", color=True)
    state = ModeNavigation.from_renderer(initial)
    state.advance_ascii()
    assert state.label == "ASCII classic"
    state.set_color(False, fallback_mode="shape")
    assert state.make_renderer(initial).mode == "simple"
    assert not state.make_renderer(initial).color
    assert state.color_mode == "shape"  # Monochrome fallback instead of blocks.
    state.advance_color()
    assert (state.make_renderer(initial).mode, state.make_renderer(initial).color) == (
        "shape", False)
    state.advance_color()
    assert state.make_renderer(initial).mode == "simple"
    state.set_color(True)
    assert state.make_renderer(initial).color
    assert state.make_renderer(initial).mode == "simple"
    state.advance_color()
    assert state.make_renderer(initial).mode == "blocks"


def _dummy_navigation_viewer(initial):
    class Session:
        info = SimpleNamespace(channels=("A",), is_rgb=False)

        def _select_channels(self, channels):
            return tuple(channels)

    viewer = Viewer(Session(), renderer=initial)
    viewer.application = SimpleNamespace(invalidate=lambda: None)
    changes = []
    viewer._set_viewport = lambda *args, **kwargs: changes.append(
        (viewer.renderer.mode, viewer.renderer.color, viewer.renderer.ramp))
    return viewer, changes


def test_viewer_t_and_m_M_preserve_monochrome_and_remember_color_preference():
    viewer, changes = _dummy_navigation_viewer(TextRenderer(mode="blocks"))
    viewer.toggle_color()
    assert (viewer.renderer.mode, viewer.renderer.color) == ("detail", False)
    viewer.cycle_mode()
    assert (viewer.renderer.mode, viewer.renderer.color) == ("shape", False)
    viewer.cycle_mode()
    assert (viewer.renderer.mode, viewer.renderer.color) == ("simple", False)
    viewer.cycle_mode()
    assert (viewer.renderer.mode, viewer.renderer.color) == ("detail", False)
    viewer.cycle_ascii()
    viewer.cycle_ascii()
    assert viewer.mode_navigation.label == "ASCII fine"
    assert viewer.renderer.ramp == viewer.mode_navigation.ascii_variants[1][1]
    count = len(changes)
    viewer.toggle_color()  # Change the remembered m preference, not strict ASCII.
    assert len(changes) == count
    assert viewer.mode_navigation.color_enabled
    assert viewer.renderer.mode == "simple" and not viewer.renderer.color
    assert viewer.renderer.ramp == viewer.mode_navigation.ascii_variants[1][1]
    viewer.cycle_mode()
    assert (viewer.renderer.mode, viewer.renderer.color) == ("detail", True)
    viewer.cycle_ascii()
    assert viewer.mode_navigation.label == "ASCII fine"
    viewer.cycle_mode()
    assert viewer.renderer.mode == "detail" and viewer.renderer.color
    viewer.toggle_color()
    assert viewer.renderer.mode == "detail" and not viewer.renderer.color
    viewer.cycle_mode()
    assert viewer.renderer.mode == "shape" and not viewer.renderer.color


def test_ascii_inversion_does_not_reset_ramp_or_remembered_artwork():
    viewer, _ = _dummy_navigation_viewer(TextRenderer(mode="shape", color=False))
    viewer.cycle_ascii()
    viewer.cycle_ascii()
    ramp = viewer.renderer.ramp
    viewer.change_renderer(invert=True)
    assert viewer.mode_navigation.label == "ASCII fine"
    assert viewer.renderer.ramp == ramp and viewer.renderer.invert
    assert not viewer.mode_navigation.color_enabled
    assert viewer.mode_navigation.color_mode == "shape"
    viewer.cycle_mode()
    assert viewer.renderer.mode == "shape" and not viewer.renderer.color
    assert viewer.renderer.invert




def test_invert_strict_ascii_does_not_overwrite_custom_artwork_ramp():
    viewer, _ = _dummy_navigation_viewer(TextRenderer(
        mode="simple", color=False, ramp=" .#", invert=False))
    viewer.cycle_ascii()
    viewer.cycle_ascii()
    assert viewer.mode_navigation.label == "ASCII fine"
    viewer.change_renderer(invert=True)
    assert viewer.mode_navigation.label == "ASCII fine"
    viewer.cycle_mode()
    assert viewer.renderer.mode == "simple" and not viewer.renderer.color
    assert viewer.renderer.ramp == " .#" and viewer.renderer.invert

def test_b_preview_restores_mono_art_and_strict_ascii_without_losing_preference():
    viewer, _ = _dummy_navigation_viewer(TextRenderer(mode="shape", color=False))
    viewer.toggle_blocks()
    assert viewer.renderer.mode == "blocks" and viewer.renderer.color
    viewer.toggle_blocks()
    assert viewer.renderer.mode == "shape" and not viewer.renderer.color
    assert viewer.mode_navigation.family == "color" and not viewer.mode_navigation.color_enabled
    viewer.cycle_ascii()
    viewer.cycle_ascii()
    before = viewer.renderer.ramp
    viewer.toggle_blocks()
    assert viewer.renderer.mode == "blocks" and viewer.renderer.color
    viewer.toggle_blocks()
    assert viewer.renderer.mode == "simple" and not viewer.renderer.color
    assert viewer.renderer.ramp == before
    assert viewer.mode_navigation.label == "ASCII fine"
    viewer.cycle_mode()
    assert viewer.renderer.mode == "shape" and not viewer.renderer.color
    viewer.toggle_blocks()
    viewer.cycle_mode()  # Abandon the temporary b preview and cycle normally.
    assert viewer.renderer.mode == "simple" and not viewer.renderer.color
    viewer.toggle_blocks()
    assert viewer.renderer.mode == "blocks" and viewer.renderer.color
    viewer.toggle_blocks()
    assert viewer.renderer.mode == "simple" and not viewer.renderer.color




def test_t_during_temporary_blocks_uses_underlying_artwork_preference():
    viewer, _ = _dummy_navigation_viewer(TextRenderer(mode="shape", color=False))
    viewer.toggle_blocks()
    viewer.toggle_color()
    assert viewer.renderer.mode == "shape" and viewer.renderer.color
    assert viewer._blocks_restore is None
    viewer.cycle_ascii()
    viewer.cycle_ascii()
    assert viewer.mode_navigation.label == "ASCII fine"
    viewer.toggle_blocks()
    assert viewer.renderer.mode == "blocks"
    viewer.toggle_color()
    assert viewer.mode_navigation.label == "ASCII fine"
    assert viewer.renderer.mode == "simple" and not viewer.renderer.color
    assert not viewer.mode_navigation.color_enabled
    viewer.cycle_mode()
    assert viewer.renderer.mode == "shape" and not viewer.renderer.color

def test_monochrome_frozen_selection_preview_has_no_colors_or_source_reads(monkeypatch):
    from ttwhy.viewport import Viewport
    pixels = np.linspace(0, 1, 20 * 32, dtype=np.float32).reshape(20, 32)
    initial = TextRenderer(mode="detail", color=False)
    frame = initial.render(pixels, rows=5, columns=8)

    class Session:
        info = SimpleNamespace(channels=("A",), shape=(20, 32), is_rgb=False)

        def _select_channels(self, channels):
            return tuple(channels)

        def prepare_styles(self, *args, **kwargs):
            from ttwhy import ChannelStyle, IntensityWindow
            return (ChannelStyle(IntensityWindow(0, 1)),)

        def render(self, samples, *, styles, renderer, rows, columns):
            return renderer.render(pixels, rows=rows, columns=columns)

        def sample(self, *args, **kwargs):
            raise AssertionError("Frozen selection must not read source pixels")

    viewer = Viewer(Session(), renderer=initial)
    viewer.application = SimpleNamespace(
        layout=SimpleNamespace(focus=lambda _: None), invalidate=lambda: None)
    viewer.canvas_size = (8, 5)
    viewer.viewport = Viewport(0, 0, 32, 20)
    viewer.frame = frame
    viewer.cell_lines = terminal_lines(frame)
    viewer.generation = viewer.displayed_generation = 1
    viewer.samples = SimpleNamespace(plan=SimpleNamespace(channels=(0,)))
    viewer.windows = [None]
    monkeypatch.setattr("ttwhy.annotation_identity.random_name", lambda: "quiet-small-otter")
    clip = viewer.clipping
    clip.start()
    clip.selection.start_drag(1, 1)
    clip.selection.drag(4, 3, finish=True)
    before = clip.selection.snapshots()
    clip.cycle_preview("color")
    assert clip.preview_modes.color_mode == "shape"
    assert clip.selection.frame.foreground is None
    clip.cycle_preview("color")
    assert clip.preview_modes.color_mode == "simple"
    assert clip.selection.frame.foreground is None
    clip.cycle_preview("color")
    assert clip.preview_modes.color_mode == "detail"
    assert clip.selection.frame.foreground is None
    clip.cycle_preview("ascii")
    assert clip.ascii_preview and clip.selection.frame.to_text().isascii()
    clip.cycle_preview("color")
    assert clip.selection.frame is frame
    assert clip.selection.snapshots() == before
    assert viewer.renderer is initial and not viewer.renderer.color
    clip.end()


@pytest.mark.parametrize("kind", ("rectangle", "polygon"))
def test_selection_copy_save_key_pairs_and_render_keys(kind):
    calls = []

    class Clip:
        selection = SimpleNamespace(kind=kind)

        def submit(self, *, ansi=False):
            calls.append(("copy", "ansi" if ansi else "text"))

        def save(self, *, ansi=False, image=False, geojson=False):
            mode = "json" if geojson else "png" if image else "ansi" if ansi else "text"
            calls.append(("save", mode))

        def submit_image(self):
            calls.append(("copy", "png"))

        def submit_geojson(self):
            calls.append(("copy", "json"))

        def cycle_preview(self, family):
            calls.append(("mode", family))

    viewer = SimpleNamespace(editing=None, selecting=True, clipping=Clip())
    bindings = create_bindings(viewer)
    for key in ("y", "Y", "i", "I", "t", "T", "j", "J", "m", "M"):
        active = [item for item in bindings.bindings
                  if item.keys == (key,) and item.filter()]
        assert len(active) == 1, (key, active)
        active[0].handler(None)
    assert calls == [
        ("copy", "text"), ("save", "text"),
        ("copy", "png"), ("save", "png"),
        ("copy", "ansi"), ("save", "ansi"),
        ("copy", "json"), ("save", "json"),
        ("mode", "color"), ("mode", "ascii"),
    ]
    for retired in ("w", "W", "P"):
        assert not any(item.keys == (retired,) and item.filter()
                       for item in bindings.bindings)


def test_frozen_multiselection_cycles_without_sampling_or_moving_areas(monkeypatch):
    from ttwhy.viewport import Viewport

    data = np.linspace(0, 1, 160 * 240).reshape(160, 240)
    renderer = TextRenderer(mode="blocks")
    frame = renderer.render(data, rows=8, columns=12)

    class Session:
        info = SimpleNamespace(channels=("A",), shape=(160, 240), is_rgb=False)

        def _select_channels(self, channels):
            return tuple(channels)

        def prepare_styles(self, *args, **kwargs):
            from ttwhy import ChannelStyle, IntensityWindow
            return (ChannelStyle(IntensityWindow(0, 1)),)

        def render(self, samples, *, styles, renderer, rows, columns):
            return renderer.render(data, rows=rows, columns=columns)

        def sample(self, *args, **kwargs):
            raise AssertionError("No image source reads allowed while selecting")

    viewer = Viewer(Session(), renderer=renderer)
    viewer.application = SimpleNamespace(
        layout=SimpleNamespace(focus=lambda _: None), invalidate=lambda: None,
    )
    viewer.canvas_size = (12, 8)
    viewer.viewport = Viewport(0, 0, 240, 160)
    viewer.frame = frame
    viewer.cell_lines = terminal_lines(frame)
    viewer.generation = viewer.displayed_generation = 1
    viewer.samples = SimpleNamespace(plan=SimpleNamespace(channels=(0,)))
    viewer.windows = [None]
    names = iter(("quiet-small-otter", "brisk-amber-owl"))
    monkeypatch.setattr("ttwhy.annotation_identity.random_name", lambda: next(names))

    clip = viewer.clipping
    clip.start()
    clip.selection.start_drag(1, 1)
    clip.selection.drag(3, 4, finish=True)
    clip.add_area("polygon")
    for x, y in ((6, 1), (10, 1), (9, 5), (6, 1)):
        clip.selection.start_drag(x, y)
        clip.selection.drag(x, y, finish=True)
    before = clip.selection.snapshots()
    assert len(before) == 2
    clip.cycle_preview("ascii")
    assert clip.ascii_preview and clip.preview_modes.label == "ASCII classic"
    assert clip.selection.frame.to_text().isascii()
    clip.cycle_preview("ascii")
    assert clip.preview_modes.label == "ASCII fine"
    clip.cycle_preview("color")
    assert not clip.ascii_preview
    assert clip.selection.frame is frame
    clip.cycle_preview("color")
    assert clip.preview_modes.label == "detail"
    clip.cycle_preview("ascii")
    assert clip.preview_modes.label == "ASCII fine"
    assert clip.selection.snapshots() == before
    assert viewer.frame is frame
    assert viewer.viewport == Viewport(0, 0, 240, 160)
    clip.end()
    assert viewer.renderer.mode == "blocks"


@pytest.mark.asyncio
async def test_real_terminal_app_m_M_switches_and_selection_saves(tmp_path, monkeypatch):
    """Exercise actual prompt_toolkit input dispatch with a file-free image session."""
    import asyncio
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from ttwhy.intensity_window import IntensityWindow
    from ttwhy.compositing import ChannelStyle
    from ttwhy.viewport import Viewport

    names = iter(("quiet-small-otter", "brisk-amber-owl"))
    monkeypatch.setattr("ttwhy.annotation_identity.random_name", lambda: next(names))

    class Session:
        info = SimpleNamespace(name="Demo", channels=("A",), shape=(96, 144),
                               is_rgb=False, pixel_aspect=1.0, pixel_size=None)
        read_calls = 0

        def _select_channels(self, channels):
            return tuple(channels)

        def fit(self, *, columns, rows, renderer, channels):
            return Viewport.fit(self.info.shape, columns=columns, rows=rows), False

        def plan(self, viewport, *, rows, columns, channels,
                 allow_sampled_overview=False):
            return SimpleNamespace(viewport=viewport, channels=tuple(channels),
                                   sample_shape=(rows, columns), sampled_overview=None,
                                   level=SimpleNamespace(index=0, approximate=False))

        def sample(self, plan):
            self.read_calls += 1
            rows, columns = plan.sample_shape
            source = np.linspace(0, 1, rows * columns).reshape(1, rows, columns)
            return SimpleNamespace(values=source, plan=plan, is_rgb=False)

        def prepare_styles(self, samples, *, windows):
            return (ChannelStyle(IntensityWindow(0, 1)),)

        def render(self, samples, *, styles, renderer, rows, columns):
            return renderer.render(samples.values[0], rows=rows, columns=columns)

        def clear_cache(self):
            pass

    class SizedOutput(DummyOutput):
        def get_size(self):
            return Size(rows=24, columns=85)

    async def wait_for(predicate, task):
        for _ in range(500):
            assert not task.done(), "Viewer exited before expected state"
            if predicate():
                return
            await asyncio.sleep(.01)
        raise AssertionError("Viewer did not reach expected state")

    session = Session()
    ansi_path = tmp_path / "clip.ansi"
    txt_path = tmp_path / "clip.txt"
    with create_pipe_input() as pipe:
        viewer = Viewer(session, renderer=TextRenderer(mode="blocks"),
                        clipboard="off", output_ansi_clip=ansi_path,
                        output_text_clip=txt_path)
        task = asyncio.create_task(viewer.run(input=pipe, output=SizedOutput()))
        try:
            await wait_for(lambda: viewer.frame is not None and not viewer.busy, task)

            async def send(key, condition):
                pipe.send_text(key)
                await wait_for(condition, task)

            await send("M", lambda: viewer.mode_navigation.label == "ASCII classic"
                       and not viewer.busy)
            assert "\x1b" not in viewer.frame.to_text(ansi=True)
            await send("M", lambda: viewer.mode_navigation.label == "ASCII fine"
                       and not viewer.busy)
            await send("m", lambda: viewer.renderer.mode == "blocks"
                       and viewer.renderer.color and not viewer.busy)
            await send("m", lambda: viewer.renderer.mode == "detail"
                       and viewer.renderer.color and not viewer.busy)
            await send("s", lambda: viewer.selecting)
            selection = viewer.clipping.selection
            selection.start_drag(5, 5)
            selection.drag(15, 12, finish=True)
            existing_id = selection.active.identity.id
            reads = session.read_calls
            await send("M", lambda: viewer.clipping.preview_modes.label == "ASCII fine")
            await send("M", lambda: viewer.clipping.preview_modes.label == "ASCII bold")
            await send("T", lambda: ansi_path.exists() and not viewer.clipping.busy)
            assert "\x1b" not in ansi_path.read_text() and ansi_path.read_text().isascii()
            await send("Y", lambda: txt_path.exists() and not viewer.clipping.busy)
            assert ansi_path.read_bytes() == txt_path.read_bytes()
            await send("m", lambda: viewer.clipping.preview_modes.family == "color")
            assert viewer.clipping.preview_modes.color_mode == "detail"
            await send("m", lambda: viewer.clipping.preview_modes.color_mode == "shape")
            assert selection.active.identity.id == existing_id
            assert session.read_calls == reads  # No re-read while the selection is frozen.
        finally:
            if not task.done():
                pipe.send_bytes(b"\x11")
            await asyncio.wait_for(task, 5)

"""Explicit clip delivery: UTF-8 files and write-only clipboard requests.

No clipboard reads, shell evaluation, or implicit files. Native clipboard helpers
are optional conveniences; OSC 52 needs only Python and terminal cooperation.
"""

from __future__ import annotations

import base64
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

CLIPBOARD_MODES = ("auto", "osc52", "off")
CLIP_SOURCES = ("auto", "screen")
# Conservative application limit, not a claim about every terminal's capacity.
MAX_OSC52_BYTES = 65_536


def validate_clip_path(path: Path, *, source: Path | None = None,
                       overwrite: bool = True) -> None:
    """Check again at publication time; never follow an output symlink or FIFO."""
    if source is not None and (path.resolve() == source.resolve() or (
        path.exists() and source.exists() and os.path.samefile(path, source)
    )):
        raise ValueError("The clip destination must not be the input image")
    if path.is_symlink():
        raise ValueError("The clip destination must not be a symlink")
    if path.exists():
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("The clip destination must be a regular file")
        if not overwrite:
            raise FileExistsError(f"Clip destination already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")


def write_text_clip(path: Path, text: str, *, source: Path | None = None,
                    overwrite: bool = True) -> None:
    """Atomically publish one clip; preserve rectangular spaces and UTF-8 characters."""
    validate_clip_path(path, source=source, overwrite=overwrite)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            if not text.endswith("\n"):
                stream.write("\n")
        validate_clip_path(path, source=source, overwrite=overwrite)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def osc52_sequence(text: str) -> str:
    """Request the terminal's CLIPBOARD selection; never query or truncate it."""
    payload = text.encode("utf-8")
    if not payload:
        raise ValueError("Nothing to copy; the clipboard was not changed")
    if len(payload) > MAX_OSC52_BYTES:
        raise ValueError("Clip exceeds the 64 KiB OSC 52 limit; save it to a file instead")
    return "\x1b]52;c;" + base64.b64encode(payload).decode("ascii") + "\x07"


@dataclass(frozen=True)
class ClipboardDelivery:
    message: str
    sequence: str = ""


def clipboard_delivery(text: str, mode: str = "auto") -> ClipboardDelivery:
    """Run optional local helpers off the UI thread, or prepare an OSC 52 write.

    The caller emits sequence on the UI's own Output, between repaints. 'Sent'
    is intentionally not 'copied': OSC 52 write requests have no success reply.
    SSH and multiplexers use the terminal route, not a remote desktop clipboard.
    """
    if mode not in CLIPBOARD_MODES:
        raise ValueError(f"clipboard must be one of {CLIPBOARD_MODES}")
    if mode == "off":
        return ClipboardDelivery("Clipboard disabled")
    if not text:
        raise ValueError("Nothing to copy; the clipboard was not changed")
    remote = any(os.environ.get(key) for key in
                 ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "TMUX", "STY"))
    commands: list[tuple[list[str], str]] = []
    if mode == "auto" and not remote:
        if sys.platform == "darwin":
            commands.append((["pbcopy"], "utf-8"))
        elif sys.platform == "win32":
            commands.append((["clip.exe"], "utf-16le"))
        else:
            if os.environ.get("WAYLAND_DISPLAY"):
                commands.append((["wl-copy", "--type", "text/plain;charset=utf-8"], "utf-8"))
            if os.environ.get("DISPLAY"):
                commands += [(["xclip", "-selection", "clipboard", "-in"], "utf-8"),
                             (["xsel", "--clipboard", "--input"], "utf-8")]
    for command, encoding in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        env = dict(os.environ)
        if command[0] == "pbcopy":
            env["LC_ALL"] = "en_US.UTF-8"
        try:
            subprocess.run([executable, *command[1:]], input=text.encode(encoding),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=2, check=True, env=env)
        except (OSError, subprocess.SubprocessError):
            continue
        return ClipboardDelivery(f"Copied via {command[0]}")
    if os.environ.get("TERM", "").lower() in ("dumb", "unknown"):
        raise RuntimeError("No usable clipboard helper; use Save or --output-text-clip")
    return ClipboardDelivery("Sent OSC 52; terminal permission required", osc52_sequence(text))

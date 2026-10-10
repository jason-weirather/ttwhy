"""Optional native PNG clipboard delivery; terminals/OSC 52 carry text, not images.

No shell, network access, or clipboard reads. On macOS a short-lived private PNG
file is needed by AppleScript and removed immediately after clipboard delivery.
On unsupported/remote sessions, callers can offer a normal PNG file export.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from .text_clip import CLIPBOARD_MODES, ClipboardDelivery, validate_clip_path


def write_png_clip(path: Path, data: bytes, *, source: Path | None = None,
                   overwrite: bool = True) -> None:
    """Atomically publish a PNG without touching its input or following a symlink."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Expected PNG bytes")
    validate_clip_path(path, source=source, overwrite=overwrite)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        validate_clip_path(path, source=source, overwrite=overwrite)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clipboard_png(data: bytes, mode: str = "auto") -> ClipboardDelivery:
    """Write a pasteable PNG using a local OS clipboard helper when available.

    OSC 52 and SSH/tmux cannot transport PNG clipboard content portably. A
    successful native helper reports request completion, not verification that
    the receiving application will accept the paste.
    """
    if mode not in CLIPBOARD_MODES:
        raise ValueError(f"clipboard must be one of {CLIPBOARD_MODES}")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Expected PNG bytes")
    if mode == "off":
        raise RuntimeError("Clipboard disabled; press I to save a PNG instead")
    if mode == "osc52":
        raise RuntimeError("OSC 52 copies text, not images; press I to save a PNG")
    if any(os.environ.get(key) for key in
           ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "TMUX", "STY")):
        raise RuntimeError("No portable image clipboard over SSH/tmux; save a PNG and upload it")

    commands: list[list[str]] = []
    if sys.platform == "darwin":
        executable = shutil.which("osascript")
        if executable:
            temporary = None
            try:
                with NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                # Positional argv, not AppleScript source string interpolation.
                subprocess.run([
                    executable, "-e", "on run argv", "-e",
                    "set the clipboard to (read (POSIX file (item 1 of argv)) as «class PNGf»)",
                    "-e", "end run", str(temporary),
                ], check=True, timeout=5, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
                return ClipboardDelivery("Copied PNG via macOS clipboard")
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    elif sys.platform == "win32":
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if executable:
            temporary = None
            try:
                with NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                command = (
                    'Add-Type -AssemblyName System.Windows.Forms; '
                    'Add-Type -AssemblyName System.Drawing; '
                    '$img = [System.Drawing.Image]::FromFile($args[0]); '
                    'try { [System.Windows.Forms.Clipboard]::SetImage($img) } '
                    'finally { $img.Dispose() }'
                )
                subprocess.run([executable, "-NoProfile", "-NonInteractive", "-Sta",
                                "-Command", command, str(temporary)],
                               check=True, timeout=5, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return ClipboardDelivery("Copied PNG via Windows clipboard")
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    else:
        if os.environ.get("WAYLAND_DISPLAY"):
            commands.append(["wl-copy", "--type", "image/png"])
        if os.environ.get("DISPLAY"):
            commands.append(["xclip", "-selection", "clipboard", "-t", "image/png", "-in"])
        for command in commands:
            executable = shutil.which(command[0])
            if executable is None:
                continue
            try:
                subprocess.run([executable, *command[1:]], input=data,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=5, check=True)
                return ClipboardDelivery(f"Copied PNG via {command[0]}")
            except (OSError, subprocess.SubprocessError):
                continue
    raise RuntimeError("No working local image clipboard helper; press I to save PNG for Discord")

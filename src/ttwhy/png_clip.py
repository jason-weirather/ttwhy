"""Portable PNG rendering of a frozen terminal-cell rectangle.

Half-block artwork uses the actual 24-bit foreground and background samples;
no ANSI escape sequences or terminal font are involved. Other text modes
are rasterized with available fonts (Braille is drawn geometrically), and
may vary from the user's terminal font. This is a shareable image, not OCR.
"""

from __future__ import annotations

from io import BytesIO

from .text_image import TextImage

CELL_WIDTH = 12
CELL_HEIGHT = 24
MAX_PNG_PIXELS = 30_000_000


def _font(height: int):
    """Use local fonts if available; never require or distribute a font file."""
    from PIL import ImageFont

    for name in ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Menlo.ttc",
                 "Consolas.ttf"):
        try:
            return ImageFont.truetype(name, size=round(height * .78))
        except (OSError, ValueError):
            continue
    return ImageFont.load_default(size=round(height * .78))


def _braille(draw, code: int, x: int, y: int, width: int, height: int,
             color: tuple[int, int, int]) -> None:
    # Unicode Braille dot bit order, arranged on a two-column, four-row grid.
    positions = ((0, 0), (0, 1), (0, 2), (1, 0),
                 (1, 1), (1, 2), (0, 3), (1, 3))
    radius_x = max(1, round(width * .11))
    radius_y = max(1, round(height * .055))
    for bit, (column, row) in enumerate(positions):
        if code & (1 << bit):
            cx = x + round((column + .5) * width / 2)
            cy = y + round((row + .5) * height / 4)
            draw.ellipse((cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y),
                         fill=color)


def render_clip_png(frame: TextImage, *, cell_width: int = CELL_WIDTH,
                    cell_height: int = CELL_HEIGHT) -> bytes:
    """Return PNG bytes of a TextImage, preserving half-block RGB at every cell.

    Only the cropped, frozen frame is used. Selection borders, UI chrome, and
    escape sequences are absent. The pixel limit prevents an accidental huge
    allocation if an embedding application supplies an enormous TextImage.
    """
    if not isinstance(frame, TextImage):
        raise TypeError("PNG clipping requires a TextImage")
    if (type(cell_width) is not int or type(cell_height) is not int or
            not 4 <= cell_width <= 128 or not 8 <= cell_height <= 256):
        raise ValueError("PNG cell width/height must be integers in 4..128 / 8..256")
    from PIL import Image, ImageDraw

    rows, columns = frame.shape
    width, height = columns * cell_width, rows * cell_height
    if width * height > MAX_PNG_PIXELS:
        raise ValueError("Selected artwork exceeds the 30 megapixel PNG export limit")
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = None
    for row in range(rows):
        y = row * cell_height
        for column in range(columns):
            x = column * cell_width
            char = str(frame.characters[row, column])
            if char == " ":
                # ttwhy's canvas is black; unpainted block cells stay black.
                continue
            foreground = ((238, 238, 238) if frame.foreground is None else
                          tuple(int(channel) for channel in frame.foreground[row, column]))
            background = ((0, 0, 0) if frame.background is None else
                          tuple(int(channel) for channel in frame.background[row, column]))
            box = (x, y, x + cell_width - 1, y + cell_height - 1)
            if char == "▀":
                mid = y + cell_height // 2
                draw.rectangle((x, y, x + cell_width - 1, mid - 1), fill=foreground)
                draw.rectangle((x, mid, x + cell_width - 1, y + cell_height - 1),
                               fill=background)
            elif char == "▄":
                mid = y + cell_height // 2
                draw.rectangle((x, y, x + cell_width - 1, mid - 1), fill=background)
                draw.rectangle((x, mid, x + cell_width - 1, y + cell_height - 1),
                               fill=foreground)
            elif char == "█":
                draw.rectangle(box, fill=foreground)
            elif char in ("▌", "▐"):
                mid = x + cell_width // 2
                draw.rectangle(box, fill=background)
                rectangle = ((x, y, mid - 1, y + cell_height - 1) if char == "▌" else
                             (mid, y, x + cell_width - 1, y + cell_height - 1))
                draw.rectangle(rectangle, fill=foreground)
            elif 0x2800 <= ord(char) <= 0x28FF:
                _braille(draw, ord(char) - 0x2800, x, y, cell_width, cell_height,
                         foreground)
            else:
                if font is None:
                    font = _font(cell_height)
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
                    draw.text((x + (cell_width - (right-left)) / 2 - left,
                               y + (cell_height - (bottom-top)) / 2 - top),
                              char, font=font, fill=foreground)
                except (UnicodeError, OSError):
                    # An unsupported font glyph must not break PNG export.
                    draw.rectangle((x + cell_width//4, y + cell_height//4,
                                    x + 3*cell_width//4, y + 3*cell_height//4),
                                   outline=foreground)
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()

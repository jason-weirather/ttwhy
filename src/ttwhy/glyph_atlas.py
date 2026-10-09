"""Small, font-independent shape descriptors for terminal character selection.

These hand-authored stroke recipes are NOT a font or a measured terminal raster.
They encode relative stroke positions; actual appearance depends on the terminal.
No font discovery, subprocess, downloaded asset, or model is used at runtime.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

# Eight rows by four columns per character. Braille dot numbers are not row-major.
SAMPLE_YX = (8, 4)
TONE_ANCHORS = 11
BRAILLE_BITS = np.array([[1, 8], [2, 16], [4, 32], [64, 128]], dtype=np.uint16)
BRAILLE_BITS.flags.writeable = False


@dataclass(frozen=True, slots=True)
class GlyphAtlas:
    """Read-only canonical masks, tonal weights, and normalized shape features."""

    characters: NDArray
    masks: NDArray
    tones: NDArray
    features: NDArray


@lru_cache(maxsize=1)
def glyph_atlas() -> GlyphAtlas:
    """Build the deterministic stroke vocabulary for shape mode."""
    height, width = SAMPLE_YX
    # Draw on an oversampled square-pixel grid, then area-average to cell samples.
    y, x = np.mgrid[:height * 4, :width * 4].astype(np.float64)
    x = (x + .5) / (width * 4)
    y = (y + .5) / (height * 4)
    chars, masks, tones = [], [], []

    def add(char, paths=(), *, tone, stroke=.10):
        mask = np.zeros_like(x)
        for path in paths:
            for (ax, ay), (bx, by) in zip(path[:-1], path[1:], strict=True):
                # In physical cell units, y spans twice the horizontal distance.
                dx, dy = bx - ax, (by - ay) * 2
                distance2 = dx * dx + dy * dy
                t = (np.zeros_like(x) if distance2 == 0 else
                     np.clip(((x - ax) * dx + (y - ay) * 2 * dy) / distance2, 0, 1))
                distance = np.hypot(x - (ax + t * dx), (y - ay) * 2 - t * dy)
                mask = np.maximum(mask, np.clip((stroke - distance) * 24 + .5, 0, 1))
        masks.append(mask.reshape(height, 4, width, 4).mean(axis=(1, 3)))
        chars.append(char)
        tones.append(tone)

    def dot(x, y):
        return [(x, y), (x, y)]

    # Tonal anchors precede structural candidates so ties are stable and quiet.
    # Tones describe visual ink density, not literal fractions of a font's pixels.
    add(' ', tone=0)
    add('.', [dot(.5, .82)], tone=.06, stroke=.07)
    add('·', [dot(.5, .50)], tone=.09, stroke=.07)
    add(':', [dot(.5, .32), dot(.5, .72)], tone=.17, stroke=.07)
    add('-', [[(.1, .5), (.9, .5)]], tone=.24)
    add('=', [[(.1, .37), (.9, .37)], [(.1, .63), (.9, .63)]], tone=.34)
    add('+', [[(.1, .5), (.9, .5)], [(.5, .2), (.5, .8)]], tone=.43)
    add('*', [[(.1, .5), (.9, .5)], [(.2, .23), (.8, .77)],
              [(.8, .23), (.2, .77)]], tone=.55)
    add('#', [[(.3, .12), (.3, .88)], [(.7, .12), (.7, .88)],
              [(.1, .37), (.9, .37)], [(.1, .63), (.9, .63)]], tone=.72)
    add('%', [[(.1, .9), (.9, .1)], [(.2, .12), (.4, .12), (.4, .35),
              (.2, .35), (.2, .12)], [(.6, .65), (.8, .65), (.8, .88),
              (.6, .88), (.6, .65)]], tone=.86)
    add('@', [[(.9, .8), (.7, .92), (.2, .9), (.1, .15), (.8, .1), (.9, .65),
              (.5, .7), (.4, .35), (.7, .3), (.7, .7)]], tone=1.0, stroke=.15)

    add(',', [[(.5, .77), (.5, .88), (.3, 1)]], tone=.10)
    add(';', [dot(.5, .32), [(.5, .7), (.5, .85), (.3, .97)]], tone=.20)
    add("'", [[(.5, .05), (.4, .26)]], tone=.07)
    add('"', [[(.3, .05), (.3, .25)], [(.7, .05), (.7, .25)]], tone=.14)
    add('_', [[(0, .91), (1, .91)]], tone=.23)
    add('^', [[(.1, .35), (.5, .08), (.9, .35)]], tone=.22)
    add('v', [[(.1, .4), (.5, .9), (.9, .4)]], tone=.30)
    add('<', [[(.85, .2), (.15, .5), (.85, .8)]], tone=.31)
    add('>', [[(.15, .2), (.85, .5), (.15, .8)]], tone=.31)
    add('|', [[(.5, .05), (.5, .95)]], tone=.31)
    add('/', [[(.05, .97), (.95, .03)]], tone=.34)
    add('\\', [[(.05, .03), (.95, .97)]], tone=.34)
    add('(', [[(.7, .03), (.35, .22), (.25, .5), (.35, .78), (.7, .97)]], tone=.33)
    add(')', [[(.3, .03), (.65, .22), (.75, .5), (.65, .78), (.3, .97)]], tone=.33)
    add('[', [[(.75, .05), (.25, .05), (.25, .95), (.75, .95)]], tone=.38)
    add(']', [[(.25, .05), (.75, .05), (.75, .95), (.25, .95)]], tone=.38)
    add('o', [[(.5, .35), (.2, .45), (.2, .8), (.5, .9), (.8, .8),
              (.8, .45), (.5, .35)]], tone=.39)
    add('O', [[(.5, .07), (.15, .2), (.15, .8), (.5, .93), (.85, .8),
              (.85, .2), (.5, .07)]], tone=.49)
    add('c', [[(.8, .4), (.4, .35), (.15, .5), (.15, .75), (.4, .9), (.8, .85)]],
        tone=.32)
    add('n', [[(.15, .9), (.15, .4), (.55, .35), (.85, .5), (.85, .9)]], tone=.42)
    add('u', [[(.15, .35), (.15, .8), (.45, .9), (.85, .85), (.85, .35)]], tone=.42)
    add('x', [[(.15, .35), (.85, .9)], [(.85, .35), (.15, .9)]], tone=.44)
    add('X', [[(.1, .08), (.9, .92)], [(.9, .08), (.1, .92)]], tone=.57)
    add('V', [[(.1, .08), (.5, .92), (.9, .08)]], tone=.48)
    add('Y', [[(.1, .08), (.5, .5), (.9, .08)], [(.5, .5), (.5, .92)]], tone=.45)
    add('T', [[(.05, .08), (.95, .08)], [(.5, .08), (.5, .92)]], tone=.43)
    add('L', [[(.2, .08), (.2, .9), (.9, .9)]], tone=.40)
    add('J', [[(.1, .08), (.8, .08), (.8, .8), (.5, .92), (.2, .8)]], tone=.45)
    add('I', [[(.15, .08), (.85, .08)], [(.5, .08), (.5, .92)],
              [(.15, .92), (.85, .92)]], tone=.48)
    add('!', [[(.5, .08), (.5, .64)], dot(.5, .9)], tone=.28)
    add('i', [dot(.5, .13), [(.3, .4), (.5, .4), (.5, .9)],
              [(.25, .9), (.75, .9)]], tone=.28)
    add('s', [[(.8, .4), (.3, .35), (.15, .5), (.8, .7), (.8, .85), (.2, .9)]],
        tone=.42)
    add('S', [[(.85, .18), (.6, .06), (.2, .15), (.15, .38), (.85, .65),
              (.8, .85), (.4, .94), (.15, .82)]], tone=.53)
    add('M', [[(.1, .92), (.1, .08), (.5, .55), (.9, .08), (.9, .92)]], tone=.75)
    add('W', [[(.1, .08), (.2, .92), (.5, .45), (.8, .92), (.9, .08)]], tone=.75)
    add('H', [[(.15, .08), (.15, .92)], [(.85, .08), (.85, .92)],
              [(.15, .5), (.85, .5)]], tone=.65)

    # Connected box/diagonal strokes are valuable for contour continuation.
    add('─', [[(0, .5), (1, .5)]], tone=.25)
    add('│', [[(.5, 0), (.5, 1)]], tone=.35)
    add('╱', [[(0, 1), (1, 0)]], tone=.36)
    add('╲', [[(0, 0), (1, 1)]], tone=.36)
    add('┌', [[(.5, 1), (.5, .5), (1, .5)]], tone=.29)
    add('┐', [[(0, .5), (.5, .5), (.5, 1)]], tone=.29)
    add('└', [[(.5, 0), (.5, .5), (1, .5)]], tone=.29)
    add('┘', [[(0, .5), (.5, .5), (.5, 0)]], tone=.29)
    add('╭', [[(.5, 1), (.5, .7), (.65, .55), (.8, .5), (1, .5)]], tone=.26)
    add('╮', [[(0, .5), (.2, .5), (.35, .55), (.5, .7), (.5, 1)]], tone=.26)
    add('╰', [[(.5, 0), (.5, .3), (.65, .45), (.8, .5), (1, .5)]], tone=.26)
    add('╯', [[(0, .5), (.2, .5), (.35, .45), (.5, .3), (.5, 0)]], tone=.26)
    add('┼', [[(0, .5), (1, .5)], [(.5, 0), (.5, 1)]], tone=.51)

    array = np.asarray(masks, dtype=np.float32)
    flat = array.reshape(len(chars), -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    features = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-7)
    result = GlyphAtlas(np.array(chars, dtype='U1'), array,
                        np.asarray(tones, dtype=np.float32), features)
    for value in (result.characters, result.masks, result.tones,
                  result.features):
        value.flags.writeable = False
    return result
